// reorder_optimized.slang
// ─────────────────────────────────────────────────────────────────────────────
// Optimized work-reordering kernel for 16×16 material-ID tiles.
// Material IDs are arbitrary uint32 values.
//
// ── Optimizations over reorder.slang ─────────────────────────────────────────
//
//   1. SoA packed layout
//      Replaces uint3 gs_data[] (12 B/elem) with separate gs_key[] and
//      gs_pos[] arrays (8 B/elem total).  Positions are packed as
//      (y << 4) | x.  Saves 33% per swap and eliminates stride-3 bank
//      conflicts in shared memory.
//
//   2. Wave-intrinsic intra-wave sort
//      The first 5 bitonic stages (strides 1–16, sorting within each
//      32-lane wave) use WaveReadLaneAt() — values stay in registers,
//      no shared memory access, no barriers.  Only the final 3 stages
//      (strides 32–128, merging across waves) use shared memory.
//
//   3. Barrier reduction: 38 → ~11
//      Phase 0 load: 0 barriers (data goes straight to registers).
//      Intra-wave sort: 0 barriers (wave-level lockstep execution).
//      Write-back: 1 barrier.
//      Cross-wave stages: 6 cross-wave stride barriers + 3 write-backs = 9.
//      Phase 2 broadcast: 1 barrier.
//      Total ≈ 11 barriers (3.5× reduction).
//
//   4. gs_key[] reuse for output indices
//      After sorting, Phase 2 overwrites gs_key[] with encoded output
//      positions (thread 0 reads each key before overwriting it).
//      Eliminates the separate gs_out_idx[256] array, saving 1024 B.
//      Total shared memory: ~2.1 KB (vs ~4.1 KB).
//
//   5. Conditional position swap
//      In cross-wave stages, gs_pos[] is loaded only when a swap is
//      actually needed, saving 2 shared-memory reads per non-swap pair.
//
// ── Requirements ─────────────────────────────────────────────────────────────
//
//   Shader Model 6.0+ (wave intrinsics: WaveGetLaneIndex, WaveReadLaneAt).
//   Wave size must be exactly 32.
//
// ── Algorithm (unchanged) ────────────────────────────────────────────────────
//
//   Phase 0   Load tile data into registers (key + packed position).
//   Phase 1a  Intra-wave bitonic sort via WaveReadLaneAt (no barriers).
//   Phase 1b  Cross-wave bitonic merge via shared memory + barriers.
//   Phase 2   Wave packing — thread 0, serial O(N).  Same encoding:
//             bit 31 = 0 → full-wave, bit 31 = 1 → overflow.
//   Phase 3   All 256 threads decode position and scatter (x, y).
//
// ── Output guarantee (identical to reorder.slang) ────────────────────────────
//
//   output[0..F-1]:   full-wave section, every 32-block is one material.
//   output[F..255]:   overflow section, sorted by material ID.
//   F = totalFullElements[0], always a multiple of 32.
//
// ── Shared memory ────────────────────────────────────────────────────────────
//
//   gs_key   256 × uint = 1024 B  (sort keys, then output indices)
//   gs_pos   256 × uint = 1024 B  (packed positions)
//   gs_total_full          4 B
//   Total               ≈ 2.1 KB
//
// ── Dispatch ─────────────────────────────────────────────────────────────────
//
//   1 group of 256 threads  →  dispatchGroups(1, 1, 1)
//
// ─────────────────────────────────────────────────────────────────────────────

static const uint TILE_W        = 16u;
static const uint TILE_H        = 16u;
static const uint TILE_ELEMENTS = 256u;
static const uint WAVE_SIZE     =  32u;
static const uint NUM_WAVES     =   8u;
static const uint OVERFLOW_FLAG = 0x80000000u;

// SoA layout — separate key and position arrays
groupshared uint gs_key[TILE_ELEMENTS];   // Phase 0–1: materialID; Phase 2–3: encoded out_idx
groupshared uint gs_pos[TILE_ELEMENTS];   // packed (y << 4) | x, preserved throughout
groupshared uint gs_total_full;           // broadcast: total full-wave elements

// ─────────────────────────────────────────────────────────────────────────────
[numthreads(256, 1, 1)]
void reorderByMaterialOpt(
    StructuredBuffer<uint>    tileData,
    RWStructuredBuffer<uint>  reorderedXY,
    RWStructuredBuffer<uint>  totalFullElements,
    uint3 groupTid : SV_GroupThreadID
)
{
    uint tid  = groupTid.x;           // 0 … 255
    uint lane = WaveGetLaneIndex();   // 0 … 31

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 0 — Load (directly into registers, no shared memory yet)
    // ══════════════════════════════════════════════════════════════════════════
    uint my_key = tileData[tid];
    uint my_pos = ((tid / TILE_W) << 4u) | (tid % TILE_W);

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 1a — Intra-wave bitonic sort (stages 1–5, strides 1–16)
    //
    // All 32 lanes within a wave execute in lockstep.  WaveReadLaneAt reads
    // the partner's register value at the moment of the call — before any
    // thread has updated its own register in this step.  No shared memory
    // access and no barriers are needed.
    // ══════════════════════════════════════════════════════════════════════════
    [unroll]
    for (uint stage = 2u; stage <= WAVE_SIZE; stage <<= 1u)
    {
        [unroll]
        for (uint stride = stage >> 1u; stride > 0u; stride >>= 1u)
        {
            uint p_lane = lane ^ stride;
            uint p_key  = WaveReadLaneAt(my_key, p_lane);
            uint p_pos  = WaveReadLaneAt(my_pos, p_lane);

            // Lower half of each pair keeps min (ascending) or max (descending).
            // Upper half keeps the opposite.  Both threads read pre-update values
            // and independently decide what to keep — no write conflict.
            bool i_am_lower = (lane & stride) == 0u;
            bool ascending  = ((tid & stage) == 0u);
            bool keep_min   = (ascending == i_am_lower);

            if (keep_min ? (p_key < my_key) : (p_key > my_key))
            {
                my_key = p_key;
                my_pos = p_pos;
            }
        }
    }

    // Flush sorted intra-wave results to shared memory for cross-wave stages
    gs_key[tid] = my_key;
    gs_pos[tid] = my_pos;
    GroupMemoryBarrierWithGroupSync();

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 1b — Cross-wave bitonic sort (stages 6–8)
    //
    // For each stage:
    //   • Cross-wave strides (≥ 32) — shared memory + barrier per stride.
    //   • Intra-wave strides (< 32)  — register + WaveReadLaneAt, no barriers.
    //
    // After all strides of a stage, results are written back and a barrier
    // ensures visibility for the next stage.
    // ══════════════════════════════════════════════════════════════════════════
    for (uint stage = WAVE_SIZE * 2u; stage <= TILE_ELEMENTS; stage <<= 1u)
    {
        // ── Cross-wave strides (shared memory) ──────────────────────────────
        for (uint stride = stage >> 1u; stride >= WAVE_SIZE; stride >>= 1u)
        {
            uint partner = tid ^ stride;
            if (tid < partner)
            {
                uint a_key = gs_key[tid];
                uint b_key = gs_key[partner];
                bool ascending = ((tid & stage) == 0u);
                bool do_swap   = ascending ? (a_key > b_key) : (a_key < b_key);
                if (do_swap)
                {
                    // Load positions only when swap is needed
                    uint a_pos = gs_pos[tid];
                    uint b_pos = gs_pos[partner];
                    gs_key[tid]     = b_key;
                    gs_key[partner] = a_key;
                    gs_pos[tid]     = b_pos;
                    gs_pos[partner] = a_pos;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // ── Intra-wave strides (registers, no barriers) ────────────────────
        my_key = gs_key[tid];
        my_pos = gs_pos[tid];

        [unroll]
        for (uint stride = WAVE_SIZE >> 1u; stride > 0u; stride >>= 1u)
        {
            uint p_lane = lane ^ stride;
            uint p_key  = WaveReadLaneAt(my_key, p_lane);
            uint p_pos  = WaveReadLaneAt(my_pos, p_lane);

            bool i_am_lower = (lane & stride) == 0u;
            bool ascending  = ((tid & stage) == 0u);
            bool keep_min   = (ascending == i_am_lower);

            if (keep_min ? (p_key < my_key) : (p_key > my_key))
            {
                my_key = p_key;
                my_pos = p_pos;
            }
        }

        // Write back for next stage
        gs_key[tid] = my_key;
        gs_pos[tid] = my_pos;
        GroupMemoryBarrierWithGroupSync();
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 2 — Wave packing  (thread 0 only, serial)
    //
    // Identical algorithm to reorder.slang.  gs_key[] is overwritten in-place
    // with encoded output positions — thread 0 reads each key before writing.
    // gs_pos[] is untouched (still holds packed x,y for Phase 3).
    // ══════════════════════════════════════════════════════════════════════════
    if (tid == 0u)
    {
        uint full_cnt = 0u;
        uint ovfl_cnt = 0u;
        uint i        = 0u;
        uint cur_mat  = gs_key[0];       // read before first overwrite

        while (i < TILE_ELEMENTS)
        {
            // Find end of this run (cur_mat already in register)
            uint run_end = i + 1u;
            while (run_end < TILE_ELEMENTS)
            {
                uint next_mat = gs_key[run_end];   // still original key here
                if (next_mat != cur_mat) break;
                run_end++;
            }

            uint run_len    = run_end - i;
            uint full_elems = (run_len / WAVE_SIZE) * WAVE_SIZE;

            // Assign full-wave positions (no flag)
            for (uint j = 0u; j < full_elems; j++)
            {
                gs_key[i + j] = full_cnt;          // overwrite key → out_idx
                full_cnt++;
            }

            // Assign overflow positions (bit-31 flag)
            for (uint j = full_elems; j < run_len; j++)
            {
                gs_key[i + j] = OVERFLOW_FLAG | ovfl_cnt;
                ovfl_cnt++;
            }

            // Save next material key before it can be overwritten next iteration
            if (run_end < TILE_ELEMENTS)
                cur_mat = gs_key[run_end];
            i = run_end;
        }

        gs_total_full = full_cnt;
    }
    GroupMemoryBarrierWithGroupSync();

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 3 — Decode and scatter
    //
    // gs_key[tid] now holds the encoded output index.
    // gs_pos[tid] still holds the packed (y, x) position.
    // ══════════════════════════════════════════════════════════════════════════
    uint encoded   = gs_key[tid];
    uint packed_xy = gs_pos[tid];

    uint is_ovfl = encoded >> 31u;
    uint out_pos = (encoded & ~OVERFLOW_FLAG) + is_ovfl * gs_total_full;

    uint x = packed_xy & 0xFu;
    uint y = (packed_xy >> 4u) & 0xFu;
    reorderedXY[out_pos] = (y << 16u) | x;

    if (tid == 0u)
        totalFullElements[0] = gs_total_full;
}
