// reorder_setassoc_v4.slang
// ─────────────────────────────────────────────────────────────────────────────
// 2-barrier wave-merge reorder with parallel merge + precomputed bases.
//
// Same per-wave ballot discovery as v3 (no hash table, no init barrier),
// but fixes v3's two main bottlenecks:
//
//   1. Parallel wave-base computation in Phase 2.
//      After serial dedup by lane 0 (~300 ops), all 32 lanes of the first
//      wave collaborate: each lane handles one global material, scanning
//      8 waves to build cumulative counts.  ~48 ops/lane vs ~300 serial.
//
//   2. Precomputed cumulative bases (gs_wave_cumul).
//      Phase 3 scatter uses a single shared-memory read per thread to get
//      its global rank base, instead of v3's per-thread prior-wave scan
//      (~42 reads per thread × 256 threads = ~10K divergent reads).
//
//   3. Wave prefix sum for allocation (WavePrefixSum + WaveActiveSum).
//      Parallel across 32 lanes instead of serial loop.
//
// ── Barriers ─────────────────────────────────────────────────────────────────
//
//   2 barriers (same as v3, vs 3 in v2)
//
// ── Shared memory ────────────────────────────────────────────────────────────
//
//   gs_wave_keys[256]       1024 B   per-wave unique material keys
//   gs_wave_counts[256]     1024 B   per-wave material counts
//   gs_wave_num[8]            32 B   unique material count per wave
//   gs_wave_cumul[256]      1024 B   cumulative count per (wave, global mat)
//   gs_global_keys[32]       128 B   global unique material IDs
//   gs_global_count[32]      128 B   global total counts
//   gs_global_full_base[32]  128 B   full-wave allocation base per material
//   gs_global_ovfl_base[32]  128 B   overflow allocation base per material
//   gs_total_full              4 B
//   gs_num_global              4 B
//   Total                 ≈ 3.6 KB
//
// ── Trade-offs vs v2 ─────────────────────────────────────────────────────────
//
//   + 1 fewer barrier (2 vs 3)
//   + Zero shared-memory atomics
//   + Fully deterministic
//   − More shared memory (3.6 KB vs 516 B)
//   − Serial dedup step (~300 ops by lane 0 of first wave)
//
// ── Trade-offs vs v3 ─────────────────────────────────────────────────────────
//
//   + Parallel wave-base computation (32 lanes vs 1 thread)
//   + Precomputed bases → O(1) Phase 3 lookup vs O(wave_id × M) scan
//   + Wave prefix sum for allocation
//   − ~1 KB more shared memory (3.6 KB vs 2.6 KB)
//
// ─────────────────────────────────────────────────────────────────────────────

static const uint TILE_W        = 16u;
static const uint TILE_H        = 16u;
static const uint TILE_ELEMENTS = 256u;
static const uint WAVE_SIZE     =  32u;
static const uint NUM_WAVES     = TILE_ELEMENTS / WAVE_SIZE;  // 8
static const uint MAX_MATS      =  32u;

// ── Shared memory ────────────────────────────────────────────────────────────

// Per-wave discovery results
groupshared uint gs_wave_keys  [NUM_WAVES * MAX_MATS];
groupshared uint gs_wave_counts[NUM_WAVES * MAX_MATS];
groupshared uint gs_wave_num   [NUM_WAVES];

// Precomputed cumulative base per (wave, global material).
// gs_wave_cumul[w * MAX_MATS + g] = total count of global material g
// across waves 0 .. w-1.
groupshared uint gs_wave_cumul [NUM_WAVES * MAX_MATS];

// Global merged material data
groupshared uint gs_global_keys     [MAX_MATS];
groupshared uint gs_global_count    [MAX_MATS];
groupshared uint gs_global_full_base[MAX_MATS];
groupshared uint gs_global_ovfl_base[MAX_MATS];
groupshared uint gs_total_full;
groupshared uint gs_num_global;

// ─────────────────────────────────────────────────────────────────────────────

[numthreads(256, 1, 1)]
void reorderByMaterialV4(
    StructuredBuffer<uint>    tileData,
    RWStructuredBuffer<uint>  reorderedXY,
    RWStructuredBuffer<uint>  totalFullElements,
    uint3 groupTid : SV_GroupThreadID
)
{
    uint tid     = groupTid.x;
    uint lane    = WaveGetLaneIndex();
    uint wave_id = tid / WAVE_SIZE;

    uint my_key = tileData[tid];

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 1 — Wave-local ballot discovery + per-wave shared-memory write
    //
    // Identical to v3 Phase 1.  Each wave independently discovers its unique
    // materials via iterative ballot.  Representatives write (key, count)
    // pairs to the wave's designated region of gs_wave_keys / gs_wave_counts.
    // No prior initialization needed — each wave writes its own region.
    //
    // Per-lane register outputs:
    //   my_wave_rank      — rank within this wave for this lane's material
    //   my_local_mat_idx  — 0-based index of this material in the wave's list
    // ══════════════════════════════════════════════════════════════════════════

    uint remaining        = 0xFFFFFFFFu;
    bool assigned         = false;
    uint my_wave_rank     = 0u;
    uint my_local_mat_idx = 0u;
    uint num_unique       = 0u;

    [loop]
    for (uint iter = 0u; iter < WAVE_SIZE && remaining != 0u; iter++)
    {
        uint src_lane      = firstbitlow(remaining);
        uint candidate_key = WaveReadLaneAt(my_key, src_lane);

        uint match_mask = WaveActiveBallot(my_key == candidate_key).x;
        match_mask &= remaining;

        uint match_count = countbits(match_mask);

        if (!assigned && my_key == candidate_key)
        {
            uint lane_bit    = 1u << lane;
            uint prefix_mask = match_mask & (lane_bit - 1u);
            my_wave_rank     = countbits(prefix_mask);
            my_local_mat_idx = iter;
            assigned         = true;
        }

        if (lane == src_lane)
        {
            uint base = wave_id * MAX_MATS;
            gs_wave_keys  [base + iter] = candidate_key;
            gs_wave_counts[base + iter] = match_count;
        }

        num_unique = iter + 1u;
        remaining &= ~match_mask;
    }

    if (lane == 0u)
        gs_wave_num[wave_id] = num_unique;

    GroupMemoryBarrierWithGroupSync();   // ── Barrier 1 ──

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 2 — Merge + wave-base computation + prefix sum  (first wave only)
    //
    // 2a) Lane 0: serial dedup of per-wave entries → gs_global_keys/count.
    //     With ~8 waves × ~6 mats/wave ≈ 48 entries and ~6 unique globals,
    //     this is ~300 comparisons — fast at GPU clocks.
    //
    // 2b) All 32 lanes: each lane g (g < num_global) builds the cumulative
    //     base table gs_wave_cumul[w][g] by scanning 8 waves for material g.
    //     ~48 reads per lane, fully parallel across ~6 active lanes.
    //
    // 2c) All 32 lanes: parallel prefix sum (WavePrefixSum) for full-wave
    //     and overflow allocation bases.
    // ══════════════════════════════════════════════════════════════════════════

    if (wave_id == 0u)
    {
        // ── 2a: Serial dedup by lane 0 ──────────────────────────────────────

        if (lane == 0u)
        {
            uint ng = 0u;

            for (uint w = 0u; w < NUM_WAVES; w++)
            {
                uint wbase = w * MAX_MATS;
                uint wnum  = gs_wave_num[w];

                for (uint m = 0u; m < wnum; m++)
                {
                    uint key = gs_wave_keys  [wbase + m];
                    uint cnt = gs_wave_counts[wbase + m];

                    uint gslot = 0xFFFFFFFFu;
                    for (uint g = 0u; g < ng; g++)
                    {
                        if (gs_global_keys[g] == key)
                        {
                            gslot = g;
                            break;
                        }
                    }

                    if (gslot == 0xFFFFFFFFu)
                    {
                        gs_global_keys [ng] = key;
                        gs_global_count[ng] = cnt;
                        ng++;
                    }
                    else
                    {
                        gs_global_count[gslot] += cnt;
                    }
                }
            }

            gs_num_global = ng;
        }

        // Lane 0's writes are visible to all lanes after reconvergence
        // (wave-internal coherence — all lanes execute in lockstep).

        uint ng = gs_num_global;

        // ── 2b: Parallel wave-base computation ──────────────────────────────
        //
        // Lane g scans all 8 waves for global material g, building
        // gs_wave_cumul[w * MAX_MATS + g] = count of material g in waves
        // 0 .. w-1.

        if (lane < ng)
        {
            uint g    = lane;
            uint gkey = gs_global_keys[g];
            uint cumul = 0u;

            for (uint w = 0u; w < NUM_WAVES; w++)
            {
                gs_wave_cumul[w * MAX_MATS + g] = cumul;

                uint wbase = w * MAX_MATS;
                uint wnum  = gs_wave_num[w];

                for (uint m = 0u; m < wnum; m++)
                {
                    if (gs_wave_keys[wbase + m] == gkey)
                    {
                        cumul += gs_wave_counts[wbase + m];
                        break;
                    }
                }
            }
        }

        // ── 2c: Parallel prefix sum for full / overflow allocation ──────────

        uint my_cnt  = (lane < ng) ? gs_global_count[lane] : 0u;
        uint my_full = (my_cnt / WAVE_SIZE) * WAVE_SIZE;
        uint my_ovfl = my_cnt - my_full;

        uint full_prefix = WavePrefixSum(my_full);
        uint ovfl_prefix = WavePrefixSum(my_ovfl);

        if (lane < ng)
        {
            gs_global_full_base[lane] = full_prefix;
            gs_global_ovfl_base[lane] = ovfl_prefix;
        }

        if (lane == 0u)
            gs_total_full = WaveActiveSum(my_full);
    }

    GroupMemoryBarrierWithGroupSync();   // ── Barrier 2 ──

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 3 — Scatter
    //
    // Each thread:
    //   1. Finds its global material slot (linear scan of gs_global_keys).
    //   2. Reads precomputed cumulative base from gs_wave_cumul.
    //   3. Computes output position and writes.
    //
    // Key improvement over v3: step 2 is a single shared-memory read
    // instead of scanning prior waves (~42 reads per thread).
    // ══════════════════════════════════════════════════════════════════════════

    uint ng          = gs_num_global;
    uint global_slot = 0u;

    for (uint g = 0u; g < ng; g++)
    {
        if (gs_global_keys[g] == my_key)
        {
            global_slot = g;
            break;
        }
    }

    // Precomputed: sum of my material's counts in waves 0 .. wave_id-1
    uint wave_cumul  = gs_wave_cumul[wave_id * MAX_MATS + global_slot];
    uint global_rank = wave_cumul + my_wave_rank;

    uint count      = gs_global_count[global_slot];
    uint full_elems = (count / WAVE_SIZE) * WAVE_SIZE;
    uint full_base  = gs_global_full_base[global_slot];
    uint ovfl_base  = gs_global_ovfl_base[global_slot];
    uint total_full = gs_total_full;

    uint out_pos;
    if (global_rank < full_elems)
    {
        out_pos = full_base + global_rank;
    }
    else
    {
        out_pos = total_full + ovfl_base + (global_rank - full_elems);
    }

    reorderedXY[out_pos] = ((tid >> 4u) << 16u) | (tid & 0xFu);

    if (tid == 0u)
        totalFullElements[0] = total_full;
}
