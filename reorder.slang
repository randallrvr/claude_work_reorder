// reorder.slang
// ─────────────────────────────────────────────────────────────────────────────
// Optimal work-reordering kernel for 16×16 material-ID tiles.
// Material IDs are arbitrary uint32 values.
//
// ── Why naïve sorting is insufficient ────────────────────────────────────────
//
//   After a plain sort, elements are grouped by material ID but output is
//   assigned consecutively.  A material whose element count is not a multiple
//   of WAVE_SIZE (32) leaves a partial wave at its end.  That partial wave is
//   then filled with the first elements of the next material, producing a
//   mixed wave even when a fully-uniform wave was achievable.
//
//   Example: material A = 50 elements, material B = 206 elements.
//     Naïve sort → Wave 1 = [18 × A | 14 × B]  (mixed, avoidable)
//     This kernel → Wave 0 = [32 × A], Wave 1..7 = B runs, overflow at end.
//
// ── Algorithm ────────────────────────────────────────────────────────────────
//
//   Phase 1  Parallel bitonic sort (ascending by materialID, all 256 threads).
//            After this phase gs_data[] is sorted and same-material elements
//            are contiguous.
//
//   Phase 2  Wave packing — thread 0, serial, O(N).
//            Scans the sorted array for runs of equal materialIDs.
//            For each run of length L:
//              •  first  floor(L / WAVE_SIZE) * WAVE_SIZE  elements  → FULL-WAVE section
//              •  remaining  L % WAVE_SIZE  elements                 → OVERFLOW section
//            Output positions are written to gs_out_idx[] using a flag bit
//            to distinguish the two sections (see encoding below).
//            No atomics are used; every output position is unique by
//            construction — there are no inter-thread collisions.
//
//   Phase 3  All 256 threads decode their pre-assigned position and scatter.
//
// ── Output guarantee ─────────────────────────────────────────────────────────
//
//   Let F = totalFullElements[0] after the kernel.
//
//   output[  0 ..  F-1 ]  FULL-WAVE section
//     Every 32-element block [32k .. 32k+31] contains exactly one material ID.
//     This achieves the theoretical maximum:  Σ  floor(count_i / 32)  uniform waves.
//
//   output[ F .. 255 ]  OVERFLOW section
//     The remaining (256 − F) elements, sorted by material ID.
//     Same-material overflow elements are always adjacent, so materials with
//     large overflows may still produce uniform overflow waves.
//
//   F is always a multiple of WAVE_SIZE.
//   The number of guaranteed-uniform waves = F / WAVE_SIZE.
//
// ── Encoding of gs_out_idx ───────────────────────────────────────────────────
//
//   bit 31 == 0  →  full-wave element,  output position = bits[30:0]
//   bit 31 == 1  →  overflow element,   output position = totalFullElements
//                                                        + bits[30:0]
//
// ── Shared memory ────────────────────────────────────────────────────────────
//
//   gs_data     256 × uint3  = 3072 B   (matID, x, y)
//   gs_out_idx  256 × uint   = 1024 B   (encoded positions)
//   gs_total_full             =    4 B
//   Total                     ≈  4.1 KB  (well within hardware limits)
//
// ── Dispatch ─────────────────────────────────────────────────────────────────
//
//   1 group of 256 threads  →  dispatchGroups(1, 1, 1)
//
// ─────────────────────────────────────────────────────────────────────────────

static const uint TILE_W        = 16u;
static const uint TILE_H        = 16u;
static const uint TILE_ELEMENTS = 256u;   // must be a power of 2
static const uint WAVE_SIZE     =  32u;
static const uint NUM_WAVES     =   8u;   // TILE_ELEMENTS / WAVE_SIZE
static const uint OVERFLOW_FLAG = 0x80000000u;  // bit 31

groupshared uint3 gs_data[TILE_ELEMENTS];       // (materialID, x, y)
groupshared uint  gs_out_idx[TILE_ELEMENTS];    // encoded output positions
groupshared uint  gs_total_full;                // broadcast: #full-wave elements

// ─────────────────────────────────────────────────────────────────────────────
[numthreads(256, 1, 1)]
void reorderByMaterial(
    StructuredBuffer<uint>    tileData,           // [TILE_ELEMENTS]  arbitrary uint32 mat IDs
    RWStructuredBuffer<uint>  reorderedXY,        // [TILE_ELEMENTS]  packed (y<<16)|x  (output)
    RWStructuredBuffer<uint>  totalFullElements,  // [1]  #full-wave elements (output)
    uint3 groupTid : SV_GroupThreadID
)
{
    uint tid = groupTid.x;   // 0 … 255

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 0 — Load
    // Pack each element as (materialID, x, y) into shared memory.
    // ══════════════════════════════════════════════════════════════════════════
    gs_data[tid] = uint3(
        tileData[tid],
        tid % TILE_W,
        tid / TILE_W
    );
    GroupMemoryBarrierWithGroupSync();

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 1 — Parallel bitonic sort (ascending by materialID)
    //
    // 8 outer stages (log₂ 256), each with up to 8 inner stride sub-steps.
    // Per inner step: one barrier + one conditional compare-swap per thread pair.
    // Only the lower-index thread of each pair (tid < partner) performs the
    // swap, so there are no write races on gs_data.
    // ══════════════════════════════════════════════════════════════════════════
    for (uint stage = 2u; stage <= TILE_ELEMENTS; stage <<= 1u)
    {
        for (uint stride = stage >> 1u; stride > 0u; stride >>= 1u)
        {
            GroupMemoryBarrierWithGroupSync();

            uint partner = tid ^ stride;
            if (tid < partner)
            {
                uint3 a    = gs_data[tid];
                uint3 b    = gs_data[partner];
                bool  asc  = ((tid & stage) == 0u);
                bool  swap = asc ? (a.x > b.x) : (a.x < b.x);
                if (swap)
                {
                    gs_data[tid]     = b;
                    gs_data[partner] = a;
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 2 — Wave packing  (thread 0 only, serial)
    //
    // Walk the sorted array run by run.  For each run of L equal materialIDs:
    //
    //   full_elems = floor(L / WAVE_SIZE) * WAVE_SIZE
    //     → positions  full_cnt … full_cnt + full_elems - 1  in the FULL section
    //
    //   ovfl_elems = L % WAVE_SIZE   (always < WAVE_SIZE)
    //     → positions  ovfl_cnt … ovfl_cnt + ovfl_elems - 1  in the OVERFLOW section
    //       (stored with OVERFLOW_FLAG set in bit 31)
    //
    // Because full_cnt and ovfl_cnt are strictly monotone counters and each
    // element maps to exactly one counter value, all output positions are
    // distinct — no collisions, no atomics required.
    // ══════════════════════════════════════════════════════════════════════════
    if (tid == 0u)
    {
        uint full_cnt = 0u;
        uint ovfl_cnt = 0u;
        uint i        = 0u;

        while (i < TILE_ELEMENTS)
        {
            // ── Find the end of this material's contiguous run ────────────────
            uint run_end = i + 1u;
            while (run_end < TILE_ELEMENTS &&
                   gs_data[run_end].x == gs_data[i].x)
            {
                run_end++;
            }

            uint run_len    = run_end - i;
            uint full_elems = (run_len / WAVE_SIZE) * WAVE_SIZE;  // 0, 32, 64, …
            uint ovfl_elems = run_len - full_elems;               // 0 … 31

            // ── Assign full-wave positions (no flag) ──────────────────────────
            for (uint j = 0u; j < full_elems; j++)
            {
                gs_out_idx[i + j] = full_cnt;
                full_cnt++;
            }

            // ── Assign overflow positions (flag bit 31) ───────────────────────
            for (uint j = full_elems; j < run_len; j++)
            {
                gs_out_idx[i + j] = OVERFLOW_FLAG | ovfl_cnt;
                ovfl_cnt++;
            }

            i = run_end;
        }

        // Broadcast total full-wave element count to all threads.
        // Always a multiple of WAVE_SIZE.
        gs_total_full = full_cnt;
    }
    GroupMemoryBarrierWithGroupSync();

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 3 — Decode and scatter
    //
    // Each thread reads its pre-assigned encoded index, decodes it into a
    // concrete output position, and writes (x, y).
    //
    //   Full-wave:  out_pos = encoded                   (bit 31 = 0)
    //   Overflow:   out_pos = gs_total_full + (encoded & ~OVERFLOW_FLAG)
    //
    // Every thread writes to a unique position → no collisions.
    // ══════════════════════════════════════════════════════════════════════════
    uint encoded = gs_out_idx[tid];
    uint is_ovfl = encoded >> 31u;                         // 0 or 1
    uint out_pos = (encoded & ~OVERFLOW_FLAG)
                 + is_ovfl * gs_total_full;

    reorderedXY[out_pos] = (gs_data[tid].z << 16u) | gs_data[tid].y;

    // Thread 0 also writes the full-element count so the caller knows
    // how many guaranteed-uniform waves were produced.
    if (tid == 0u)
        totalFullElements[0] = gs_total_full;
}
