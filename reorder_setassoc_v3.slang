// reorder_setassoc_v3.slang
// ─────────────────────────────────────────────────────────────────────────────
// Hash-table-free wave-merge reorder kernel for 16×16 material-ID tiles.
//
// Eliminates the hash table and all shared-memory atomics from v2.
// Cross-wave coordination uses deterministic per-wave summaries + serial merge.
//
// Improvements over reorder_setassoc_v2.slang:
//
//   1. No hash table — no CAS, no AtomicAdd.
//      Each wave discovers materials purely via ballot, writes (key, count)
//      pairs to a designated per-wave region of shared memory.
//
//   2. Fused Phase 0 + Phase 1 — no initialization barrier needed.
//      Since there's no hash table to clear, per-wave writes go directly to
//      pre-allocated regions.  Saves one barrier (2 total vs 3 in v2).
//
//   3. Deterministic merge (Phase 2, thread 0).
//      Scans per-wave summaries, deduplicates into a global material list,
//      computes full/overflow prefix sums.  ~300 operations for ~6 materials
//      across 8 waves.
//
//   4. On-the-fly global rank computation (Phase 3).
//      Each thread finds its global material slot and computes its global rank
//      by scanning prior waves' per-wave data.  No precomputed lookup tables.
//
// ── Shared memory ────────────────────────────────────────────────────────────
//
//   gs_wave_keys[256]       1024 B   per-wave unique material keys
//   gs_wave_counts[256]     1024 B   per-wave material counts
//   gs_wave_num[8]            32 B   unique material count per wave
//   gs_global_keys[32]       128 B   global material IDs
//   gs_global_count[32]      128 B   global material counts
//   gs_global_full_base[32]  128 B   full-wave base per material
//   gs_global_ovfl_base[32]  128 B   overflow base per material
//   gs_total_full              4 B   broadcast
//   gs_num_global              4 B   broadcast
//   Total                 ≈ 2.6 KB   (vs 516 B v2, 5.1 KB v1)
//
// ── Barriers ─────────────────────────────────────────────────────────────────
//
//   2 barriers (vs 3 in v2, v1)
//
// ── Trade-offs ───────────────────────────────────────────────────────────────
//
//   + Zero shared-memory atomics (no CAS, no AtomicAdd)
//   + One fewer barrier than v2
//   + Fully deterministic execution
//   − More shared memory (2.6 KB vs 516 B)
//   − Serial merge step (~300 ops by thread 0)
//   − Per-thread prior-wave scan in Phase 3
//
// ─────────────────────────────────────────────────────────────────────────────

static const uint TILE_W        = 16u;
static const uint TILE_H        = 16u;
static const uint TILE_ELEMENTS = 256u;
static const uint WAVE_SIZE     =  32u;
static const uint NUM_WAVES     = TILE_ELEMENTS / WAVE_SIZE;  // 8
static const uint MAX_MATS      =  32u;   // max unique materials per wave (and globally)

// ── Shared memory ────────────────────────────────────────────────────────────

// Per-wave discovery results (each wave writes to its own region)
groupshared uint gs_wave_keys[NUM_WAVES * MAX_MATS];     // per-wave unique material keys
groupshared uint gs_wave_counts[NUM_WAVES * MAX_MATS];   // per-wave material counts
groupshared uint gs_wave_num[NUM_WAVES];                  // # unique materials per wave

// Global merged material data (written by thread 0 in Phase 2)
groupshared uint gs_global_keys[MAX_MATS];
groupshared uint gs_global_count[MAX_MATS];
groupshared uint gs_global_full_base[MAX_MATS];
groupshared uint gs_global_ovfl_base[MAX_MATS];
groupshared uint gs_total_full;
groupshared uint gs_num_global;

// ─────────────────────────────────────────────────────────────────────────────
[numthreads(256, 1, 1)]
void reorderByMaterialWaveMerge(
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
    // Phase 1 — Wave-local discovery
    //
    // Each wave independently discovers its unique materials using iterative
    // ballot.  No shared memory needs to be initialized beforehand — each wave
    // writes to its own designated region gs_wave_*[wave_id * MAX_MATS + ...].
    //
    // Per-lane outputs:
    //   my_wave_rank      — this lane's rank within its wave for its material
    //   my_local_mat_idx  — 0-based index of this lane's material within its
    //                       wave's unique-material list
    // ══════════════════════════════════════════════════════════════════════════

    uint remaining       = 0xFFFFFFFFu;  // all 32 lanes active
    bool assigned        = false;
    uint my_wave_rank    = 0u;
    uint my_local_mat_idx = 0u;
    uint num_unique      = 0u;

    [loop]
    for (uint iter = 0u; iter < WAVE_SIZE && remaining != 0u; iter++)
    {
        // Pick the key from the lowest remaining lane
        uint src_lane      = firstbitlow(remaining);
        uint candidate_key = WaveReadLaneAt(my_key, src_lane);

        // Which remaining lanes have this key?
        uint match_mask = WaveActiveBallot(my_key == candidate_key).x;
        match_mask &= remaining;

        uint match_count = countbits(match_mask);

        // Lanes matching this material: record wave-local rank
        if (!assigned && my_key == candidate_key)
        {
            uint lane_bit    = 1u << lane;
            uint prefix_mask = match_mask & (lane_bit - 1u);
            my_wave_rank     = countbits(prefix_mask);
            my_local_mat_idx = iter;
            assigned         = true;
        }

        // Representative (src_lane) writes per-wave summary
        if (lane == src_lane)
        {
            uint base = wave_id * MAX_MATS;
            gs_wave_keys[base + iter]   = candidate_key;
            gs_wave_counts[base + iter] = match_count;
        }

        num_unique = iter + 1u;
        remaining &= ~match_mask;
    }

    // One lane per wave writes the unique-material count
    if (lane == 0u)
        gs_wave_num[wave_id] = num_unique;

    GroupMemoryBarrierWithGroupSync();   // ── Barrier 1 ──

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 2 — Serial merge + allocation  (thread 0 only)
    //
    // Scans all per-wave summaries, builds a deduplicated global material list
    // with total counts, then computes full/overflow prefix sums.
    //
    // Total work: ~8 waves × ~6 mats/wave = ~48 entries to merge against
    // ~6 unique globals ⇒ roughly ~300 operations.  Negligible at GPU clocks.
    // ══════════════════════════════════════════════════════════════════════════

    if (tid == 0u)
    {
        uint ng = 0u;

        for (uint w = 0u; w < NUM_WAVES; w++)
        {
            uint wbase = w * MAX_MATS;
            uint wnum  = gs_wave_num[w];

            for (uint m = 0u; m < wnum; m++)
            {
                uint key = gs_wave_keys[wbase + m];
                uint cnt = gs_wave_counts[wbase + m];

                // Linear search in global list
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
                    // New material — append to global list
                    gs_global_keys[ng]   = key;
                    gs_global_count[ng]  = cnt;
                    ng++;
                }
                else
                {
                    gs_global_count[gslot] += cnt;
                }
            }
        }

        gs_num_global = ng;

        // Prefix sums for full-wave / overflow allocation
        uint full_cnt = 0u;
        uint ovfl_cnt = 0u;

        for (uint g = 0u; g < ng; g++)
        {
            uint cnt       = gs_global_count[g];
            uint full_elems = (cnt / WAVE_SIZE) * WAVE_SIZE;
            uint ovfl_elems = cnt - full_elems;

            gs_global_full_base[g] = full_cnt;
            gs_global_ovfl_base[g] = ovfl_cnt;

            full_cnt += full_elems;
            ovfl_cnt += ovfl_elems;
        }

        gs_total_full = full_cnt;
    }

    GroupMemoryBarrierWithGroupSync();   // ── Barrier 2 ──

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 3 — Scatter
    //
    // Each thread:
    //   1. Finds its global material slot (linear scan of gs_global_keys).
    //   2. Computes its global rank by scanning prior waves' per-wave data
    //      for the same material, summing their counts, then adding its
    //      wave-local rank.
    //   3. Looks up allocation (full_base / ovfl_base) and writes output.
    // ══════════════════════════════════════════════════════════════════════════

    // 1. Find global material slot for my material
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

    // 2. Compute global rank: sum counts for my material in waves 0..wave_id-1
    uint prior_count = 0u;

    for (uint w = 0u; w < wave_id; w++)
    {
        uint wbase = w * MAX_MATS;
        uint wnum  = gs_wave_num[w];

        for (uint m = 0u; m < wnum; m++)
        {
            if (gs_wave_keys[wbase + m] == my_key)
            {
                prior_count += gs_wave_counts[wbase + m];
                break;
            }
        }
    }

    uint global_rank = prior_count + my_wave_rank;

    // 3. Look up allocation and compute output position
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
