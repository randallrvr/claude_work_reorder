// reorder_setassoc_v2.slang
// ─────────────────────────────────────────────────────────────────────────────
// Optimized set-associative work-reordering kernel for 16×16 material-ID tiles.
//
// Improvements over reorder_setassoc.slang:
//
//   1. Removed gs_pos[] array — x,y derived directly from tid.
//      Saves 1 KB shared memory and eliminates a write+read pair.
//
//   2. Wave-cooperative hash-table insertion (Phase 1).
//      Within each wave, lanes with the same material are grouped via ballot.
//      Only one representative per unique material per wave performs the CAS +
//      AtomicAdd.  All lanes derive their rank from the wave-local prefix.
//      Reduces atomic contention by up to 32x.
//
//   3. Shrunk hash table from 256 → 32 slots.
//      With ≤ ~10 materials, 32 slots (load factor ≤ 0.31) is plenty.
//      Phase 0 only needs 32 threads to clear; Phase 2 scans 32 slots.
//      Saves 3.5 KB shared memory.
//
//   4. Parallel Phase 2 — single-wave prefix sum over 32 slots.
//      Instead of thread-0 serial scan, the first wave reads all 32 slots
//      and does a wave-level prefix sum in one step.  Eliminates the serial
//      O(M) bottleneck entirely for M ≤ 32.
//
// ── Shared memory ────────────────────────────────────────────────────────────
//
//   gs_ht_key[32]          128 B   hash table keys
//   gs_ht_count[32]        128 B   element counts
//   gs_ht_full_base[32]    128 B   full-wave base per material
//   gs_ht_ovfl_base[32]    128 B   overflow base per material
//   gs_total_full            4 B   broadcast
//   Total                ≈ 516 B   (vs 5.1 KB original)
//
// ── Barriers ─────────────────────────────────────────────────────────────────
//
//   3 barriers (same count, but each phase is cheaper)
//
// ─────────────────────────────────────────────────────────────────────────────

static const uint TILE_W        = 16u;
static const uint TILE_H        = 16u;
static const uint TILE_ELEMENTS = 256u;
static const uint WAVE_SIZE     =  32u;
static const uint HT_SIZE       =  32u;    // shrunk from 256
static const uint HT_EMPTY      = 0xFFFFFFFFu;

// ── Shared memory ────────────────────────────────────────────────────────────
groupshared uint gs_ht_key[HT_SIZE];
groupshared uint gs_ht_count[HT_SIZE];
groupshared uint gs_ht_full_base[HT_SIZE];
groupshared uint gs_ht_ovfl_base[HT_SIZE];
groupshared uint gs_total_full;

// ── Hash function ────────────────────────────────────────────────────────────
uint hashKey(uint key)
{
    return (key * 2654435761u) >> 27u;   // top 5 bits → [0, 31]
}

// ─────────────────────────────────────────────────────────────────────────────
[numthreads(256, 1, 1)]
void reorderByMaterialSetAssocV2(
    StructuredBuffer<uint>    tileData,
    RWStructuredBuffer<uint>  reorderedXY,
    RWStructuredBuffer<uint>  totalFullElements,
    uint3 groupTid : SV_GroupThreadID
)
{
    uint tid  = groupTid.x;
    uint lane = WaveGetLaneIndex();

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 0 — Initialize hash table
    //
    // Only 32 slots to clear.  First 32 threads handle it.
    // ══════════════════════════════════════════════════════════════════════════
    if (tid < HT_SIZE)
    {
        gs_ht_key[tid]   = HT_EMPTY;
        gs_ht_count[tid] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 1 — Wave-cooperative hash-table insert + count
    //
    // Strategy: within each wave of 32 lanes, group lanes by material using
    // iterative ballot.  For each unique material in the wave:
    //   - Count how many lanes have it          (popcount of mask)
    //   - Elect one representative              (first set bit)
    //   - Representative does CAS + AtomicAdd   (one atomic instead of ~N)
    //   - All lanes derive rank = global_base + prefix_within_wave
    //
    // This reduces atomic traffic from 256 ops to roughly 8*M (8 waves, M
    // materials), which for M=6 is ~48 atomics instead of 256.
    // ══════════════════════════════════════════════════════════════════════════
    uint my_key = tileData[tid];

    // Wave-cooperative grouping via iterative ballot
    uint my_slot;
    uint my_rank;

    // Track which lanes still need processing
    uint remaining = 0xFFFFFFFFu;  // all 32 lanes active

    // Temporary storage for this lane's result
    bool assigned = false;

    [loop]
    for (uint iter = 0u; iter < WAVE_SIZE && remaining != 0u; iter++)
    {
        // Pick the key from the lowest remaining lane
        uint src_lane = firstbitlow(remaining);
        uint candidate_key = WaveReadLaneAt(my_key, src_lane);

        // Which lanes in this wave have this key?
        uint match_mask = WaveActiveBallot(my_key == candidate_key).x;
        match_mask &= remaining;  // only consider unprocessed lanes

        uint match_count = countbits(match_mask);

        if (!assigned && my_key == candidate_key)
        {
            // Lanes matching this key: compute wave-local prefix
            // (count of set bits below our lane in match_mask)
            uint lane_bit      = 1u << lane;
            uint prefix_mask   = match_mask & (lane_bit - 1u);
            uint wave_prefix   = countbits(prefix_mask);
            bool is_rep        = (lane == src_lane);

            // Representative inserts into hash table
            uint ht_slot;
            uint global_base = 0u;

            if (is_rep)
            {
                // CAS insertion with linear probing
                uint h = hashKey(candidate_key);
                [loop]
                for (uint probe = 0u; probe < HT_SIZE; probe++)
                {
                    uint idx = (h + probe) & (HT_SIZE - 1u);
                    uint original;
                    InterlockedCompareExchange(gs_ht_key[idx], HT_EMPTY,
                                              candidate_key, original);
                    if (original == HT_EMPTY || original == candidate_key)
                    {
                        ht_slot = idx;
                        InterlockedAdd(gs_ht_count[idx], match_count, global_base);
                        break;
                    }
                }
            }

            // Broadcast slot and base from representative to all matching lanes
            ht_slot     = WaveReadLaneAt(ht_slot, src_lane);
            global_base = WaveReadLaneAt(global_base, src_lane);

            my_slot = ht_slot;
            my_rank = global_base + wave_prefix;
            assigned = true;
        }

        // Remove processed lanes
        remaining &= ~match_mask;
    }

    GroupMemoryBarrierWithGroupSync();

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 2 — Parallel wave allocation (first wave, 32 threads)
    //
    // With HT_SIZE == WAVE_SIZE == 32, the first wave reads all slots and
    // does a wave-level prefix sum — no serial loop needed.
    // ══════════════════════════════════════════════════════════════════════════
    if (tid < WAVE_SIZE)
    {
        // Each lane reads one hash-table slot
        uint key   = gs_ht_key[lane];
        uint count = (key != HT_EMPTY) ? gs_ht_count[lane] : 0u;

        uint full_elems = (count / WAVE_SIZE) * WAVE_SIZE;
        uint ovfl_elems = count - full_elems;

        // Wave-level exclusive prefix sum for full-wave positions
        uint full_prefix = 0u;
        uint ovfl_prefix = 0u;
        [unroll]
        for (uint offset = 1u; offset < WAVE_SIZE; offset <<= 1u)
        {
            uint f = WaveReadLaneAt(full_elems, (lane >= offset) ? lane - offset : lane);
            uint o = WaveReadLaneAt(ovfl_elems, (lane >= offset) ? lane - offset : lane);
            if (lane >= offset)
            {
                full_prefix += f;
                ovfl_prefix += o;
            }
            full_elems += (lane >= offset) ? f : 0u;
            ovfl_elems += (lane >= offset) ? o : 0u;
        }

        // full_prefix / ovfl_prefix are now inclusive prefix sums.
        // We need exclusive, so subtract our own contribution.
        uint my_full = (key != HT_EMPTY) ? (gs_ht_count[lane] / WAVE_SIZE) * WAVE_SIZE : 0u;
        uint my_ovfl = (key != HT_EMPTY) ? gs_ht_count[lane] - my_full : 0u;

        gs_ht_full_base[lane] = full_elems - my_full;   // exclusive prefix
        gs_ht_ovfl_base[lane] = ovfl_elems - my_ovfl;

        // Last lane has the total
        uint total_full = WaveReadLaneAt(full_elems, WAVE_SIZE - 1u);
        if (lane == 0u)
            gs_total_full = total_full;
    }
    GroupMemoryBarrierWithGroupSync();

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 3 — Scatter
    //
    // Same logic as v1, but x,y derived directly from tid (no gs_pos).
    // ══════════════════════════════════════════════════════════════════════════
    uint count      = gs_ht_count[my_slot];
    uint full_elems = (count / WAVE_SIZE) * WAVE_SIZE;
    uint full_base  = gs_ht_full_base[my_slot];
    uint ovfl_base  = gs_ht_ovfl_base[my_slot];
    uint total_full = gs_total_full;

    uint out_pos;
    if (my_rank < full_elems)
    {
        out_pos = full_base + my_rank;
    }
    else
    {
        out_pos = total_full + ovfl_base + (my_rank - full_elems);
    }

    reorderedXY[out_pos] = ((tid >> 4u) << 16u) | (tid & 0xFu);

    if (tid == 0u)
        totalFullElements[0] = total_full;
}
