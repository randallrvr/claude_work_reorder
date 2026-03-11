// reorder_setassoc.slang
// ─────────────────────────────────────────────────────────────────────────────
// Set-associative work-reordering kernel for 16×16 material-ID tiles.
// Material IDs are arbitrary uint32 values.
//
// ── Analogy ────────────────────────────────────────────────────────────────────
//
//   This kernel treats the reordering problem like a set-associative cache:
//
//     Cache concept          Reorder concept
//     ─────────────          ───────────────
//     Cache set              Output wave (32 lanes)
//     Cache line / way       Lane slot within a wave
//     Tag                    Material ID
//     Placement policy       Hash-table → prefix-sum allocation
//
//   Instead of sorting all 256 elements (O(N log²N) bitonic sort), we build
//   a shared-memory hash table to count elements per material, then allocate
//   wave slots via prefix sums — O(N) parallel work + O(M) serial allocation
//   where M ≪ N is the number of unique materials.
//
// ── Algorithm ──────────────────────────────────────────────────────────────────
//
//   Phase 0   Initialize hash table (all threads, 1 barrier).
//
//   Phase 1   Hash-table insert + count (all threads, parallel).
//             Each thread atomically inserts its material ID into an
//             open-addressing hash table and atomically increments the
//             count for that material.  The atomicAdd return value gives
//             each thread its rank within its material.  1 barrier.
//
//   Phase 2   Wave allocation (thread 0, serial O(M)).
//             Scans occupied hash-table slots.  For each material with
//             count C:
//               full-wave slots = floor(C / 32) * 32
//               overflow slots  = C - full-wave slots
//             Computes prefix sums → full_base[] and ovfl_base[] per slot.
//             1 barrier.
//
//   Phase 3   Scatter (all threads, parallel).
//             Each thread reads its material's allocation, uses its rank
//             to determine full-wave vs overflow, and writes (x, y) to
//             the computed output position.  0 barriers.
//
//   Total: 3 barriers  (vs 38 original, ~11 optimized)
//
// ── Output guarantee (identical to reorder.slang) ──────────────────────────────
//
//   output[0..F-1]:   full-wave section, every 32-block is one material.
//   output[F..255]:   overflow section, packed by material.
//   F = totalFullElements[0], always a multiple of 32.
//
// ── Hash table ─────────────────────────────────────────────────────────────────
//
//   Open-addressing with linear probing.
//   256 slots (load factor ≤ 1.0; in practice much lower with ~5–10 materials).
//   Empty sentinel = 0xFFFFFFFF.
//   Insertion uses InterlockedCompareExchange (CAS) for lock-free correctness.
//
// ── Shared memory ──────────────────────────────────────────────────────────────
//
//   gs_ht_key[256]        1024 B   hash table keys (material IDs)
//   gs_ht_count[256]      1024 B   element count per material
//   gs_ht_full_base[256]  1024 B   full-wave base position per material
//   gs_ht_ovfl_base[256]  1024 B   overflow base position per material
//   gs_pos[256]           1024 B   packed (y << 4) | x per thread
//   gs_total_full            4 B   broadcast: total full-wave elements
//   Total                ≈ 5.1 KB
//
// ── Dispatch ───────────────────────────────────────────────────────────────────
//
//   1 group of 256 threads  →  dispatchGroups(1, 1, 1)
//
// ─────────────────────────────────────────────────────────────────────────────

static const uint TILE_W        = 16u;
static const uint TILE_H        = 16u;
static const uint TILE_ELEMENTS = 256u;
static const uint WAVE_SIZE     =  32u;
static const uint HT_SIZE       = 256u;   // hash table slots
static const uint HT_EMPTY      = 0xFFFFFFFFu;  // empty sentinel

// ── Shared memory ──────────────────────────────────────────────────────────────
groupshared uint gs_ht_key[HT_SIZE];          // hash table: material IDs
groupshared uint gs_ht_count[HT_SIZE];        // hash table: element counts
groupshared uint gs_ht_full_base[HT_SIZE];    // full-wave base per material
groupshared uint gs_ht_ovfl_base[HT_SIZE];    // overflow base per material
groupshared uint gs_pos[TILE_ELEMENTS];       // packed (y << 4) | x
groupshared uint gs_total_full;               // broadcast: total full-wave elements

// ── Hash function ──────────────────────────────────────────────────────────────
// Simple multiplicative hash (Knuth's golden-ratio method).
// Distributes arbitrary uint32 keys across 256 slots with minimal clustering.
uint hashKey(uint key)
{
    return (key * 2654435761u) >> 24u;   // top 8 bits of product → [0, 255]
}

// ─────────────────────────────────────────────────────────────────────────────
[numthreads(256, 1, 1)]
void reorderByMaterialSetAssoc(
    StructuredBuffer<uint>    tileData,
    RWStructuredBuffer<uint>  reorderedXY,
    RWStructuredBuffer<uint>  totalFullElements,
    uint3 groupTid : SV_GroupThreadID
)
{
    uint tid = groupTid.x;

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 0 — Initialize hash table
    //
    // Every thread clears one hash-table slot.  Since HT_SIZE == TILE_ELEMENTS
    // == 256, this is a simple 1:1 mapping.
    // ══════════════════════════════════════════════════════════════════════════
    gs_ht_key[tid]   = HT_EMPTY;
    gs_ht_count[tid] = 0u;
    GroupMemoryBarrierWithGroupSync();

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 1 — Hash-table insert + count
    //
    // Each thread inserts its material ID into the hash table using CAS
    // (compare-and-swap) with linear probing.  Once the slot is secured,
    // an atomic increment on the count gives this thread its rank within
    // its material group.
    //
    // Thread-local outputs:
    //   my_slot  — which hash-table slot holds this thread's material
    //   my_rank  — this thread's 0-based rank within its material
    // ══════════════════════════════════════════════════════════════════════════
    uint my_key = tileData[tid];
    uint my_pos = ((tid / TILE_W) << 4u) | (tid % TILE_W);

    uint slot = hashKey(my_key);
    uint my_slot;
    uint my_rank;

    // Linear-probe insertion
    [loop]
    for (uint probe = 0u; probe < HT_SIZE; probe++)
    {
        uint idx = (slot + probe) & (HT_SIZE - 1u);   // mod 256 (power of 2)

        // Try to claim this slot for our key
        uint original;
        InterlockedCompareExchange(gs_ht_key[idx], HT_EMPTY, my_key, original);

        if (original == HT_EMPTY || original == my_key)
        {
            // Slot is ours (we inserted) or already holds our key
            my_slot = idx;
            InterlockedAdd(gs_ht_count[idx], 1u, my_rank);
            break;
        }
        // Slot holds a different key — probe next
    }

    gs_pos[tid] = my_pos;
    GroupMemoryBarrierWithGroupSync();

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 2 — Wave allocation  (thread 0 only, serial O(M))
    //
    // Scans all hash-table slots.  For each occupied slot (unique material):
    //   count      = gs_ht_count[slot]
    //   full_elems = floor(count / 32) * 32
    //   ovfl_elems = count - full_elems
    //
    // Computes prefix sums for full-wave and overflow base positions.
    // ══════════════════════════════════════════════════════════════════════════
    if (tid == 0u)
    {
        uint full_cnt = 0u;
        uint ovfl_cnt = 0u;

        for (uint s = 0u; s < HT_SIZE; s++)
        {
            if (gs_ht_key[s] == HT_EMPTY)
            {
                // Mark unused slots so Phase 3 doesn't read garbage
                gs_ht_full_base[s] = 0u;
                gs_ht_ovfl_base[s] = 0u;
                continue;
            }

            uint count      = gs_ht_count[s];
            uint full_elems = (count / WAVE_SIZE) * WAVE_SIZE;
            uint ovfl_elems = count - full_elems;

            gs_ht_full_base[s] = full_cnt;
            gs_ht_ovfl_base[s] = ovfl_cnt;

            full_cnt += full_elems;
            ovfl_cnt += ovfl_elems;
        }

        gs_total_full = full_cnt;
    }
    GroupMemoryBarrierWithGroupSync();

    // ══════════════════════════════════════════════════════════════════════════
    // Phase 3 — Scatter
    //
    // Each thread uses its material's allocation (full_base, ovfl_base) and
    // its rank within the material to compute its output position.
    //
    //   If rank < full_elems → full-wave section:  out_pos = full_base + rank
    //   Else                 → overflow section:   out_pos = total_full
    //                                                       + ovfl_base
    //                                                       + (rank - full_elems)
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

    uint packed_xy = gs_pos[tid];
    uint x = packed_xy & 0xFu;
    uint y = (packed_xy >> 4u) & 0xFu;
    reorderedXY[out_pos] = (y << 16u) | x;

    if (tid == 0u)
        totalFullElements[0] = total_full;
}
