#pragma once

// ============================================================
//  WAVEFRONT ENGINE (A‑MODE)
// ============================================================

__device__ __forceinline__ uint32_t bd_tile_signature(bool match)
{
    return __ballot_sync(0xffffffff, match);
}

__device__ __forceinline__ bool bd_should_skip_tile(
    uint32_t sig,
    uint32_t& last_sig)
{
    if (sig == last_sig) return true;
    last_sig = sig;
    return false;
}

template<typename RuleEvalFn>
__device__ void bd_wavefront_pass(
    const void* __restrict emb_raw,
    int dim,
    const BitDropBank& bank,
    int lane,
    uint64_t& local_bits,
    int* tag_counter,
    int max_tags,
    std::int32_t* tags_row,
    RuleEvalFn eval_fn)
{
    const int rules_per_tile = RULES_PER_TILE;
    uint32_t last_sig = 0xffffffffu;

    // Forward sweep
    for (int tile_base = 0; tile_base < bank.num_rules; tile_base += rules_per_tile) {
        int tile_end = min(tile_base + rules_per_tile, bank.num_rules);

        for (int r = tile_base + lane; r < tile_end; r += 32 * UNROLL) {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                int idx = r + u * 32;
                if (idx >= tile_end) break;

                bool match = eval_fn(idx);

                uint32_t sig = bd_tile_signature(match);
                if (bd_should_skip_tile(sig, last_sig)) continue;

                if (match) {
                    local_bits |= (1ull << bank.rules[idx].bit);
                    if (tags_row) {
                        int pos = atomicAdd(tag_counter, 1);
                        if (pos < max_tags) tags_row[pos] = idx;
                    }
                }
            }
        }
    }

    // Backward sweep (refinement)
    for (int tile_base = bank.num_rules - rules_per_tile;
         tile_base >= 0;
         tile_base -= rules_per_tile)
    {
        int tile_end = min(tile_base + rules_per_tile, bank.num_rules);

        for (int r = tile_base + lane; r < tile_end; r += 32) {
            bool match = eval_fn(r);

            if (match) {
                local_bits |= (1ull << bank.rules[r].bit);
                if (tags_row) {
                    int pos = atomicAdd(tag_counter, 1);
                    if (pos < max_tags) tags_row[pos] = r;
                }
            }
        }
    }
}
