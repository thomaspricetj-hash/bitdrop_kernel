#pragma once

// ============================================================
//  SLIDING‑WINDOW REFINEMENT (A‑MODE)
// ============================================================

template<typename RuleEvalFn>
__device__ void bd_refine_tile(
    int tile_base,
    int tile_end,
    int lane,
    const BitDropBank& bank,
    uint64_t& local_bits,
    int* tag_counter,
    int max_tags,
    std::int32_t* tags_row,
    RuleEvalFn eval_fn)
{
    // Two refinement passes with shifted offsets
    for (int pass = 0; pass < 2; ++pass) {
        int offset = pass * 4; // small shift

        for (int r = tile_base + lane + offset; r < tile_end; r += 32) {
            if (r >= tile_end) break;

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
