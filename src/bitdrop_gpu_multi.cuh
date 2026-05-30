#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// ------------------------------------------------------------
// RuleBank descriptor (one per bank)
// ------------------------------------------------------------
struct RuleBank {
    const int32_t* rules;      // encoded rules buffer
    const int32_t* payloads;   // encoded payload buffer
    int32_t        num_rules;  // number of rules in this bank
    int32_t        bits_per_bank;     // 32 / 64 / 128
    int32_t        bit_offset;        // global bit offset in final bitstream
};

// ------------------------------------------------------------
// Device-side helpers (you will implement these)
// ------------------------------------------------------------

// Evaluate all rules in a bank for a single vector.
// Writes bits into `bank_bits` (as uint64_t or uint32_t mask).
// Optionally writes tags into `tag_buf` and returns number of tags written.
__device__ int eval_rule_bank(
    const half* emb_vec,   // [D]
    int D,
    const RuleBank& bank,
    uint64_t& bank_bits,   // output bit mask for this bank
    int32_t* tag_buf,      // per-thread scratch for tags
    int max_tags_per_vec
);

// ------------------------------------------------------------
// Kernel + host wrapper
// ------------------------------------------------------------

void bitdrop_collapse_multi(
    const half* d_emb,          // [N, D]
    int N,
    int D,
    const RuleBank* d_banks,
    int num_banks,
    uint8_t* d_out_bits,        // [N, total_bits/8]
    int32_t* d_out_tags,        // [N, max_tags_per_vec] or nullptr
    uint8_t* d_skim_mask,       // [N] or nullptr
    int max_tags_per_vec,
    bool skim,
    float skim_threshold,
    bool auto_chunk,
    int chunk_size,
    cudaStream_t stream = 0
);
