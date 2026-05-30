#pragma once

#include <cstdint>

// ------------------------------------------------------------
// Global feature / tuning flags (host-visible)
// ------------------------------------------------------------

// Tag type hook (kernel can reinterpret_cast if we later pack tags)
// For now we keep the C API as int32_t* to avoid ABI break.
#ifndef BITDROP_TAG_TYPE
    #define BITDROP_TAG_TYPE std::int32_t
#endif

// Future payload compression flags (used by host builders / kernels)
#ifndef BITDROP_USE_FP8_PAYLOAD
    #define BITDROP_USE_FP8_PAYLOAD 0
#endif

#ifndef BITDROP_FP8_FORMAT
    // 0 = E4M3, 1 = E5M2 (placeholder for future use)
    #define BITDROP_FP8_FORMAT 0
#endif

// ------------------------------------------------------------
// Export macro (static lib friendly)
// ------------------------------------------------------------
#if defined(_WIN32) || defined(_WIN64)
  #if defined(BITDROP_GPU_EXPORTS)
    // Building the library (if ever turned into a DLL)
    #define BITDROP_API __declspec(dllexport)
  #else
    // Linking from static lib or same module → no import decoration
    #define BITDROP_API
  #endif
#else
  #define BITDROP_API __attribute__((visibility("default")))
#endif

// ------------------------------------------------------------
// Status codes
// ------------------------------------------------------------
enum BitDropStatus : int {
    BITDROP_OK              = 0,
    BITDROP_ERR_INVALID_ARG = 1,
    BITDROP_ERR_NOT_INIT    = 2,
    BITDROP_ERR_CUDA        = 3,
    BITDROP_ERR_INTERNAL    = 4,
};

// ------------------------------------------------------------
// Rule types (must match Python)
// ------------------------------------------------------------
enum BitDropRuleType : std::uint8_t {
    BITDROP_RULE_THRESHOLD  = 0,
    BITDROP_RULE_SIMILARITY = 1,
};

// ------------------------------------------------------------
// BitDropRule layout (must match Python)
// ------------------------------------------------------------
#pragma pack(push, 1)
struct BitDropRule {
    std::uint8_t  type;           // BitDropRuleType
    std::uint8_t  bit;            // 0..63
    std::uint16_t dim;            // dimension index (threshold rules)
    std::uint32_t payload_offset; // byte offset into payload[]
};
#pragma pack(pop)

static_assert(sizeof(BitDropRule) == 8, "BitDropRule must be 8 bytes");

// ------------------------------------------------------------
// Bank descriptor (host + device)
// ------------------------------------------------------------
#pragma pack(push, 1)
struct BitDropBank {
    const BitDropRule*  rules;         // device pointer
    const std::uint8_t* payload;       // device pointer
    int                 num_rules;
    int                 bits_per_bank; // must be 64 (enforced in bindings)
    int                 bit_offset;    // global bit offset
    int                 payload_bytes; // size of payload buffer
};
#pragma pack(pop)

// 64-bit: 8 + 8 + 4*4 = 32
static_assert(sizeof(BitDropBank) == 32, "BitDropBank must be 32 bytes on 64-bit");

// ------------------------------------------------------------
// Tuning parameters
// ------------------------------------------------------------
struct BitDropTuning {
    int block_size;
    int rules_per_tile;
    int warps_per_vector;
    int unroll;
};

// ------------------------------------------------------------
// C API
// ------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

BITDROP_API BitDropStatus
bitdrop_init(int dim, int num_rules, int rule_bits);

BITDROP_API BitDropStatus
bitdrop_autotune(int dim, int num_rules);

BITDROP_API BitDropStatus
bitdrop_collapse(const void*        embeddings_f16,
                 int                num_vecs,
                 const BitDropRule* rules,
                 int                num_rules,
                 const void*        rule_payload,
                 std::uint64_t*     out_bits,
                 void*              out_tags,
                 int*               out_tags_bytes);

BITDROP_API BitDropStatus
bitdrop_collapse_multi(const void*        embeddings_f16,
                       int                num_vecs,
                       int                dim,
                       const BitDropBank* banks_host,
                       int                num_banks,
                       int                total_bits,
                       std::uint8_t*      out_bits,
                       std::int32_t*      out_tags,
                       int                max_tags_per_vec,
                       std::uint8_t*      skim_mask,
                       int                skim,
                       float              skim_threshold,
                       int                auto_chunk,
                       int                chunk_size);

BITDROP_API BitDropTuning
bitdrop_get_tuning(void);

BITDROP_API void
bitdrop_shutdown(void);

BITDROP_API void
bitdrop_set_similarity_chunk(int chunk);

#ifdef __cplusplus
}
#endif

// ------------------------------------------------------------
// CUDA kernel declaration (C++ linkage, shared by host and device)
// ------------------------------------------------------------
#ifdef __CUDACC__
#define BITDROP_KERNEL_DECL __global__
#else
#define BITDROP_KERNEL_DECL
#endif

BITDROP_KERNEL_DECL
void bitdrop_kernel_multi(
    const void*         emb_raw,
    int                 dim,
    int                 num_vecs,
    const BitDropBank*  banks,
    int                 num_banks,
    std::uint8_t*       out_bits,
    int                 total_bytes,
    std::int32_t*       out_tags,
    int                 max_tags_per_vec,
    std::uint8_t*       skim_mask,
    bool                skim,
    float               skim_threshold);









