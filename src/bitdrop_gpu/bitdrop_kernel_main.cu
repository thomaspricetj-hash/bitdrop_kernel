// ============================================================
//  BitDrop Ultra Kernel (A‑Mode) — Persistent Kernel + 2‑Vector Warp
//  src/bitdrop_gpu/bitdrop_kernel_main.cu
//  Design: 2 vectors per warp, FP8‑friendly, autotuner‑ready.
// ============================================================

#include "bitdrop_gpu.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <unordered_map>
#include <array>
#include <cstring>
#include <cstdint>

// ============================================================
//  AUTOTUNER MACROS / TUNING KNOBS
// ============================================================

#define USE_HALF2 0
#define MULTI_BANK 0

#ifndef RULES_PER_TILE
#define RULES_PER_TILE 48
#endif

#ifndef TILE
#define TILE 32
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#ifndef UNROLL
#define UNROLL 1
#endif

#ifndef ENABLE_SIMILARITY
#define ENABLE_SIMILARITY 1
#endif

#ifndef USE_WARP_OR
#define USE_WARP_OR 1
#endif

#ifndef USE_FP8_SIM_PAYLOAD
#define USE_FP8_SIM_PAYLOAD 1
#endif

#ifndef USE_FP8_EMB
#define USE_FP8_EMB 1
#endif

#ifndef BITDROP_FP8_FORMAT
#define BITDROP_FP8_FORMAT 0  // E4M3
#endif

#ifndef VEC_WIDTH
#define VEC_WIDTH 4
#endif

#if defined(_WIN32) || defined(_WIN64)
#define BITDROP_WINDOWS 1
#else
#define BITDROP_WINDOWS 0
#endif

// ============================================================
//  GLOBAL STATE (unchanged, host‑side metadata)
// ============================================================

static bool          g_initialized                = false;
static int           g_dim                        = 0;
static int           g_num_rules                  = 0;
static int           g_rule_bits                  = 0;
static BitDropTuning g_tuning                     = {128, RULES_PER_TILE, 1, UNROLL};
static int           g_similarity_chunk_override  = 0;
static int           g_similarity_chunk_autotuned = 0;

// ============================================================
//  SMART REUSE CACHE (host-side, unchanged)
// ============================================================

using Sig256 = std::array<uint8_t, 32>;

struct CachedCollapse {
    std::vector<uint8_t> bits;
    std::vector<int32_t> tags;
    uint8_t              skim;
};

namespace std {
template<>
struct hash<Sig256> {
    std::size_t operator()(const Sig256& s) const noexcept {
        const std::uint64_t* p = reinterpret_cast<const std::uint64_t*>(s.data());
        std::size_t h = p[0];
        h ^= (p[1] << 1);
        h ^= (p[2] << 2);
        h ^= (p[3] << 3);
        return h;
    }
};
} // namespace std

static Sig256 hash_embedding_fast(const __half* emb, int dim) {
    Sig256 sig{};
    const uint8_t* bytes =
        reinterpret_cast<const uint8_t*>(emb);
    std::size_t len = (std::size_t)dim * sizeof(__half);

    std::uint64_t h[4] = {0,0,0,0};
    for (std::size_t i = 0; i < len; ++i) {
        h[i & 3] = (h[i & 3] * 131) ^ bytes[i];
    }
    std::memcpy(sig.data() + 0,  &h[0], 8);
    std::memcpy(sig.data() + 8,  &h[1], 8);
    std::memcpy(sig.data() + 16, &h[2], 8);
    std::memcpy(sig.data() + 24, &h[3], 8);
    return sig;
}

static std::unordered_map<Sig256, CachedCollapse> g_reuse_cache;

// ============================================================
//  FP8 E4M3 → float
// ============================================================

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t v)
{
    std::uint32_t sign = (v >> 7) & 0x1;
    std::uint32_t exp  = (v >> 3) & 0xF;
    std::uint32_t man  = v & 0x7;

    int exp16 = (int)exp - 7 + 15;
    if (exp16 < 0) exp16 = 0;
    if (exp16 > 0x1F) exp16 = 0x1F;

    std::uint16_t f16 = (sign << 15) | (exp16 << 10) | (man << 7);
    __half h = __ushort_as_half(f16);
    return __half2float(h);
}

// ============================================================
//  MICRO HELPERS: LOAD VALUE FROM EMBEDDING
// ============================================================

__device__ __forceinline__ float load_emb_value(
    const void* __restrict__ emb_raw,
    int idx)
{
#if USE_FP8_EMB
    const uint8_t* vec_fp8 =
        reinterpret_cast<const uint8_t*>(emb_raw);
    return fp8_e4m3_to_float(vec_fp8[idx]);
#else
    const __half* vec =
        reinterpret_cast<const __half*>(emb_raw);
    return __half2float(vec[idx]);
#endif
}

// Vectorized load of VEC_WIDTH values from embedding
__device__ __forceinline__ void load_emb_vec(
    const void* __restrict__ emb_raw,
    int base_idx,
    float vals[VEC_WIDTH],
    int dim)
{
#pragma unroll
    for (int k = 0; k < VEC_WIDTH; ++k) {
        int idx = base_idx + k;
        if (idx < dim)
            vals[k] = load_emb_value(emb_raw, idx);
        else
            vals[k] = 0.0f;
    }
}

// ============================================================
//  PARALLEL HELPERS (warp‑cooperative)
// ============================================================

__device__ __forceinline__ void warp_zero_bytes(
    uint8_t* __restrict__ ptr,
    int total_bytes,
    int lane)
{
    for (int i = lane; i < total_bytes; i += 32) {
        ptr[i] = 0;
    }
}

__device__ __forceinline__ void warp_zero_ints(
    int32_t* __restrict__ ptr,
    int count,
    int lane)
{
    for (int i = lane; i < count; i += 32) {
        ptr[i] = -1;
    }
}

__device__ __forceinline__ std::uint64_t warp_or_reduce(std::uint64_t v)
{
#if USE_WARP_OR
    unsigned mask = __activemask();
    for (int off = 16; off > 0; off >>= 1) {
        std::uint64_t other = __shfl_down_sync(mask, v, off);
        v |= other;
    }
#endif
    return v;
}

// 16‑lane subwarp OR‑reduction (for 2‑vector kernel)
__device__ __forceinline__ std::uint64_t subwarp_or_reduce(
    std::uint64_t v,
    int subwarp_id)
{
#if USE_WARP_OR
    unsigned sub_mask = (subwarp_id == 0) ? 0x0000FFFFu : 0xFFFF0000u;
    for (int off = 8; off > 0; off >>= 1) {
        std::uint64_t other = __shfl_down_sync(sub_mask, v, off, 16);
        v |= other;
    }
#endif
    return v;
}

// ============================================================
//  MICRO HELPERS: THRESHOLD RULE EVAL
// ============================================================

__device__ __forceinline__ bool eval_threshold_rule(
    const void* __restrict__ emb_raw,
    int dim,
    const BitDropBank& bank,
    const BitDropRule& rule)
{
    int dim_idx = rule.dim;
    if (dim_idx < 0 || dim_idx >= dim)
        return false;

    float v = load_emb_value(emb_raw, dim_idx);

    const uint8_t* p = bank.payload + rule.payload_offset;
    const __half* thr_ptr = reinterpret_cast<const __half*>(p);
    float thr = __half2float(thr_ptr[0]);

    return (v >= thr);
}

// ============================================================
//  PREDICTOR FIBER (activity estimator used for skim/early-exit)
// ============================================================

__device__ __forceinline__ float predictor_fiber_activity(
    const void* __restrict__ emb_raw,
    int dim)
{
    int lane = threadIdx.x & 31;
    float activity = 0.0f;

    for (int i = lane; i < dim; i += 32 * 4) {
#if USE_FP8_EMB
        const uint8_t* vec_fp8 =
            reinterpret_cast<const uint8_t*>(emb_raw);
        activity += (vec_fp8[i] != 0);
#else
        const __half* vec =
            reinterpret_cast<const __half*>(emb_raw);
        float v = __half2float(vec[i]);
        activity += (fabsf(v) > 1e-6f);
#endif
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        activity += __shfl_down_sync(0xffffffff, activity, offset);

    return activity;
}

// ============================================================
//  MODULE IMPORTS (Ultra engine)
// ============================================================

#include "bitdrop_kernel_activity.cuh"
#include "bitdrop_kernel_similarity.cuh"
#include "bitdrop_kernel_wavefront.cuh"
#include "bitdrop_kernel_refine.cuh"
#include "bitdrop_kernel_locality.cuh"

// ============================================================
//  RULE EVALUATION DISPATCH
// ============================================================

struct RuleEvalContext {
    const void*        emb_raw;
    int                dim;
    const BitDropBank* bank;
};

__device__ __forceinline__ bool bd_eval_rule(
    int rule_idx,
    const RuleEvalContext& ctx)
{
    const BitDropRule& rule = ctx.bank->rules[rule_idx];

    if (rule.type == BITDROP_RULE_THRESHOLD) {
        return eval_threshold_rule(ctx.emb_raw, ctx.dim, *ctx.bank, rule);
    }

#if ENABLE_SIMILARITY
    if (rule.type == BITDROP_RULE_SIMILARITY) {
        const uint8_t* payload = ctx.bank->payload + rule.payload_offset;

        float acc = bd_similarity_coarse(ctx.emb_raw, ctx.dim, payload);

        const __half* thr_ptr = reinterpret_cast<const __half*>(payload + ctx.dim);
        float thr = __half2float(thr_ptr[0]);

        if (acc < thr * 0.8f) return false;

        float acc2 = bd_similarity_refine(ctx.emb_raw, ctx.dim, payload);
        return (acc2 >= thr);
    }
#endif

    return false;
}

// ============================================================
//  PERSISTENT KERNEL — 2 VECTORS PER WARP
// ============================================================
//
//  Warp split:
//    subwarp 0: lanes 0–15 → vector A
//    subwarp 1: lanes 16–31 → vector B
//
//  Each subwarp has its own:
//    vec_id, bits_row, tags_row, tag_counter, any_hit
//
//  Warps stride over pairs of vectors:
//    vec_base = warp_idx * 2
// ============================================================

__global__ __launch_bounds__(BLOCK_SIZE, 2)
void bitdrop_kernel_multi(
    const void*        __restrict__ emb_raw,
    int                dim,
    int                num_vecs,
    const BitDropBank* __restrict__ banks,
    int                num_banks,
    uint8_t*           __restrict__ out_bits,
    int                total_bytes,
    int32_t*           __restrict__ out_tags,
    int                max_tags_per_vec,
    uint8_t*           __restrict__ skim_mask,
    bool               skim,
    float              skim_threshold)
{
    extern __shared__ int shared_int[];
    const int threads_per_block = blockDim.x;
    const int warps_per_block   = threads_per_block / 32;
    const int warp_id_in_block  = threadIdx.x / 32;
    const int lane              = threadIdx.x & 31;

    const int subwarp_id      = lane >> 4;      // 0 or 1
    const int lane_in_subwarp = lane & 15;      // 0..15

    const int global_warp_base = (blockIdx.x * warps_per_block) + warp_id_in_block;
    const int warp_stride      = gridDim.x * warps_per_block;

    const std::size_t emb_stride =
        (std::size_t)dim * (USE_FP8_EMB ? sizeof(uint8_t) : sizeof(__half));

    // 2 counters per warp in shared memory:
    // index = warp_id_in_block * 2 + subwarp_id
    int* tag_counter_base = shared_int;

    for (int warp_idx = global_warp_base; (warp_idx * 2) < num_vecs; warp_idx += warp_stride) {
        int vec_base = warp_idx * 2;
        int vec_id   = vec_base + subwarp_id;
        if (vec_id >= num_vecs)
            continue;

        const void* vec_base_ptr =
            reinterpret_cast<const uint8_t*>(emb_raw) +
            (std::size_t)vec_id * emb_stride;

        uint8_t* bits_row =
            out_bits + (std::size_t)vec_id * total_bytes;

        int32_t* tags_row = out_tags
            ? out_tags + (std::size_t)vec_id * max_tags_per_vec
            : nullptr;

        // Zero bits row (16‑lane subwarp)
        for (int i = lane_in_subwarp; i < total_bytes; i += 16) {
            bits_row[i] = 0;
        }

        // Zero tags row (16‑lane subwarp)
        if (tags_row) {
            for (int i = lane_in_subwarp; i < max_tags_per_vec; i += 16) {
                tags_row[i] = -1;
            }
        }

        __syncwarp();

        int tag_counter_idx = warp_id_in_block * 2 + subwarp_id;
        int* tag_counter    = &tag_counter_base[tag_counter_idx];

        if (lane_in_subwarp == 0)
            *tag_counter = 0;
        __syncwarp();

        float activity = bd_activity(vec_base_ptr, dim);
        __syncwarp();

        if (skim && bd_should_skip_vector(activity, skim_threshold)) {
            if (lane_in_subwarp == 0 && skim_mask)
                skim_mask[vec_id] = 0;
            continue;
        }

        bool any_hit = false;

        for (int b = 0; b < num_banks; ++b) {
            const BitDropBank& bank = banks[b];

            std::uint64_t* bank_bits =
                reinterpret_cast<std::uint64_t*>(bits_row + bank.bit_offset / 8);

            std::uint64_t local_bits = 0;

            RuleEvalContext ctx { vec_base_ptr, dim, &bank };

            bd_wavefront_pass(
                vec_base_ptr,
                dim,
                bank,
                lane_in_subwarp,
                local_bits,
                tag_counter,
                max_tags_per_vec,
                tags_row,
                [&] __device__ (int idx) {
                    return bd_eval_rule(idx, ctx);
                }
            );

            const int rules_per_tile = RULES_PER_TILE;
#pragma unroll
            for (int tile_base = 0; tile_base < bank.num_rules; tile_base += rules_per_tile) {
                int tile_end = min(tile_base + rules_per_tile, bank.num_rules);

                bd_refine_tile(
                    tile_base,
                    tile_end,
                    lane_in_subwarp,
                    bank,
                    local_bits,
                    tag_counter,
                    max_tags_per_vec,
                    tags_row,
                    [&] __device__ (int idx) {
                        return bd_eval_rule(idx, ctx);
                    }
                );
            }

            local_bits = subwarp_or_reduce(local_bits, subwarp_id);

            if (lane_in_subwarp == 0 && local_bits) {
                atomicOr(reinterpret_cast<unsigned long long*>(bank_bits), local_bits);
                any_hit = true;
            }

            __syncwarp();
        }

        if (skim && lane_in_subwarp == 0 && skim_mask) {
            skim_mask[vec_id] = any_hit ? 1 : 0;
        }

        __syncwarp();
    }
}

// ============================================================
//  PUBLIC API (host TU unchanged)
// ============================================================

extern "C" {
    // Host-side functions remain in bitdrop_gpu.cpp
}