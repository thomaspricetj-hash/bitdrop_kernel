// src/bitdrop_gpu/bitdrop_gpu_host.cpp
#include "bitdrop_gpu.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>

// ----------------------------------------------------------
// Local config mirrors (in case not visible from this TU)
// ----------------------------------------------------------
#ifndef USE_FP8_EMB
#define USE_FP8_EMB 1
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

#ifndef UNROLL
#define UNROLL 1
#endif

#ifndef RULES_PER_TILE
#define RULES_PER_TILE 24
#endif

#if defined(_WIN32) || defined(_WIN64)
#define BITDROP_WINDOWS 1
#else
#define BITDROP_WINDOWS 0
#endif

// ----------------------------------------------------------
// Shared state (matches original kernel/global design)
// ----------------------------------------------------------
bool          g_initialized                = false;
int           g_dim                        = 0;
int           g_num_rules                  = 0;
int           g_rule_bits                  = 0;
BitDropTuning g_tuning                     = {128, RULES_PER_TILE, 1, UNROLL};
int           g_similarity_chunk_override  = 0;
int           g_similarity_chunk_autotuned = 0;

// ----------------------------------------------------------
// Simple reuse cache (C-style) to replace unordered_map usage
// This is a minimal, fixed-size LRU-like cache for embeddings.
// It avoids templates and STL so MSVC won't choke on '<'.
// ----------------------------------------------------------
#define REUSE_CACHE_CAPACITY 1024
#define SIG256_BYTES 32

typedef struct {
    unsigned char sig[SIG256_BYTES];
    unsigned char* bits;   // pointer to bits array (total_bytes)
    int32_t*       tags;   // pointer to tags array (max_tags_per_vec)
    unsigned char  skim;
    int            used;   // 0 = free, 1 = used
} CachedCollapseEntry;

static CachedCollapseEntry g_reuse_cache[REUSE_CACHE_CAPACITY];

// ----------------------------------------------------------
// Helper: compute 256-bit signature (same algorithm as backup)
// ----------------------------------------------------------
static void hash_embedding_fast_c(const __half* emb, int dim, unsigned char out_sig[SIG256_BYTES]) {
    // produce 32 bytes by folding bytes of the half array
    const unsigned char* bytes = (const unsigned char*)emb;
    size_t len = (size_t)dim * sizeof(__half);

    uint64_t h[4] = {0,0,0,0};
    for (size_t i = 0; i < len; ++i) {
        h[i & 3] = (h[i & 3] * 131) ^ bytes[i];
    }
    memcpy(out_sig + 0,  &h[0], 8);
    memcpy(out_sig + 8,  &h[1], 8);
    memcpy(out_sig + 16, &h[2], 8);
    memcpy(out_sig + 24, &h[3], 8);
}

// ----------------------------------------------------------
// Helper: lookup cache entry by signature
// Returns index or -1 if not found
// ----------------------------------------------------------
static int reuse_cache_find(const unsigned char sig[SIG256_BYTES]) {
    for (int i = 0; i < REUSE_CACHE_CAPACITY; ++i) {
        if (!g_reuse_cache[i].used) continue;
        if (memcmp(g_reuse_cache[i].sig, sig, SIG256_BYTES) == 0)
            return i;
    }
    return -1;
}

// ----------------------------------------------------------
// Helper: insert into cache (simple first-free policy)
// Caller must provide copies of bits/tags (or they will be allocated here)
// ----------------------------------------------------------
static int reuse_cache_insert(const unsigned char sig[SIG256_BYTES],
                              const uint8_t* bits_src, int total_bytes,
                              const int32_t* tags_src, int max_tags_per_vec,
                              uint8_t skim_val)
{
    int idx = -1;
    for (int i = 0; i < REUSE_CACHE_CAPACITY; ++i) {
        if (!g_reuse_cache[i].used) { idx = i; break; }
    }
    if (idx == -1) {
        // simple eviction: overwrite index 0
        idx = 0;
        if (g_reuse_cache[idx].bits) free(g_reuse_cache[idx].bits);
        if (g_reuse_cache[idx].tags) free(g_reuse_cache[idx].tags);
    }

    memcpy(g_reuse_cache[idx].sig, sig, SIG256_BYTES);
    g_reuse_cache[idx].bits = NULL;
    g_reuse_cache[idx].tags = NULL;

    if (total_bytes > 0 && bits_src) {
        g_reuse_cache[idx].bits = (unsigned char*)malloc((size_t)total_bytes);
        if (g_reuse_cache[idx].bits) memcpy(g_reuse_cache[idx].bits, bits_src, (size_t)total_bytes);
    }

    if (max_tags_per_vec > 0 && tags_src) {
        g_reuse_cache[idx].tags = (int32_t*)malloc((size_t)max_tags_per_vec * sizeof(int32_t));
        if (g_reuse_cache[idx].tags) memcpy(g_reuse_cache[idx].tags, tags_src, (size_t)max_tags_per_vec * sizeof(int32_t));
    }

    g_reuse_cache[idx].skim = skim_val;
    g_reuse_cache[idx].used = 1;
    return idx;
}

// ----------------------------------------------------------
// Autotune helpers (kept same semantics)
// ----------------------------------------------------------
static bool banks_have_similarity(const BitDropBank* banks_host, int num_banks) {
    for (int b = 0; b < num_banks; ++b) {
        const BitDropBank* bh = &banks_host[b];
        for (int r = 0; r < bh->num_rules; ++r) {
            if (bh->rules[r].type == BITDROP_RULE_SIMILARITY)
                return true;
        }
    }
    return false;
}

static int autotune_similarity_chunk(
    const void*         d_emb,
    int                 dim,
    int                 num_vecs,
    const BitDropBank*  d_banks,
    int                 num_banks,
    uint8_t*            d_bits,
    int                 total_bytes,
    int32_t*            d_tags,
    int                 max_tags_per_vec,
    uint8_t*            d_skim,
    bool                skim,
    float               skim_threshold,
    int                 threads)
{
    if (g_similarity_chunk_override > 0)
        return g_similarity_chunk_override;

    if (g_similarity_chunk_autotuned > 0)
        return g_similarity_chunk_autotuned;

    if (num_vecs <= 0) {
        g_similarity_chunk_autotuned = 256;
        return 256;
    }

    const int candidates_small_dim[]  = {128, 256, 512, 1024};
    const int candidates_large_dim[]  = {64, 128, 192, 256};
    const int* candidates             = (dim <= 512) ? candidates_small_dim : candidates_large_dim;
    const int  n_candidates           = (dim <= 512)
                                       ? (int)(sizeof(candidates_small_dim) / sizeof(candidates_small_dim[0]))
                                       : (int)(sizeof(candidates_large_dim) / sizeof(candidates_large_dim[0]));

    int tune_vecs = num_vecs < 4096 ? num_vecs : 4096;

    float best_ms    = 1e30f;
    int   best_chunk = 256;

    for (int i = 0; i < n_candidates; ++i) {
        int chunk = candidates[i];
        if (chunk > tune_vecs) chunk = tune_vecs;
        if (chunk <= 0) continue;

        int warps_needed   = chunk;
        int threads_needed = warps_needed * 32;
        int blocks         = (threads_needed + threads - 1) / threads;
        if (blocks <= 0) blocks = 1;

        int warps_per_block = threads / 32;
        size_t shmem_bytes  = (size_t)warps_per_block * 2 * sizeof(int);

        cudaEvent_t start_ev, stop_ev;
        cudaEventCreate(&start_ev);
        cudaEventCreate(&stop_ev);

        cudaEventRecord(start_ev, 0);

        bitdrop_kernel_multi<<<blocks, threads, shmem_bytes>>>(
            d_emb,
            dim,
            chunk,
            d_banks,
            num_banks,
            d_bits,
            total_bytes,
            d_tags,
            max_tags_per_vec,
            d_skim,
            skim,
            skim_threshold
        );

        cudaEventRecord(stop_ev, 0);
        cudaEventSynchronize(stop_ev);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_ev, stop_ev);

        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);

        double limit_ms = BITDROP_WINDOWS ? 5.0 : 10.0;
        if (ms > limit_ms)
            continue;

        if (ms < best_ms) {
            best_ms   = ms;
            best_chunk = chunk;
        }
    }

    if (best_chunk <= 0)
        best_chunk = 64;

    g_similarity_chunk_autotuned = best_chunk;
    return best_chunk;
}

// ----------------------------------------------------------
// PUBLIC API — C linkage
// ----------------------------------------------------------
extern "C" {

BITDROP_API BitDropStatus
bitdrop_init(int dim, int num_rules, int rule_bits)
{
    if (dim <= 0 || num_rules < 0 || rule_bits <= 0 || rule_bits > 64)
        return BITDROP_ERR_INVALID_ARG;

    g_dim         = dim;
    g_num_rules   = num_rules;
    g_rule_bits   = rule_bits;
    g_initialized = true;

    g_similarity_chunk_override  = 0;
    g_similarity_chunk_autotuned = 0;

    // clear reuse cache
    for (int i = 0; i < REUSE_CACHE_CAPACITY; ++i) {
        if (g_reuse_cache[i].used) {
            if (g_reuse_cache[i].bits) free(g_reuse_cache[i].bits);
            if (g_reuse_cache[i].tags) free(g_reuse_cache[i].tags);
            g_reuse_cache[i].bits = NULL;
            g_reuse_cache[i].tags = NULL;
            g_reuse_cache[i].used = 0;
        }
    }

    return BITDROP_OK;
}

BITDROP_API BitDropStatus
bitdrop_autotune(int dim, int num_rules)
{
    if (!g_initialized) return BITDROP_ERR_NOT_INIT;
    if (dim <= 0 || num_rules < 0) return BITDROP_ERR_INVALID_ARG;

    const int test_vecs = 1024;

    void* d_emb = NULL;
    size_t emb_bytes = (size_t)test_vecs * (size_t)dim *
                       (USE_FP8_EMB ? sizeof(uint8_t) : sizeof(__half));
    if (emb_bytes == 0) return BITDROP_ERR_INVALID_ARG;

    if (cudaMalloc(&d_emb, emb_bytes) != cudaSuccess)
        return BITDROP_ERR_CUDA;

    BitDropBank h_bank;
    memset(&h_bank, 0, sizeof(h_bank));

    BitDropBank* d_banks = NULL;
    if (cudaMalloc(&d_banks, sizeof(BitDropBank)) != cudaSuccess) {
        cudaFree(d_emb);
        return BITDROP_ERR_CUDA;
    }
    if (cudaMemcpy(d_banks, &h_bank, sizeof(BitDropBank), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_emb);
        cudaFree(d_banks);
        return BITDROP_ERR_CUDA;
    }

    const int total_bits  = 64;
    const int total_bytes = total_bits / 8;

    uint8_t* d_bits = NULL;
    if (cudaMalloc(&d_bits, (size_t)test_vecs * total_bytes) != cudaSuccess) {
        cudaFree(d_emb);
        cudaFree(d_banks);
        return BITDROP_ERR_CUDA;
    }

    int32_t* d_tags = NULL;
    uint8_t* d_skim = NULL;

    const int block_candidates[] = {128, 256, 512};
    const int n_blocks = (int)(sizeof(block_candidates) / sizeof(block_candidates[0]));

    float best_ms    = 1e30f;
    int   best_block = 128;

    for (int i = 0; i < n_blocks; ++i) {
        int block = block_candidates[i];

        int warps_needed   = test_vecs;
        int threads_needed = warps_needed * 32;
        int blocks         = (threads_needed + block - 1) / block;
        if (blocks <= 0) blocks = 1;

        int warps_per_block = block / 32;
        size_t shmem_bytes  = (size_t)warps_per_block * 2 * sizeof(int);

        cudaEvent_t start_ev, stop_ev;
        cudaEventCreate(&start_ev);
        cudaEventCreate(&stop_ev);

        cudaEventRecord(start_ev, 0);

        bitdrop_kernel_multi<<<blocks, block, shmem_bytes>>>(
            d_emb,
            dim,
            test_vecs,
            d_banks,
            1,
            d_bits,
            total_bytes,
            d_tags,
            0,
            d_skim,
            false,
            0.0f
        );

        cudaEventRecord(stop_ev, 0);
        cudaEventSynchronize(stop_ev);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_ev, stop_ev);

        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);

        if (ms < best_ms) {
            best_ms    = ms;
            best_block = block;
        }
    }

    cudaFree(d_bits);
    cudaFree(d_banks);
    cudaFree(d_emb);

    g_tuning.block_size       = best_block;
    g_tuning.rules_per_tile   = RULES_PER_TILE;
    g_tuning.warps_per_vector = 1;
    g_tuning.unroll           = UNROLL;

    return BITDROP_OK;
}

BITDROP_API BitDropStatus
bitdrop_collapse(const void*, int, const BitDropRule*, int,
                 const void*, uint64_t*, void*, int*)
{
    return BITDROP_ERR_INTERNAL;
}

BITDROP_API BitDropStatus
bitdrop_collapse_multi(const void*        embeddings_raw,
                       int                num_vecs,
                       int                dim,
                       const BitDropBank* banks_host,
                       int                num_banks,
                       int                total_bits,
                       uint8_t*           out_bits,
                       int32_t*           out_tags,
                       int                max_tags_per_vec,
                       uint8_t*           skim_mask,
                       int                skim,
                       float              skim_threshold,
                       int                auto_chunk,
                       int                chunk_size)
{
    if (!g_initialized) return BITDROP_ERR_NOT_INIT;
    if (!embeddings_raw || !banks_host || !out_bits) return BITDROP_ERR_INVALID_ARG;
    if (num_vecs <= 0 || num_banks <= 0 || total_bits <= 0) return BITDROP_ERR_INVALID_ARG;
    if (dim != g_dim) return BITDROP_ERR_INVALID_ARG;

    BitDropStatus status = BITDROP_OK;

    const void* h_emb = embeddings_raw;

    void*    d_emb   = NULL;
    uint8_t* d_bits  = NULL;
    int32_t* d_tags  = NULL;
    uint8_t* d_skim  = NULL;
    BitDropBank* d_banks = NULL;

    // allocate arrays for device rules/payload pointers (C-style)
    BitDropBank* banks_tmp = (BitDropBank*)malloc((size_t)num_banks * sizeof(BitDropBank));
    if (!banks_tmp) return BITDROP_ERR_INTERNAL;

    BitDropRule** d_rules = (BitDropRule**)calloc((size_t)num_banks, sizeof(BitDropRule*));
    uint8_t**     d_payloads = (uint8_t**)calloc((size_t)num_banks, sizeof(uint8_t*));
    if (!d_rules || !d_payloads) { status = BITDROP_ERR_INTERNAL; goto cleanup_host_allocs; }

    size_t emb_bytes = (size_t)num_vecs * (size_t)dim *
                       (USE_FP8_EMB ? sizeof(uint8_t) : sizeof(__half));

    int total_bytes = total_bits / 8;

    if (!USE_FP8_EMB) {
        int all_cached = 1;
        for (int i = 0; i < num_vecs; ++i) {
            const __half* vec_i = (const __half*)h_emb + (size_t)i * dim;
            unsigned char sig[SIG256_BYTES];
            hash_embedding_fast_c(vec_i, dim, sig);

            int found = reuse_cache_find(sig);
            if (found >= 0) {
                CachedCollapseEntry* cc = &g_reuse_cache[found];
                if (cc->bits && total_bytes > 0) {
                    memcpy(out_bits + (size_t)i * total_bytes, cc->bits, (size_t)total_bytes);
                }
                if (out_tags && max_tags_per_vec > 0 && cc->tags) {
                    memcpy(out_tags + (size_t)i * max_tags_per_vec, cc->tags, (size_t)max_tags_per_vec * sizeof(int32_t));
                }
                if (skim && skim_mask) skim_mask[i] = cc->skim;
            } else {
                all_cached = 0;
            }
        }
        if (all_cached) {
            status = BITDROP_OK;
            goto cleanup_host_allocs;
        }
    }

    if (emb_bytes > 0) {
        if (cudaMalloc(&d_emb, emb_bytes) != cudaSuccess) { status = BITDROP_ERR_CUDA; goto cleanup; }
        if (cudaMemcpy(d_emb, h_emb, emb_bytes, cudaMemcpyHostToDevice) != cudaSuccess) { status = BITDROP_ERR_CUDA; goto cleanup; }
    }

    {
        size_t bits_bytes = (size_t)num_vecs * total_bytes;
        if (bits_bytes > 0) {
            if (cudaMalloc(&d_bits, bits_bytes) != cudaSuccess) { status = BITDROP_ERR_CUDA; goto cleanup; }
        }
    }

    if (out_tags && max_tags_per_vec > 0) {
        size_t tags_bytes = (size_t)num_vecs * max_tags_per_vec * sizeof(int32_t);
        if (tags_bytes > 0) {
            if (cudaMalloc(&d_tags, tags_bytes) != cudaSuccess) { status = BITDROP_ERR_CUDA; goto cleanup; }
        }
    }

    if (skim && skim_mask) {
        size_t skim_bytes = (size_t)num_vecs * sizeof(uint8_t);
        if (skim_bytes > 0) {
            if (cudaMalloc(&d_skim, skim_bytes) != cudaSuccess) { status = BITDROP_ERR_CUDA; goto cleanup; }
        }
    }

    for (int i = 0; i < num_banks; ++i) {
        const BitDropBank* bh = &banks_host[i];

        if (bh->num_rules > 0) {
            size_t rules_bytes = (size_t)bh->num_rules * sizeof(BitDropRule);
            if (cudaMalloc(&d_rules[i], rules_bytes) != cudaSuccess) { status = BITDROP_ERR_CUDA; goto cleanup; }
            if (cudaMemcpy(d_rules[i], bh->rules, rules_bytes, cudaMemcpyHostToDevice) != cudaSuccess) { status = BITDROP_ERR_CUDA; goto cleanup; }
        }

        if (bh->payload_bytes > 0) {
            size_t payload_bytes = (size_t)bh->payload_bytes;
            if (cudaMalloc(&d_payloads[i], payload_bytes) != cudaSuccess) { status = BITDROP_ERR_CUDA; goto cleanup; }
            if (cudaMemcpy(d_payloads[i], bh->payload, payload_bytes, cudaMemcpyHostToDevice) != cudaSuccess) { status = BITDROP_ERR_CUDA; goto cleanup; }
        }

        BitDropBank bdev = *bh;
        bdev.rules   = d_rules[i];
        bdev.payload = d_payloads[i];
        banks_tmp[i] = bdev;
    }

    {
        size_t banks_bytes = (size_t)num_banks * sizeof(BitDropBank);
        if (cudaMalloc(&d_banks, banks_bytes) != cudaSuccess) { status = BITDROP_ERR_CUDA; goto cleanup; }
        if (cudaMemcpy(d_banks, banks_tmp, banks_bytes, cudaMemcpyHostToDevice) != cudaSuccess) { status = BITDROP_ERR_CUDA; goto cleanup; }
    }

    {
        int threads = g_tuning.block_size > 0 ? g_tuning.block_size : BLOCK_SIZE;
        if (threads <= 0) threads = 128;

        if (!auto_chunk || chunk_size <= 0)
            chunk_size = num_vecs;

        if (banks_have_similarity(banks_host, num_banks)) {
            int tuned = autotune_similarity_chunk(
                d_emb,
                dim,
                num_vecs,
                d_banks,
                num_banks,
                d_bits,
                total_bytes,
                d_tags,
                max_tags_per_vec,
                d_skim,
                skim != 0,
                skim_threshold,
                threads
            );
            if (tuned > 0 && tuned < chunk_size)
                chunk_size = tuned;
        }

        for (int start = 0; start < num_vecs; start += chunk_size) {
            int remaining = num_vecs - start;
            int n_chunk   = remaining < chunk_size ? remaining : chunk_size;

            const void*   emb_chunk  = (const uint8_t*)d_emb + (size_t)start * dim *
                                       (USE_FP8_EMB ? sizeof(uint8_t) : sizeof(__half));
            uint8_t* bits_chunk = d_bits + (size_t)start * total_bytes;
            int32_t* tags_chunk = d_tags
                                       ? d_tags + (size_t)start * max_tags_per_vec
                                       : NULL;
            uint8_t* skim_chunk = d_skim ? d_skim + start : NULL;

            int warps_needed   = n_chunk;
            int threads_needed = warps_needed * 32;
            int blocks         = (threads_needed + threads - 1) / threads;
            if (blocks <= 0) blocks = 1;

            int warps_per_block = threads / 32;
            size_t shmem_bytes  = (size_t)warps_per_block * 2 * sizeof(int);

            bitdrop_kernel_multi<<<blocks, threads, shmem_bytes>>>(
                emb_chunk,
                dim,
                n_chunk,
                d_banks,
                num_banks,
                bits_chunk,
                total_bytes,
                tags_chunk,
                max_tags_per_vec,
                skim_chunk,
                skim != 0,
                skim_threshold
            );

            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                fprintf(stderr, "BitDrop kernel failed: %s\n", cudaGetErrorString(err));
                status = BITDROP_ERR_CUDA;
                goto cleanup;
            }
        }
    }

    {
        size_t bits_bytes = (size_t)num_vecs * total_bytes;
        if (bits_bytes > 0 && d_bits) {
            if (cudaMemcpy(out_bits, d_bits, bits_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) { status = BITDROP_ERR_CUDA; goto cleanup; }
        }
    }

    if (out_tags && max_tags_per_vec > 0 && d_tags) {
        size_t tags_bytes = (size_t)num_vecs * max_tags_per_vec * sizeof(int32_t);
        if (tags_bytes > 0) {
            if (cudaMemcpy(out_tags, d_tags, tags_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) { status = BITDROP_ERR_CUDA; goto cleanup; }
        }
    }

    if (skim && skim_mask && d_skim) {
        size_t skim_bytes = (size_t)num_vecs * sizeof(uint8_t);
        if (skim_bytes > 0) {
            if (cudaMemcpy(skim_mask, d_skim, skim_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) { status = BITDROP_ERR_CUDA; goto cleanup; }
        }
    }

    if (!USE_FP8_EMB) {
        for (int i = 0; i < num_vecs; ++i) {
            const __half* vec_i = (const __half*)h_emb + (size_t)i * dim;
            unsigned char sig[SIG256_BYTES];
            hash_embedding_fast_c(vec_i, dim, sig);

            int found = reuse_cache_find(sig);
            if (found >= 0) continue;

            // insert into cache: copy bits and tags
            uint8_t* bits_copy = NULL;
            int32_t* tags_copy = NULL;
            if (total_bytes > 0) {
                bits_copy = (uint8_t*)malloc((size_t)total_bytes);
                if (bits_copy) memcpy(bits_copy, out_bits + (size_t)i * total_bytes, (size_t)total_bytes);
            }
            if (out_tags && max_tags_per_vec > 0) {
                tags_copy = (int32_t*)malloc((size_t)max_tags_per_vec * sizeof(int32_t));
                if (tags_copy) memcpy(tags_copy, out_tags + (size_t)i * max_tags_per_vec, (size_t)max_tags_per_vec * sizeof(int32_t));
            }
            uint8_t skim_val = (skim && skim_mask) ? skim_mask[i] : 0;
            reuse_cache_insert(sig, bits_copy, total_bytes, tags_copy, max_tags_per_vec, skim_val);
            // note: reuse_cache_insert copies the data again; free temporary copies
            if (bits_copy) free(bits_copy);
            if (tags_copy) free(tags_copy);
        }
    }

cleanup:
    if (d_emb)   cudaFree(d_emb);
    if (d_bits)  cudaFree(d_bits);
    if (d_tags)  cudaFree(d_tags);
    if (d_skim)  cudaFree(d_skim);
    if (d_banks) cudaFree(d_banks);

    for (int i = 0; i < num_banks; ++i) {
        if (d_rules[i])    cudaFree(d_rules[i]);
        if (d_payloads[i]) cudaFree(d_payloads[i]);
    }

    free(d_rules);
    free(d_payloads);
    free(banks_tmp);

cleanup_host_allocs:
    // nothing else to free here (already handled)
    return status;
}

BITDROP_API BitDropTuning
bitdrop_get_tuning()
{
    return g_tuning;
}

BITDROP_API void
bitdrop_shutdown()
{
    g_initialized = false;
    g_dim         = 0;
    g_num_rules   = 0;
    g_rule_bits   = 0;
    g_similarity_chunk_override  = 0;
    g_similarity_chunk_autotuned = 0;

    // free cache entries
    for (int i = 0; i < REUSE_CACHE_CAPACITY; ++i) {
        if (g_reuse_cache[i].used) {
            if (g_reuse_cache[i].bits) free(g_reuse_cache[i].bits);
            if (g_reuse_cache[i].tags) free(g_reuse_cache[i].tags);
            g_reuse_cache[i].bits = NULL;
            g_reuse_cache[i].tags = NULL;
            g_reuse_cache[i].used = 0;
        }
    }
}

BITDROP_API void
bitdrop_set_similarity_chunk(int chunk)
{
    g_similarity_chunk_override  = chunk;
    g_similarity_chunk_autotuned = 0;
}

} // extern "C"



