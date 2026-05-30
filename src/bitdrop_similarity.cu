// bitdrop_similarity.cu
// Similarity rule execution with micro-kernel splitting and autotuning.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <algorithm>

#if defined(_WIN32) || defined(_WIN64)
#define BITDROP_WINDOWS 1
#else
#define BITDROP_WINDOWS 0
#endif

// -----------------------------------------------------------------------------
// Global autotuned micro-chunk for similarity kernels
// -----------------------------------------------------------------------------

static int g_similarity_micro_chunk = 0;

// Optional: expose this to Python via C API
extern "C" void bitdrop_set_similarity_micro_chunk(int micro) {
    g_similarity_micro_chunk = micro;
}

// -----------------------------------------------------------------------------
// Kernel: simple FP16 similarity (dot product) example
// You should adapt this to your actual rule layout and output format.
// -----------------------------------------------------------------------------

// Each thread handles one vector, computes dot with a single rule vector.
// n_vectors is the batch dimension we split on.
__global__ void collapse_similarity_kernel(
    const __half* __restrict__ emb,    // [n_vectors, dim]
    const uint8_t* __restrict__ rules, // rule payload (e.g., packed vectors)
    uint8_t* __restrict__ out,         // output bits/tags
    int n_vectors,
    int dim,
    int rule_index,                    // which similarity rule to apply
    int out_stride                     // bytes per vector in out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vectors) return;

    const __half* v = emb + idx * dim;

    // For simplicity, assume rule payload is a contiguous FP16 vector per rule.
    const __half* rule_vec = reinterpret_cast<const __half*>(rules) + rule_index * dim;

    float acc = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float a = __half2float(v[d]);
        float b = __half2float(rule_vec[d]);
        acc += a * b;
    }

    // Simple threshold at 0.0 for demo; adapt to your real rule logic.
    bool fire = (acc > 0.0f);

    // Write one bit/byte for this rule; adapt to your bit-packing.
    uint8_t* out_row = out + idx * out_stride;
    out_row[rule_index] = fire ? 1 : 0;
}

// -----------------------------------------------------------------------------
// Micro-batch launcher
// -----------------------------------------------------------------------------

static void launch_similarity_micro_batch(
    const __half* emb,
    const uint8_t* rules,
    uint8_t* out,
    int n_vectors,
    int dim,
    int rule_index,
    int out_stride,
    cudaStream_t stream
) {
    if (n_vectors <= 0) return;

    dim3 block(128);
    dim3 grid((n_vectors + block.x - 1) / block.x);

    collapse_similarity_kernel<<<grid, block, 0, stream>>>(
        emb,
        rules,
        out,
        n_vectors,
        dim,
        rule_index,
        out_stride
    );
}

// -----------------------------------------------------------------------------
// Simple autotuner for similarity micro-chunk
// -----------------------------------------------------------------------------

static int autotune_similarity_micro_chunk(
    const __half* emb,
    const uint8_t* rules,
    uint8_t* out,
    int dim,
    int n_vectors,
    int rule_index,
    int out_stride,
    cudaStream_t stream
) {
    if (g_similarity_micro_chunk > 0) {
        return g_similarity_micro_chunk;
    }

    if (n_vectors <= 0) {
        g_similarity_micro_chunk = 64;
        return g_similarity_micro_chunk;
    }

    // Candidate micro-chunk sizes (vectors per kernel)
    const int candidates[] = {32, 64, 96, 128};
    const int n_candidates = sizeof(candidates) / sizeof(candidates[0]);

    // Tune on up to 4096 vectors or the available count
    int tune_n = std::min(n_vectors, 4096);

    for (int i = 0; i < n_candidates; ++i) {
        int micro = candidates[i];
        if (micro > tune_n) micro = tune_n;

        // Warmup
        for (int w = 0; w < 2; ++w) {
            int processed = 0;
            while (processed < tune_n) {
                int count = std::min(micro, tune_n - processed);
                launch_similarity_micro_batch(
                    emb + processed * dim,
                    rules,
                    out + processed * out_stride,
                    count,
                    dim,
                    rule_index,
                    out_stride,
                    stream
                );
                processed += count;
            }
        }
        cudaDeviceSynchronize();

        // Time one full pass over tune_n vectors with this micro size
        cudaEvent_t start_ev, stop_ev;
        cudaEventCreate(&start_ev);
        cudaEventCreate(&stop_ev);

        cudaEventRecord(start_ev, stream);

        int processed = 0;
        while (processed < tune_n) {
            int count = std::min(micro, tune_n - processed);
            launch_similarity_micro_batch(
                emb + processed * dim,
                rules,
                out + processed * out_stride,
                count,
                dim,
                rule_index,
                out_stride,
                stream
            );
            processed += count;
        }

        cudaEventRecord(stop_ev, stream);
        cudaEventSynchronize(stop_ev);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_ev, stop_ev);

        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);

        // Heuristic: keep total tuning batch under ~5 ms on Windows, ~10 ms on Linux.
        double limit_ms = BITDROP_WINDOWS ? 5.0 : 10.0;

        if (ms <= limit_ms) {
            g_similarity_micro_chunk = micro;
            return g_similarity_micro_chunk;
        }
    }

    // Fallback: smallest micro-chunk
    g_similarity_micro_chunk = candidates[0];
    return g_similarity_micro_chunk;
}

// -----------------------------------------------------------------------------
// Public launcher: similarity path with micro-kernel splitting
// Call this from your main collapse_multi implementation when handling
// similarity rules.
// -----------------------------------------------------------------------------

extern "C" void bitdrop_launch_similarity(
    const __half* emb,      // [n_vectors, dim]
    const uint8_t* rules,   // rule payload
    uint8_t* out,           // [n_vectors, out_stride]
    int n_vectors,
    int dim,
    int rule_index,
    int out_stride,
    cudaStream_t stream
) {
    if (n_vectors <= 0) return;

    int micro = autotune_similarity_micro_chunk(
        emb,
        rules,
        out,
        dim,
        n_vectors,
        rule_index,
        out_stride,
        stream
    );

    int processed = 0;
    while (processed < n_vectors) {
        int count = std::min(micro, n_vectors - processed);

        launch_similarity_micro_batch(
            emb + processed * dim,
            rules,
            out + processed * out_stride,
            count,
            dim,
            rule_index,
            out_stride,
            stream
        );

        processed += count;
    }
}

