#include "bitdrop_gpu.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cstdint>

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t _err = (expr);                                           \
        if (_err != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",             \
                         #expr, __FILE__, __LINE__, cudaGetErrorString(_err)); \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// Host-safe FP32 -> FP16 conversion (no CUDA intrinsics)
static inline std::uint16_t float_to_half(float f) {
    std::uint32_t x = *reinterpret_cast<std::uint32_t*>(&f);
    std::uint32_t sign = (x >> 16) & 0x8000u;
    std::uint32_t mant = x & 0x007FFFFFu;
    std::uint32_t exp  = x & 0x7F800000u;

    if (exp >= 0x47800000u) {
        // Inf / NaN -> Inf
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    if (exp <= 0x38000000u) {
        // Underflow -> signed zero
        return static_cast<std::uint16_t>(sign);
    }

    std::uint32_t half = sign | ((exp - 0x38000000u) >> 13) | (mant >> 13);
    return static_cast<std::uint16_t>(half);
}

int main() {
    const int dim           = 128;
    const int num_vecs      = 1'000'000;
    const int num_rules     = 64;
    const int bits_per_bank = 64;
    const int num_banks     = 1;
    const int total_bits    = bits_per_bank * num_banks;
    const int iters         = 50;

    std::printf("BitDrop GPU benchmark\n");
    std::printf("dim=%d, num_vecs=%d, num_rules=%d, iters=%d\n",
                dim, num_vecs, num_rules, iters);

    // ------------------------------------------------------------
    // Host embeddings (FP16)
    // ------------------------------------------------------------
    std::vector<std::uint16_t> h_emb(num_vecs * dim);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < num_vecs * dim; ++i) {
        h_emb[i] = float_to_half(dist(rng));
    }

    // ------------------------------------------------------------
    // Device embeddings
    // ------------------------------------------------------------
    void* d_emb = nullptr;
    std::size_t emb_bytes = h_emb.size() * sizeof(std::uint16_t);
    CUDA_CHECK(cudaMalloc(&d_emb, emb_bytes));
    CUDA_CHECK(cudaMemcpy(d_emb, h_emb.data(), emb_bytes, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------
    // Rules (device)
    // ------------------------------------------------------------
    std::vector<BitDropRule> h_rules(num_rules);
    for (int i = 0; i < num_rules; ++i) {
        h_rules[i].type           = BITDROP_RULE_THRESHOLD;
        h_rules[i].bit            = static_cast<std::uint8_t>(i % 64);
        h_rules[i].dim            = static_cast<std::uint16_t>(i % dim);
        h_rules[i].payload_offset = 0;
    }

    BitDropRule* d_rules = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rules, num_rules * sizeof(BitDropRule)));
    CUDA_CHECK(cudaMemcpy(d_rules, h_rules.data(),
                          num_rules * sizeof(BitDropRule),
                          cudaMemcpyHostToDevice));

    const std::uint8_t* d_payload = nullptr;

    // ------------------------------------------------------------
    // Bank descriptor (HOST)
    // ------------------------------------------------------------
    BitDropBank bank_host{};
    bank_host.rules         = d_rules;
    bank_host.payload       = d_payload;
    bank_host.num_rules     = num_rules;
    bank_host.bits_per_bank = bits_per_bank;
    bank_host.bit_offset    = 0;
    bank_host.payload_bytes = 0;

    const BitDropBank* banks_host = &bank_host;

    // ------------------------------------------------------------
    // Output buffers (device)
    // ------------------------------------------------------------
    std::uint8_t* d_out_bits  = nullptr;
    std::int32_t* d_out_tags  = nullptr;
    std::uint8_t* d_skim_mask = nullptr;

    std::size_t out_bits_bytes =
        static_cast<std::size_t>(num_vecs) * total_bits / 8;

    CUDA_CHECK(cudaMalloc(&d_out_bits, out_bits_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_tags, sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&d_skim_mask, num_vecs));

    // ------------------------------------------------------------
    // Initialize BitDrop
    // ------------------------------------------------------------
    BitDropStatus st = bitdrop_init(dim, num_rules, bits_per_bank);
    if (st != BITDROP_OK) {
        std::fprintf(stderr, "bitdrop_init failed: %d\n", st);
        return 1;
    }

    // ------------------------------------------------------------
    // Warmup
    // ------------------------------------------------------------
    st = bitdrop_collapse_multi(
        d_emb,
        num_vecs,
        dim,
        banks_host,
        num_banks,
        total_bits,
        d_out_bits,
        d_out_tags,
        0,
        d_skim_mask,
        0,
        0.0f,
        1,
        0
    );
    if (st != BITDROP_OK) {
        std::fprintf(stderr, "bitdrop_collapse_multi (warmup) failed: %d\n", st);
        return 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------
    // Timed loop
    // ------------------------------------------------------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        st = bitdrop_collapse_multi(
            d_emb,
            num_vecs,
            dim,
            banks_host,
            num_banks,
            total_bits,
            d_out_bits,
            d_out_tags,
            0,
            d_skim_mask,
            0,
            0.0f,
            1,
            0
        );
        if (st != BITDROP_OK) {
            std::fprintf(stderr, "bitdrop_collapse_multi (iter %d) failed: %d\n", i, st);
            return 1;
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    double seconds     = ms / 1000.0;
    double total_vecs  = static_cast<double>(num_vecs) * iters;
    double total_bytes = static_cast<double>(num_vecs) * dim *
                         sizeof(std::uint16_t) * iters;

    double vecs_per_sec = total_vecs / seconds;
    double gb_per_sec   = (total_bytes / 1e9) / seconds;

    std::printf("Elapsed: %.3f ms over %d iters\n", ms, iters);
    std::printf("Vectors/sec: %.3f M\n", vecs_per_sec / 1e6);
    std::printf("Throughput: %.3f GB/s (FP16 input only)\n", gb_per_sec);

    // ------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------
    bitdrop_shutdown();

    CUDA_CHECK(cudaFree(d_emb));
    CUDA_CHECK(cudaFree(d_rules));
    CUDA_CHECK(cudaFree(d_out_bits));
    CUDA_CHECK(cudaFree(d_out_tags));
    CUDA_CHECK(cudaFree(d_skim_mask));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}



