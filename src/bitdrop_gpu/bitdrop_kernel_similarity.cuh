#pragma once

// ============================================================
//  SIMILARITY ENGINE (A‑MODE)
// ============================================================

__device__ __forceinline__ float bd_similarity_coarse(
    const void* __restrict emb_raw,
    int dim,
    const uint8_t* __restrict payload)
{
    float acc = 0.0f;

    for (int t = 0; t < dim; t += TILE * VEC_WIDTH) {
#pragma unroll
        for (int inner = 0; inner < TILE; ++inner) {
            int base_idx = t + inner * VEC_WIDTH;
            if (base_idx >= dim) break;

            float vvals[VEC_WIDTH];
            load_emb_vec(emb_raw, base_idx, vvals, dim);

#pragma unroll
            for (int k = 0; k < VEC_WIDTH; ++k) {
                int idx = base_idx + k;
                if (idx >= dim) break;

#if USE_FP8_SIM_PAYLOAD
                float pval = fp8_e4m3_to_float(payload[idx]);
#else
                const __half* hp = reinterpret_cast<const __half*>(payload);
                float pval = __half2float(hp[idx]);
#endif
                acc += vvals[k] * pval;
            }
        }
    }

    return acc;
}

__device__ __forceinline__ float bd_similarity_refine(
    const void* __restrict emb_raw,
    int dim,
    const uint8_t* __restrict payload)
{
    // A‑MODE: smaller TILE for refinement
    const int REF_TILE = TILE / 2;
    float acc = 0.0f;

    for (int t = 0; t < dim; t += REF_TILE * VEC_WIDTH) {
#pragma unroll
        for (int inner = 0; inner < REF_TILE; ++inner) {
            int base_idx = t + inner * VEC_WIDTH;
            if (base_idx >= dim) break;

            float vvals[VEC_WIDTH];
            load_emb_vec(emb_raw, base_idx, vvals, dim);

#pragma unroll
            for (int k = 0; k < VEC_WIDTH; ++k) {
                int idx = base_idx + k;
                if (idx >= dim) break;

#if USE_FP8_SIM_PAYLOAD
                float pval = fp8_e4m3_to_float(payload[idx]);
#else
                const __half* hp = reinterpret_cast<const __half*>(payload);
                float pval = __half2float(hp[idx]);
#endif
                acc += vvals[k] * pval;
            }
        }
    }

    return acc;
}
