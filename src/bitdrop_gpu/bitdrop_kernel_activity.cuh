#pragma once

// ============================================================
//  ACTIVITY + SKIM PRUNING (A‑MODE)
// ============================================================

__device__ __forceinline__ float bd_activity(
    const void* __restrict emb_raw,
    int dim)
{
    int lane = threadIdx.x & 31;
    float activity = 0.0f;

    for (int i = lane; i < dim; i += 32 * 8) {
#if USE_FP8_EMB
        const uint8_t* v = reinterpret_cast<const uint8_t*>(emb_raw);
        activity += (v[i] != 0);
#else
        const __half* v = reinterpret_cast<const __half*>(emb_raw);
        activity += (__half2float(v[i]) != 0.0f);
#endif
    }

    for (int off = 16; off > 0; off >>= 1)
        activity += __shfl_down_sync(0xffffffff, activity, off);

    return activity;
}

__device__ __forceinline__ bool bd_should_skip_vector(
    float activity,
    float skim_threshold)
{
    // A‑MODE: aggressive pruning
    return (activity < skim_threshold);
}
