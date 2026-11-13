// Optional pooled CUDA allocations for immutable weights, outside every graph capture.
#pragma once
#include <cuda_runtime.h>

namespace strata::core {
inline cudaError_t allocate_weight(void** out, size_t bytes, bool pooled) {
    if (!pooled) return cudaMalloc(out, bytes); // The original allocation path.
    *out = nullptr;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020 && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
    // The load's consumers include nonblocking streams. Finish the allocation before publishing
    // its pointer: relying on legacy-default-stream ordering would not order those consumers.
    cudaError_t status = cudaMallocAsync(out, bytes, nullptr);
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(nullptr);
        if (status != cudaSuccess) {
            (void) cudaFree(*out);
            *out = nullptr;
        }
    }
    // cudaFree accepts pooled allocations too. The owner must finish all graph readers first,
    // as NativeDense already requires. No release-threshold or other process-wide pool policy changes.
    return status;
#elif defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL)
    // The compatibility runtimes expose this unsupported-operation alias. NativeDense rejects
    // a pooled request with a specific message before calling us on these backends.
    return cudaErrorStreamCaptureUnsupported;
#else
    return cudaErrorInvalidValue; // CUDA toolkit older than 11.2.
#endif
}
} // namespace strata::core
