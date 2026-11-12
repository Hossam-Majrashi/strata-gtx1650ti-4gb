// Opt-in CUDA weight storage for cards whose fixed weights leave no room for a session.
#pragma once

#include "strata/core/pin_budget.hpp"
#include <cuda_runtime.h>

namespace strata::core {

// The device alias is read by the existing kernels. No conversion or requantization is done.
// Keep this owner alive until all readers and captured graphs have finished using the weights.
class MappedWeightArena {
public:
    MappedWeightArena() = default;
    MappedWeightArena(const MappedWeightArena&) = delete;
    MappedWeightArena& operator=(const MappedWeightArena&) = delete;
    ~MappedWeightArena() { if (host_) cudaFreeHost(host_); }

    cudaError_t allocate(void** device, size_t bytes) {
        *device = nullptr;
        if (host_ || bytes == 0) return cudaErrorInvalidValue;
        cudaError_t e = host_alloc_guarded(&host_, bytes, cudaHostAllocMapped | cudaHostAllocPortable);
        if (e != cudaSuccess) return e;
        e = cudaHostGetDevicePointer(device, host_, 0);
        if (e != cudaSuccess) {
            cudaFreeHost(host_);
            host_ = nullptr;
        }
        return e;
    }

private:
    void* host_ = nullptr;
};

}  // namespace strata::core
