// Opt-in CUDA weight storage for cards whose fixed weights leave no room for a session.
#pragma once

#include "strata/core/pin_budget.hpp"
#include "strata/core/weights.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace strata::core {

// The device alias is read by the existing kernels. No conversion or requantization is done.
// Keep this owner alive until all readers and captured graphs have finished using the weights.
class MappedWeightArena {
public:
    MappedWeightArena() = default;
    MappedWeightArena(const MappedWeightArena&) = delete;
    MappedWeightArena& operator=(const MappedWeightArena&) = delete;
    ~MappedWeightArena() {
        if (cache_) cudaFree(cache_);
        if (host_) cudaFreeHost(host_);
    }

    cudaError_t allocate(void** device, size_t bytes) {
        *device = nullptr;
        if (host_ || bytes == 0) return cudaErrorInvalidValue;
        cudaError_t e = host_alloc_guarded(&host_, bytes, cudaHostAllocMapped | cudaHostAllocPortable);
        if (e != cudaSuccess) return e;
        e = cudaHostGetDevicePointer(device, host_, 0);
        if (e != cudaSuccess) {
            cudaFreeHost(host_);
            host_ = nullptr;
        } else {
            alias_ = *device;
            bytes_ = bytes;
        }
        return e;
    }

    // Optional exact-byte VRAM cache. Call once, after loading, BEFORE sessions/graphs bind the
    // table. A boundary inside a tensor is rounded back: a tensor is never split across memories.
    // Prefix selection follows the pack's storage order, not a calibration/quality heuristic.
    cudaError_t cache_prefix(WeightTable& table, size_t budget) {
        if (!host_ || !alias_ || cache_ || !budget) return cudaErrorInvalidValue;
        const size_t limit = std::min(budget, bytes_);
        const uintptr_t base = reinterpret_cast<uintptr_t>(alias_);
        size_t end = 0;
        for (const auto& [name, weight] : table.table_) {
            const uintptr_t address = reinterpret_cast<uintptr_t>(weight.data);
            if (!weight.data || address < base || address - base > limit) continue;
            const size_t offset = (size_t) (address - base);
            if (weight.bytes && weight.bytes <= limit - offset)
                end = std::max(end, offset + (size_t) weight.bytes);
        }
        if (!end) return cudaErrorInvalidValue;
        void* copy = nullptr;
        cudaError_t status = cudaMalloc(&copy, end);
        if (status != cudaSuccess) return status;
        status = cudaMemcpy(copy, host_, end, cudaMemcpyHostToDevice);
        if (status != cudaSuccess) { cudaFree(copy); return status; }
        // Publish only after the complete copy succeeds. Native overrides, skipped tensors and
        // tensors outside this arena keep their pointers and metadata unchanged.
        for (auto& [name, weight] : table.table_) {
            const uintptr_t address = reinterpret_cast<uintptr_t>(weight.data);
            if (!weight.data || address < base || address - base > end) continue;
            const size_t offset = (size_t) (address - base);
            if (weight.bytes && weight.bytes <= end - offset) {
                weight.data = static_cast<const uint8_t*>(copy) + offset;
                ++cached_tensors_;
            }
        }
        cache_ = copy;
        cached_bytes_ = end;
        return cudaSuccess;
    }

    size_t cached_bytes() const { return cached_bytes_; }
    size_t cached_tensors() const { return cached_tensors_; }

private:
    void* host_ = nullptr;
    void* alias_ = nullptr;
    void* cache_ = nullptr;
    size_t bytes_ = 0;
    size_t cached_bytes_ = 0;
    size_t cached_tensors_ = 0;
};

}  // namespace strata::core
