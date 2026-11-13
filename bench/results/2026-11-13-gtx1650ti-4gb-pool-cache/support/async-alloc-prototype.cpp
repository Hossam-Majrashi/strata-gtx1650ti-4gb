// Local measurement shim only. Not a production implementation.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
extern "C" cudaError_t cudaMalloc(void** p, size_t n) {
    using Alloc = cudaError_t(*)(void**,size_t,cudaStream_t);
    using Sync = cudaError_t(*)(cudaStream_t);
    static auto alloc=(Alloc)dlsym(RTLD_NEXT,"cudaMallocAsync");
    static auto sync=(Sync)dlsym(RTLD_NEXT,"cudaStreamSynchronize");
    if(!alloc||!sync)return cudaErrorNotSupported;
    auto result=alloc(p,n,nullptr);
    if(result==cudaSuccess)result=sync(nullptr);
    return result;
}
extern "C" cudaError_t cudaFree(void* p) {
    using Free = cudaError_t(*)(void*);
    using Sync = cudaError_t(*)();
    static auto freep=(Free)dlsym(RTLD_NEXT,"cudaFree");
    static auto sync=(Sync)dlsym(RTLD_NEXT,"cudaDeviceSynchronize");
    if(!freep||!sync)return cudaErrorNotSupported;
    auto status=sync();
    if(status!=cudaSuccess)return status;
    return freep(p);
}
