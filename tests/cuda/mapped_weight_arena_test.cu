// The loader's H2D copy into a mapped alias, followed by real kernel and graph reads.
#include "strata/core/mapped_weight_arena.hpp"
#include <cstdio>
#include <vector>

__global__ void read_weights(const unsigned char* weights, unsigned char* output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = weights[i];
}

#define CHECK_CUDA(expr) do { const cudaError_t e = (expr); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); return 1; } } while (0)

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    constexpr int n = 1 << 20;
    std::vector<unsigned char> input(n), got(n);
    for (int i = 0; i < n; ++i) input[i] = (unsigned char) (i * 37 + i / 256);
    strata::core::MappedWeightArena arena;
    void* device = nullptr;
    if (arena.allocate(&device, 0) != cudaErrorInvalidValue || device != nullptr) return 1;
    CHECK_CUDA(arena.allocate(&device, n));
    unsigned char* output = nullptr;
    CHECK_CUDA(cudaMalloc(&output, n));
    // This is how WeightTable::load writes transformed pack bytes to its arena.
    CHECK_CUDA(cudaMemcpy(device, input.data(), n, cudaMemcpyHostToDevice));
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    read_weights<<<n / 256, 256, 0, stream>>>((const unsigned char*) device, output, n);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(got.data(), output, n, cudaMemcpyDeviceToHost));
    if (got != input) return 1;
    // A repeated allocation must not replace or leak the first allocation.
    void* second = nullptr;
    if (arena.allocate(&second, n) != cudaErrorInvalidValue || second != nullptr) return 1;
    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(output));
    std::puts("mapped weight arena: loader copy and captured GPU read match every byte");
    return 0;
}
