// Exact-byte prefix relocation plus pooled weights consumed on a nonblocking graph stream.
#include "strata/core/mapped_weight_arena.hpp"
#include "strata/core/cuda_weight_alloc.hpp"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "failed: %s\n", #expr); return 1; } } while (0)
#define CUDA(expr) CHECK((expr) == cudaSuccess)

__global__ void read_pair(const unsigned char* a, const unsigned char* b, unsigned char* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = a[i]; out[n + i] = b[i]; }
}

struct TempPack {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("strata-weight-cache-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TempPack() { std::filesystem::create_directories(path); }
    ~TempPack() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    constexpr int n = 4096;
    std::vector<unsigned char> source(3 * n), got(2 * n);
    for (int i = 0; i < 3 * n; ++i) source[i] = (unsigned char) (i * 37 + i / 256);
    TempPack pack;
    {
        std::ofstream index(pack.path / "index.txt");
        index << "# align 256 pool 12288 tensors 3\n";
        const char* names[] = {"first.weight", "second.weight", "skip.weight"};
        for (int i = 0; i < 3; ++i)
            index << names[i] << " 0 2 " << i * n << " 4096 " << i * n
                  << " 4096 1024 0 0 0 1 0 0 0 0 0 0 0\n";
        std::ofstream file(pack.path / "dense.bin", std::ios::binary);
        file.write((const char*) source.data(), source.size());
    }
    strata::core::MappedWeightArena arena;
    strata::core::WeightTable table;
    std::string err;
    std::set<std::string> skip{"skip.weight"};
    uint64_t bytes = 0;
    CHECK(strata::core::WeightTable::pool_bytes(pack.path.string(), bytes, err, &skip));
    CHECK(bytes == 2 * n);
    void* mapped = nullptr;
    CUDA(arena.allocate(&mapped, bytes));
    CHECK(table.load(pack.path.string(), mapped, bytes, err, &skip));
    const auto* a = table.find("first.weight");
    const auto* b = table.find("second.weight");
    const auto* skipped = table.find("skip.weight");
    CHECK(a && b && skipped && !skipped->data && !skipped->resident);
    const void* old_a = a->data;
    const void* old_b = b->data;
    CHECK(arena.cache_prefix(table, n / 2) == cudaErrorInvalidValue);
    CHECK(a->data == old_a && b->data == old_b);
    // Budget ends halfway through tensor 2. Only tensor 1 moves; no split tensors.
    CUDA(arena.cache_prefix(table, n + n / 2));
    CHECK(arena.cached_bytes() == n && arena.cached_tensors() == 1);
    CHECK(a->data != old_a && b->data == old_b && !skipped->data);
    CHECK(a->bytes == n && a->ne0 == 1024 && a->kind == strata::core::WeightKind::F32);
    cudaPointerAttributes attributes{};
    CUDA(cudaPointerGetAttributes(&attributes, a->data));
    CHECK(attributes.type == cudaMemoryTypeDevice);
    CHECK(arena.cache_prefix(table, bytes) == cudaErrorInvalidValue);
    unsigned char* output = nullptr;
    CUDA(cudaMalloc(&output, got.size()));
    cudaStream_t stream;
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    read_pair<<<n / 256, 256, 0, stream>>>((const unsigned char*) a->data,
                                         (const unsigned char*) b->data, output, n);
    CUDA(cudaStreamEndCapture(stream, &graph));
    CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CUDA(cudaGraphLaunch(exec, stream));
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(got.data(), output, got.size(), cudaMemcpyDeviceToHost));
    CHECK(std::equal(got.begin(), got.end(), source.begin()));
    CUDA(cudaGraphExecDestroy(exec));
    CUDA(cudaGraphDestroy(graph));

    int device = 0, supported = 0;
    CUDA(cudaGetDevice(&device));
    CUDA(cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device));
    if (supported) {
        void* pooled_a = nullptr;
        void* pooled_b = nullptr;
        CUDA(strata::core::allocate_weight(&pooled_a, n, true));
        CUDA(strata::core::allocate_weight(&pooled_b, n, true));
        CUDA(cudaMemcpy(pooled_a, source.data(), n, cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(pooled_b, source.data() + n, n, cudaMemcpyHostToDevice));
        CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        read_pair<<<n / 256, 256, 0, stream>>>((const unsigned char*) pooled_a,
                                             (const unsigned char*) pooled_b, output, n);
        CUDA(cudaStreamEndCapture(stream, &graph));
        CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        CUDA(cudaGraphLaunch(exec, stream));
        CUDA(cudaStreamSynchronize(stream));
        CUDA(cudaMemcpy(got.data(), output, got.size(), cudaMemcpyDeviceToHost));
        CHECK(std::equal(got.begin(), got.end(), source.begin()));
        CUDA(cudaGraphExecDestroy(exec));
        CUDA(cudaGraphDestroy(graph));
        CUDA(cudaFree(pooled_a));
        CUDA(cudaFree(pooled_b));
    } else std::puts("pooled subtest skipped: device does not support CUDA memory pools");
    CUDA(cudaStreamDestroy(stream));
    CUDA(cudaFree(output));
    std::puts("weight cache: full-tensor boundaries, metadata, mixed host/device graph and pooled bytes passed");
    return 0;
}
