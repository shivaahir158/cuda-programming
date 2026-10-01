#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                       \
    do {                                                       \
        cudaError_t error = (call);                            \
        if (error != cudaSuccess) {                            \
            std::cerr << "CUDA error: "                        \
                      << cudaGetErrorString(error)             \
                      << " at line " << __LINE__ << '\n';      \
            std::exit(EXIT_FAILURE);                           \
        }                                                      \
    } while (0)

constexpr int TILE = 16;

// A: M x K, B: K x N, C: M x N. All matrices are row-major.
__global__ void matmul(const float* A, const float* B, float* C,
                       int M, int K, int N) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        // Zero padding handles dimensions not divisible by TILE.
        tileA[ty][tx] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        tileB[ty][tx] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        // Wait until every thread has loaded its input elements.
        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        // Wait before overwriting shared memory with the next tiles.
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 256;
    const int K = 128;
    const int N = 192;

    const size_t bytesA = size_t(M) * K * sizeof(float);
    const size_t bytesB = size_t(K) * N * sizeof(float);
    const size_t bytesC = size_t(M) * N * sizeof(float);

    std::vector<float> A(size_t(M) * K);
    std::vector<float> B(size_t(K) * N);
    std::vector<float> C(size_t(M) * N);

    for (size_t i = 0; i < A.size(); ++i)
        A[i] = (int(i % 17) - 8) / 17.0f;

    for (size_t i = 0; i < B.size(); ++i)
        B[i] = (int(i % 13) - 6) / 13.0f;

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dA), bytesA));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dB), bytesB));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dC), bytesC));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytesB, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);  // 256 threads per block
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // Warm up the GPU.
    matmul<<<grid, block>>>(dA, dB, dC, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    constexpr int repetitions = 100;
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repetitions; ++i)
        matmul<<<grid, block>>>(dA, dB, dC, M, K, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

    CUDA_CHECK(cudaMemcpy(C.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    // Verify every output against a CPU calculation.
    bool passed = true;
    double maxError = 0.0;

    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double reference = 0.0;

            for (int k = 0; k < K; ++k)
                reference += double(A[row * K + k]) * B[k * N + col];

            double actual = C[row * N + col];
            double error = std::abs(actual - reference);

            if (error > maxError)
                maxError = error;

            if (!std::isfinite(actual) ||
                error > 1e-4 + 1e-4 * std::abs(reference))
                passed = false;
        }
    }

    std::cout << "A: " << M << " x " << K << '\n'
              << "B: " << K << " x " << N << '\n'
              << "C: " << M << " x " << N << '\n'
              << "Average GPU time: " << elapsedMs / repetitions
              << " ms (excludes memory transfers)\n"
              << "Maximum absolute error: " << maxError << '\n'
              << "Verification: " << (passed ? "PASS" : "FAIL") << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
