// @file: ./task-1/src/f32-v2.cu
#include <cuda_runtime.h>
#include "playground/matmul.hpp"

namespace playground {

__global__ void matmul_v2(const float *A, const float *B, float *C, 
                          int M, int N, int K) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (ty < M && tx < N) {
        float c = 0;
        for (int i = 0; i < K; ++i) {
            c += A[ty * K + i] * B[i * N + tx];
        }
        C[ty * N + tx] = c;
    }
}

PG_MATMUL_SIG(float32_t, 2, M, N, K, A, B, C) {
    dim3 blocks(N / 32, M / 32, 1);
    dim3 threads(32, 32, 1);
    playground::matmul_v2<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

} // namespace playground