// @file: ./task-1/src/f32-v3.cu
#include <cuda_runtime.h>
#include "playground/matmul.hpp"

const int TILE_SIZE = 32;

namespace playground {

__global__ void matmul_v3(const float *A, const float *B, float *C, 
                          int M, int N, int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    float *begin_a = const_cast<float*>(A) + by * blockDim.y * K;
    float *begin_b = const_cast<float*>(B) + bx * blockDim.x;
    float *end_a = begin_a + K;

    float sum = 0.f;
    for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
        a_ptr += blockDim.y, b_ptr += blockDim.x * N) {
        __shared__ float ashare[TILE_SIZE][TILE_SIZE];
        __shared__ float bshare[TILE_SIZE][TILE_SIZE];

        ashare[ty][tx] = a_ptr[ty * K + tx];
        bshare[ty][tx] = b_ptr[ty * M + tx];
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < blockDim.y; ++kk) {
            sum += ashare[ty][kk] * bshare[kk][tx];
        }
        __syncthreads();
    }

    C[(blockDim.y * by + ty) * N + blockDim.x * bx + tx] = sum;
}

PG_MATMUL_SIG(float32_t, 3, M, N, K, A, B, C) {
    dim3 blocks(N / 32, M / 32, 1);
    dim3 threads(32, 32, 1);
    playground::matmul_v3<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

} // namespace playground