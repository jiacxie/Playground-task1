// @file: ./task-1/src/f32-v4.cu
#include <cstring>
#include <cuda_runtime.h>
#include "playground/matmul.hpp"

namespace playground {

__device__ inline int offset(int row, int col, int ld) {
    return row * ld + col;
}

__device__ inline float4& float4_ref(float& pointer) {
    return reinterpret_cast<float4*>(&pointer)[0];
}

__device__ inline const float4& float4_const_ref(const float& pointer) {
    return reinterpret_cast<const float4*>(&pointer)[0];
}

__global__ void matmul_v4(const float *A, const float *B, float *C, int M, int N, int K) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ __align__(32 * 1024) float s_a[BM][BK];
    __shared__ __align__(32 * 1024) float s_b[BK][BN];

    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = offset(load_a_gmem_m, load_a_gmem_k, K);
        float4_ref(s_a[load_a_smem_m][load_a_smem_k]) = float4_const_ref(A[load_a_gmem_addr]);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = offset(load_b_gmem_k, load_b_gmem_n, N);
        float4_ref(s_b[load_b_smem_k][load_b_smem_n]) = float4_const_ref(B[load_b_gmem_addr]);

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = offset(store_c_gmem_m, store_c_gmem_n, N);
            float4_ref(C[store_c_gmem_addr]) = float4_ref(r_c[i][j]);
        }
    }
}

PG_MATMUL_SIG(float32_t, 4, M, N, K, A, B, C) {
    dim3 blocks((N + 127) / 128, (M + 127) / 128, 1);
    dim3 threads(16, 16, 1);
    playground::matmul_v4<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

} // namespace playground
