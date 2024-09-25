// @file: ./task-1/src/f32-v5.cu
#include "playground/matmul.hpp"
#include <cstring>
#include <cuda_runtime.h>

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

__global__ void matmul_v5(const float* A, const float* B, float* C, int M,
                          int N, int K) {
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

    __shared__ float s_a[BK][BM];
    __shared__ float s_b[BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

#pragma unroll
    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = offset(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = offset(load_b_gmem_k, load_b_gmem_n, N);
        float4_ref(r_load_a[0]) = float4_const_ref(A[load_a_gmem_addr]);
        float4_ref(r_load_b[0]) = float4_const_ref(B[load_b_gmem_addr]);

        s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        float4_ref(s_b[load_b_smem_k][load_b_smem_n]) = float4_ref(r_load_b[0]);

        __syncthreads();

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            float4_ref(r_comp_a[0]) = float4_ref(s_a[tk][ty * TM / 2]);
            float4_ref(r_comp_a[4]) = float4_ref(s_a[tk][ty * TM / 2 + BM / 2]);
            float4_ref(r_comp_b[0]) = float4_ref(s_b[tk][tx * TN / 2]);
            float4_ref(r_comp_b[4]) = float4_ref(s_b[tk][tx * TN / 2 + BN / 2]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = offset(store_c_gmem_m, store_c_gmem_n, N);
        float4_ref(C[store_c_gmem_addr]) = float4_ref(r_c[i][0]);
        float4_ref(C[store_c_gmem_addr + BN / 2]) = float4_ref(r_c[i][4]);
    }
    
#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = offset(store_c_gmem_m, store_c_gmem_n, N);
        float4_ref(C[store_c_gmem_addr]) = float4_ref(r_c[i + TM / 2][0]);
        float4_ref(C[store_c_gmem_addr + BN / 2]) = float4_ref(r_c[i + TM / 2][4]);
    }
}

PG_MATMUL_SIG(float32_t, 5, M, N, K, A, B, C) {
    dim3 blocks((N + 127) / 128, (M + 127) / 128, 1);
    dim3 threads(16, 16, 1);
    playground::matmul_v5<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

} // namespace playground
