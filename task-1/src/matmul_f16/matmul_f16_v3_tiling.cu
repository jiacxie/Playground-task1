// @file: ./task-1/src/f16-v2.cu
#include "playground/matmul.hpp"
#include "playground/system.hpp"
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FLOAT4_CONST(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

namespace playground
{
using namespace nvcuda;
__global__ void matmul_v8(const float16_t* A, const float16_t* B, float16_t* C,
                          int M, int N, int K) {

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2]
                                                                            [4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2]
                                                                            [4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

#pragma unroll
    for (auto & i : frag_c) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(i[j], 0.0);
        }
    }

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid & 3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    for (int bk = 0; bk < K / BK; bk++) {
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4_CONST(A[load_a_gmem_addr]);
        FLOAT4(s_a[load_a_smem_m + 1][load_a_smem_k]) =
            FLOAT4_CONST(A[load_a_gmem_addr + K]);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4_CONST(B[load_b_gmem_addr]);
        FLOAT4(s_b[load_b_smem_k + 1][load_b_smem_n]) =
            FLOAT4_CONST(B[load_b_gmem_addr + N]);
        FLOAT4(s_b[load_b_smem_k + 2][load_b_smem_n]) =
            FLOAT4_CONST(B[load_b_gmem_addr + 2 * N]);
        FLOAT4(s_b[load_b_smem_k + 3][load_b_smem_n]) =
            FLOAT4_CONST(B[load_b_gmem_addr + 3 * N]);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        __syncthreads();

        wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64][0],
                               BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][0],
                               BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][0],
                               BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][0],
                               BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64][16],
                               BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16],
                               BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16],
                               BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16],
                               BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[0][comp_c_frag_n * 64],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[0][comp_c_frag_n * 64 + 16],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[0][comp_c_frag_n * 64 + 32],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[0][comp_c_frag_n * 64 + 48],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48],
                               BN + BPAD);

#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j],
                               frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j],
                               frag_c[i][j]);
            }
        }

        __syncthreads();
    }

    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&C[store_c_gmem_addr + i * 16 * N + j * 16],
                                    frag_c[i][j], N, wmma::mem_row_major);
        }
    }
}

PG_MATMUL_SIG(float16_t, 3, M, N, K, A, B, C)
{
    dim3 blocks((N + 256 - 1) / 256, (M + 128 - 1) / 128, 1);
    dim3 threads(256, 1, 1);
    playground::matmul_v8<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
}  // namespace playground
