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
__global__ void matmul_v11(const float16_t* A, const float16_t* B, float16_t* C,
                          int M, int N, int K)
{

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    const int APAD = 8;
    const int BPAD = 8;

    extern __shared__ half smem[];
    half* s_a = smem;
    half* s_b = smem + 2 * BM * (BK + APAD);
    int s_a_db_offset = BM * (BK + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2]
                                                                            [4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2]
                                                                            [4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid & 3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 =
        s_a_base_addr +
        OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(half);
    int load_a_smem_addr_1 = load_a_smem_addr_0 + (BK + APAD) * sizeof(half);
    int load_b_smem_addr_0 =
        s_b_base_addr +
        OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(half);
    int load_b_smem_addr_1 = load_b_smem_addr_0 + (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_2 =
        load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(half);
    int load_b_smem_addr_3 =
        load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(half);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    {
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(load_a_smem_addr_0), "l"(&A[load_a_gmem_addr]));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(load_a_smem_addr_1), "l"(&A[load_a_gmem_addr + K]));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(load_b_smem_addr_0), "l"(&B[load_b_gmem_addr]));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(load_b_smem_addr_1), "l"(&B[load_b_gmem_addr + N]));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(load_b_smem_addr_2), "l"(&B[load_b_gmem_addr + 2 * N]));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(load_b_smem_addr_3), "l"(&B[load_b_gmem_addr + 3 * N]));

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

#pragma unroll 32
    for (int bk = 1; bk < K / BK; bk++) {

        int smem_sel = (bk & 1) ^ 1;
        int smem_sel_next = ((bk - 1) & 1) ^ 1;

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(load_a_smem_addr_0 +
                  smem_sel_next * s_a_db_offset * (int) sizeof(half)),
              "l"(&A[load_a_gmem_addr]));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(load_a_smem_addr_1 +
                  smem_sel_next * s_a_db_offset * (int) sizeof(half)),
              "l"(&A[load_a_gmem_addr + K]));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(load_b_smem_addr_0 +
                  smem_sel_next * s_b_db_offset * (int) sizeof(half)),
              "l"(&B[load_b_gmem_addr]));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(load_b_smem_addr_1 +
                  smem_sel_next * s_b_db_offset * (int) sizeof(half)),
              "l"(&B[load_b_gmem_addr + N]));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(load_b_smem_addr_2 +
                  smem_sel_next * s_b_db_offset * (int) sizeof(half)),
              "l"(&B[load_b_gmem_addr + 2 * N]));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(load_b_smem_addr_3 +
                  smem_sel_next * s_b_db_offset * (int) sizeof(half)),
              "l"(&B[load_b_gmem_addr + 3 * N]));

        wmma::load_matrix_sync(frag_a[0][0],
                               &s_a[smem_sel * s_a_db_offset +
                                    (comp_c_frag_m * 64) * (BK + APAD) + 0],
                               BK + APAD);
        wmma::load_matrix_sync(
            frag_a[0][1],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 16) * (BK + APAD) + 0],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[0][2],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 32) * (BK + APAD) + 0],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[0][3],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 48) * (BK + APAD) + 0],
            BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0],
                               &s_a[smem_sel * s_a_db_offset +
                                    (comp_c_frag_m * 64) * (BK + APAD) + 16],
                               BK + APAD);
        wmma::load_matrix_sync(
            frag_a[1][1],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[1][2],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[1][3],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16],
            BK + APAD);

        wmma::load_matrix_sync(
            frag_b[0][0], &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64],
            BN + BPAD);
        wmma::load_matrix_sync(
            frag_b[0][1],
            &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 16],
            BN + BPAD);
        wmma::load_matrix_sync(
            frag_b[0][2],
            &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 32],
            BN + BPAD);
        wmma::load_matrix_sync(
            frag_b[0][3],
            &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 48],
            BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64 + 16],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64 + 32],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64 + 48],
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

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);

        __syncthreads();
    }

    int smem_sel = ((K / BK) & 1) ^ 1;

    wmma::load_matrix_sync(
        frag_a[0][0],
        &s_a[smem_sel * s_a_db_offset + (comp_c_frag_m * 64) * (BK + APAD) + 0],
        BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1],
                           &s_a[smem_sel * s_a_db_offset +
                                (comp_c_frag_m * 64 + 16) * (BK + APAD) + 0],
                           BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2],
                           &s_a[smem_sel * s_a_db_offset +
                                (comp_c_frag_m * 64 + 32) * (BK + APAD) + 0],
                           BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3],
                           &s_a[smem_sel * s_a_db_offset +
                                (comp_c_frag_m * 64 + 48) * (BK + APAD) + 0],
                           BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0],
                           &s_a[smem_sel * s_a_db_offset +
                                (comp_c_frag_m * 64) * (BK + APAD) + 16],
                           BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1],
                           &s_a[smem_sel * s_a_db_offset +
                                (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16],
                           BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2],
                           &s_a[smem_sel * s_a_db_offset +
                                (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16],
                           BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3],
                           &s_a[smem_sel * s_a_db_offset +
                                (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16],
                           BK + APAD);

    wmma::load_matrix_sync(frag_b[0][0],
                           &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64],
                           BN + BPAD);
    wmma::load_matrix_sync(
        frag_b[0][1], &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 16],
        BN + BPAD);
    wmma::load_matrix_sync(
        frag_b[0][2], &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 32],
        BN + BPAD);
    wmma::load_matrix_sync(
        frag_b[0][3], &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 48],
        BN + BPAD);
    wmma::load_matrix_sync(
        frag_b[1][0],
        &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) + comp_c_frag_n * 64],
        BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][1],
                           &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) +
                                comp_c_frag_n * 64 + 16],
                           BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][2],
                           &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) +
                                comp_c_frag_n * 64 + 32],
                           BN + BPAD);
    wmma::load_matrix_sync(frag_b[1][3],
                           &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) +
                                comp_c_frag_n * 64 + 48],
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

PG_MATMUL_SIG(float16_t, 6, M, N, K, A, B, C)
{
    const int BM = 128, BN = 256, BK = 32;
    dim3 blockDim(256);
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;
    dim3 gridDim(BX, BY);

    cudaFuncSetAttribute(matmul_v11,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
    matmul_v11<<<gridDim, blockDim, dsmem>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
}  // namespace playground
 