// // @file: ./task-1/src/f16-v2.cu
// #include "playground/matmul.hpp"
// #include "playground/system.hpp"
// #include <cassert>
// #include <cstring>
// #include <cuda_runtime.h>
// #include <iostream>
// #include <mma.h>

// #define OFFSET(row, col, ld) ((row) * (ld) + (col))
// #define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
// #define FLOAT4_CONST(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

// namespace playground
// {
// using namespace nvcuda;
// __global__ void matmul_v11_mma(const half* A, const half* B, half* C, int M,
//                                int N, int K)
// {
//     const int BM = 128;
//     const int BN = 256;
//     const int BK = 32;

//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int tid = threadIdx.x;
//     int wid = tid >> 5;

//     const int APAD = 8;
//     const int BPAD = 8;

//     extern __shared__ half smem[];

//     half* s_a[3];
//     half* s_b[3];

//     s_a[0] = smem;
//     s_a[1] = s_a[0] + BM * (BK + APAD);
//     s_a[2] = s_a[1] + BM * (BK + APAD);

//     s_b[0] = s_a[2] + BM * (BK + APAD);
//     s_b[1] = s_b[0] + BK * (BN + BPAD);
//     s_b[2] = s_b[1] + BK * (BN + BPAD);

//     int s_a_db_offset = BM * (BK + APAD);
//     int s_b_db_offset = BK * (BN + BPAD);

//     wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[3]
//                                                                             [4];
//     wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[3]
//                                                                             [4];
//     wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

// // 碎片初始化
// #pragma unroll
//     for (int i = 0; i < 4; i++) {
// #pragma unroll
//         for (int j = 0; j < 4; j++) {
//             wmma::fill_fragment(frag_c[i][j], 0.0f);
//         }
//     }

//     int load_a_smem_m = (tid >> 2) << 1;
//     int load_a_smem_k = (tid & 3) << 3;
//     int load_b_smem_k = (tid >> 5) << 2;
//     int load_b_smem_n = (tid & 31) << 3;

//     int load_a_gmem_m = by * BM + load_a_smem_m;
//     int load_b_gmem_n = bx * BN + load_b_smem_n;

//     int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
//     int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

//     int comp_c_frag_m = wid & 1;
//     int comp_c_frag_n = wid >> 1;

//     // Tallies for next load
//     int next_load_a_gmem_addr = load_a_gmem_addr + BK;
//     int next_load_b_gmem_addr = load_b_gmem_addr + BK * N;

//     // Load initial data to buffer 0
//     asm("cp.async.bulk.shared.global [%0], [%1], 16;\n"
//         :
//         : "l"(__cvta_generic_to_shared(
//               s_a[0] + OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD))),
//           "l"(A + load_a_gmem_addr));
//     asm("cp.async.bulk.shared.global [%0], [%1], 16;\n"
//         :
//         : "l"(__cvta_generic_to_shared(
//               s_a[0] + OFFSET(load_a_smem_m + 1, load_a_smem_k, BK + APAD))),
//           "l"(A + load_a_gmem_addr + K));
//     asm("cp.async.bulk.shared.global [%0], [%1], 16;\n"
//         :
//         : "l"(__cvta_generic_to_shared(
//               s_b[0] + OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD))),
//           "l"(B + load_b_gmem_addr));
//     asm("cp.async.bulk.shared.global [%0], [%1], 16;\n"
//         :
//         : "l"(__cvta_generic_to_shared(
//               s_b[0] + OFFSET(load_b_smem_k + 1, load_b_smem_n, BN + BPAD))),
//           "l"(B + load_b_gmem_addr + N));
//     asm("cp.async.bulk.shared.global [%0], [%1], 16;\n"
//         :
//         : "l"(__cvta_generic_to_shared(
//               s_b[0] + OFFSET(load_b_smem_k + 2, load_b_smem_n, BN + BPAD))),
//           "l"(B + load_b_gmem_addr + 2 * N));
//     asm("cp.async.bulk.shared.global [%0], [%1], 16;\n"
//         :
//         : "l"(__cvta_generic_to_shared(
//               s_b[0] + OFFSET(load_b_smem_k + 3, load_b_smem_n, BN + BPAD))),
//           "l"(B + load_b_gmem_addr + 3 * N));

//     asm("cp.async.bulk.commit_group 0;\n" ::);
//     asm("cp.async.bulk.wait_group 0;\n" ::);

//     __syncthreads();

//     // 定义buffer索引
//     int current_buffer = 0;
//     int next_buffer = 1;
//     int compute_buffer = 2;

//     for (int bk = 1; bk < K / BK; bk++) {
//         // 交换buffer的顺序
//         int temp = compute_buffer;
//         compute_buffer = current_buffer;
//         current_buffer = next_buffer;
//         next_buffer = temp;

//         // 加载下一组片段
//         load_a_gmem_addr += BK;
//         load_b_gmem_addr += BK * N;

//         asm("cp.async.bulk.shared.global [%0], [%1], 16;\n"
//             :
//             : "l"(__cvta_generic_to_shared(
//                   s_a[next_buffer] +
//                   OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD))),
//               "l"(A + load_a_gmem_addr));
//         asm("cp.async.bulk.shared.global [%0], [%1], 16;\n"
//             :
//             : "l"(__cvta_generic_to_shared(
//                   s_a[next_buffer] +
//                   OFFSET(load_a_smem_m + 1, load_a_smem_k, BK + APAD))),
//               "l"(A + load_a_gmem_addr + K));
//         asm("cp.async.bulk.shared.global [%0], [%1], 16;\n"
//             :
//             : "l"(__cvta_generic_to_shared(
//                   s_b[next_buffer] +
//                   OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD))),
//               "l"(B + load_b_gmem_addr));
//         asm("cp.async.bulk.shared.global [%0], [%1], 16;\n"
//             :
//             : "l"(__cvta_generic_to_shared(
//                   s_b[next_buffer] +
//                   OFFSET(load_b_smem_k + 1, load_b_smem_n, BN + BPAD))),
//               "l"(B + load_b_gmem_addr + N));
//         asm("cp.async.bulk.shared.global [%0], [%1], 16;\n"
//             :
//             : "l"(__cvta_generic_to_shared(
//                   s_b[next_buffer] +
//                   OFFSET(load_b_smem_k + 2, load_b_smem_n, BN + BPAD))),
//               "l"(B + load_b_gmem_addr + 2 * N));
//         asm("cp.async.bulk.shared.global [%0], [%1], 16;\n"
//             :
//             : "l"(__cvta_generic_to_shared(
//                   s_b[next_buffer] +
//                   OFFSET(load_b_smem_k + 3, load_b_smem_n, BN + BPAD))),
//               "l"(B + load_b_gmem_addr + 3 * N));

//         // 执行计算
//         wmma::load_matrix_sync(frag_a[compute_buffer][0],
//                                s_a[compute_buffer] +
//                                    (comp_c_frag_m * 64 + 0) * (BK + APAD),
//                                BK + APAD);
//         wmma::load_matrix_sync(frag_a[compute_buffer][1],
//                                s_a[compute_buffer] +
//                                    (comp_c_frag_m * 64 + 16) * (BK + APAD),
//                                BK + APAD);
//         wmma::load_matrix_sync(frag_a[compute_buffer][2],
//                                s_a[compute_buffer] +
//                                    (comp_c_frag_m * 64 + 32) * (BK + APAD),
//                                BK + APAD);
//         wmma::load_matrix_sync(frag_a[compute_buffer][3],
//                                s_a[compute_buffer] +
//                                    (comp_c_frag_m * 64 + 48) * (BK + APAD),
//                                BK + APAD);

//         wmma::load_matrix_sync(frag_b[compute_buffer][0],
//                                s_b[compute_buffer] + (comp_c_frag_n * 64 + 0),
//                                BN + BPAD);
//         wmma::load_matrix_sync(frag_b[compute_buffer][1],
//                                s_b[compute_buffer] + (comp_c_frag_n * 64 + 16),
//                                BN + BPAD);
//         wmma::load_matrix_sync(frag_b[compute_buffer][2],
//                                s_b[compute_buffer] + (comp_c_frag_n * 64 + 32),
//                                BN + BPAD);
//         wmma::load_matrix_sync(frag_b[compute_buffer][3],
//                                s_b[compute_buffer] + (comp_c_frag_n * 64 + 48),
//                                BN + BPAD);

// #pragma unroll
//         for (int i = 0; i < 4; i++) {
// #pragma unroll
//             for (int j = 0; j < 4; j++) {
//                 wmma::mma_sync(frag_c[i][j], frag_a[compute_buffer][i],
//                                frag_b[compute_buffer][j], frag_c[i][j]);
//             }
//         }

//         asm("cp.async.bulk.commit_group 0;\n" ::);
//         asm("cp.async.bulk.wait_group 0;\n" ::);

//         __syncthreads();
//     }

//     // 处理最后一组片段
//     wmma::load_matrix_sync(frag_a[compute_buffer][0],
//                            s_a[compute_buffer] +
//                                (comp_c_frag_m * 64 + 0) * (BK + APAD),
//                            BK + APAD);
//     wmma::load_matrix_sync(frag_a[compute_buffer][1],
//                            s_a[compute_buffer] +
//                                (comp_c_frag_m * 64 + 16) * (BK + APAD),
//                            BK + APAD);
//     wmma::load_matrix_sync(frag_a[compute_buffer][2],
//                            s_a[compute_buffer] +
//                                (comp_c_frag_m * 64 + 32) * (BK + APAD),
//                            BK + APAD);
//     wmma::load_matrix_sync(frag_a[compute_buffer][3],
//                            s_a[compute_buffer] +
//                                (comp_c_frag_m * 64 + 48) * (BK + APAD),
//                            BK + APAD);

//     wmma::load_matrix_sync(frag_b[compute_buffer][0],
//                            s_b[compute_buffer] + (comp_c_frag_n * 64 + 0),
//                            BN + BPAD);
//     wmma::load_matrix_sync(frag_b[compute_buffer][1],
//                            s_b[compute_buffer] + (comp_c_frag_n * 64 + 16),
//                            BN + BPAD);
//     wmma::load_matrix_sync(frag_b[compute_buffer][2],
//                            s_b[compute_buffer] + (comp_c_frag_n * 64 + 32),
//                            BN + BPAD);
//     wmma::load_matrix_sync(frag_b[compute_buffer][3],
//                            s_b[compute_buffer] + (comp_c_frag_n * 64 + 48),
//                            BN + BPAD);

// #pragma unroll
//     for (int i = 0; i < 4; i++) {
// #pragma unroll
//         for (int j = 0; j < 4; j++) {
//             wmma::mma_sync(frag_c[i][j], frag_a[compute_buffer][i],
//                            frag_b[compute_buffer][j], frag_c[i][j]);
//         }
//     }

//     int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
//     int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
//     int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);

// // 将计算结果存储到全局内存中
// #pragma unroll
//     for (int i = 0; i < 4; i++) {
// #pragma unroll
//         for (int j = 0; j < 4; j++) {
//             wmma::store_matrix_sync(&C[store_c_gmem_addr + i * 16 * N + j * 16],
//                                     frag_c[i][j], N, wmma::mem_row_major);
//         }
//     }
// }

// PG_MATMUL_SIG(float16_t, 6, M, N, K, A, B, C)
// {
//     const int BM = 128, BN = 256, BK = 32;
//     dim3 blockDim(256);
//     int BX = (N + BN - 1) / BN;
//     int BY = (M + BM - 1) / BM;
//     dim3 gridDim(BX, BY);

//     cudaFuncSetAttribute(matmul_v11_mma,
//                          cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

//     unsigned int dsmem = 3 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
//     matmul_v11_mma<<<gridDim, blockDim, dsmem>>>(A, B, C, M, N, K);
//     cudaDeviceSynchronize();
// }
// }  // namespace playground
