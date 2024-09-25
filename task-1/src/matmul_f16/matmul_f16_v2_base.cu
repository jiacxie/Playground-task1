// @file: ./task-1/src/f16-v2.cu
#include "playground/matmul.hpp"
#include "playground/system.hpp"
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>

namespace playground
{
using namespace nvcuda;

constexpr int WARP_SIZE = 32;
constexpr int M_TILE = 16;
constexpr int N_TILE = 16;
constexpr int K_TILE = 16;

__host__ __device__ int pad(int x, int y) {
    return (x % y) ? ((x / y + 1) * y) : x;
}

__global__ void matmul_v7(const float16_t* A, const float16_t* B, float16_t* C, int M, int N, int K) {
    int N_PAD = pad(N, N_TILE);
    int K_PAD = pad(K, K_TILE);
    int idx, midx, nidx, ndim, kdim;
    ndim = N_PAD / N_TILE;
    kdim = K_PAD / K_TILE;
    idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    nidx = idx % ndim;
    midx = idx / ndim;
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, M_TILE, N_TILE, K_TILE, float16_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M_TILE, N_TILE, K_TILE, float16_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, M_TILE, N_TILE, K_TILE, float16_t> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    float16_t* c_unique = C + nidx * N_TILE + midx * M_TILE * ndim * N_TILE;

    for (int kidx = 0; kidx < kdim; kidx++) {
        // Load the inputs
        const float16_t* a_unique = A + kidx * K_TILE + midx * M_TILE * kdim * K_TILE;
        const float16_t* b_unique = B + nidx * N_TILE + kidx * K_TILE * ndim * N_TILE;

        wmma::load_matrix_sync(a_frag, a_unique, K_PAD);
        wmma::load_matrix_sync(b_frag, b_unique, N_PAD);

        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    // Store the output
    wmma::store_matrix_sync(c_unique, c_frag, N_PAD, wmma::mem_row_major);
}

PG_MATMUL_SIG(float16_t, 2, M, N, K, A, B, C) {
    constexpr int TILE = 16;
    int M_PAD = pad(M, TILE);
    int N_PAD = pad(N, TILE);
    int nwarp = (M_PAD / TILE) * (N_PAD / TILE);
    int GRID_DIM = (nwarp * WARP_SIZE) % 512 ? nwarp * WARP_SIZE / 512 + 1 : nwarp * WARP_SIZE / 512;
    playground::matmul_v7<<<GRID_DIM, 512>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
}  // namespace playground