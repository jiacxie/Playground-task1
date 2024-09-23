#pragma once

#include "playground/system.hpp"

namespace playground
{

template <typename DType, uint8_t Version>
void matmul(const size_t M, const size_t N, const size_t K,
            const DType* const A, const DType* const B,
            DType* const C) = delete;

constexpr uint8_t CBLAS_VER = 0;
constexpr uint8_t CUBLAS_VER = 1;

// Playground Matmul Signature.
#define PG_MATMUL_SIG(DType, Version, M, N, K, A, B, C)                        \
    template <>                                                                \
    void matmul<DType, Version>(const size_t M, const size_t N,                \
                                const size_t K, const DType* const A,          \
                                const DType* const B, DType* const C)

// =============================================================================
// Declaration of library matmul functions.
// -----------------------------------------------------------------------------
/**
 * @brief Matrix multiplication, fp16-v0, cBLAS.
 */
PG_MATMUL_SIG(float16_t, CBLAS_VER, M, N, K, A, B, C);

/**
 * @brief Matrix multiplication, fp32-v0, cBLAS.
 */
PG_MATMUL_SIG(float32_t, CBLAS_VER, M, N, K, A, B, C);

/**
 * @brief Matrix multiplication, fp16-v1, cuBLAS.
 */
PG_MATMUL_SIG(float16_t, CUBLAS_VER, M, N, K, A, B, C);

/**
 * @brief Matrix multiplication, fp32-v1, cuBLAS.
 */
PG_MATMUL_SIG(float32_t, CUBLAS_VER, M, N, K, A, B, C);


PG_MATMUL_SIG(float32_t, 2, M, N, K, A, B, C);

PG_MATMUL_SIG(float32_t, 3, M, N, K, A, B, C);

PG_MATMUL_SIG(float32_t, 4, M, N, K, A, B, C);

PG_MATMUL_SIG(float32_t, 5, M, N, K, A, B, C);

PG_MATMUL_SIG(float32_t, 6, M, N, K, A, B, C);

PG_MATMUL_SIG(float16_t, 2, M, N, K, A, B, C);

PG_MATMUL_SIG(float16_t, 3, M, N, K, A, B, C);

PG_MATMUL_SIG(float16_t, 4, M, N, K, A, B, C);

PG_MATMUL_SIG(float16_t, 5, M, N, K, A, B, C);

PG_MATMUL_SIG(float16_t, 6, M, N, K, A, B, C);

// =============================================================================
// Declaration of self-implemented matmul functions.
// e.g. PG_MATMUL_SIG(float16_t, 2, M, N, K, A, B, C);
// -----------------------------------------------------------------------------

// ...

}  // namespace playground