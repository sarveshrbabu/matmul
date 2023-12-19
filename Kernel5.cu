// vectorization for into SMEM and other Memory 
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
sgemm_vectorized(int M, int N, int K, float alpha, const float *A,
                   const float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint totalResultsBlockTile = BM * BN;
    const uint numThreadsBlockTile = totalResultsBlockTile / (TM * TN);

    assert(numThreadsBlockTile == blockDim.x);

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float4 AsVectorized[BM * BK / 4];  // Adjusted for float4
    __shared__ float4 BsTransposedVectorized[BK * BN / 4];  // Adjusted for float4

    const float4 *Avec = reinterpret_cast<const float4*>(A);
    const float4 *Bvec = reinterpret_cast<const float4*>(B);
    float4 *Cvec = reinterpret_cast<float4*>(C);

    Avec += cRow * BM * K / 4;
    Bvec += cCol * BN / 4;  // Adjust for transposed access
    Cvec += cRow * BM * N / 4 + cCol * BN / 4;

    const uint strideA = numThreadsBlockTile / BK;
    const uint strideB = numThreadsBlockTile / BN;

    float threadResults[TM * TN] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Vectorized Load A
        for (uint loadOffset = 0; loadOffset < BM / 4; loadOffset += strideA) {
            AsVectorized[(threadIdx.x / BK + loadOffset) * BK / 4 + threadIdx.x % BK / 4] =
                Avec[(threadIdx.x / BK + loadOffset) * K / 4 + threadIdx.x % BK / 4];
        }

        // Vectorized Load B with transposed access pattern
        for (uint loadOffset = 0; loadOffset < BK / 4; loadOffset += strideB) {
            BsTransposedVectorized[(threadIdx.x / BN + loadOffset) * BN / 4 + threadIdx.x % BN / 4] =
                Bvec[(threadIdx.x % BN + loadOffset) * N / 4 + threadIdx.x / BN / 4];
        }
        __syncthreads();

        // Advance A and B pointers
        Avec += BK / 4;
        Bvec += BK * N / 4;

        // Calculation using warp shuffle
        for (uint k = 0; k < BK; ++k) {
            float aValue = reinterpret_cast<float*>(AsVectorized)[threadRow * TM * BK + k];
            float bValue = reinterpret_cast<float*>(BsTransposedVectorized)[k * BN + threadCol * TN];

            for (uint i = 0; i < TM; ++i) {
                for (uint j = 0; j < TN; ++j) {
                    float aElement = __shfl_sync(0xFFFFFFFF, aValue, threadRow * TM + i);
                    float bElement = __shfl_sync(0xFFFFFFFF, bValue, threadCol * TN + j);
                    threadResults[i * TN + j] += aElement * bElement;
                }
            }
        }
        __syncthreads();
    }

    // Vectorized Write the results back to C
    for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN / 4; ++j) {
            float4 result = make_float4(
                alpha * threadResults[i * TN + j * 4] + beta * Cvec[(threadRow * TM + i) * N / 4 + threadCol * TN / 4 + j].x,
                alpha * threadResults[i * TN + j * 4 + 1] + beta * Cvec[(threadRow * TM + i) * N / 4 + threadCol * TN / 4 + j].y,
                alpha * threadResults[i * TN + j * 4 + 2] + beta * Cvec[(threadRow * TM + i) * N / 4 + threadCol * TN / 4 + j].z,
                alpha * threadResults[i * TN + j * 4 + 3] + beta * Cvec[(threadRow * TM + i) * N / 4 + threadCol * TN / 4 + j].w);
            Cvec[(threadRow * TM + i) * N / 4 + threadCol * TN / 4 + j] = result;
        }
    }
}
