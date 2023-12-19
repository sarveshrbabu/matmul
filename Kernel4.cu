#pragma once
//Transposed matrix and warpshuffling 
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
sgemm_warpshuffling(int M, int N, int K, float alpha, const float *A,
                   const float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint totalResultsBlockTile = BM * BN;
    const uint numThreadsBlockTile = totalResultsBlockTile / (TM * TN);

    assert(numThreadsBlockTile == blockDim.x);

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float BsTransposed[BK * BN];  // Transposed B matrix in shared memory

    A += cRow * BM * K;
    B += cCol * BN;  // Adjust for transposed access
    C += cRow * BM * N + cCol * BN;

    const uint strideA = numThreadsBlockTile / BK;
    const uint strideB = numThreadsBlockTile / BN;

    float threadResults[TM * TN] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load A normally
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(threadIdx.x / BK + loadOffset) * BK + threadIdx.x % BK] =
                A[(threadIdx.x / BK + loadOffset) * K + threadIdx.x % BK];
        }

        // Load B with transposed access pattern
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            BsTransposed[(threadIdx.x / BN + loadOffset) * BN + threadIdx.x % BN] =
                B[(threadIdx.x % BN + loadOffset) * N + threadIdx.x / BN];
        }
        __syncthreads();

        // Advance A and B pointers
        A += BK;
        B += BK * N;

        // Calculation using warp shuffle
        for (uint k = 0; k < BK; ++k) {
            float aValue = As[threadRow * TM * BK + k];
            float bValue = BsTransposed[k * BN + threadCol * TN];

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

    // Write the results back to C
    for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; ++j) {
            C[(threadRow * TM + i) * N + threadCol * TN + j] =
                alpha * threadResults[i * TN + j] +
                beta * C[(threadRow * TM + i) * N + threadCol * TN + j];
        }
    }
}
