//autotuning 
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "common.h"

const int NUM_THREADS = 256;  // Number of threads per block

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(NUM_THREADS)
sgemm2DBlockTilingAutotuned(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // Define the size of the warptile and iterations
    constexpr int WM = TM * 16;
    constexpr int WN = TN * 16;
    constexpr int WMITER = CEIL_DIV(BM, WM);
    constexpr int WNITER = CEIL_DIV(BN, WN);

    const int threadCol = threadIdx.x % (WN / TN);
    const int threadRow = threadIdx.x / (WN / TN);

    __shared__ float4 AsVectorized[BM * BK / 4];  // Adjusted for float4
    __shared__ float4 BsTransposedVectorized[BK * BN / 4];  // Adjusted for float4

    const float4 *Avec = reinterpret_cast<const float4*>(A);
    const float4 *Bvec = reinterpret_cast<const float4*>(B);
    float4 *Cvec = reinterpret_cast<float4*>(C);

    Avec += cRow * BM * K / 4;
    Bvec += cCol * BN / 4;
    Cvec += cRow * BM * N / 4 + cCol * BN / 4;

    float threadResults[WMITER * WNITER * TM * TN] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Vectorized and transposed load for A
        for (uint offset = 0; offset < BM / 4; offset += NUM_THREADS / (BK / 4)) {
            float4 tmp = Avec[(threadIdx.x / (BK / 4) + offset) * K / 4 + threadIdx.x % (BK / 4)];
            // Transpose and store in shared memory
            reinterpret_cast<float*>(&AsVectorized)[(offset + threadIdx.x / (BK / 4)) * BK + threadIdx.x % (BK / 4) * 4 + 0] = tmp.x;
            reinterpret_cast<float*>(&AsVectorized)[(offset + threadIdx.x / (BK / 4)) * BK + threadIdx.x % (BK / 4) * 4 + 1] = tmp.y;
            reinterpret_cast<float*>(&AsVectorized)[(offset + threadIdx.x / (BK / 4)) * BK + threadIdx.x % (BK / 4) * 4 + 2] = tmp.z;
            reinterpret_cast<float*>(&AsVectorized)[(offset + threadIdx.x / (BK / 4)) * BK + threadIdx.x % (BK / 4) * 4 + 3] = tmp.w;
        }

        // Vectorized load for B
        for (uint offset = 0; offset < BK / 4; offset += NUM_THREADS / (BN / 4)) {
            BsTransposedVectorized[(threadIdx.x / BN + offset) * BN / 4 + threadIdx.x % BN / 4] =
                Bvec[(threadIdx.x % BN + offset) * N / 4 + threadIdx.x / BN / 4];
        }
        __syncthreads();

        // Perform calculations over warptile iterations
        for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
            for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
                for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
                    float aValue = reinterpret_cast<float*>(AsVectorized)[dotIdx * BM + wmIdx * WM + threadRow * TM];
                    float bValue = reinterpret_cast<float*>(BsTransposedVectorized)[dotIdx * BN + wnIdx * WN + threadCol * TN];

                    for (uint i = 0; i < TM; ++i) {
                        for (uint j = 0; j < TN; ++j) {
                            float aElement = __shfl_sync(0xFFFFFFFF, aValue, threadRow * TM + i);
                            float bElement = __shfl_sync(0xFFFFFFFF, bValue, threadCol * TN + j);
                            threadResults[(wmIdx * TM + i) * WNITER * TN + wnIdx * TN + j] += aElement * bElement;
                        }
                    }
                }
            }
        }
        __syncthreads();

        Avec += BK / 4;
        Bvec += BK * N / 4;
    }

    // Vectorized write back to global memory C
    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
        for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
            float4 *Cinterim = &Cvec[(wmIdx * WM * N / 4) + (wnIdx * WN / 4)];
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN / 4; ++resIdxN) {
                    int idx = (wmIdx * TM + resIdxM) * WNITER * TN + wnIdx * TN + resIdxN * 4;
                    float4 cVal = Cinterim[(threadRow * TM + resIdxM) * N / 4 + threadCol * TN / 4 + resIdxN];
                    cVal.x = alpha * threadResults[idx + 0] + beta * cVal.x;
                    cVal.y = alpha * threadResults[idx + 1] + beta * cVal.y;
                    cVal.z = alpha * threadResults[idx + 2] + beta * cVal.z;
                    cVal.w = alpha * threadResults[idx + 3] + beta * cVal.w;
                    Cinterim[(threadRow * TM + resIdxM) * N / 4 + threadCol * TN / 4 + resIdxN] = cVal;
                }
            }
        }
    }
}
