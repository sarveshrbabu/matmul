#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_multi_entry_per_thread(int M, int N, int K, float alpha,
                                             const float *A, const float *B, float beta,
                                             float *C) {
  const uint cRow = blockIdx.y * BM;
  const uint cCol = blockIdx.x * BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const uint threadCol = threadIdx.x % BN;
  const uint threadRow = threadIdx.x / BN;

  A += cRow * K;
  B += cCol;
  C += cRow * N + cCol;

  float threadResults[TM] = {0.0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Adjusted loading with boundary checks
    for (uint i = 0; i < TM; ++i) {
      uint aIndex = (cRow + threadRow * TM + i) * K + bkIdx + threadCol;
      uint bIndex = bkIdx * N + (cCol + threadCol) + (threadRow * TM + i) * BN;
      As[(threadRow * TM + i) * BK + threadCol] = (aIndex / K < M && aIndex % K < K) ? A[aIndex] : 0;
      Bs[(threadRow * TM + i) * BK + threadCol] = (bIndex / N < K && bIndex % N < N) ? B[bIndex] : 0;
    }

    __syncthreads();

    // Calculate the effective block size for the last block
    uint effectiveBK = min(BK, K - bkIdx);

    for (uint dotIdx = 0; dotIdx < effectiveBK; ++dotIdx) {
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // Writing out results with boundary check
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    uint cIndex = (cRow + threadRow * TM + resIdx) * N + cCol + threadCol;
    if (cRow + threadRow * TM + resIdx < M && cCol + threadCol < N) {
      C[cIndex] = alpha * threadResults[resIdx] + beta * C[cIndex];
    }
  }
}

