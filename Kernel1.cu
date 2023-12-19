#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "common.h"


template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  const uint cRow = blockIdx.x * BLOCKSIZE;
  const uint cCol = blockIdx.y * BLOCKSIZE;

  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  float tmp = 0.0;

  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    int aRow = cRow + threadRow;
    int aCol = bkIdx + threadCol;
    int bRow = bkIdx + threadRow;
    int bCol = cCol + threadCol;

    // Adjust for the edge blocks
    if (aRow < M && aCol < K) {
      As[threadRow * BLOCKSIZE + threadCol] = A[aRow * K + aCol];
    } else {
      As[threadRow * BLOCKSIZE + threadCol] = 0.0;
    }

    if (bRow < K && bCol < N) {
      Bs[threadRow * BLOCKSIZE + threadCol] = B[bRow * N + bCol];
    } else {
      Bs[threadRow * BLOCKSIZE + threadCol] = 0.0;
    }

    __syncthreads();

    // Calculate the effective block size for the last block
    int effectiveBlockSize = min(BLOCKSIZE, K - bkIdx);

    for (int dotIdx = 0; dotIdx < effectiveBlockSize; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
    }

    __syncthreads();
  }

  int cIdx = cRow * N + cCol + threadRow * N + threadCol;
  if (cRow + threadRow < M && cCol + threadCol < N) {
    C[cIdx] = alpha * tmp + beta * C[cIdx];
  }
}




