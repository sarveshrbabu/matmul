#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <functional>  // Add this line
#include <cublas_v2.h>

// Include your custom kernel(s)
#include "Kernel1.cu"
#include "Kernel2.cu"
#include "Kernel3.cu"
#include "Kernel4.cu"
#include "Kernel5.cu"
#include "Kernel6.cu"

void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

double benchmarkMatMul(std::function<void()> matmulFunc, int M, int N, int K) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmulFunc();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 2.0 * M * N * K / (milliseconds / 1000.0);
}

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    cudaSetDevice(0);
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeof(float) * M * K);
    cudaMalloc((void**)&d_B, sizeof(float) * K * N);
    cudaMalloc((void**)&d_C, sizeof(float) * M * N);

    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    initializeMatrix(h_A, M * K);
    initializeMatrix(h_B, K * N);

    cudaMemcpy(d_A, h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    dim3 blockSize(16, 16); // Example block size
    dim3 gridSize(CEIL_DIV(M, blockSize.x), CEIL_DIV(N, blockSize.y)); // Example grid size

    // Lambda for CUBLAS
    auto matmulFunc_0 = [&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    };

    // Lambda for Kernel 1
    auto matmulFunc_1 = [&]() {
        sgemm_shared_mem_block<32><<<gridSize, blockSize>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    };

    // Lambda for Kernel 2
    auto matmulFunc_2 = [&]() {
        sgemm_multi_entry_per_thread<64, 64, 8, 8><<<gridSize, blockSize>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    };

    // Add similar lambda functions for Kernel 3, Kernel 4, Kernel 5, Kernel 6

    double flops_0 = benchmarkMatMul(matmulFunc_0, M, N, K);
    double flops_1 = benchmarkMatMul(matmulFunc_1, M, N, K);
    double flops_2 = benchmarkMatMul(matmulFunc_2, M, N, K);
    // Calculate FLOPs for Kernel 3, Kernel 4, Kernel 5, Kernel 6

    std::cout << "FLOPs for CUBLAS: " << flops_0 << std::endl;
    std::cout << "FLOPs for Custom Kernel 1: " << flops_1 << std::endl;
    std::cout << "FLOPs for Custom Kernel 2: " << flops_2 << std::endl;
    // Print FLOPs for Kernel 3, Kernel 4, Kernel 5, Kernel 6

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
