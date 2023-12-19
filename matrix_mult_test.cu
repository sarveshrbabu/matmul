#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Include your custom kernel(s)
#include "kernel1.cu"
#include "kernel2.cu"

void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void printMatrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Matrix dimensions
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    // Initialize CUDA and CUBLAS
    cudaSetDevice(0);
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate device memory for matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeof(float) * M * K);
    cudaMalloc((void**)&d_B, sizeof(float) * K * N);
    cudaMalloc((void**)&d_C, sizeof(float) * M * N);

    // Allocate host memory for matrices
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    // Initialize matrices with random values
    initializeMatrix(h_A, M * K);
    initializeMatrix(h_B, K * N);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    // Test custom kernel
    dim3 blockSize(16, 16);
    dim3 gridSize(CEIL_DIV(M, blockSize.x), CEIL_DIV(N, blockSize.y));

    float alpha = 1.0;
    float beta = 0.0;

    // Test custom kernel1
    clock_t start_custom1 = clock();
    sgemm_shared_mem_block<16><<<gridSize, blockSize>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    clock_t end_custom1 = clock();

    // Copy result back to host for kernel1
    cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Test custom kernel2
    clock_t start_custom2 = clock();
    sgemm_shared_mem_block<32><<<gridSize, blockSize>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    clock_t end_custom2 = clock();

    // Copy result back to host for kernel2
    cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Test CUBLAS
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
    cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Calculate FLOPs for custom kernel1
    double flops_custom1 = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) / (end_custom1 - start_custom1) * CLOCKS_PER_SEC;

    // Calculate FLOPs for custom kernel2
    double flops_custom2 = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) / (end_custom2 - start_custom2) * CLOCKS_PER_SEC;

    // Print FLOPs for kernel1 and kernel2
    std::cout << "FLOPs for Custom Kernel1: " << flops_custom1 << std::endl;
    std::cout << "FLOPs for Custom Kernel2: " << flops_custom2 << std::endl;

    // Print proportion of FLOPs for kernel1 and kernel2
    std::cout << "Proportion of FLOPs for Kernel1: " << flops_custom1 / flops_cublas << std::endl;
    std::cout << "Proportion of FLOPs for Kernel2: " << flops_custom2 / flops_cublas << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
