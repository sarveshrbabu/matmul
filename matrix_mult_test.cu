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

double calculateFLOPs(float milliseconds, int M, int N, int K) {
    // Number of floating-point operations for matrix multiplication is 2 * M * N * K
    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

    // Convert milliseconds to seconds
    double seconds = milliseconds / 1000.0;

    // Calculate FLOPs: Operations per second
    double flopsPerSecond = flops / seconds;

    return flopsPerSecond;
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

/*
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
    //double flops_1 = benchmarkMatMul(matmulFunc_1, M, N, K);
    //double flops_2 = benchmarkMatMul(matmulFunc_2, M, N, K);
    // Calculate FLOPs for Kernel 3, Kernel 4, Kernel 5, Kernel 6

    std::cout << "FLOPs for CUBLAS: " << flops_0 << std::endl;
    //std::cout << "FLOPs for Custom Kernel 1: " << flops_1 << std::endl;
    //std::cout << "FLOPs for Custom Kernel 2: " << flops_2 << std::endl;
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
*/

int main(){
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
    dim3 blockSize(32, 32);
    dim3 gridSize(CEIL_DIV(M, blockSize.x), CEIL_DIV(N, blockSize.y));

    float alpha = 1.0;
    float beta = 0.0;

    // Test custom kernel1
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_shared_mem_block<32><<<gridSize, blockSize>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 
    float milliseconds_1 = 0;
    cudaEventElapsedTime(&milliseconds_1, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy result back to host for kernel1
    cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    double flops_1 = calculateFLOPs(milliseconds_1,  M,  N,  K);
    // Output the elapsed time or use it to calculate FLOPs
    std::cout << "Elapsed time for kernel: " << milliseconds_1 << " ms" << std::endl;
    std::cout<< "FLOPS:" <<flops_1 << " lol " << std::endl;

    /*
    // Test custom kernel2
    clock_t start_custom2 = clock();
    sgemm_shared_mem_block<32><<<gridSize, blockSize>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    clock_t end_custom2 = clock();

    // Copy result back to host for kernel2
    cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    */

    /*
    // Test custom kernel3
    clock_t start_custom3 = clock();
    sgemm2DBlocktiling<<<gridSize, blockSize>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    clock_t end_custom3 = clock();

    // Copy result back to host for kernel3
    cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Calculate FLOPs for custom kernel3
    double flops_custom3 = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) / (end_custom3 - start_custom3) * CLOCKS_PER_SEC;

    // Print FLOPs for kernel3
    std::cout << "FLOPs for Custom Kernel3: " << flops_custom3 << std::endl;

    // Test custom kernel4
    clock_t start_custom4 = clock();
    sgemm_warpshuffling<<<gridSize, blockSize>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    clock_t end_custom4 = clock();

    // Copy result back to host for kernel4
    cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Calculate FLOPs for custom kernel4
    double flops_custom4 = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) / (end_custom4 - start_custom4) * CLOCKS_PER_SEC;

    // Print FLOPs for kernel4
    std::cout << "FLOPs for Custom Kernel4: " << flops_custom4 << std::endl;

    // Test custom kernel5
    clock_t start_custom5 = clock();
    sgemm_vectorized<<<gridSize, blockSize>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    clock_t end_custom5 = clock();

    // Copy result back to host for kernel5
    cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Calculate FLOPs for custom kernel5
    double flops_custom5 = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) / (end_custom5 - start_custom5) * CLOCKS_PER_SEC;

    // Print FLOPs for kernel5
    std::cout << "FLOPs for Custom Kernel5: " << flops_custom5 << std::endl;

    // Test custom kernel6
    clock_t start_custom6 = clock();
    sgemm2DBlockTilingAutotuned<<<gridSize, blockSize>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();
    clock_t end_custom6 = clock();

    // Copy result back to host for kernel6
    cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Calculate FLOPs for custom kernel6
    double flops_custom6 = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) / (end_custom6 - start_custom6) * CLOCKS_PER_SEC;

    // Print FLOPs for kernel6
    std::cout << "FLOPs for Custom Kernel6: " << flops_custom6 << std::endl;
    */

    // Test CUBLAS
    cudaEvent_t startCublas, stopCublas;
    cudaEventCreate(&startCublas);
    cudaEventCreate(&stopCublas);
    cudaEventRecord(startCublas);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
    cudaEventRecord(stopCublas);
    cudaEventSynchronize(stopCublas);
    float millisecondsCublas = 0;
    cudaEventElapsedTime(&millisecondsCublas, startCublas, stopCublas);
    cudaEventDestroy(startCublas);
    cudaEventDestroy(stopCublas);
    cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    std::cout << "Elapsed time for CUBLAS: " << millisecondsCublas << " ms" << std::endl;
    double flopsCublas = calculateFLOPs(millisecondsCublas, M, N, K);
    std::cout << "FLOPS for CUBLAS: " << flopsCublas << " FLOPs/s" << std::endl;

    // Print proportion of FLOPs for kernel1 and kernel2
    std::cout << "Proportion of FLOPs for Kernel1: " << flops_1 / flopsCublas << std::endl;
    //std::cout << "Proportion of FLOPs for Kernel2: " << flops_custom2 / flops_cublas << std::endl;

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