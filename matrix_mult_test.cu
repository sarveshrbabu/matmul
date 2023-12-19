#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
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

//some useful variables 
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float alpha = 1.0f;
    float beta = 0.0f;

    std::vector<std::function<void()>> kernelExecutions = {
        //CUBLAS Default 
        [&]() {
            cudaEventRecord(start);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            double seconds = milliseconds / 1000.0;
            double flops = 2.0 * M * N * K / seconds;
            std::cout << "FLOPs for CUBLAS: " << flops << std::endl;
        },
        // Kernel 1
        [&]() {
            dim3 blockSize1(16, 16);
            dim3 gridSize1(CEIL_DIV(M, blockSize1.x), CEIL_DIV(N, blockSize1.y));
            cudaEventRecord(start);
            sgemm_shared_mem_block<16><<<gridSize1, blockSize1>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds1 = 0;
            cudaEventElapsedTime(&milliseconds1, start, stop);
            double seconds1 = milliseconds1 / 1000.0;
            double flops1 = 2.0 * M * N * K / seconds1;
            std::cout << "FLOPs for Custom Kernel1: " << flops1 << std::endl;
        },
        // Kernel 2
        // Repeat the above structure for each kernel with their specific configurations
        // For example:
        [&]() {
            const int BM = 64; // arbitrary value for BM
            const int BN = 64; // arbitrary value for BN
            const int BK = 8; // arbitrary value for BK
            const int TM = 8;  // arbitrary value for TM

            dim3 blockSize1(16, 16);
            dim3 gridSize1(CEIL_DIV(M, blockSize1.x), CEIL_DIV(N, blockSize1.y));
            cudaEventRecord(start);
            sgemm_multi_entry_per_thread(int M, int N, int K, float alpha,
                                             const float *A, const float *B, float beta,
                                             float *C)
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliseconds2 = 0;
            cudaEventElapsedTime(&milliseconds2, start, stop);
            double seconds2 = milliseconds2 / 1000.0;
            double flops2 = 2.0 * M * N * K / seconds2;
            std::cout << "FLOPs for Custom Kernel1: " << flops1 << std::endl;
        },
        // Kernel 3 and beyond will be soon 
    };

    for (auto& exec : kernelExecutions) {
        exec();
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}






