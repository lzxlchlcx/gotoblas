#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "myblas.h"

#ifdef USE_CUDA
#include <cublas_v2.h>
#include "kernel/cuda/gemm_gpu.h"
#endif

static double get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_random_d(double *A, int n)
{
    for (int i = 0; i < n; i++) A[i] = (double)rand() / RAND_MAX - 0.5;
}

static void fill_random_s(float *A, int n)
{
    for (int i = 0; i < n; i++) A[i] = (float)rand() / RAND_MAX - 0.5f;
}

#ifdef USE_CUDA
static double bench_my_dgemm(int n, int version, int niter)
{
    double *A = (double *)malloc((size_t)n * n * sizeof(double));
    double *B = (double *)malloc((size_t)n * n * sizeof(double));
    fill_random_d(A, n * n);
    fill_random_d(B, n * n);
    double *dA = 0, *dB = 0, *dC = 0;
    size_t bytes = (size_t)n * n * sizeof(double);
    cudaMalloc((void **)&dA, bytes);
    cudaMalloc((void **)&dB, bytes);
    cudaMalloc((void **)&dC, bytes);
    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);
    gpu_dgemm_device_version(version, 'N', 'N', n, n, n, 1.0, dA, n, dB, n, 0.0, dC, n);
    cudaDeviceSynchronize();
    double t0 = get_time();
    for (int i = 0; i < niter; i++)
        gpu_dgemm_device_version(version, 'N', 'N', n, n, n, 1.0, dA, n, dB, n, 0.0, dC, n);
    cudaDeviceSynchronize();
    double gflops = 2.0 * n * n * n * niter / ((get_time() - t0) * 1e9);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B);
    return gflops;
}

static double bench_my_sgemm(int n, int version, int niter)
{
    float *A = (float *)malloc((size_t)n * n * sizeof(float));
    float *B = (float *)malloc((size_t)n * n * sizeof(float));
    fill_random_s(A, n * n);
    fill_random_s(B, n * n);
    float *dA = 0, *dB = 0, *dC = 0;
    size_t bytes = (size_t)n * n * sizeof(float);
    cudaMalloc((void **)&dA, bytes);
    cudaMalloc((void **)&dB, bytes);
    cudaMalloc((void **)&dC, bytes);
    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);
    gpu_sgemm_device_version(version, 'N', 'N', n, n, n, 1.0f, dA, n, dB, n, 0.0f, dC, n);
    cudaDeviceSynchronize();
    double t0 = get_time();
    for (int i = 0; i < niter; i++)
        gpu_sgemm_device_version(version, 'N', 'N', n, n, n, 1.0f, dA, n, dB, n, 0.0f, dC, n);
    cudaDeviceSynchronize();
    double gflops = 2.0 * n * n * n * niter / ((get_time() - t0) * 1e9);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B);
    return gflops;
}

static double bench_cublas_dgemm(int n, int niter)
{
    cublasHandle_t h;
    double *A, *B, *C, *dA, *dB, *dC;
    double alpha = 1.0, beta = 0.0;
    A = (double *)malloc((size_t)n * n * sizeof(double));
    B = (double *)malloc((size_t)n * n * sizeof(double));
    C = (double *)calloc((size_t)n * n, sizeof(double));
    fill_random_d(A, n * n);
    fill_random_d(B, n * n);
    cudaMalloc((void **)&dA, (size_t)n * n * sizeof(double));
    cudaMalloc((void **)&dB, (size_t)n * n * sizeof(double));
    cudaMalloc((void **)&dC, (size_t)n * n * sizeof(double));
    cudaMemcpy(dA, A, (size_t)n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, (size_t)n * n * sizeof(double), cudaMemcpyHostToDevice);
    cublasCreate(&h);
    cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dA, n, dB, n, &beta, dC, n);
    cudaDeviceSynchronize();
    double t0 = get_time();
    for (int i = 0; i < niter; i++) cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dA, n, dB, n, &beta, dC, n);
    cudaDeviceSynchronize();
    double gflops = 2.0 * n * n * n * niter / ((get_time() - t0) * 1e9);
    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C);
    return gflops;
}

static double bench_cublas_sgemm(int n, int niter)
{
    cublasHandle_t h;
    float *A, *B, *C, *dA, *dB, *dC;
    float alpha = 1.0f, beta = 0.0f;
    A = (float *)malloc((size_t)n * n * sizeof(float));
    B = (float *)malloc((size_t)n * n * sizeof(float));
    C = (float *)calloc((size_t)n * n, sizeof(float));
    fill_random_s(A, n * n);
    fill_random_s(B, n * n);
    cudaMalloc((void **)&dA, (size_t)n * n * sizeof(float));
    cudaMalloc((void **)&dB, (size_t)n * n * sizeof(float));
    cudaMalloc((void **)&dC, (size_t)n * n * sizeof(float));
    cudaMemcpy(dA, A, (size_t)n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, (size_t)n * n * sizeof(float), cudaMemcpyHostToDevice);
    cublasCreate(&h);
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dA, n, dB, n, &beta, dC, n);
    cudaDeviceSynchronize();
    double t0 = get_time();
    for (int i = 0; i < niter; i++) cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dA, n, dB, n, &beta, dC, n);
    cudaDeviceSynchronize();
    double gflops = 2.0 * n * n * n * niter / ((get_time() - t0) * 1e9);
    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C);
    return gflops;
}
#endif

int main(void)
{
#ifndef USE_CUDA
    printf("CUDA support is not enabled in this build.\n");
    return 0;
#else
    if (!gpu_is_available()) {
        printf("No CUDA GPU available.\n");
        return 0;
    }

    int sizes[] = {128, 256, 512, 1024, 2048};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    printf("=== GPU GEMM Benchmark ===\n");
    printf("GPU: SM %d, %.1f GiB\n\n", gpu_get_compute_capability(), gpu_get_memory() / 1073741824.0);

    for (int i = 0; i < nsizes; i++) {
        int n = sizes[i];
        int niter = n <= 256 ? 5 : (n <= 1024 ? 3 : 1);
        double cublas_d = bench_cublas_dgemm(n, niter);
        double cublas_s = bench_cublas_sgemm(n, niter);
        printf("n=%d\n", n);
        printf("  DGEMM cuBLAS: %8.2f GFLOPS\n", cublas_d);
        for (int v = 0; v <= 3; v++) {
            double g = bench_my_dgemm(n, v, niter);
            printf("  DGEMM v%d:     %8.2f GFLOPS (%5.1f%% cuBLAS)\n", v, g, 100.0 * g / cublas_d);
        }
        printf("  SGEMM cuBLAS: %8.2f GFLOPS\n", cublas_s);
        for (int v = 0; v <= 4; v++) {
            double g = bench_my_sgemm(n, v, niter);
            const char *tag = (v == 4) ? " TC/TF32" : "";
            printf("  SGEMM v%d%s: %8.2f GFLOPS (%5.1f%% cuBLAS)\n", v, tag, g, 100.0 * g / cublas_s);
        }
        printf("\n");
    }
    return 0;
#endif
}
