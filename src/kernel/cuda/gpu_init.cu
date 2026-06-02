#include <cuda_runtime.h>
#include <stdlib.h>
#include "kernel/cuda/gemm_gpu.h"
#include "kernel/cuda/gpu_common.cuh"

struct gpu_gemm_handle_t {
    cudaStream_t stream;
};

extern "C" int gpu_is_available(void)
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count > 0;
}

extern "C" int gpu_get_compute_capability(void)
{
    int device = 0;
    cudaDeviceProp prop;
    if (!gpu_is_available()) return 0;
    if (cudaGetDevice(&device) != cudaSuccess) device = 0;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0;
    return prop.major * 10 + prop.minor;
}

extern "C" size_t gpu_get_memory(void)
{
    int device = 0;
    cudaDeviceProp prop;
    if (!gpu_is_available()) return 0;
    if (cudaGetDevice(&device) != cudaSuccess) device = 0;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0;
    return prop.totalGlobalMem;
}

extern "C" gpu_gemm_handle_t *gpu_gemm_handle_create(void)
{
    gpu_gemm_handle_t *handle = (gpu_gemm_handle_t *)calloc(1, sizeof(gpu_gemm_handle_t));
    if (!handle) return 0;
    if (cudaStreamCreate(&handle->stream) != cudaSuccess) {
        free(handle);
        return 0;
    }
    return handle;
}

extern "C" void gpu_gemm_handle_destroy(gpu_gemm_handle_t *handle)
{
    if (!handle) return;
    cudaStreamDestroy(handle->stream);
    free(handle);
}

extern "C" int gpu_dgemm(gpu_gemm_handle_t *handle,
                         char transa, char transb,
                         int m, int n, int k,
                         double alpha, const double *A, int lda,
                         const double *B, int ldb,
                         double beta, double *C, int ldc)
{
    (void)handle;
    return gpu_dgemm_version(3, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

extern "C" int gpu_sgemm(gpu_gemm_handle_t *handle,
                         char transa, char transb,
                         int m, int n, int k,
                         float alpha, const float *A, int lda,
                         const float *B, int ldb,
                         float beta, float *C, int ldc)
{
    (void)handle;
    return gpu_sgemm_version(2, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
