#ifndef MYBLAS_GPU_COMMON_CUH
#define MYBLAS_GPU_COMMON_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include "config/ampere.h"

#define GPU_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        return -1; \
    } \
} while (0)

static inline int gpu_ceil_div(int x, int y)
{
    return (x + y - 1) / y;
}

#endif
