#ifdef USE_CUDA

#include "kernel/cuda/gemm_gpu.h"
#include "config/ampere.h"

static gpu_gemm_handle_t *gpu_handle = 0;
static int gpu_available = -1;

int gpu_dispatch_init(void)
{
    if (gpu_available < 0) {
        gpu_available = gpu_is_available() ? 1 : 0;
    }
    if (!gpu_available) return -1;
    if (!gpu_handle) {
        gpu_handle = gpu_gemm_handle_create();
        if (!gpu_handle) {
            gpu_available = 0;
            return -1;
        }
    }
    return 0;
}

void gpu_dispatch_cleanup(void)
{
    if (gpu_handle) {
        gpu_gemm_handle_destroy(gpu_handle);
        gpu_handle = 0;
    }
}

gpu_gemm_handle_t *gpu_dispatch_handle(void)
{
    return gpu_handle;
}

int gpu_should_dispatch(int m, int n, int k, int precision)
{
    int min_dim = precision == 0 ? GPU_DGEMM_DISPATCH_MIN : GPU_SGEMM_DISPATCH_MIN;
    if (m < min_dim || n < min_dim || k < min_dim) return 0;
    if (gpu_available == 0) return 0;
    if (gpu_available < 0) gpu_available = gpu_is_available() ? 1 : 0;
    return gpu_available;
}

#else

#include "kernel/cuda/gemm_gpu.h"

int gpu_dispatch_init(void) { return -1; }
void gpu_dispatch_cleanup(void) { }
gpu_gemm_handle_t *gpu_dispatch_handle(void) { return 0; }
int gpu_should_dispatch(int m, int n, int k, int precision)
{
    (void)m; (void)n; (void)k; (void)precision;
    return 0;
}

#endif
