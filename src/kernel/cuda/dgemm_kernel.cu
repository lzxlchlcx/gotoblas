#include <cuda_runtime.h>
#include "kernel/cuda/gemm_gpu.h"
#include "kernel/cuda/gpu_common.cuh"

__device__ static inline double dgemm_load_a(const double *A, int lda, int transa, int row, int p)
{
    return transa ? A[p + row * lda] : A[row + p * lda];
}

__device__ static inline double dgemm_load_b(const double *B, int ldb, int transb, int p, int col)
{
    return transb ? B[col + p * ldb] : B[p + col * ldb];
}

__global__ void dgemm_naive_kernel(int m, int n, int k, int transa, int transb,
                                   double alpha, const double *A, int lda,
                                   const double *B, int ldb,
                                   double beta, double *C, int ldc)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    double sum = 0.0;
    for (int p = 0; p < k; p++) {
        sum += dgemm_load_a(A, lda, transa, row, p) * dgemm_load_b(B, ldb, transb, p, col);
    }
    C[row + col * ldc] = beta * C[row + col * ldc] + alpha * sum;
}

template<int TILE>
__global__ void dgemm_shared_kernel(int m, int n, int k, int transa, int transb,
                                    double alpha, const double *A, int lda,
                                    const double *B, int ldb,
                                    double beta, double *C, int ldc)
{
    __shared__ double sA[TILE][TILE];
    __shared__ double sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    double sum = 0.0;

    for (int kk = 0; kk < k; kk += TILE) {
        int ap = kk + threadIdx.x;
        int bp = kk + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < m && ap < k) ? dgemm_load_a(A, lda, transa, row, ap) : 0.0;
        sB[threadIdx.y][threadIdx.x] = (bp < k && col < n) ? dgemm_load_b(B, ldb, transb, bp, col) : 0.0;
        __syncthreads();

        for (int p = 0; p < TILE; p++) {
            sum += sA[threadIdx.y][p] * sB[p][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row + col * ldc] = beta * C[row + col * ldc] + alpha * sum;
    }
}

template<int TM, int TN, int TK, int MR, int NR, int DBUF>
__global__ void dgemm_regblock_kernel(int m, int n, int k, int transa, int transb,
                                      double alpha, const double *A, int lda,
                                      const double *B, int ldb,
                                      double beta, double *C, int ldc)
{
    __shared__ double sA[DBUF][TM][TK + 1];
    __shared__ double sB[DBUF][TK][TN + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int nthreads = blockDim.x * blockDim.y;
    int row0 = blockIdx.y * TM + ty * MR;
    int col0 = blockIdx.x * TN + tx * NR;
    double acc[MR][NR];

    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < NR; j++) acc[i][j] = 0.0;
    }

    for (int kk = 0; kk < k; kk += TK) {
        int buf = DBUF == 1 ? 0 : ((kk / TK) & 1);
        for (int idx = tid; idx < TM * TK; idx += nthreads) {
            int r = idx / TK;
            int p = idx - r * TK;
            int gr = blockIdx.y * TM + r;
            int gp = kk + p;
            sA[buf][r][p] = (gr < m && gp < k) ? dgemm_load_a(A, lda, transa, gr, gp) : 0.0;
        }
        for (int idx = tid; idx < TK * TN; idx += nthreads) {
            int p = idx / TN;
            int c = idx - p * TN;
            int gp = kk + p;
            int gc = blockIdx.x * TN + c;
            sB[buf][p][c] = (gp < k && gc < n) ? dgemm_load_b(B, ldb, transb, gp, gc) : 0.0;
        }
        __syncthreads();

        for (int p = 0; p < TK; p++) {
            double ar[MR], br[NR];
            for (int i = 0; i < MR; i++) ar[i] = sA[buf][ty * MR + i][p];
            for (int j = 0; j < NR; j++) br[j] = sB[buf][p][tx * NR + j];
            for (int i = 0; i < MR; i++) {
                for (int j = 0; j < NR; j++) acc[i][j] += ar[i] * br[j];
            }
        }
        __syncthreads();
    }

    for (int j = 0; j < NR; j++) {
        int col = col0 + j;
        if (col >= n) continue;
        for (int i = 0; i < MR; i++) {
            int row = row0 + i;
            if (row < m) C[row + col * ldc] = beta * C[row + col * ldc] + alpha * acc[i][j];
        }
    }
}

extern "C" int gpu_dgemm_device_version(int version,
                                        char transa, char transb,
                                        int m, int n, int k,
                                        double alpha, const double *dA, int lda,
                                        const double *dB, int ldb,
                                        double beta, double *dC, int ldc)
{
    int ta = !(transa == 'N' || transa == 'n');
    int tb = !(transb == 'N' || transb == 'n');

    if (version <= 0) {
        dim3 block(32, 32);
        dim3 grid(gpu_ceil_div(n, 32), gpu_ceil_div(m, 32));
        dgemm_naive_kernel<<<grid, block>>>(m, n, k, ta, tb, alpha, dA, lda, dB, ldb, beta, dC, ldc);
    } else if (version == 1) {
        dim3 block(16, 16);
        dim3 grid(gpu_ceil_div(n, 16), gpu_ceil_div(m, 16));
        dgemm_shared_kernel<16><<<grid, block>>>(m, n, k, ta, tb, alpha, dA, lda, dB, ldb, beta, dC, ldc);
    } else if (version == 2) {
        dim3 block(GPU_DGEMM_TILE_N / GPU_DGEMM_THREAD_N, GPU_DGEMM_TILE_M / GPU_DGEMM_THREAD_M);
        dim3 grid(gpu_ceil_div(n, GPU_DGEMM_TILE_N), gpu_ceil_div(m, GPU_DGEMM_TILE_M));
        dgemm_regblock_kernel<GPU_DGEMM_TILE_M, GPU_DGEMM_TILE_N, GPU_DGEMM_TILE_K,
                              GPU_DGEMM_THREAD_M, GPU_DGEMM_THREAD_N, 1>
            <<<grid, block>>>(m, n, k, ta, tb, alpha, dA, lda, dB, ldb, beta, dC, ldc);
    } else {
        dim3 block(GPU_DGEMM_TILE_N / GPU_DGEMM_THREAD_N, GPU_DGEMM_TILE_M / GPU_DGEMM_THREAD_M);
        dim3 grid(gpu_ceil_div(n, GPU_DGEMM_TILE_N), gpu_ceil_div(m, GPU_DGEMM_TILE_M));
        dgemm_regblock_kernel<GPU_DGEMM_TILE_M, GPU_DGEMM_TILE_N, GPU_DGEMM_TILE_K,
                              GPU_DGEMM_THREAD_M, GPU_DGEMM_THREAD_N, 2>
            <<<grid, block>>>(m, n, k, ta, tb, alpha, dA, lda, dB, ldb, beta, dC, ldc);
    }

    GPU_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int gpu_dgemm_version(int version,
                                 char transa, char transb,
                                 int m, int n, int k,
                                 double alpha, const double *A, int lda,
                                 const double *B, int ldb,
                                 double beta, double *C, int ldc)
{
    int ta = !(transa == 'N' || transa == 'n');
    int tb = !(transb == 'N' || transb == 'n');
    size_t asize = (size_t)lda * (ta ? m : k) * sizeof(double);
    size_t bsize = (size_t)ldb * (tb ? k : n) * sizeof(double);
    size_t csize = (size_t)ldc * n * sizeof(double);
    double *dA = 0, *dB = 0, *dC = 0;

    GPU_CHECK(cudaMalloc((void **)&dA, asize));
    GPU_CHECK(cudaMalloc((void **)&dB, bsize));
    GPU_CHECK(cudaMalloc((void **)&dC, csize));
    GPU_CHECK(cudaMemcpy(dA, A, asize, cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(dB, B, bsize, cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(dC, C, csize, cudaMemcpyHostToDevice));

    int ret = gpu_dgemm_device_version(version, transa, transb, m, n, k,
                                       alpha, dA, lda, dB, ldb, beta, dC, ldc);
    if (ret != 0) {
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        return ret;
    }

    GPU_CHECK(cudaMemcpy(C, dC, csize, cudaMemcpyDeviceToHost));
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
