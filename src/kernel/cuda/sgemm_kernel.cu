#include <cuda_runtime.h>
#include <mma.h>
#include "kernel/cuda/gemm_gpu.h"
#include "kernel/cuda/gpu_common.cuh"

using namespace nvcuda;

__device__ static inline float sgemm_load_a(const float *A, int lda, int transa, int row, int p)
{
    return transa ? A[p + row * lda] : A[row + p * lda];
}

__device__ static inline float sgemm_load_b(const float *B, int ldb, int transb, int p, int col)
{
    return transb ? B[col + p * ldb] : B[p + col * ldb];
}

__global__ void sgemm_naive_kernel(int m, int n, int k, int transa, int transb,
                                   float alpha, const float *A, int lda,
                                   const float *B, int ldb,
                                   float beta, float *C, int ldc)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (int p = 0; p < k; p++) {
        sum += sgemm_load_a(A, lda, transa, row, p) * sgemm_load_b(B, ldb, transb, p, col);
    }
    C[row + col * ldc] = beta * C[row + col * ldc] + alpha * sum;
}

template<int TILE>
__global__ void sgemm_shared_kernel(int m, int n, int k, int transa, int transb,
                                    float alpha, const float *A, int lda,
                                    const float *B, int ldb,
                                    float beta, float *C, int ldc)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int kk = 0; kk < k; kk += TILE) {
        int ap = kk + threadIdx.x;
        int bp = kk + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < m && ap < k) ? sgemm_load_a(A, lda, transa, row, ap) : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bp < k && col < n) ? sgemm_load_b(B, ldb, transb, bp, col) : 0.0f;
        __syncthreads();

        for (int p = 0; p < TILE; p++) sum += sA[threadIdx.y][p] * sB[p][threadIdx.x];
        __syncthreads();
    }

    if (row < m && col < n) C[row + col * ldc] = beta * C[row + col * ldc] + alpha * sum;
}

template<int TM, int TN, int TK, int MR, int NR, int DBUF>
__global__ void sgemm_regblock_kernel(int m, int n, int k, int transa, int transb,
                                      float alpha, const float *A, int lda,
                                      const float *B, int ldb,
                                      float beta, float *C, int ldc)
{
    __shared__ float sA[DBUF][TM][TK + 1];
    __shared__ float sB[DBUF][TK][TN + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int nthreads = blockDim.x * blockDim.y;
    int row0 = blockIdx.y * TM + ty * MR;
    int col0 = blockIdx.x * TN + tx * NR;
    float acc[MR][NR];

    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < NR; j++) acc[i][j] = 0.0f;
    }

    for (int kk = 0; kk < k; kk += TK) {
        int buf = DBUF == 1 ? 0 : ((kk / TK) & 1);
        for (int idx = tid; idx < TM * TK; idx += nthreads) {
            int r = idx / TK;
            int p = idx - r * TK;
            int gr = blockIdx.y * TM + r;
            int gp = kk + p;
            sA[buf][r][p] = (gr < m && gp < k) ? sgemm_load_a(A, lda, transa, gr, gp) : 0.0f;
        }
        for (int idx = tid; idx < TK * TN; idx += nthreads) {
            int p = idx / TN;
            int c = idx - p * TN;
            int gp = kk + p;
            int gc = blockIdx.x * TN + c;
            sB[buf][p][c] = (gp < k && gc < n) ? sgemm_load_b(B, ldb, transb, gp, gc) : 0.0f;
        }
        __syncthreads();

        for (int p = 0; p < TK; p++) {
            float ar[MR], br[NR];
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

template<int TM, int TN, int TK>
__global__ void sgemm_tensorcore_tf32_kernel(int m, int n, int k, int transa, int transb,
                                             float alpha, const float *A, int lda,
                                             const float *B, int ldb,
                                             float beta, float *C, int ldc)
{
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 8;
    constexpr int WARP_SIZE = 32;
    constexpr int NWARPS_M = 2;
    constexpr int NWARPS_N = 4;
    constexpr int WARPS = NWARPS_M * NWARPS_N;
    constexpr int WMMA_PER_WARP_M = (TM / NWARPS_M) / WMMA_M;
    constexpr int WMMA_PER_WARP_N = (TN / NWARPS_N) / WMMA_N;
    static_assert(WARPS * WARP_SIZE <= 1024, "too many threads");
    static_assert((TM / NWARPS_M) % WMMA_M == 0, "warp M must be multiple of WMMA_M");
    static_assert((TN / NWARPS_N) % WMMA_N == 0, "warp N must be multiple of WMMA_N");

    __shared__ float sA[TM][TK + 1];
    __shared__ float sB[TK][TN + 1];
    __shared__ float sC_out[WARPS][WMMA_M * WMMA_N];

    int warp_id = threadIdx.x / WARP_SIZE;
    int wm = warp_id / NWARPS_N;
    int wn = warp_id % NWARPS_N;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int nthreads = WARPS * WARP_SIZE;

    int warp_row0 = wm * (TM / NWARPS_M);
    int warp_col0 = wn * (TN / NWARPS_N);

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WMMA_PER_WARP_M][WMMA_PER_WARP_N];
    for (int ii = 0; ii < WMMA_PER_WARP_M; ii++) {
        for (int jj = 0; jj < WMMA_PER_WARP_N; jj++) wmma::fill_fragment(c_frag[ii][jj], 0.0f);
    }

    for (int kk = 0; kk < k; kk += TK) {
        if (!transa) {
            for (int idx = tid; idx < TM * TK; idx += nthreads) {
                int p = idx / TM;
                int r = idx - p * TM;
                int gr = blockIdx.y * TM + r;
                int gp = kk + p;
                float av = (gr < m && gp < k) ? A[gr + gp * lda] : 0.0f;
                sA[r][p] = wmma::__float_to_tf32(av);
            }
        } else {
            for (int idx = tid; idx < TM * TK; idx += nthreads) {
                int r = idx / TK;
                int p = idx - r * TK;
                int gr = blockIdx.y * TM + r;
                int gp = kk + p;
                float av = (gr < m && gp < k) ? A[gp + gr * lda] : 0.0f;
                sA[r][p] = wmma::__float_to_tf32(av);
            }
        }
        if (!transb) {
            for (int idx = tid; idx < TK * TN; idx += nthreads) {
                int c = idx / TK;
                int p = idx - c * TK;
                int gp = kk + p;
                int gc = blockIdx.x * TN + c;
                float bv = (gp < k && gc < n) ? B[gp + gc * ldb] : 0.0f;
                sB[p][c] = wmma::__float_to_tf32(bv);
            }
        } else {
            for (int idx = tid; idx < TK * TN; idx += nthreads) {
                int p = idx / TN;
                int c = idx - p * TN;
                int gp = kk + p;
                int gc = blockIdx.x * TN + c;
                float bv = (gp < k && gc < n) ? B[gc + gp * ldb] : 0.0f;
                sB[p][c] = wmma::__float_to_tf32(bv);
            }
        }
        __syncthreads();

        for (int p = 0; p < TK; p += WMMA_K) {
            for (int jj = 0; jj < WMMA_PER_WARP_N; jj++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &sB[p][warp_col0 + jj * WMMA_N], TN + 1);

                for (int ii = 0; ii < WMMA_PER_WARP_M; ii++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
                    wmma::load_matrix_sync(a_frag, &sA[warp_row0 + ii * WMMA_M][p], TK + 1);
                    wmma::mma_sync(c_frag[ii][jj], a_frag, b_frag, c_frag[ii][jj]);
                }
            }
        }
        __syncthreads();
    }

    int direct_store = (alpha == 1.0f && beta == 0.0f && (ldc & 3) == 0);
    for (int ii = 0; ii < WMMA_PER_WARP_M; ii++) {
        for (int jj = 0; jj < WMMA_PER_WARP_N; jj++) {
            int base_row = blockIdx.y * TM + warp_row0 + ii * WMMA_M;
            int base_col = blockIdx.x * TN + warp_col0 + jj * WMMA_N;

            if (direct_store && base_row + WMMA_M <= m && base_col + WMMA_N <= n) {
                wmma::store_matrix_sync(&C[base_row + base_col * ldc], c_frag[ii][jj], ldc, wmma::mem_col_major);
            } else {
                wmma::store_matrix_sync(sC_out[warp_id], c_frag[ii][jj], WMMA_N, wmma::mem_row_major);
                __syncwarp();

                for (int idx = lane; idx < WMMA_M * WMMA_N; idx += WARP_SIZE) {
                    int i = idx / WMMA_N;
                    int j = idx % WMMA_N;
                    int row = base_row + i;
                    int col = base_col + j;
                    if (row < m && col < n) {
                        C[row + col * ldc] = beta * C[row + col * ldc] + alpha * sC_out[warp_id][i * WMMA_N + j];
                    }
                }
                __syncwarp();
            }
        }
    }
}

extern "C" int gpu_sgemm_device_version(int version,
                                        char transa, char transb,
                                        int m, int n, int k,
                                        float alpha, const float *dA, int lda,
                                        const float *dB, int ldb,
                                        float beta, float *dC, int ldc)
{
    int ta = !(transa == 'N' || transa == 'n');
    int tb = !(transb == 'N' || transb == 'n');

    if (version <= 0) {
        dim3 block(32, 32);
        dim3 grid(gpu_ceil_div(n, 32), gpu_ceil_div(m, 32));
        sgemm_naive_kernel<<<grid, block>>>(m, n, k, ta, tb, alpha, dA, lda, dB, ldb, beta, dC, ldc);
    } else if (version == 1) {
        dim3 block(16, 16);
        dim3 grid(gpu_ceil_div(n, 16), gpu_ceil_div(m, 16));
        sgemm_shared_kernel<16><<<grid, block>>>(m, n, k, ta, tb, alpha, dA, lda, dB, ldb, beta, dC, ldc);
    } else if (version == 2) {
        dim3 block(GPU_SGEMM_TILE_N / GPU_SGEMM_THREAD_N, GPU_SGEMM_TILE_M / GPU_SGEMM_THREAD_M);
        dim3 grid(gpu_ceil_div(n, GPU_SGEMM_TILE_N), gpu_ceil_div(m, GPU_SGEMM_TILE_M));
        sgemm_regblock_kernel<GPU_SGEMM_TILE_M, GPU_SGEMM_TILE_N, GPU_SGEMM_TILE_K,
                              GPU_SGEMM_THREAD_M, GPU_SGEMM_THREAD_N, 1>
            <<<grid, block>>>(m, n, k, ta, tb, alpha, dA, lda, dB, ldb, beta, dC, ldc);
    } else if (version == 3) {
        dim3 block(GPU_SGEMM_TILE_N / GPU_SGEMM_THREAD_N, GPU_SGEMM_TILE_M / GPU_SGEMM_THREAD_M);
        dim3 grid(gpu_ceil_div(n, GPU_SGEMM_TILE_N), gpu_ceil_div(m, GPU_SGEMM_TILE_M));
        sgemm_regblock_kernel<GPU_SGEMM_TILE_M, GPU_SGEMM_TILE_N, GPU_SGEMM_TILE_K,
                              GPU_SGEMM_THREAD_M, GPU_SGEMM_THREAD_N, 2>
            <<<grid, block>>>(m, n, k, ta, tb, alpha, dA, lda, dB, ldb, beta, dC, ldc);
    } else {
        dim3 block(8 * 32);
        dim3 grid(gpu_ceil_div(n, GPU_SGEMM_TC_TILE_N), gpu_ceil_div(m, GPU_SGEMM_TC_TILE_M));
        sgemm_tensorcore_tf32_kernel<GPU_SGEMM_TC_TILE_M, GPU_SGEMM_TC_TILE_N, GPU_SGEMM_TC_TILE_K>
            <<<grid, block>>>(m, n, k, ta, tb, alpha, dA, lda, dB, ldb, beta, dC, ldc);
    }

    GPU_CHECK(cudaGetLastError());
    return 0;
}

extern "C" int gpu_sgemm_version(int version,
                                 char transa, char transb,
                                 int m, int n, int k,
                                 float alpha, const float *A, int lda,
                                 const float *B, int ldb,
                                 float beta, float *C, int ldc)
{
    int ta = !(transa == 'N' || transa == 'n');
    int tb = !(transb == 'N' || transb == 'n');
    size_t asize = (size_t)lda * (ta ? m : k) * sizeof(float);
    size_t bsize = (size_t)ldb * (tb ? k : n) * sizeof(float);
    size_t csize = (size_t)ldc * n * sizeof(float);
    float *dA = 0, *dB = 0, *dC = 0;

    GPU_CHECK(cudaMalloc((void **)&dA, asize));
    GPU_CHECK(cudaMalloc((void **)&dB, bsize));
    GPU_CHECK(cudaMalloc((void **)&dC, csize));
    GPU_CHECK(cudaMemcpy(dA, A, asize, cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(dB, B, bsize, cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(dC, C, csize, cudaMemcpyHostToDevice));

    int ret = gpu_sgemm_device_version(version, transa, transb, m, n, k,
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
