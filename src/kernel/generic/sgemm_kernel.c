int sgemm_kernel_generic(int m, int n, int k,
                         float alpha,
                         const float *A,
                         const float *B,
                         float *C, int ldc)
{
    int i, j, p;
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            float sum = 0.0f;
            for (p = 0; p < k; p++) {
                sum += A[i + p * m] * B[p + j * k];
            }
            C[i + j * ldc] += alpha * sum;
        }
    }
    return 0;
}
