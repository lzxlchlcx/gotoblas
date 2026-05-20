int dgemm_kernel_generic(int m, int n, int k,
                         double alpha,
                         const double *A,
                         const double *B,
                         double *C, int ldc)
{
    int i, j, p;
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            double sum = 0.0;
            for (p = 0; p < k; p++) {
                sum += A[i + p * m] * B[p + j * k];
            }
            C[i + j * ldc] += alpha * sum;
        }
    }
    return 0;
}
