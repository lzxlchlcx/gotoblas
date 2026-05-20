int dgemm_beta_generic(int m, int n, double beta, double *C, int ldc)
{
    int i, j;
    if (beta == 0.0) {
        for (j = 0; j < n; j++) {
            for (i = 0; i < m; i++) {
                C[i + j * ldc] = 0.0;
            }
        }
    } else if (beta != 1.0) {
        for (j = 0; j < n; j++) {
            for (i = 0; i < m; i++) {
                C[i + j * ldc] *= beta;
            }
        }
    }
    return 0;
}
