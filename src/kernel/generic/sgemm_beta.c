int sgemm_beta_generic(int m, int n, float beta, float *C, int ldc)
{
    int i, j;
    if (beta == 0.0f) {
        for (j = 0; j < n; j++) {
            for (i = 0; i < m; i++) {
                C[i + j * ldc] = 0.0f;
            }
        }
    } else if (beta != 1.0f) {
        for (j = 0; j < n; j++) {
            for (i = 0; i < m; i++) {
                C[i + j * ldc] *= beta;
            }
        }
    }
    return 0;
}
