int dgemm_pack_a_nn(int m, int k, const double *A, int lda, double *A_packed)
{
    int i, p;
    for (p = 0; p < k; p++) {
        for (i = 0; i < m; i++) {
            A_packed[i + p * m] = A[i + p * lda];
        }
    }
    return 0;
}

int dgemm_pack_a_tn(int m, int k, const double *A, int lda, double *A_packed)
{
    int i, p;
    for (p = 0; p < k; p++) {
        for (i = 0; i < m; i++) {
            A_packed[i + p * m] = A[p + i * lda];
        }
    }
    return 0;
}

int dgemm_pack_b_nn(int k, int n, const double *B, int ldb, double *B_packed)
{
    int p, j;
    for (j = 0; j < n; j++) {
        for (p = 0; p < k; p++) {
            B_packed[p + j * k] = B[p + j * ldb];
        }
    }
    return 0;
}

int dgemm_pack_b_tn(int k, int n, const double *B, int ldb, double *B_packed)
{
    int p, j;
    for (j = 0; j < n; j++) {
        for (p = 0; p < k; p++) {
            B_packed[p + j * k] = B[j + p * ldb];
        }
    }
    return 0;
}
