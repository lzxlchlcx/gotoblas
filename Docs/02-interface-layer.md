# 02 - 公共接口层 (interface/)

## 文件组织

每个 BLAS 函数在 `interface/` 下有独立的 `.c` 文件：

| 文件 | 对应函数 |
|------|----------|
| `gemm.c` | sgemm, dgemm, cgemm, zgemm（通过宏复用同一份代码） |
| `gemv.c` | sgemv, dgemv, cgemv, zgemv |
| `trsm.c` | strsm, dtrsm, ctrsm, ztrsm |
| `syrk.c` | ssyrk, dsyrk, csyrk, zsyrk |
| ... | ... |

验证来源：`interface/` 目录列表

## 双接口设计

### Fortran BLAS 接口

所有参数通过指针传递（Fortran 约定），列主序：

```c
// interface/gemm.c 中的 NAME() 函数
void NAME(char *TRANSA, char *TRANSB,
          blasint *M, blasint *N, blasint *K,
          FLOAT *alpha,
          IFLOAT *a, blasint *ldA,
          IFLOAT *b, blasint *ldB,
          FLOAT *beta,
          FLOAT *c, blasint *ldC)
```

验证来源：`interface/gemm.c:109-117`（非 CBLAS 分支）

### CBLAS 接口

值传递，支持行/列主序：

```c
void CNAME(enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE TransA,
           enum CBLAS_TRANSPOSE TransB,
           blasint m, blasint n, blasint k,
           FLOAT alpha, IFLOAT *a, blasint lda,
           IFLOAT *b, blasint ldb,
           FLOAT beta, FLOAT *c, blasint ldc)
```

验证来源：`interface/gemm.c:119-131`（CBLAS 分支）

## 转置映射

OpenBLAS 用 2-bit 编码支持 4 种转置状态：

```
transa/transb:
  0 = NoTrans (N)
  1 = Trans (T)
  2 = ConjNoTrans (R)  ← 仅复数有效
  3 = ConjTrans (C)    ← 仅复数有效
```

实数时的退化处理：
```c
// 实数时 R 等价于 N，C 等价于 T
if (transA == 'R') transa = 0;
if (transA == 'C') transa = 1;
```

16 种组合通过 `(transb << 2) | transa` 索引函数指针表：

```c
static int (*gemm[])(...) = {
    GEMM_NN, GEMM_TN, GEMM_RN, GEMM_CN,  // transb=0
    GEMM_NT, GEMM_TT, GEMM_RT, GEMM_CT,  // transb=1
    GEMM_NR, GEMM_TR, GEMM_RR, GEMM_CR,  // transb=2
    GEMM_NC, GEMM_TC, GEMM_RC, GEMM_CC,  // transb=3
};
```

验证来源：`interface/gemm.c:78-95`

## 参数校验

遵循 BLAS 标准的 info 编码：

```c
info = 0;
if (transb < 0)        info = 2;
if (args.m < 0)        info = 3;
if (args.n < 0)        info = 4;
if (args.k < 0)        info = 5;
if (args.lda < nrowa)  info = 8;
if (args.ldb < nrowb)  info = 10;
if (args.ldc < args.m) info = 13;
```

验证来源：`interface/gemm.c:152-160`

## GEMM → GEMV 优化转发

当 n=1 或 m=1 时，GEMM 自动转发为 GEMV，避免进入重量级 Level-3 路径：

```c
if (args.n == 1) {
    // C(m×1) = α * A(m×k) * B(k×1) + β * C(m×1) → dgemv
    GEMV(&NT, &m, &n, args.alpha, args.a, &lda,
         args.b, &inc_x, args.beta, args.c, &inc_y);
    return;
}
```

验证来源：`interface/gemm.c:290-326`

## 小矩阵优化

当 `SMALL_MATRIX_OPT` 启用时，小矩阵走专用快速路径：

```c
if (GEMM_SMALL_MATRIX_PERMIT(transa, transb, m, n, k, alpha, beta)) {
    if (beta == 0.0)
        (GEMM_SMALL_KERNEL_B0((transb << 2) | transa))(...);
    else
        (GEMM_SMALL_KERNEL((transb << 2) | transa))(...);
    return;
}
```

验证来源：`interface/gemm.c:337-354`

## 线程数决策

```c
MNK = (double) args.m * (double) args.n * (double) args.k;
args.nthreads = get_gemm_optimal_nthreads(MNK);
```

`get_gemm_optimal_nthreads` 基于 MNK 阈值表决定线程数，不同 CPU 有不同策略（如 NeoverseV1/V2 有专门的阈值表）。

验证来源：`interface/gemm.c:89-107`

## CBLAS RowMajor 处理

当 `order == CblasRowMajor` 时，OpenBLAS 通过交换 A/B 和 m/n 将行主序转换为列主序：

```c
if (order == CblasRowMajor) {
    args.m = n;  args.n = m;  args.k = k;
    args.a = (void *)b;  args.b = (void *)a;  args.c = (void *)c;
    args.lda = ldb;  args.ldb = lda;  args.ldc = ldc;
    // 转置也需要交换
}
```

验证来源：`interface/gemm.c:187-223`
