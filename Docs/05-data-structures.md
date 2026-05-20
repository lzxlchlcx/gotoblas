# 05 - 核心数据结构

## blas_arg_t — BLAS 参数封装

所有 Level-3 BLAS 函数通过 `blas_arg_t` 结构体传递参数：

```c
typedef struct {
    BLASLONG m, n, k;       // 矩阵维度
    void *a, *b, *c;        // 矩阵指针（void* 支持多精度）
    BLASLONG lda, ldb, ldc; // leading dimensions
    void *alpha, *beta;     // 标量参数（void* 支持复数）
    int nthreads;           // 线程数
    void *common;           // 线程共享数据
} blas_arg_t;
```

验证来源：`common.h` 中的使用，`interface/gemm.c:133-146`

## BLASLONG / blasint — 整数类型

```c
// 64-bit Windows: long long
// 其他平台: long
typedef long BLASLONG;
typedef unsigned long BLASULONG;

// 默认 32-bit int，USE64BITINT 时为 BLASLONG
typedef int blasint;
```

验证来源：`common.h:256-288`

## FLOAT / IFLOAT / XFLOAT — 精度类型

通过编译时宏定义：

```c
#ifdef DOUBLE
#define FLOAT   double       // 主精度
#define SIZE    8            // 字节数
#define BASE_SHIFT 3         // 2^3 = 8
#elif defined(BFLOAT16)
#define IFLOAT  bfloat16     // 输入精度
#define FLOAT   float        // 计算精度（SBGEMM 时）
#define SIZE    2
#else
#define FLOAT   float
#define SIZE    4
#endif

#ifndef XFLOAT
#define XFLOAT  FLOAT        // 扩展精度（packing 用）
#endif

#ifndef IFLOAT
#define IFLOAT  FLOAT        // 输入精度（与 FLOAT 相同除非混合精度）
#endif
```

验证来源：`common.h:299-349`

## COMPSIZE — 复数大小

```c
#ifndef COMPLEX
#define COMPSIZE  1          // 实数：1 个元素
#else
#define COMPSIZE  2          // 复数：2 个元素（实部+虚部）
#endif
```

验证来源：`common.h:351-355`

## BLASFUNC — 符号名装饰

```c
#ifdef NEEDBUNDERSCORE
#define BLASFUNC(FUNC) FUNC##_    // Fortran 下划线约定
#else
#define BLASFUNC(FUNC) FUNC
#endif
```

验证来源：`common.h:194-198`

## gotoblas_t — DYNAMIC_ARCH 函数指针表

当 `DYNAMIC_ARCH` 启用时，所有内核函数通过此结构体的函数指针调用：

```c
typedef struct {
    int dtb_entries;
    int switch_ratio;
    int preferred_size;
    int exclusive_cache;
    int offsetA, offsetB, align;

    // SGEMM 阻塞参数
    int sgemm_p, sgemm_q, sgemm_r;
    int sgemm_unroll_m, sgemm_unroll_n, sgemm_unroll_mn;

    // SGEMM 函数指针
    int (*sgemm_kernel)(BLASLONG, BLASLONG, BLASLONG, float, float *, float *, float *, BLASLONG);
    int (*sgemm_beta)(BLASLONG, BLASLONG, BLASLONG, float, float *, BLASLONG, float *, BLASLONG, float *, BLASLONG);
    int (*sgemm_incopy)(BLASLONG, BLASLONG, float *, BLASLONG, float *);
    int (*sgemm_itcopy)(BLASLONG, BLASLONG, float *, BLASLONG, float *);
    int (*sgemm_oncopy)(BLASLONG, BLASLONG, float *, BLASLONG, float *);
    int (*sgemm_otcopy)(BLASLONG, BLASLONG, float *, BLASLONG, float *);

    // DGEMM 函数指针（类似结构）
    int dgemm_p, dgemm_q, dgemm_r;
    int (*dgemm_kernel)(...);
    int (*dgemm_beta)(...);
    int (*dgemm_incopy)(...);
    int (*dgemm_itcopy)(...);
    int (*dgemm_oncopy)(...);
    int (*dgemm_otcopy)(...);

    // Level 1/2 函数指针
    int (*dgemv_n)(...);
    int (*dgemv_t)(...);
    int (*dger_k)(...);
    // ... 几百个函数指针
} gotoblas_t;
```

验证来源：`common_param.h`

## 宏映射链

以 `dgemm NN` 为例，从接口层到内核的完整宏映射链：

```
interface/gemm.c:
  gemm[(0 << 2) | 0]  →  GEMM_NN

common_macro.h (实数 double):
  GEMM_NN  →  DGEMM_NN

common_d.h:
  DGEMM_NN  →  dgemm_nn

driver/level3/level3.c:
  KERNEL_FUNC  →  GEMM_KERNEL_N  →  DGEMM_KERNEL_N  →  dgemm_kernel

kernel/x86_64/KERNEL.HASWELL:
  DGEMMKERNEL  →  dgemm_kernel_4x8_haswell.S
```

验证来源：`common_macro.h:175`, `common_d.h`, `driver/level3/level3.c:81-84`
