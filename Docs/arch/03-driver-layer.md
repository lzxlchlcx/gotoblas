# 03 - 驱动层 (driver/level3/)

## 文件组织

| 文件 | 职责 |
|------|------|
| `gemm.c` | 入口：选择单线程/多线程路径 |
| `level3.c` | 单线程 GEMM 驱动（模板） |
| `level3_thread.c` | 多线程 GEMM 驱动（模板） |
| `gemm_thread_m.c` | M 维度并行 |
| `gemm_thread_n.c` | N 维度并行 |
| `gemm_thread_mn.c` | M×N 维度并行 |
| `level3_syrk.c` | SYRK 驱动 |
| `trsm_L.c` / `trsm_R.c` | TRSM 驱动（左/右） |

验证来源：`driver/level3/` 目录列表

## gemm.c 入口

`driver/level3/gemm.c` 仅 82 行，核心逻辑是选择编译路径：

```c
#ifdef THREADED_LEVEL3
#include "level3_thread.c"    // 多线程版本
#else
#include "level3.c"           // 单线程版本
#endif
```

验证来源：`driver/level3/gemm.c:78-82`

## level3.c 阻塞循环（核心）

这是 GotoBLAS 算法的核心实现。采用三级阻塞循环：

```
for js = 0 to N step GEMM_R:           ← N 维度阻塞（适合 L3）
  for ls = 0 to K step GEMM_Q:         ← K 维度阻塞（适合 L2）
    ICOPY_OPERATION: pack A[ls:ls+Q, m_from:m_to] → sa
    for jjs = js to js+R step min_jj:  ← N 细分（寄存器级别）
      OCOPY_OPERATION: pack B[ls:ls+Q, jjs:jjs+jj] → sb
      KERNEL_OPERATION: sa × sb → C[m_from:m_to, jjs:jjs+jj]
    for is = m_from+P to M step P:     ← M 维度阻塞（适合 L2）
      ICOPY_OPERATION: pack A[ls:ls+Q, is:is+P] → sa
      KERNEL_OPERATION: sa × sb → C[is:is+P, js:js+R]
```

### 关键宏

| 宏 | 含义 | 映射（以 dgemm NN 为例） |
|----|------|--------------------------|
| `ICOPY_OPERATION` | 打包 A | `GEMM_ITCOPY`（NoTrans 时转置拷贝） |
| `OCOPY_OPERATION` | 打包 B | `GEMM_ONCOPY`（NoTrans 时直接拷贝） |
| `KERNEL_OPERATION` | 微内核计算 | `GEMM_KERNEL_N` → `dgemm_kernel` |
| `BETA_OPERATION` | beta 缩放 C | `GEMM_BETA` → `dgemm_beta` |

### ICOPY/OCOPY 选择逻辑

根据转置组合选择不同的 packing 方式：

```c
// A 的 packing：NoTrans 时用 ITCOPY（转置拷贝），Trans 时用 INCOPY（直接拷贝）
#if defined(NN) || defined(NT) || defined(NC) || defined(NR) || ...
#define ICOPY_OPERATION(M, N, A, LDA, X, Y, BUFFER) GEMM_ITCOPY(...)
#else
#define ICOPY_OPERATION(M, N, A, LDA, X, Y, BUFFER) GEMM_INCOPY(...)
#endif

// B 的 packing：NoTrans 时用 ONCOPY，Trans 时用 OTCOPY
#if defined(NN) || defined(TN) || defined(CN) || ...
#define OCOPY_OPERATION(M, N, A, LDA, X, Y, BUFFER) GEMM_ONCOPY(...)
#else
#define OCOPY_OPERATION(M, N, A, LDA, X, Y, BUFFER) GEMM_OTCOPY(...)
#endif
```

验证来源：`driver/level3/level3.c:63-79`

### KERNEL_FUNC 选择

根据复数的共轭组合选择不同的内核变体：

```c
// NN/TT/TN/NT → KERNEL_N (alpha 为实数)
// CN/CT/RN/RT → KERNEL_L (alpha 的左乘变体)
// NC/TC/NR/TR → KERNEL_R (alpha 的右乘变体)
// CC/CR/RC/RR → KERNEL_B (alpha 的双向变体)
```

验证来源：`driver/level3/level3.c:81-94`

## 阻塞参数使用

```c
// 阻塞参数在 param.h 中按 CPU 定义
#define GEMM_P  512   // M 维度阻塞（适合 L2 缓存）
#define GEMM_Q  256   // K 维度阻塞（适合 L2 缓存）
#define GEMM_R  13824 // N 维度阻塞（适合 L3 缓存）

// 阻塞循环中的使用：
l2size = GEMM_P * GEMM_Q;  // L2 适配的总元素数

// 自适应阻塞：当剩余维度不足时，调整 P 以适配 L2
if (min_l >= GEMM_Q * 2) {
    min_l = GEMM_Q;
} else {
    if (min_l > GEMM_Q) {
        min_l = ((min_l / 2 + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
    }
    gemm_p = ((l2size / min_l + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
}
```

验证来源：`driver/level3/level3.c:289-341`

## ICOPY/OCOPY 宏的实际展开

以 dgemm NN 为例：

```c
// ICOPY: 打包 A 的一个块到 sa
// NN 时 A 是 NoTrans，需要转置拷贝（ITCOPY）
ICOPY_OPERATION(min_l, min_i, a, lda, ls, m_from, sa)
// 展开为：
GEMM_ITCOPY(min_l, min_i, (IFLOAT *)(a) + ((ls) + (m_from) * (lda)) * COMPSIZE, lda, sa)
// 即 dgemm_itcopy(min_l, min_i, &a[ls + m_from*lda], lda, sa)

// OCOPY: 打包 B 的一个块到 sb
// NN 时 B 是 NoTrans，需要直接拷贝（ONCOPY）
OCOPY_OPERATION(min_l, min_jj, b, ldb, ls, jjs, sb + offset)
// 展开为：
GEMM_ONCOPY(min_l, min_jj, (IFLOAT *)(b) + ((ls) + (jjs) * (ldb)) * COMPSIZE, ldb, sb + offset)
// 即 dgemm_oncopy(min_l, min_jj, &b[ls + jjs*ldb], ldb, sb + offset)
```

验证来源：`driver/level3/level3.c:63-79`
