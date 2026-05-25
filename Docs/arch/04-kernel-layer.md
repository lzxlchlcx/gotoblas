# 04 - 内核层 (kernel/)

## 目录结构

```
kernel/
├── generic/        # 通用 C 实现（后备）
├── x86_64/         # x86-64 架构内核（515 个文件）
│   ├── KERNEL.HASWELL    # Haswell CPU 的内核配置
│   ├── KERNEL.SKYLAKEX   # Skylake-X CPU 的内核配置
│   ├── dgemm_kernel_4x8_haswell.S    # dgemm 微内核（汇编）
│   ├── dgemm_ncopy_8_skylakex.c      # A 的 packing（NoTrans）
│   ├── dgemm_tcopy_8.S               # A 的 packing（Trans）
│   ├── gemm_beta.S                   # beta 缩放
│   └── ...
├── arm64/          # ARM64 架构内核
├── power/          # POWER 架构内核
└── ...
```

验证来源：`kernel/` 目录列表，`kernel/x86_64/` 目录列表

## KERNEL.{CPU} 配置文件

每个 CPU 型号有一个 `KERNEL.{CPU}` 文件，定义该 CPU 使用的具体内核源文件。

以 Haswell 的 dgemm 为例：

```makefile
# KERNEL.HASWELL
DGEMMKERNEL    = dgemm_kernel_4x8_haswell.S      # 微内核（4×8 分块）
DGEMM_BETA     = dgemm_beta_skylakex.c           # beta 缩放
DGEMMINCOPY    = ../generic/gemm_ncopy_4.c       # A 的 NoTrans packing
DGEMMITCOPY    = ../generic/gemm_tcopy_4.c       # A 的 Trans packing
DGEMMONCOPY    = dgemm_ncopy_8_skylakex.c        # B 的 NoTrans packing
DGEMMOTCOPY    = ../generic/gemm_tcopy_8.c       # B 的 Trans packing
```

验证来源：`kernel/x86_64/KERNEL.HASWELL:48-57`

## 内核命名规则

```
{精度}gemm_kernel_{M}x{N}_{cpu}.{ext}

示例：
dgemm_kernel_4x8_haswell.S    # double, 4×8 分块, Haswell
sgemm_kernel_8x4_haswell_2.c  # float, 8×4 分块, Haswell（C 实现）
zgemm_kernel_4x2_haswell.S    # complex double, 4×2 分块, Haswell
```

内核的 M×N 含义：
- M = GEMM_UNROLL_M（寄存器行阻塞）
- N = GEMM_UNROLL_N（寄存器列阻塞）

Haswell 的配置：
- dgemm: 4×8（MR=4, NR=8）
- sgemm: 8×4（MR=8, NR=4）
- zgemm: 4×2（MR=4, NR=2）

验证来源：`param.h:1566-1578`（UNROLL 参数），`KERNEL.HASWELL:48,36`

## Packing 函数

OpenBLAS 中 packing 函数有两种命名：

| 名称 | 含义 | 示例 |
|------|------|------|
| `ncopy` | NoTrans 拷贝（直接拷贝） | `dgemm_ncopy_8.c` |
| `tcopy` | Trans 拷贝（转置拷贝） | `dgemm_tcopy_4.c` |

命名中的数字表示打包的列/行数：
- `dgemm_ncopy_8`：打包 8 列（适合 NR=8）
- `dgemm_tcopy_4`：转置打包 4 行（适合 MR=4）

ICOPY/OCOPY 的选择逻辑：

```
ICOPY（打包 A）:
  NoTrans → ITCOPY（转置拷贝）  // 因为内核期望 A 按列连续
  Trans   → INCOPY（直接拷贝）

OCOPY（打包 B）:
  NoTrans → ONCOPY（直接拷贝）  // 因为内核期望 B 按行连续
  Trans   → OTCOPY（转置拷贝）
```

验证来源：`driver/level3/level3.c:63-79`

## 微内核接口

```c
// 实数 GEMM 微内核
int dgemm_kernel(
    BLASLONG m,      // 行数（MR 的倍数）
    BLASLONG n,      // 列数（NR 的倍数）
    BLASLONG k,      // 共享维度（KC）
    double alpha,    // 缩放因子
    double *A,       // 已打包的 A（MR × KC 连续）
    double *B,       // 已打包的 B（KC × NR 连续）
    double *C,       // 输出矩阵
    BLASLONG ldc     // C 的 leading dimension
);

// 复数 GEMM 微内核（4 种变体）
int cgemm_kernel_n(...);  // alpha 为实数
int cgemm_kernel_l(...);  // 左乘变体
int cgemm_kernel_r(...);  // 右乘变体
int cgemm_kernel_b(...);  // 双向变体
```

验证来源：`common_level3.h`

## Beta 缩放内核

```c
// C = beta * C
int dgemm_beta(
    BLASLONG m, BLASLONG n, BLASLONG k,
    double beta,
    double *C, BLASLONG ldc,
    double *dummy1, BLASLONG dummy2,
    double *dummy3, BLASLONG dummy4
);
```

验证来源：`common_level3.h`

## GEMM 小矩阵优化内核

当 `SMALL_MATRIX_OPT` 启用时，小矩阵走专用内核：

```c
int dgemm_small_kernel_nn(BLASLONG m, BLASLONG n, BLASLONG k,
    double *A, BLASLONG lda, double alpha,
    double *B, BLASLONG ldb, double beta,
    double *C, BLASLONG ldc);
```

每种转置组合有独立的内核（nn, nt, tn, tt），以及 beta=0 的优化版本（b0_nn, b0_nt 等）。

验证来源：`common_level3.h`，`kernel/x86_64/dgemm_small_kernel_nn_skylakex.c`
