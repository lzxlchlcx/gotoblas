# 08 - MyBLAS 分块算法新手教学

本文面向第一次学习本仓库的人，目标是讲清楚 `src/` 目录里 GEMM 的分块算法如何工作。

这里的 GEMM 指矩阵乘法：

```text
C = alpha * op(A) * op(B) + beta * C
```

其中 `op(A)` 和 `op(B)` 可以是不转置矩阵，也可以是转置矩阵。

## 先看源码入口

本仓库的主要源码在 `src/`：

```text
src/
├── api/       用户可调用的 my_dgemm / my_sgemm 入口
├── config/    不同后端的分块参数
├── driver/    GEMM 分块循环和多线程调度
├── kernel/    pack 函数、micro-kernel、beta 函数
├── util/      日志和辅助工具
└── myblas.h   对外头文件
```

学习分块算法时，建议按这个顺序阅读：

1. `src/api/dgemm.c`
2. `src/driver/gemm_driver.c`
3. `src/config/generic.h` 和 `src/config/haswell.h`
4. `src/kernel/generic/dgemm_pack.c`
5. `src/kernel/generic/dgemm_kernel.c`
6. `src/driver/gemm_thread.c`

`sgemm` 和 `dgemm` 的结构基本相同。`dgemm` 使用 `double`，`sgemm` 使用 `float`。

## 为什么需要分块

最朴素的矩阵乘法通常长这样：

```c
for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        for (int p = 0; p < k; p++) {
            sum += A[i + p * lda] * B[p + j * ldb];
        }
        C[i + j * ldc] += alpha * sum;
    }
}
```

这个写法容易理解，但对现代 CPU 不友好。

主要问题是：

1. 数据太大，不能全部放进缓存。
2. A、B 的访问步长可能不连续。
3. CPU 的 SIMD 和寄存器很难充分利用。
4. 多线程时需要避免多个线程写同一块 C。

所以高性能 GEMM 通常会做三件事：

1. blocking：把大矩阵切成小块。
2. packing：把小块复制成连续内存。
3. micro-kernel：用很小的固定形状做核心计算。

本仓库的 `src/` 实现就是这个思路。

## 分块参数

分块参数定义在：

```text
src/config/generic.h
src/config/haswell.h
```

核心参数有五个：

| 参数 | 方向 | 含义 |
|------|------|------|
| `P` | M | M 方向的大块高度 |
| `Q` | K | K 方向的大块深度 |
| `R` | N | N 方向的大块宽度 |
| `MR` | M | micro-kernel 的 M 方向高度 |
| `NR` | N | micro-kernel 的 N 方向宽度 |

以 `generic` 双精度为例：

```c
#define GEMM_GENERIC_D_P  128
#define GEMM_GENERIC_D_Q  128
#define GEMM_GENERIC_D_R  4096
#define GEMM_GENERIC_D_MR 4
#define GEMM_GENERIC_D_NR 4
```

以 `avx2/haswell` 双精度为例：

```c
#define GEMM_HASWELL_D_P  256
#define GEMM_HASWELL_D_Q  256
#define GEMM_HASWELL_D_R  4096
#define GEMM_HASWELL_D_MR 8
#define GEMM_HASWELL_D_NR 6
```

可以先把它们理解成：

```text
P/Q/R 控制大块大小
MR/NR 控制最小计算块大小
```

## 调用流程总览

以 `my_dgemm()` 为例，入口在 `src/api/dgemm.c`。

整体流程如下：

```text
my_dgemm
  |
  |-- 解析 transa/transb
  |-- 检查 m/n/k/lda/ldb/ldc
  |-- 处理 k == 0 或 alpha == 0 的特殊情况
  |-- 选择 generic 或 avx2 kernel table
  |-- 分配 sa / sb packing buffer
  |-- 先处理 beta * C
  |-- 单线程调用 gemm_driver_double
  |-- 多线程调用 gemm_parallel_double
```

`my_sgemm()` 的流程一样，只是数据类型从 `double` 变成 `float`。

## beta 是先处理的

GEMM 的完整公式是：

```text
C = alpha * A * B + beta * C
```

当前实现里，`beta * C` 不是放进 micro-kernel 里做，而是在 API 层提前处理。

代码在 `src/api/dgemm.c`：

```c
if (beta == 0.0) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            C[i + j * ldc] = 0.0;
} else if (beta != 1.0) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            C[i + j * ldc] *= beta;
}
```

处理完以后，后面的 driver 只负责累加：

```text
C += alpha * A * B
```

## driver 是分块算法的核心

真正的分块循环在：

```text
src/driver/gemm_driver.c
```

双精度版本是 `gemm_driver_double()`，单精度版本是 `gemm_driver_float()`。

双精度的主要循环结构可以简化成：

```text
for col0 in 0..n step R:          N 方向大块
  for k0 in 0..k step Q:          K 方向大块
    for col1 in col0 block step NR:
      pack B 的 k_rem x col1_rem 小面板
      for row0 in 0..m step P:    M 方向大块
        for row1 in row0 block step MR:
          pack A 的 row1_rem x k_rem 小面板
          micro-kernel 计算 C(row1, col1)
```

对应源码是：

```c
for (col0 = 0; col0 < n; col0 += R) {
    for (k0 = 0; k0 < k; k0 += Q) {
        for (col1 = col0; col1 < col0 + col_rem; col1 += col1_rem) {
            pack_b(...);

            for (row0 = 0; row0 < m; row0 += P) {
                for (row1 = row0; row1 < row0 + row_rem; row1 += row1_rem) {
                    pack_a(...);
                    kernels->kernel(...);
                }
            }
        }
    }
}
```

注意实际代码里 `col1_rem` 和 `row1_rem` 会在每次循环中计算，用来处理边界块。

## 三层大块是什么意思

矩阵乘法可以看成：

```text
C[M x N] = A[M x K] * B[K x N]
```

本仓库按三个方向切块：

```text
M 方向用 P 切
K 方向用 Q 切
N 方向用 R 切
```

图示：

```text
A: M x K                 B: K x N                 C: M x N

   K                         N                         N
   |                         |                         |
M--+                     K---+                     M---+

A 块大小: P x Q
B 块大小: Q x R
C 块大小: P x R
```

这样做的目的不是改变数学结果，而是改变计算顺序，让当前正在用的数据更容易留在缓存里。

## 微块 MR x NR

在大块内部，代码还会继续切成 micro-kernel 需要的小块。

M 方向微块：

```c
if (row1_rem > MR) row1_rem = MR;
```

N 方向微块：

```c
if (col1_rem > NR) col1_rem = NR;
```

所以一次 micro-kernel 计算的是：

```text
C[row1 : row1 + MR, col1 : col1 + NR]
```

它会沿着 K 方向累加 `k_rem` 次。

逻辑上等价于：

```text
C_micro[MR x NR] += alpha * A_micro[MR x k_rem] * B_micro[k_rem x NR]
```

最后一块可能不足 `MR` 或 `NR`，所以代码传入的是 `row1_rem` 和 `col1_rem`，不是固定传入 `MR` 和 `NR`。

## packing buffer 是什么

`sa` 和 `sb` 是临时缓冲区。

它们在 API 层分配：

```c
size_t sa_size = (size_t)cfg.P * cfg.Q * sizeof(double) + 4096;
size_t sb_size = (size_t)cfg.Q * cfg.R * sizeof(double) + 4096;
double *sa = (double *)malloc(sa_size + cfg.offset_a);
double *sb = (double *)malloc(sb_size + cfg.offset_b);
```

含义是：

| buffer | 用途 | 大小近似 |
|--------|------|----------|
| `sa` | 保存 packed A | `P * Q` |
| `sb` | 保存 packed B | `Q * R` |

为什么要 pack？

1. 把原矩阵中可能不连续的数据整理成连续内存。
2. 让 micro-kernel 的访问模式稳定。
3. 减少复杂的 `lda/ldb` 步长访问。
4. 为 SIMD 优化创造条件。

## A 的 packing 布局

通用双精度 A packing 在：

```text
src/kernel/generic/dgemm_pack.c
```

不转置版本：

```c
int dgemm_pack_a_nn(int m, int k, const double *A, int lda, double *A_packed)
{
    for (int p = 0; p < k; p++) {
        for (int i = 0; i < m; i++) {
            A_packed[i + p * m] = A[i + p * lda];
        }
    }
    return 0;
}
```

packed 后的布局是：

```text
A_packed[i + p * m]
```

也就是第 `p` 列里的 `m` 个元素连续存放。

转置版本：

```c
A_packed[i + p * m] = A[p + i * lda];
```

它读的是转置视角下的元素，但写入 packed buffer 的格式保持一致。

## B 的 packing 布局

B packing 的不转置版本：

```c
int dgemm_pack_b_nn(int k, int n, const double *B, int ldb, double *B_packed)
{
    for (int j = 0; j < n; j++) {
        for (int p = 0; p < k; p++) {
            B_packed[p + j * k] = B[p + j * ldb];
        }
    }
    return 0;
}
```

packed 后的布局是：

```text
B_packed[p + j * k]
```

也就是第 `j` 列里的 `k` 个元素连续存放。

转置版本：

```c
B_packed[p + j * k] = B[j + p * ldb];
```

同样，读法根据转置变化，但写入 packed buffer 的格式保持一致。

## micro-kernel 怎么计算

通用双精度 kernel 在：

```text
src/kernel/generic/dgemm_kernel.c
```

代码如下：

```c
int dgemm_kernel_generic(int m, int n, int k,
                         double alpha,
                         const double *A,
                         const double *B,
                         double *C, int ldc)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += A[i + p * m] * B[p + j * k];
            }
            C[i + j * ldc] += alpha * sum;
        }
    }
    return 0;
}
```

这里的 `A` 和 `B` 已经不是原始矩阵，而是 packed buffer。

所以 kernel 内部不需要关心原始矩阵的 `lda` 和 `ldb`，只需要用规则的 packed 布局访问。

## 一次 micro-kernel 的数据关系

假设当前微块大小是 `m = MR`，`n = NR`，`k = k_rem`。

那么 kernel 做的是：

```text
for j in 0..NR:
  for i in 0..MR:
    C[i, j] += alpha * sum_p(A_packed[i, p] * B_packed[p, j])
```

对应内存下标：

```text
A_packed: A[i + p * m]
B_packed: B[p + j * k]
C:        C[i + j * ldc]
```

可以把它想象成：

```text
      B micro-panel
      k_rem x NR

        NR
     +------+
     |      |
K    |      |
     |      |
     +------+

A micro-panel        C micro-tile
MR x k_rem           MR x NR

     K                  NR
  +------+            +------+
M |      |      =>  M |      |
  +------+            +------+
```

## generic 和 avx2 的区别

`generic` 版本更容易读：

```text
src/kernel/generic/
```

它主要是普通 C 循环，适合学习算法结构。

`avx2` 版本用于性能优化：

```text
src/kernel/avx2/
```

它使用 AVX2 指令和更适合 Haswell 的参数。

选择逻辑在 `src/api/dgemm.c`：

```c
#ifdef __AVX2__
if (has_avx2) {
    gemm_config_avx2_double(&cfg);
    kernels = &gemm_kernel_avx2_double;
} else {
    gemm_config_generic_double(&cfg);
    kernels = &gemm_kernel_generic_double;
}
#else
gemm_config_generic_double(&cfg);
kernels = &gemm_kernel_generic_double;
#endif
```

这里有两个东西一起切换：

| 内容 | generic | avx2 |
|------|---------|------|
| 分块参数 | `config/generic.h` | `config/haswell.h` |
| kernel table | `gemm_kernel_generic_double` | `gemm_kernel_avx2_double` |
| pack/kernel 实现 | 普通 C | AVX2 优化 |

## kernel table 是什么

kernel table 定义在 `src/driver/gemm_internal.h`：

```c
typedef struct {
    gemm_kernel_func   kernel;
    pack_func          pack_a_nn;
    pack_func          pack_a_tn;
    pack_func          pack_b_nn;
    pack_func          pack_b_tn;
    gemm_beta_func     beta;
} gemm_kernel_table_t;
```

它的作用是把不同后端的函数组织起来。

例如 generic 双精度表在 `src/kernel/generic/kernel_init.c`：

```c
gemm_kernel_table_t gemm_kernel_generic_double = {
    .kernel    = dgemm_kernel_generic,
    .pack_a_nn = dgemm_pack_a_nn,
    .pack_a_tn = dgemm_pack_a_tn,
    .pack_b_nn = dgemm_pack_b_nn,
    .pack_b_tn = dgemm_pack_b_tn,
    .beta      = dgemm_beta_generic,
};
```

driver 不直接写死调用哪个 kernel，而是通过函数指针调用：

```c
kernels->kernel(...);
```

这使得 driver 逻辑可以复用，后端实现可以替换。

## 转置是怎么处理的

API 层会把 `transa` 和 `transb` 转成整数标记：

```c
ta = 0 表示 A 不转置
ta = 1 表示 A 转置或共轭转置
tb = 0 表示 B 不转置
tb = 1 表示 B 转置或共轭转置
```

driver 根据这个标记选择不同 pack 函数：

```c
pack_func pack_a = (transa == 0) ? kernels->pack_a_nn : kernels->pack_a_tn;
pack_func pack_b = (transb == 0) ? kernels->pack_b_nn : kernels->pack_b_tn;
```

关键点是：

```text
转置差异主要在 pack 阶段消化
kernel 看到的 A_packed / B_packed 布局保持一致
```

这样 micro-kernel 就不用为 NN、NT、TN、TT 写四套通用逻辑。

## 边界块如何处理

矩阵尺寸不一定刚好被 `P/Q/R/MR/NR` 整除。

所以代码每层都会算剩余长度。

N 大块：

```c
col_rem = n - col0;
if (col_rem > R) col_rem = R;
```

K 大块：

```c
k_rem = k - k0;
if (k_rem > Q) k_rem = Q;
```

M 大块：

```c
row_rem = m - row0;
if (row_rem > P) row_rem = P;
```

N 微块：

```c
col1_rem = (col0 + col_rem) - col1;
if (col1_rem > NR) col1_rem = NR;
```

M 微块：

```c
row1_rem = (row0 + row_rem) - row1;
if (row1_rem > MR) row1_rem = MR;
```

所以边界块不会越界，只是传给 pack 和 kernel 的尺寸变小。

## 多线程按 N 方向切分

多线程逻辑在：

```text
src/driver/gemm_thread.c
```

线程数来自：

```c
MYBLAS_NUM_THREADS
```

或者：

```c
myblas_set_num_threads(int num_threads)
```

多线程切分的是 N 方向，也就是 C 的列方向。

核心代码：

```c
int chunk = n / nthreads;
int remainder = n % nthreads;

local_arg.n = task->n_to - task->n_from;
local_arg.B = (const double *)task->arg->B + task->n_from * task->arg->ldb;
local_arg.C = (double *)task->arg->C + task->n_from * task->arg->ldc;
```

可以理解成：

```text
C 的列被分给不同线程
每个线程只写自己负责的列
A 被所有线程只读共享
B 根据线程负责的列偏移
每个线程有自己的 sa / sb
```

这种切法的优点是简单安全，因为不同线程不会写同一个 C 元素。

## 完整算法伪代码

把上面的内容合起来，可以写成：

```text
function my_dgemm(transa, transb, m, n, k, alpha, A, B, beta, C):
  check arguments
  choose config and kernel table
  scale C by beta

  for each N block of width R:
    for each K block of depth Q:
      for each N micro-panel of width NR:
        pack B[k_block, n_micro] into sb

        for each M block of height P:
          for each M micro-panel of height MR:
            pack A[m_micro, k_block] into sa
            kernel(sa, sb, C[m_micro, n_micro])
```

## 一个具体例子

假设使用 generic 双精度参数：

```text
P = 128
Q = 128
R = 4096
MR = 4
NR = 4
```

计算：

```text
C[300 x 10] = A[300 x 200] * B[200 x 10]
```

因为 `n = 10` 小于 `R = 4096`，所以 N 方向只有一个大块。

K 方向会分成两块：

```text
K block 0: k0 = 0,   k_rem = 128
K block 1: k0 = 128, k_rem = 72
```

M 方向会分成三块：

```text
M block 0: row0 = 0,   row_rem = 128
M block 1: row0 = 128, row_rem = 128
M block 2: row0 = 256, row_rem = 44
```

每个 M 大块内部再按 `MR = 4` 切。

每个 N 大块内部再按 `NR = 4` 切，所以 `n = 10` 会变成：

```text
N micro 0: col1 = 0, col1_rem = 4
N micro 1: col1 = 4, col1_rem = 4
N micro 2: col1 = 8, col1_rem = 2
```

最后一个 N 微块宽度是 2，不足 `NR`，但 kernel 仍然可以处理，因为传入的是实际宽度 `col1_rem`。

## 新手阅读建议

第一次读这个仓库，不建议直接从 AVX2 kernel 开始。

推荐顺序是：

1. 先读 `src/api/dgemm.c`，理解入口和参数检查。
2. 再读 `src/config/generic.h`，记住 `P/Q/R/MR/NR` 的含义。
3. 再读 `src/driver/gemm_driver.c`，对照本文理解循环顺序。
4. 再读 `src/kernel/generic/dgemm_pack.c`，理解 packed buffer 布局。
5. 再读 `src/kernel/generic/dgemm_kernel.c`，理解 kernel 如何消费 packed buffer。
6. 最后读 `src/kernel/avx2/`，看 SIMD 版本如何优化同一套结构。

## 最重要的结论

本仓库当前的 GEMM 分块算法可以概括为：

```text
按 N/K/M 三个方向切大块
按 NR/MR 切微块
先把 A 和 B pack 成连续内存
再用 micro-kernel 做 C += alpha * A * B
多线程时按 N 方向切分 C 的列
```

理解这句话后，再看 `src/driver/gemm_driver.c` 的循环结构就会清晰很多。
