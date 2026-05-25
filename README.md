# MyBLAS — GotoBLAS 风格的 GEMM 实现

从零实现的矩阵乘法库，基于 GotoBLAS 论文的算法架构。

## 特性

- **纯 C 实现**，无外部依赖（仅 pthreads 用于多线程）
- **GotoBLAS 三级阻塞算法**：R/Q/P 分块 + explicit packing + 微内核 GEBP
- **四种转置组合**：NN / NT / TN / TT
- **双精度 + 单精度**：`my_dgemm` / `my_sgemm`
- **pthread 多线程**：N 维度并行，线程私有缓冲区
- **可插拔内核表**：通过函数指针切换不同 CPU 的内核实现

## 构建

```bash
make          # 编译生成 libmyblas.a
make test     # 运行正确性测试
make bench    # 运行性能基准
make clean    # 清理
```

## 使用

```c
#include "myblas.h"

// C = 1.0 * A * B + 0.0 * C  (双精度, 无转置)
my_dgemm('N', 'N', m, n, k, 1.0, A, lda, B, ldb, 0.0, C, ldc);

// 设置线程数
myblas_set_num_threads(4);
// 或通过环境变量: MYBLAS_NUM_THREADS=4
```

## 项目结构

```
src/
  myblas.h                    # 公共 API
  api/
    dgemm.c                   # my_dgemm: 参数校验 + 特殊情况处理
    sgemm.c                   # my_sgemm
  driver/
    gemm_internal.h           # 核心数据结构 (gemm_arg_t, gemm_config_t, gemm_kernel_t)
    gemm_driver.c             # 三级阻塞循环 (js/ls/is)
    gemm_thread.c             # pthread 多线程分发
  kernel/generic/
    dgemm_kernel.c            # 纯 C 微内核 (MR=4, NR=4)
    sgemm_kernel.c            # 纯 C 微内核 (MR=8, NR=4)
    dgemm_pack.c              # 4 种打包函数 (pack_a_nn/tn, pack_b_nn/tn)
    sgemm_pack.c
    dgemm_beta.c              # C = beta * C
    sgemm_beta.c
    kernel_init.c             # 内核表初始化
  kernel/avx2/
    dgemm_kernel.c            # AVX2 微内核 (MR=8, NR=6, 配置驱动多路径)
    cpuid.c                   # 运行时 AVX2+FMA 检测
    dgemm_pack.c              # AVX2 打包函数
    kernel_init.c             # AVX2 内核表初始化
  config/
    generic.h                 # 阻塞参数: P=128, Q=128, R=4096, MR=4, NR=4
    haswell.h                 # AVX2 参数: P=256, Q=256, R=4096, MR=8, NR=6
  util/
    myblas_log.h              # 编译宏控制的日志系统
    myblas_log.c
test/
  test_gemm.c                 # 20 个正确性测试
  bench_gemm.c                # GFLOPS 基准测试
  bench_compare.c             # MyBLAS vs OpenBLAS 对比基准
```

## 架构

```
┌───────────────────────────────────────────────┐
│  API 层: my_dgemm() / my_sgemm()              │
│  参数校验 → 特殊情况 → 分配缓冲区 → 调度      │
├───────────────────────────────────────────────┤
│  驱动层: gemm_driver()                         │
│  三级阻塞循环 (R/Q/P) → packing → 微内核      │
├───────────────────────────────────────────────┤
│  内核层: gemm_kernel_t 函数指针表               │
│  微内核 GEBP + 打包函数 + beta 缩放            │
├───────────────────────────────────────────────┤
│  线程层: gemm_parallel()                       │
│  N 维度划分 → pthread → 线程私有 sa/sb         │
└───────────────────────────────────────────────┘
```

## 算法核心

GotoBLAS 的关键思想是将矩阵乘法分解为适合各级缓存的小块：

```
for js = 0 to n step R:           ← L3 缓存
  for ls = 0 to k step Q:         ← L2 缓存
    pack B → sb
    for is = 0 to m step P:       ← L2 缓存
      pack A → sa
      微内核: C[is,js] += sa × sb  ← 寄存器
```

阻塞参数同时满足缓存大小和 TLB reach 约束（详见 `Docs/06-gemm-parameters.md`）。

## 性能

测试环境：Intel i9-13900K (AVX2/FMA, 无 AVX-512)，单线程。

### 版本 1：纯 C 通用内核（无 SIMD）

初始版本，纯标量实现 MR=4×NR=4。

| 矩阵大小 | GFLOPS |
|----------|--------|
| 64×64    | 6.4    |
| 128×128  | 5.9    |
| 256×256  | 6.1    |
| 512×512  | 6.1    |
| 1024×1024| 6.1    |

### 版本 2：修复 AVX2 检测 + MR=4×NR=8 内核

修复 `cpuid.c` 中 `__get_cpuid` → `__get_cpuid_count`，使 AVX2 内核首次生效。

| 矩阵大小 | GFLOPS | vs 版本 1 |
|----------|--------|-----------|
| 1024×1024| 25.5   | 4.5x      |

### 版本 3：MR=8×NR=6 微内核（当前）

15/16 YMM 寄存器利用率，配置驱动的 `#if` 多路径框架。

| 矩阵大小 | GFLOPS |
|----------|--------|
| 64×64    | 17.3   |
| 128×128  | 14.3   |
| 256×256  | 27.2   |
| 512×512  | 26.3   |
| 1024×1024| 28.4   |
| 2048×2048| 26.5   |

4 线程 n=2048：103.8 GFLOPS。

### 性能演进总结

```
v1 纯 C 标量:                    ~6.1 GFLOPS  (n=1024, 1T)
v2 AVX2 MR=4×NR=8:              ~25.5 GFLOPS  → 4.5x
v3 AVX2 MR=8×NR=6 (当前):       ~28.4 GFLOPS  → +11%
─────────────────────────────────────────────────────────
待优化: 交换 driver 循环顺序，预计 ~60+ GFLOPS (2x+)
OpenBLAS 参考:                  ~81 GFLOPS  (n=1024, 1T)
```

## 参考资料

- Goto, K., & van de Geijn, R. A. (2008). *Anatomy of High-Performance Matrix Multiplication*. ACM TOMS.
- [OpenBLAS 源码](https://github.com/OpenMathLib/OpenBLAS) — 项目中的 `OpenBLAS/` 目录
- `Docs/` 目录包含详细的接口设计分析文档
