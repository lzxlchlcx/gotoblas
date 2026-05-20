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
include/
  myblas.h                    # 公共 API
src/
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
  kernel/avx2/                # (预留) AVX2 汇编内核
  config/
    generic.h                 # 阻塞参数: P=128, Q=128, R=4096, MR=4, NR=4
test/
  test_gemm.c                 # 20 个正确性测试
  bench_gemm.c                # GFLOPS 基准测试
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

纯 C 通用内核（无 SIMD 优化）：

| 矩阵大小 | GFLOPS |
|----------|--------|
| 64×64    | 6.4    |
| 128×128  | 5.9    |
| 256×256  | 6.1    |
| 512×512  | 6.1    |
| 1024×1024| 6.1    |

参考：OpenBLAS (AVX2) 约 50-100 GFLOPS。

## 参考资料

- Goto, K., & van de Geijn, R. A. (2008). *Anatomy of High-Performance Matrix Multiplication*. ACM TOMS.
- [OpenBLAS 源码](https://github.com/OpenMathLib/OpenBLAS) — 项目中的 `OpenBLAS/` 目录
- `Docs/` 目录包含详细的接口设计分析文档
