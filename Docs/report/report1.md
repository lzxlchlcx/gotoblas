# MyBLAS AVX2 DGEMM 性能分析与优化报告

## 1. 背景

### 1.1 项目目标

优化 MyBLAS 库的 DGEMM（双精度矩阵乘法）性能，逐步逼近乃至超越 OpenBLAS。

### 1.2 运行环境

| 项目 | 值 |
|------|-----|
| CPU | Intel i9-13900K (Raptor Lake) |
| P-core | 8 核，最高 5.8 GHz |
| E-core | 16 核 |
| SIMD | AVX2 / FMA（不支持 AVX-512） |
| YMM 寄存器 | 16 个 × 256-bit |
| L1D | 48 KB / P-core |
| L2 | 2 MB / P-core |
| L3 | 36 MB（共享） |

### 1.3 初始状态

项目初始时，benchmark 结果（`test/bench_compare`，n=1024，1 线程）：

```
MyBLAS:   5.6 GFLOPS
OpenBLAS: 326 GFLOPS
差距:     58x
```

---

## 2. 已完成的优化

### 2.1 修复 AVX2 检测 Bug（cpuid.c）

**问题**：`src/kernel/avx2/cpuid.c` 使用 `__get_cpuid(7, ...)` 检测 AVX2，但该函数对 CPUID leaf 7 返回全零 EBX，导致 AVX2 **永远检测不到**，始终回退到标量通用 kernel。

**验证**：

```
__get_cpuid(7)         → EAX=0 EBX=0 ECX=0 EDX=0     ← 全零！AVX2=false
__get_cpuid_count(7,0) → EBX=0x219c27ab, bit 5=1     ← AVX2=true ✓
```

**修复**：将 `__get_cpuid(7, ...)` 改为 `__get_cpuid_count(7, 0, ...)`。

**效果**（n=1024，1 线程）：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| MyBLAS | 5.6 GFLOPS | 25.5 GFLOPS |
| 提升 | — | **4.5x** |

### 2.2 MR=8×NR=6 微内核 + 配置驱动框架

**动机**：原 MR=4×NR=8 kernel 仅使用 12/16 个 YMM 寄存器（8 累加器 + 4 辅助），每 k 次迭代产出 32 FLOPs，计算密度不足。

**方案**：

1. 建立配置驱动的 kernel 框架：`dgemm_kernel.c` 通过 `#include "config/haswell.h"` 引入 MR/NR 宏，使用 `#if` 预处理器选择不同实现路径
2. 新增 MR=8×NR=6 优化路径（12 累加器 + 2 A 加载，15/16 YMM）
3. 保留 MR=4×NR=8 备选路径
4. 新增通用 AVX2 fallback（4×4 子块循环，支持任意 MR/NR）

**寄存器分配（MR=8, NR=6）**：

| 用途 | 数量 | 说明 |
|------|------|------|
| 累加器 c[j][r] | 12 | NR=6 × ceil(MR/4)=2 |
| A 加载 a0, a1 | 2 | rows 0-3 和 4-7 |
| alpha | 1 | 标量广播 |
| 合计 | 15/16 | c_old 复用 a0/a1 |

**效果**（n=2048，4 线程）：

| 指标 | MR=4×NR=8 | MR=8×NR=6 |
|------|-----------|-----------|
| MyBLAS | 78.7 GFLOPS | 103.8 GFLOPS |
| 提升 | — | **+32%** |

---

## 3. 性能瓶颈分析

### 3.1 现象：GFLOPS 不随 N 增长

Benchmark 结果显示 MyBLAS 的 GFLOPS 在 N 增大时几乎不变：

```
--- 1 thread ---
   64:  17.3 GFLOPS
  128:  14.3 GFLOPS
  256:  27.2 GFLOPS
  512:  26.3 GFLOPS
 1024:  28.4 GFLOPS
 2048:  26.5 GFLOPS    ← 几乎不变
```

对比 OpenBLAS：

```
  256:  137 GFLOPS
  512:  131 GFLOPS
 1024:  145 GFLOPS
 2048:  611 GFLOPS     ← 持续增长
```

### 3.2 量化瓶颈：pack_a 占 66% 时间

**测量方法**：在 driver 的三层嵌套循环（col1 → row1 → kernel）中插入 `clock_gettime(CLOCK_MONOTONIC)` 计时点，分别累计 pack_a、pack_b、kernel 三个阶段的耗时。具体做法：

1. 在 `gemm_driver_double` 的循环体内，对每个 `pack_a` 调用前后记录时间戳，差值累加到 `t_pack_a`
2. 同理对 `pack_b` 和 `kernel` 调用分别累加 `t_pack_b` 和 `t_kernel`
3. 调用开销和分支预测开销归入最近的前一阶段（因计时点紧贴函数调用）
4. 使用 `niter` 次迭代取总和，避免单次测量的异常值

计时开销约 ~50ns/次 `clock_gettime` 调用，而每次 pack_a/kernel 调用耗时 ~300-3000ns，计时误差 < 10%。

测试代码复现了与 `gemm_driver_double` 相同的循环结构，但用独立计时代替实际计算，确保测量结果反映真实的 packing 与 kernel 时间占比。

实测耗时分解如下：

| N | pack_a | pack_b | kernel | GFLOPS |
|---|--------|--------|--------|--------|
| 64 | 27% | 13% | 60% | 18.8 |
| 128 | 39% | 6% | 55% | 30.8 |
| 256 | 39% | 5% | 56% | 30.5 |
| 512 | 50% | 3% | 48% | 29.8 |
| 1024 | **60%** | 1% | 38% | 23.1 |
| 2048 | **66%** | 1% | 33% | 22.1 |

**关键发现**：N 越大，pack_a 占比越高。到 n=2048 时，**打包 A 占了三分之二的时间，kernel 计算只占三分之一**。

### 3.3 Kernel 本身的性能

隔离测试 kernel（数据在 L1，无 packing 开销）：

```
Kernel-only (MR=8 NR=6 K=256): 81.3 GFLOPS
  - 计算耗时: 92%（1536 cycles）
  - 调用开销: 8%（126 cycles）
```

Kernel 本身可达 81.3 GFLOPS，但经过 packing 后整体降至 22-30 GFLOPS，**效率仅为 27-37%**。

### 3.4 根因：Driver 循环顺序导致 pack_a 大量重复

当前 `src/driver/gemm_driver.c` 的循环结构：

```
for col0 (R=4096 列块):
  for k0 (Q=256 K-块):
    for col1 (NR=6 列):          ← 外层
      pack_B(k_rem, col1_rem)    ← 打包 B 的 6 列
      for row0 (P=256 行块):
        for row1 (MR=8 行):      ← 内层
          pack_A(row1, k0)       ← ⚠️ 每换一组 B 列就重新打包 A！
          kernel(A, B(col1))
```

问题：对于同一个 A 微块 `(row1, k0)`，内层 pack_A 的输出**完全相同**（相同的行、相同的 k 范围），但由于 col1 是外层循环，每当处理新的 6 列 B 时，整个行循环重新执行，导致 pack_A 被重复调用。

以 n=2048, m=2048, k=2048 为例：

| 指标 | 当前 | 理论最优 |
|------|------|----------|
| pack_a 调用次数 | 256 × 341 × 4 = **349,184** | 256 × 4 = **1,024** |
| 重复倍数 | 341x | 1x |

### 3.5 Cache 驱逐加剧问题

除重复打包外，pack_a 本身也有 cache 问题：

- `pack_a_nn_avx2` 从原始矩阵 A 读取数据，访问模式为 `A[i + p * lda]`
- 对于 n=2048，lda=2048（stride=16KB），每次 pack_a 读取 8×256=2048 个 double（16KB），但分散在 256 个不同的 cache line 上
- 256 次 stride 访问触发 256 次 cache line fill，总读取量约 256 × 64B = 16KB（实际访问的 A 数据），但这些访问会**污染 L2**，驱逐已打包的 B 数据

当 N 增大时：
- N=256：原始矩阵 512KB，fit in L2，pack_a 的读取不会过多驱逐 sb
- N=2048：原始矩阵 32MB，超出 L3，每次 pack_a 的 strided 读取驱逐 sb → kernel 读 sb 产生 L3 miss

---

## 4. 优化方案

### 4.1 方案一：交换循环顺序（推荐，预计效果最大）

将 col1 和 row1 的循环层级交换，使 pack_a 在外层执行一次，内层复用：

```
for col0:
  for k0:
    // 先打包所有 B 列块
    for col1:
      pack_B(col1)

    // 再按行处理，每个 row1 只打包一次 A
    for row0:
      for row1:
        pack_A(row1, k0)          ← 每个 (row1, k0) 只执行 1 次
        for col1:                 ← 内层复用 packed A
          kernel(A, B(col1))
```

**预期效果**：
- pack_a 调用减少 341x（n=2048 时）
- pack_a 时间占比从 66% 降至 ~5%
- 理论 GFLOPS 提升至 ~60+（kernel 占比从 33% 提升到 ~90%）

**影响范围**：仅修改 `src/driver/gemm_driver.c`，kernel 和 pack 函数不变。

### 4.2 方案二：优化 pack_a 的 cache 行为

- 使用非时序存储（`_mm256_stream_pd`）写 packed A，避免污染 L2
- 对原始 A 的读取使用软件预取（`_mm_prefetch`）
- 可与方案一叠加使用

### 4.3 方案三：增大微内核规模

- 当前 8×6 kernel 每次调用计算 48 个 multiply-accumulate
- OpenBLAS 通常使用更大微内核（如 8×12），或在一个调用中处理多个 MR 块
- 需要更多寄存器或更巧妙的寄存器分配

### 4.4 方案优先级

| 优先级 | 方案 | 预期提升 | 实现难度 |
|--------|------|----------|----------|
| 1 | 交换循环顺序 | 2-3x | 低（改 driver 循环） |
| 2 | pack_a cache 优化 | 10-20% | 低（改 pack 函数） |
| 3 | 增大微内核 | 30-50% | 高（重写 kernel） |

---

## 5. 已知问题

### 5.1 pack_b_tn 转置打包 Bug（已有）

NT（transb='T'）和 TT 场景下结果不正确，该 bug 在 MR=4 和 MR=8 配置下均复现，与本次优化无关。

### 5.2 OpenBLAS 线程控制

Benchmark 中 `OPENBLAS_NUM_THREADS=1` 可能未生效（n=2048 时 OpenBLAS 达到 611 GFLOPS，远超单核理论峰值 ~88 GFLOPS），实际可能使用了多核。

---

## 6. 文件变更清单

### 6.1 已修改

| 文件 | 变更 |
|------|------|
| `src/kernel/avx2/cpuid.c` | `__get_cpuid(7)` → `__get_cpuid_count(7, 0)` |
| `src/config/haswell.h` | `D_MR=4→8`, `D_NR=8→6` |
| `src/kernel/avx2/dgemm_kernel.c` | 配置驱动 `#if` 框架 + MR=8×NR=6 优化路径 + 通用 fallback |

### 6.2 待修改（下一步优化）

| 文件 | 计划 |
|------|------|
| `src/driver/gemm_driver.c` | 交换 col1/row1 循环顺序 |
| `src/kernel/avx2/dgemm_pack.c` | 可选：非时序存储 + 软件预取 |

---

## 7. 性能演进总结

```
初始状态（AVX2 未检测到）:    ~5.6 GFLOPS (n=1024, 1T)
修复 cpuid 后 (MR=4×NR=8):  ~25.5 GFLOPS  → 4.5x
MR=8×NR=6 kernel:            ~28.4 GFLOPS  → +11%
─────────────────────────────────────────────
理论下一步（交换循环顺序）:   ~60+ GFLOPS   → 预计 2x+
OpenBLAS 参考值:             ~145 GFLOPS (n=1024, 1T)
```
