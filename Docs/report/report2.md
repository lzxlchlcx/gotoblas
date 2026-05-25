# MyBLAS AVX2 DGEMM 后续优化空间复盘报告（二）

## 0. 版本信息

| 项目 | 值 |
|------|-----|
| 记录时间 | 2026-05-25 23:26:04 +08:00 |
| 当前分支 | main |
| 当前版本 | 2f38072 |
| 最近整理时间 | 2026-05-26 00:12:01 +08:00 |
| 前置报告 | `Docs/report/avx2-dgemm-performance-report.md` |
| 关联技术笔记 | `Docs/report/avx2-dgemm-assembly-kernel-notes.md`, `Docs/report/avx2-dgemm-p-loop-optimization-notes.md`, `Docs/report/avx2-dgemm-kernel-call-overhead-notes.md` |
| 对比日志 | `logs/d0f1507_compare.log` -> `logs/2f38072_compare.log` |

本报告记录在完成 `report1` 中的 `4.1 交换 col1/row1 循环顺序` 后，对当前性能状态和后续优化空间的判断，便于后续历史复盘。

---

## 1. 当前状态总结

### 1.1 已完成的关键优化

已经实现前置报告中的 `4.1 方案一：交换循环顺序`。

核心变化是：

```c
// 旧结构：每个 col1 都重新遍历 row1，导致同一个 A 微块重复 pack
for col1:
  pack_B(col1)
  for row1:
    pack_A(row1, k0)
    kernel(A, B_col1)

// 新结构：先打包 B，再每个 row1 只打包一次 A，并在内层复用
for col1:
  pack_B(col1)

for row1:
  pack_A(row1, k0)
  for col1:
    kernel(A, B_col1)
```

### 1.2 性能变化

来自 `logs/d0f1507_compare.log` 和 `logs/2f38072_compare.log`。

#### 1 线程

| N | 4.1 前 MyBLAS | 4.1 后 MyBLAS | 提升 |
|---|--------------:|--------------:|-----:|
| 64 | 17.67 | 17.49 | -1.0% |
| 128 | 7.26 | 37.47 | 5.16x |
| 256 | 13.30 | 10.68 | -19.7% |
| 512 | 31.04 | 60.16 | 1.94x |
| 1024 | 27.94 | 50.03 | 1.79x |
| 2048 | 26.34 | 41.31 | 1.57x |

#### 4 线程

| N | 4.1 前 MyBLAS | 4.1 后 MyBLAS | 提升 |
|---|--------------:|--------------:|-----:|
| 64 | 4.47 | 4.51 | +0.9% |
| 128 | 25.45 | 29.47 | +15.8% |
| 256 | 64.02 | 57.09 | -10.8% |
| 512 | 98.08 | 125.76 | 1.28x |
| 1024 | 93.78 | 152.53 | 1.63x |
| 2048 | 96.77 | 170.77 | 1.76x |

### 1.3 瓶颈转移

4.1 前，`pack_a` 是主要瓶颈：

```text
1T n=2048:
pack_a: 57.0%
pack_b: 0.9%
kernel: 42.1%
```

4.1 后，`pack_a` 大幅下降，kernel 成为绝对主瓶颈：

```text
1T n=2048:
pack_a: 1.8%
pack_b: 1.2%
kernel: 97.1%

4T n=2048:
pack_a: 3.0%
pack_b: 1.3%
kernel: 95.6%
```

结论：当前主线优化目标应从 driver/packing 转向 `DGEMM micro-kernel`、cache blocking 和 full-tile fast path。

---

## 2. 对方案 4.2 的重新评估

前置报告中的 `4.2 方案二：优化 pack_a 的 cache 行为` 包括：

- 使用非时序存储 `_mm256_stream_pd` 写 packed A，避免污染 L2
- 对原始 A 的读取使用软件预取 `_mm_prefetch`

### 2.1 原理

当前 `dgemm_pack_a_nn_avx2` 使用普通 store：

```c
__m256d a = _mm256_loadu_pd(&A[i + p * lda]);
_mm256_storeu_pd(&A_packed[i + p * m], a);
```

普通 store 会将目标 cache line 分配进 L1/L2。对于 packed A，这可能带来 cache 污染：

- pack_a 写入 `A_packed` 时占用 L1/L2
- 原始 A 的 stride 读取也会拉入 cache line
- packed B 或 kernel 即将使用的数据可能被驱逐

非时序存储 `_mm256_stream_pd` 会通过 write-combining buffer 写出，通常不分配进 L1/L2，从而降低 cache 污染。

软件预取 `_mm_prefetch` 可以提前把即将读取的原始 A cache line 拉入 cache，减少 load miss 延迟。

### 2.2 当前收益预期下降

4.1 后 pack_a 占比已经从约 57% 降到 1.8%-3.0%。因此即使 pack_a 自身加速 50%，整体收益也很有限：

```text
1T n=2048: pack_a 1.8%  -> 理论整体收益 < 1%
4T n=2048: pack_a 3.0%  -> 理论整体收益约 1%-2%
```

### 2.3 风险

`_mm256_stream_pd` 不一定适合 packed A，因为 packed A 是“写完马上读”的数据。

非时序存储更适合：

```text
写完短期不读、只输出到内存的数据
```

但 packed A 的使用模式是：

```text
pack_A(row1, k0) -> 立即进入多个 col1 kernel 调用读取 sa
```

如果绕过 cache，kernel 随后读取 `sa` 时可能需要重新从内存或较低层级加载，反而变慢。

结论：4.2 可以作为实验项，但不应作为当前主线优化。若实施，应通过 benchmark 判断是否有间接 cache 收益。

---

## 3. 当前主要优化空间

### 3.1 优先级最高：优化 DGEMM micro-kernel

当前 `MR=8, NR=6` kernel 每个 `p` 迭代执行：

```text
2 个 A vector load
6 个 B scalar broadcast
12 个 FMA
```

每个 kernel 调用计算一个 `8x6` C tile。

```text
每 p 计算量 = 8 * 6 * 2 = 96 FLOPs
K=256 时每次 kernel 调用 = 8 * 6 * 256 * 2 = 24576 FLOPs
```

当前瓶颈已经集中在 kernel：

```text
1T n=2048: kernel 97.1%
4T n=2048: kernel 95.6%
```

因此进一步追 OpenBLAS，必须提高 kernel 的单核吞吐。

可尝试方向：

| 方向 | 原理 | 预期 |
|------|------|------|
| p-loop 2x/4x unroll | 提前加载下一组 A/B，增加指令调度空间，隐藏 FMA/load latency | 中等收益，风险低 |
| 重排 load/broadcast/FMA 顺序 | 减少依赖链阻塞，提高端口利用率 | 小到中等收益 |
| 尝试 MR=4, NR=12 | 仍使用 12 个 accumulator，但改变 A/B load 压力分布 | 中等收益，需实测 |
| 手写 assembly kernel | 精确控制寄存器、调度、unroll、prefetch | 潜在收益最大，成本最高 |

建议第一步从 `8x6 p-loop unroll` 开始，因为它改动小、验证快。

---

### 3.2 减少 kernel 调用和入口分支开销

该方向已拆分为独立技术笔记：`Docs/report/avx2-dgemm-kernel-call-overhead-notes.md`。

当前 n=2048 的 kernel 调用次数仍然很高：

```text
1T n=2048: 2,101,248 次 kernel 调用（3 次 DGEMM 累计）
```

单次 DGEMM 约：

```text
(2048 / 8) * ceil(2048 / 6) * (2048 / 256)
= 256 * 342 * 8
= 700416 次 kernel 调用
```

虽然每次 kernel 做 24576 FLOPs，不算特别小，但调用次数非常多，函数调用、参数传递、full/edge 判断仍有累计开销。

可尝试方向：

| 方向 | 原理 | 预期 |
|------|------|------|
| full-tile fast path | driver 中区分 full tile 与 edge tile，满 `8x6` 直接走专用 kernel | 小到中等收益 |
| 拆分 kernel function pointer | 提供 `kernel_full_8x6` 和 `kernel_edge` 两个入口 | 小到中等收益 |
| edge tile 单独处理 | 主循环只处理完整 tile，尾部再走 fallback | 减少主路径分支 |

该方向风险低，但收益通常不如 micro-kernel 本身。

---

### 3.3 调整 blocking 参数 P/Q/R

当前配置：

```c
#define GEMM_HASWELL_D_P  256
#define GEMM_HASWELL_D_Q  256
#define GEMM_HASWELL_D_R  4096
#define GEMM_HASWELL_D_MR 8
#define GEMM_HASWELL_D_NR 6
```

其中 `R=4096` 会导致 packed B 缓冲区很大：

```text
sb size = Q * R * sizeof(double)
        = 256 * 4096 * 8
        = 8 MB
```

对 i9-13900K 的每核 2MB L2 来说，8MB packed B 明显过大，只能主要依赖 L3。虽然这减少了 pack_b 次数，但可能降低 kernel 访问 B 的 cache 命中率。

建议 sweep 以下组合：

| Q | R | packed B 大小 | 备注 |
|---:|---:|--------------:|------|
| 256 | 512 | 1 MB | 更适合 L2 |
| 256 | 1024 | 2 MB | 接近单核 L2 |
| 192 | 1024 | 1.5 MB | 降低 B 工作集 |
| 128 | 1024 | 1 MB | 更小 KC，可能增加 pack_b 次数 |
| 256 | 2048 | 4 MB | 折中方案 |

调参应重点看：

- 1T n=1024/2048
- 4T n=1024/2048
- pack_b 占比是否显著上升
- kernel 占比和整体 GFLOPS 是否改善

---

### 3.4 MR/NR 形状探索

当前 `8x6` 使用 12 个 accumulator：

```text
NR=6 * ceil(MR/4)=2 -> 12 accumulators
```

寄存器使用大致是：

```text
12 accumulator + 2 A vectors + 1 alpha = 15/16 YMM
```

可尝试 `MR=4, NR=12`：

```text
12 accumulator + 1 A vector + 1 alpha + 临时 = 14-15 YMM
```

它的特点：

| 项目 | 8x6 | 4x12 |
|------|-----|------|
| C tile 元素 | 48 | 48 |
| A vector load / p | 2 | 1 |
| B broadcast / p | 6 | 12 |
| accumulator | 12 | 12 |
| C 写回列数 | 6 | 12 |

`4x12` 不一定优于 `8x6`，因为它增加了 B broadcast 数量。但 B 是 packed 连续访问，A load 减半后可能改善调度和 load 压力。该方向值得单独 benchmark。

`8x8` 理论 tile 更大，但需要：

```text
NR=8 * 2 = 16 accumulators
```

仅 accumulator 就占满 16 个 YMM，无法给 A、B、alpha、临时寄存器留空间。用 C intrinsics 很可能出现 spill，风险较高。

---

### 3.5 线程和 OpenBLAS 对比口径

当前 4 线程扩展性较好：

```text
1T n=2048: 41.31 GFLOPS
4T n=2048: 170.77 GFLOPS
扩展倍数: 4.13x
```

这说明 MyBLAS 的线程扩展目前不是第一瓶颈。

但 OpenBLAS 对比仍需确认线程控制是否可靠。建议固定环境变量做复测：

```bash
OPENBLAS_NUM_THREADS=1 ./test/bench_compare
OPENBLAS_NUM_THREADS=4 ./test/bench_compare
```

如果 OpenBLAS 实际线程数不受控，MyBLAS/OpenBLAS 的差距会被误判。

---

## 4. 手写汇编路线说明

关于是否需要手写 AVX2 汇编 micro-kernel，已拆分为独立技术笔记：`Docs/report/avx2-dgemm-assembly-kernel-notes.md`。

当前结论保持不变：不要过早进入手写汇编。短期应先继续使用 C intrinsics 完成 `8x6 p-loop unroll`、`P/Q/R blocking sweep`、full-tile fast path 和 `MR=4, NR=12` 实验；如果这些优化后仍明显落后 OpenBLAS，再通过检查生成汇编中的 spill、FMA 调度和 load/broadcast 排布，判断是否进入手写 assembly kernel 阶段。

---

## 5. 建议优化路线

### 5.1 短期路线

| 优先级 | 任务 | 目标 | 风险 |
|-------:|------|------|------|
| 1 | `8x6` kernel p-loop 2x/4x unroll | 提升 kernel 吞吐 | 低 |
| 2 | sweep `Q/R/P` blocking 参数 | 改善 packed B cache 行为 | 低 |
| 3 | full-tile fast path | 降低主路径分支/调用开销 | 低 |
| 4 | 4.2 pack_a prefetch/stream store 实验 | 验证是否有间接 cache 收益 | 中 |

### 5.2 中期路线

| 优先级 | 任务 | 目标 | 风险 |
|-------:|------|------|------|
| 1 | 尝试 `MR=4, NR=12` kernel | 寻找更优寄存器块形状 | 中 |
| 2 | 修复 `pack_b_tn` | 保证 NT/TT 正确性 | 中 |
| 3 | 分离 NN 专用快路径 | 避免通用转置逻辑影响主路径 | 中 |

### 5.3 长期路线

| 优先级 | 任务 | 目标 | 风险 |
|-------:|------|------|------|
| 1 | 手写 AVX2 assembly micro-kernel | 接近 OpenBLAS 单核效率 | 高 |
| 2 | 更完整的 cache blocking 模型 | 系统调优 MC/KC/NC | 中到高 |
| 3 | 线程亲和性和 NUMA 策略 | 改善多线程稳定性 | 中 |

---

## 6. 当前判断

4.1 已经完成了最明显的 driver 层优化，性能提升显著，特别是大矩阵和 4 线程场景：

```text
4T n=2048: 96.77 -> 170.77 GFLOPS，提升 1.76x
```

当前 `pack_a` 已经不是主瓶颈：

```text
pack_a: 约 2%-3%
kernel: 约 95%-97%
```

因此后续最值得投入的方向是：

```text
1. 优化 8x6 micro-kernel 的 p-loop 调度和 unroll
2. 调整 Q/R blocking，降低 packed B 工作集
3. 为完整 8x6 tile 增加 fast path
4. 将 4.2 作为小规模实验，而不是主线
```

如果目标是继续逼近 OpenBLAS，最终瓶颈会集中到 micro-kernel 的指令级效率。C intrinsics 可以继续推进一段，但若要追到 OpenBLAS 的核心性能，后续大概率需要手写 AVX2 汇编 kernel。
