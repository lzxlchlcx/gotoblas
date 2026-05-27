# AVX2 DGEMM Micro-Kernel 与手写汇编必要性笔记

## 0. 版本信息

| 项目 | 值 |
|------|-----|
| 记录时间 | 2026-05-25 23:35:23 +08:00 |
| 当前分支 | main |
| 当前版本 | 2f38072 |
| 最近补充时间 | 2026-05-26 21:11:47 +08:00 |
| 关联复盘 | `Docs/report/report2.md` |
| 关联源码 | `src/kernel/avx2/dgemm_kernel.c` |

本文档单独记录关于 DGEMM micro-kernel 是否需要手写 AVX2 汇编的判断，以及相关背景知识。结论不是“现在必须写汇编”，而是：当前应先继续推进 C intrinsics 和 blocking 调优；当实测证明编译器生成代码成为瓶颈时，再进入手写汇编阶段。

---

## 1. 当前判断

手写汇编不是当前阶段的必选项。

更准确的判断是：

```text
当前阶段：优先用 C intrinsics + blocking 调参继续推进
追极限阶段：如果 kernel 指令级效率成为明确瓶颈，再考虑手写 AVX2 汇编
```

原因是当前还有一些低风险优化没有完成：

| 优化项 | 是否需要汇编 | 说明 |
|--------|--------------|------|
| `8x6` p-loop 2x/4x unroll | 不需要 | 先给编译器更多调度空间 |
| `P/Q/R` blocking sweep | 不需要 | 这是 cache 工作集问题 |
| full-tile fast path | 不需要 | 减少主路径分支和边界处理 |
| `MR=4, NR=12` 实验 | 不需要 | 可以继续用 intrinsics 实现 |
| 检查编译器生成代码 | 不需要 | 先判断是否真的存在 spill 或调度问题 |
| 极限 micro-kernel 调度 | 可能需要 | 需要精确控制寄存器和指令顺序 |

结论：不要过早进入手写汇编。应先通过 intrinsics 版本的实验确认瓶颈位置，再决定是否值得投入汇编成本。

---

## 2. 为什么追 OpenBLAS 时可能需要汇编

DGEMM 的核心性能主要由 micro-kernel 决定。当前 4.1 优化后，耗时已经高度集中在 kernel：

```text
1T n=2048: kernel 97.1%
4T n=2048: kernel 95.6%
```

这说明后续优化不是简单减少 packing 或 driver 开销，而是要提高这段极热循环的指令级效率。

当前 `8x6` kernel 每个 `p` 迭代大致执行：

```text
2 个 A vector load
6 个 B scalar broadcast
12 个 vector FMA
```

Raptor Lake P-core 的 AVX2/FMA 理论能力约为：

```text
每条 256-bit double FMA = 4 lanes * multiply-add 2 FLOPs = 8 FLOPs
每周期最多 2 条 256-bit FMA
理论峰值 = 16 double FLOPs / cycle
若频率按 5.8 GHz 估算，单 P-core 理论峰值约 92.8 GFLOPS
```

实际运行时频率可能因为 AVX2 负载下降，因此真实峰值通常低于按最高睿频计算的值。

要接近这个峰值，kernel 必须持续满足：

```text
FMA pipeline 尽量不空
load/broadcast 不阻塞 FMA
FMA latency 被足够多的 independent accumulators 隐藏
寄存器不 spill 到栈
循环分支和地址计算开销足够低
C 写回阶段不破坏主循环寄存器分配
```

这些细节正是编译器不一定能稳定做到最优的地方。

---

## 3. Intrinsics 和汇编的区别

C intrinsics 不是完全等价于手写汇编。

例如：

```c
c00 = _mm256_fmadd_pd(a0, b0, c00);
```

这通常会让编译器生成一条 AVX2/FMA 指令，但以下决策仍由编译器负责：

| 决策 | 影响 |
|------|------|
| 哪个变量放在哪个 YMM 寄存器 | 决定是否发生 spill |
| load A 和 broadcast B 的时机 | 决定是否等待内存数据 |
| FMA 指令顺序 | 决定能否隐藏 FMA latency |
| 循环是否展开 | 决定分支开销和调度窗口 |
| 地址计算如何安排 | 决定 AGU 和整数端口压力 |
| C 写回时如何复用临时寄存器 | 决定是否挤占 accumulator |

手写汇编的价值在于可以明确规定寄存器用途：

```text
ymm0-ymm11  = 12 个 C accumulators
ymm12       = A rows 0-3
ymm13       = A rows 4-7
ymm14       = B broadcast / scratch
ymm15       = alpha / scratch
```

同时可以精确安排指令顺序，例如：

```text
提前 load A[p+1]
broadcast B0[p]
FMA c00/c01
broadcast B1[p]
FMA c10/c11
穿插 prefetch
控制 loop branch 的位置
```

这类控制是 OpenBLAS、BLIS 等高性能 BLAS kernel 的主要优势来源。

---

## 4. 当前 8x6 kernel 的寄存器压力

当前配置来自 `src/config/haswell.h`：

```c
#define GEMM_HASWELL_D_MR 8
#define GEMM_HASWELL_D_NR 6
```

对应的 fast path 在 `src/kernel/avx2/dgemm_kernel.c` 中只在 `MR == 8 && NR == 6` 时启用。这个 kernel 的核心主循环结构是：

```c
for (p = 0; p < k; p++) {
    __m256d a0 = _mm256_loadu_pd(&A[p * 8]);
    __m256d a1 = _mm256_loadu_pd(&A[p * 8 + 4]);

    __m256d b0 = _mm256_set1_pd(B[p + 0 * k]);
    c00 = _mm256_fmadd_pd(a0, b0, c00);
    c01 = _mm256_fmadd_pd(a1, b0, c01);

    ...

    __m256d b5 = _mm256_set1_pd(B[p + 5 * k]);
    c50 = _mm256_fmadd_pd(a0, b5, c50);
    c51 = _mm256_fmadd_pd(a1, b5, c51);
}
```

### 4.1 accumulator 数量

`MR=8, NR=6` 的 accumulator 数量为：

```text
NR * ceil(MR / 4) = 6 * 2 = 12 个 YMM accumulator
```

这 12 个 accumulator 在代码中明确对应：

| C tile 列 | rows 0-3 | rows 4-7 |
|-----------|----------|----------|
| col 0 | `c00` | `c01` |
| col 1 | `c10` | `c11` |
| col 2 | `c20` | `c21` |
| col 3 | `c30` | `c31` |
| col 4 | `c40` | `c41` |
| col 5 | `c50` | `c51` |

这些 accumulator 的生命周期跨越整个 `p` loop。也就是说，从进入 `for (p = 0; p < k; p++)` 前清零开始，到 loop 结束后乘 `alpha` 和写回 `C` 之前，它们都必须保持 live。

这是当前寄存器压力的核心原因：这 12 个 YMM 不能在主循环内部释放。

### 4.2 主循环内部的临时寄存器

每个 `p` 迭代还需要：

```text
2 个 A vector: a0, a1
1 个 B broadcast: b0/b1/b2/b3/b4/b5 逐个复用
```

因此主循环热路径的最小 YMM 需求是：

```text
12 accumulators + 2 A vectors + 1 B broadcast = 15 个 YMM
```

这只是主循环内的理想下限。x86-64 + AVX2 一共只有 `ymm0-ymm15` 共 16 个 YMM，因此主循环理论上只剩 1 个 YMM 余量。

这个 1 个 YMM 余量通常要留给编译器做以下事情：

```text
地址计算或 load/broadcast 形成的中间值
编译器为了调度提前生成的下一个 broadcast
循环展开后下一组 A load 或 B broadcast
主循环和写回阶段之间的寄存器重命名/分配余量
```

所以当前 `8x6` kernel 不是“刚好宽松”，而是“在不展开时勉强可放下”。

### 4.3 `valpha` 和 C 写回阶段的压力

代码在 loop 前生成：

```c
__m256d valpha = _mm256_set1_pd(alpha);
```

但 `valpha` 只在 `p` loop 结束后使用：

```c
c00 = _mm256_mul_pd(c00, valpha);
...
c51 = _mm256_mul_pd(c51, valpha);
```

理想情况下，编译器不应该让 `valpha` 在整个 `p` loop 中长期占用一个 YMM。更合理的做法是：

```text
主循环期间只保留 12 个 C accumulator、2 个 A vector、1 个 B broadcast
loop 结束后再 broadcast alpha，或者从 GPR/栈重新生成 valpha
```

如果编译器把 `valpha` 也跨 loop 固定放在 YMM 中，主循环需求会变成：

```text
12 accumulators + 2 A vectors + 1 B broadcast + 1 valpha = 16 个 YMM
```

这会完全占满 AVX2 的 16 个 YMM，没有任何调度余量。

C 写回阶段还需要一个 `t` 临时寄存器：

```c
__m256d t;
t = _mm256_loadu_pd(&C[0 + 0 * ldc]);
c00 = _mm256_add_pd(t, c00);
_mm256_storeu_pd(&C[0 + 0 * ldc], c00);
```

不过 `t` 只在 loop 之后使用，不与 `a0/a1/bj` 的主循环临时变量重叠。因此它主要影响写回阶段，不是主循环 FMA throughput 的第一瓶颈。

### 4.4 展开后的寄存器风险

如果对 `p` loop 做 2x unroll，直觉上会想同时保留：

```text
当前 p 的 a0/a1 和 b broadcast
下一 p 的 a0/a1 和 b broadcast
同一组 12 个 accumulator
```

这会使寄存器需求很容易超过 16 个 YMM：

```text
12 accumulators + 4 A vectors + 1-2 B broadcasts = 17-18 个 YMM
```

因此 2x unroll 不能简单写成“同时保留两整组 A/B”。比较安全的方向是：

```text
保持 12 个 accumulator 常驻
只提前 load 下一组 A，或只提前部分 B broadcast
让 B broadcast 尽量短生命周期、逐列复用
检查生成汇编是否出现 accumulator spill
```

4x unroll 的风险更高。如果没有手写汇编控制寄存器分配，编译器可能为了调度窗口生成更多 live temporary，导致 accumulator 被 spill。

### 4.5 如何在汇编中识别寄存器不够

AVX2 `8x6` 主循环里最不希望看到的是 accumulator spill。典型模式是：

```text
vmovupd %ymm?, -0x??(%rsp)
vmovupd -0x??(%rsp), %ymm?
vmovapd %ymm?, -0x??(%rsp)
vmovapd -0x??(%rsp), %ymm?
```

需要注意，不是所有访问栈的 `vmov*` 都同样严重。判断重点是：

| 栈访问位置 | 严重程度 | 说明 |
|------------|----------|------|
| 出现在 `p` loop 内部，夹在 `vfmadd*pd` 之间 | 高 | 很可能是 accumulator 或热临时变量 spill |
| 出现在函数入口/出口 | 中 | 可能是 ABI、对齐或编译器保存临时状态 |
| 出现在 loop 结束后的 alpha/C 写回阶段 | 低到中 | 影响写回，但通常不是主循环瓶颈 |

一旦 accumulator 在 `p` loop 内 spill，DGEMM 性能会明显下降，因为原本应保存在寄存器中的热数据被迫反复读写内存。

### 4.6 当前 8x6 的结论

当前 `8x6` 形状的优点是 accumulator 数量足够多，可以帮助隐藏 FMA latency。缺点是寄存器使用已经贴近 AVX2 上限。

总结为：

| 项目 | 数量 | 生命周期 | 压力判断 |
|------|------|----------|----------|
| C accumulator | 12 YMM | 整个 `p` loop | 必须常驻，不能 spill |
| A vector | 2 YMM | 每个 `p` 迭代 | 可复用，但最好提前 load |
| B broadcast | 1 YMM | 每列 2 条 FMA | 应短生命周期复用 |
| alpha vector | 0-1 YMM | loop 后才需要 | 不应长期占用主循环寄存器 |
| C old/scratch | 1 YMM | loop 后写回 | 不影响主循环寄存器压力 |

因此，当前 kernel 的关键不是“12 个 accumulator 是否太多”，而是：

```text
12 个 accumulator 是否能稳定常驻 YMM
编译器是否避免让 valpha 或展开临时变量挤占主循环寄存器
unroll 是否在提升调度窗口前先触发 spill
```

---

## 5. 必要知识：FMA throughput 与 latency

优化 micro-kernel 时需要区分 throughput 和 latency。

| 概念 | 含义 | 对 DGEMM 的影响 |
|------|------|-----------------|
| throughput | 每周期最多能发射多少条同类指令 | 决定理论峰值 |
| latency | 一条指令的结果多久后可用 | 决定需要多少独立 accumulator 来隐藏等待 |

对于 AVX2 FMA，CPU 可以高吞吐地执行 FMA，但单条 FMA 结果不是下一周期立刻可用。因此 kernel 不能反复更新同一个 accumulator：

```text
差：c0 = fma(a, b, c0) 后很快又依赖 c0
好：轮流更新 c00, c01, c10, c11, ... 多个 independent accumulators
```

当前 `8x6` 有 12 个 accumulator，理论上有助于隐藏 FMA latency。但是否能充分利用，还取决于 load/broadcast/FMA 的排列顺序。

### 5.1 Raptor Lake P-core / Haswell 类 AVX2 的关键执行资源

本项目机器是 i9-13900K。对当前 AVX2 DGEMM kernel，最关键的不是全部端口，而是 load、FMA、store/address 这些资源。可以用如下简化模型理解：

| 资源 | 典型数量 | 对应操作 | 对 kernel 的意义 |
|------|----------|----------|------------------|
| 256-bit FMA pipeline | 2 条 | `vfmadd*pd` | 理想每周期 2 条 256-bit FMA |
| load pipeline / load port | 2 条 | `vmovupd` load、内存源 `vbroadcastsd` | 理想每周期最多 2 个 load 类 uop |
| store data pipeline | 1 条 | `vmovupd` store | 主要影响 C 写回阶段 |
| store address / AGU | 1 条以上 | store 地址生成 | 写回阶段和复杂地址会受影响 |
| load AGU | 2 条左右 | load 地址生成 | A load 与 B broadcast 都需要地址生成 |

这里说的“2 个 loader”更准确地说是：核心通常能在一个周期内处理两个 load 类内存操作，且 L1D 到寄存器的带宽可支持两个 256-bit load 的量级。不同资料会按“load port”“load AGU”“L1D bandwidth”分别描述，但对当前 kernel 的实践含义一致：A load 和 B broadcast 会争用 load 侧资源。

### 5.2 关键指令的吞吐和 latency

不同 Intel 微架构的精确数字会有差异，应以 Intel Intrinsics Guide、Agner Fog、uops.info 和实测为准。用于当前分析时，可以采用以下近似：

| 指令/操作 | 吞吐上限 | 典型 latency | 说明 |
|-----------|----------|--------------|------|
| `vfmadd*pd ymm, ymm, ymm` | 2 条/cycle | 约 4 cycles | 依赖链上的 accumulator 约 4 cycle 后可再次使用 |
| `vmulpd ymm` / `vaddpd ymm` | 通常 1-2 条/cycle | 约 3-4 cycles | loop 后 alpha/C 写回使用，不是主循环主体 |
| `vmovupd ymm, [mem]` 256-bit load | 最多 2 条/cycle | L1 命中约 4-5 cycles | `a0/a1` load 使用 |
| `vbroadcastsd ymm, [mem]` | load 侧通常最多约 1-2 条/cycle | L1 命中约 5 cycles 量级 | B 标量 load + broadcast，既占 load 资源也占 shuffle/broadcast 资源 |
| `vmovupd [mem], ymm` 256-bit store | 通常 1 条/cycle | store buffer 异步吸收 | C 写回阶段使用 |

对主循环最重要的是 `vfmadd*pd` 的 latency。若 FMA latency 约为 4 cycles，而核心最多每周期发射 2 条 FMA，那么为了完全隐藏 latency，粗略需要：

```text
2 FMA/cycle * 4 cycles = 8 条 independent accumulator 链
```

当前 `8x6` 有 12 条 independent accumulator 链，因此从 FMA latency 隐藏角度看是足够的。

但这不等于一定能跑满 FMA pipeline，因为每个 `p` 迭代还必须提供 A/B 数据，并完成地址更新和循环控制。

### 5.3 当前 8x6 每个 p 迭代的理论周期下界

当前每个 `p` 迭代执行：

```text
2 条 256-bit A load
6 条 B scalar broadcast
12 条 256-bit FMA
```

只看 FMA pipeline：

```text
12 FMA / 2 FMA per cycle = 6 cycles
```

只看 load 类操作，粗略有：

```text
2 A load + 6 B broadcast = 8 load-like operations
8 / 2 load ports = 4 cycles
```

所以在理想情况下，主循环更可能先受 FMA throughput 限制，理论下界约为 6 cycles / p。每个 `p` 迭代完成的浮点工作量是：

```text
8 * 6 multiply-add = 48 FMA scalar operations
48 * 2 FLOPs = 96 FLOPs
```

如果 6 cycles 完成，则刚好对应：

```text
96 FLOPs / 6 cycles = 16 FLOPs/cycle
```

这就是 2 条 256-bit FMA pipeline 的 AVX2 double 理论峰值。

### 5.4 为什么 latency 仍然会影响实际表现

虽然 12 个 accumulator 多于隐藏 FMA latency 所需的约 8 条链，但实际还有几个问题：

| 问题 | 影响 |
|------|------|
| B broadcast latency | `bj` 生成后才能执行对应的两条 FMA |
| A load latency | `a0/a1` load 太晚会让第一组 FMA 等待 |
| FMA 排列 | 如果过早回到同一个 accumulator，会暴露 FMA latency |
| 地址计算 | `B[p + j * k]` 的 6 路地址可能增加整数/AGU 压力 |
| 寄存器压力 | 为提前 load/broadcast 增加临时寄存器，可能诱发 spill |

因此较好的指令调度目标是：

```text
尽早 load a0/a1
每次 broadcast 一个 bj 后，连续消费到 c?0/c?1 两个 accumulator
在 FMA 序列中穿插后续 broadcast 或下一 p 的 A load
避免为了提前太多 load/broadcast 而增加 live YMM 数量
```

这也是当前阶段建议先检查编译器生成汇编的原因：如果编译器已经把 load/broadcast/FMA 穿插得足够好，手写汇编收益会变小；如果它把 broadcast 集中在一起、FMA 排列差、或者出现 spill，手写汇编或更精细的 intrinsics 改写才有明确依据。

---

## 6. 必要知识：load/broadcast 压力

当前每个 `p` 迭代：

```text
A load: 2 * 32B = 64B
B broadcast: 6 * 8B = 48B 标量读取，同时生成 vector broadcast
FMA: 12 条
```

如果只看 FMA，12 条 FMA 在理想情况下约 6 cycles 可以发射完，因为每周期最多 2 条 FMA。

如果只看 load/broadcast，2 条 A load 加 6 条 B broadcast 是 8 个 load-like 操作。在 L1 命中且地址生成不拖后腿时，2 个 load port 给出的粗略下界约为 4 cycles。因此当前 `8x6` 的理想主循环下界主要由 FMA 侧的 6 cycles 决定，而不是 load 侧的 4 cycles。

但实际循环还要承受：

```text
A/B load 端口压力
地址生成 AGU 压力
loop branch 开销
指令解码/发射压力
```

因此 kernel 是否接近峰值，不只取决于 FMA 数量，也取决于能否把 load、broadcast 和 FMA 合理穿插。

更具体地说，当前 `8x6` 的 load/broadcast 压力有三个特点：

| 特点 | 说明 | 优化含义 |
|------|------|----------|
| A 是连续 packed load | `A[p * 8]` 和 `A[p * 8 + 4]` 是连续 64B | 对 L1 友好，适合提前 load |
| B 是 6 条 stride-k 标量 load | `B[p + j * k]`，j=0..5 | 依赖 B packing 后的布局和 cache 命中 |
| 每个 B broadcast 被两条 FMA 消费 | 一个 `bj` 同时用于 `a0` 和 `a1` | broadcast 生命周期可很短，不必保留多个 `bj` |

理想调度不应该一次性保留 `b0..b5` 六个 broadcast。那样会把寄存器需求从 15 个 YMM 推高到：

```text
12 accumulators + 2 A vectors + 6 B broadcasts = 20 个 YMM
```

这在 AVX2 下必然无法容纳。当前 C intrinsics 写法逐个声明并消费 `b0..b5`，从源码意图上是希望编译器复用同一个 broadcast 寄存器。但最终是否真的复用，仍需要看生成汇编确认。

### 6.1 对 2x unroll 的直接影响

2x unroll 的收益来自：

```text
减少 loop branch 和地址更新占比
扩大编译器调度窗口
让下一 p 的 A load 或 B broadcast 更早发出
```

但它也会增加 load/broadcast 的瞬时压力。2 个 `p` 迭代合计有：

```text
4 条 A load
12 条 B broadcast
24 条 FMA
```

吞吐下界仍然近似是：

```text
24 FMA / 2 = 12 cycles
16 load-like ops / 2 = 8 cycles
```

从纯吞吐看，2x unroll 仍是 FMA-bound。但如果 unroll 导致多个 A/B 临时值同时 live，寄存器压力会先变成问题。因此 2x unroll 的正确验证顺序是：

```text
先看汇编是否 spill
再看 FMA 是否更连续
最后看 benchmark 是否提升
```

---

## 7. 知识来源与验证路径

这些知识不是靠猜，也不是只能靠经验记忆，而是来自一套可以反复验证的来源。

### 7.1 怎么知道有几个 YMM 寄存器

YMM 寄存器数量来自 x86-64 + AVX/AVX2 架构定义。

在 x86-64 模式下：

```text
XMM 寄存器：xmm0 - xmm15，共 16 个
YMM 寄存器：ymm0 - ymm15，共 16 个
```

YMM 是 XMM 的 256-bit 扩展：

```text
xmm0 = ymm0 的低 128-bit
ymm0 = 256-bit
```

所以 AVX2 double kernel 里可直接使用的 256-bit vector register 是：

```text
ymm0, ymm1, ..., ymm15
共 16 个
```

这不是本项目的特殊结论，而是 Intel/AMD x86-64 架构规则。

补充说明：

| 环境 | 向量寄存器数量 |
|------|----------------|
| 32-bit x86 + AVX | 通常只有 `ymm0-ymm7`，共 8 个 |
| x86-64 + AVX/AVX2 | `ymm0-ymm15`，共 16 个 |
| x86-64 + AVX-512 | 通常有 `zmm0-zmm31`，共 32 个 |

当前 i9-13900K 不支持 AVX-512，因此本项目按 AVX2 的 16 个 YMM 寄存器设计 kernel。

### 7.2 怎么知道该用哪些指令

DGEMM micro-kernel 的指令选择可以从算法需求推导。

DGEMM 计算的是：

```text
C += A * B
```

当前 `8x6` kernel 每个 `p` 迭代处理：

```text
A: 8 个 double
B: 6 个 double
C: 8x6 个 double accumulator
```

AVX2 一个 YMM 是 256-bit，一个 double 是 64-bit，因此：

```text
256 / 64 = 4 个 double / YMM
```

所以 8 行 A 需要两个 vector：

```text
a0 = A[0:3]
a1 = A[4:7]
```

对应常见指令：

```text
vmovupd / vmovapd
```

B 的每个元素需要乘到 A 的 8 行上，所以 B 是 scalar，但需要广播到 4 个 lane：

```text
bj = broadcast(B[p, j])
```

对应常见指令：

```text
vbroadcastsd
```

核心乘加是：

```text
c += a * b
```

对应 FMA 指令：

```text
vfmadd132pd
vfmadd213pd
vfmadd231pd
```

因此 DGEMM AVX2 kernel 的核心指令自然是：

| 操作 | 常见指令 |
|------|----------|
| load A | `vmovupd` / `vmovapd` |
| broadcast B | `vbroadcastsd` |
| FMA | `vfmadd*pd` |
| load C | `vmovupd` / `vmovapd` |
| store C | `vmovupd` / `vmovapd` |
| zero accumulator | `vxorpd` / `vpxor` |
| prefetch | `prefetcht0` / `prefetcht1` / `prefetchnta` |
| non-temporal store | `vmovntpd` |

### 7.3 Intrinsics 到指令的映射怎么查

最直接的来源是 Intel Intrinsics Guide。

常用映射如下：

| Intrinsic | 常见汇编指令 |
|-----------|--------------|
| `_mm256_loadu_pd` | `vmovupd` |
| `_mm256_storeu_pd` | `vmovupd` |
| `_mm256_load_pd` | `vmovapd` |
| `_mm256_store_pd` | `vmovapd` |
| `_mm256_set1_pd` | `vbroadcastsd` 或 load + shuffle/broadcast |
| `_mm256_broadcast_sd` | `vbroadcastsd` |
| `_mm256_fmadd_pd` | `vfmadd132pd` / `vfmadd213pd` / `vfmadd231pd` |
| `_mm256_mul_pd` | `vmulpd` |
| `_mm256_add_pd` | `vaddpd` |
| `_mm256_setzero_pd` | 常见为 `vxorpd` |
| `_mm_prefetch` | `prefetcht0` / `prefetcht1` / `prefetchnta` |
| `_mm256_stream_pd` | `vmovntpd` |

注意：intrinsics 不保证一一固定映射到某一条编码。例如 `_mm256_fmadd_pd` 可能生成：

```text
vfmadd132pd
vfmadd213pd
vfmadd231pd
```

这三类 FMA 数学含义相近，但 operand 顺序不同。编译器会根据寄存器分配和破坏哪个 operand 更方便来选择。

### 7.4 怎么从 MR/NR 推导 accumulator 数量

当前 kernel：

```text
MR = 8
NR = 6
```

C tile 是：

```text
8 行 x 6 列
```

每个 YMM 保存 4 个 double，因此 8 行需要 2 个 YMM：

```text
rows 0-3 -> 1 个 accumulator
rows 4-7 -> 1 个 accumulator
```

每一列都需要 2 个 accumulator。6 列总共需要：

```text
6 * 2 = 12 个 accumulator
```

这正好对应当前代码中的：

```text
c00 c01
c10 c11
c20 c21
c30 c31
c40 c41
c50 c51
```

一般公式是：

```text
accumulator_count = NR * ceil(MR / vector_lanes)
double + AVX2: vector_lanes = 4
```

因此：

```text
8x6  -> 6 * ceil(8 / 4) = 12 accumulators
4x12 -> 12 * ceil(4 / 4) = 12 accumulators
8x8  -> 8 * ceil(8 / 4) = 16 accumulators
```

`8x8` 风险高的原因也由此可见：16 个 accumulator 已经占满全部 16 个 YMM，没地方放 A、B、alpha 和 scratch。

### 7.5 怎么知道寄存器是否够用

可以先做静态估算：

```text
寄存器需求 = accumulator + A vectors + B broadcast/scratch + alpha/C old scratch
```

当前 `8x6`：

```text
12 accumulators
2 A vectors
1 B broadcast/scratch
1 alpha/C old scratch
= 16 YMM 左右
```

由于 AVX2 x86-64 只有 16 个 YMM，这已经非常紧张。

但最终不能只看估算，要看编译器真实生成的汇编。如果出现栈上的 YMM 保存和恢复，说明寄存器分配已经不够理想。

### 7.6 如何验证编译器实际用了什么指令

生成汇编：

```bash
gcc -O2 -Wall -Isrc -D__AVX2__ -mavx2 -mfma -S src/kernel/avx2/dgemm_kernel.c
```

或者反汇编目标文件：

```bash
objdump -d src/kernel/avx2/dgemm_kernel.o
```

重点看是否出现目标指令：

```text
vbroadcastsd
vfmadd132pd / vfmadd213pd / vfmadd231pd
vmovupd / vmovapd
vaddpd
vmulpd
vxorpd
```

也要看是否出现不希望看到的 YMM spill：

```text
vmovupd %ymm?, -0x??(%rsp)
vmovupd -0x??(%rsp), %ymm?
vmovapd %ymm?, -0x??(%rsp)
vmovapd -0x??(%rsp), %ymm?
```

这类访问通常说明编译器把 YMM 寄存器内容临时放到了栈上。

### 7.7 推荐资料

| 资料 | 用途 |
|------|------|
| Intel Intrinsics Guide | 查 intrinsic 对应指令、ISA、latency/throughput 参考 |
| Intel 64 and IA-32 Architectures Software Developer's Manual | 查架构规则、寄存器、指令语义 |
| Intel Optimization Reference Manual | 查优化建议、cache、prefetch、SIMD 性能建议 |
| Agner Fog instruction tables | 查不同微架构上的 latency、throughput、端口信息 |
| uops.info | 查具体 CPU 上的 uop 数量、端口、latency、throughput |
| Compiler Explorer 或 `gcc -S` | 看 C/intrinsics 实际生成的汇编 |
| `objdump -d` | 看项目真实目标文件中的机器码/汇编 |
| `perf stat` / `perf record` | 看实际运行时瓶颈 |

对当前项目，最实用的闭环是：

```text
1. 用 Intel Intrinsics Guide 确认 intrinsic 语义
2. 用 MR/NR 和 vector_lanes 推导 accumulator 数量
3. 用 16 个 YMM 的限制估算寄存器压力
4. 用 gcc -S / objdump 看真实汇编
5. 用 benchmark / perf 验证性能变化
```

---

## 8. 如何判断是否真的需要汇编

在决定写汇编前，应先检查编译器生成的汇编质量。

可以生成汇编：

```bash
gcc -O2 -Wall -Isrc -D__AVX2__ -mavx2 -mfma -S src/kernel/avx2/dgemm_kernel.c
```

也可以反汇编目标文件：

```bash
objdump -d src/kernel/avx2/dgemm_kernel.o
```

重点检查：

| 检查项 | 说明 |
|--------|------|
| 是否有 YMM spill 到栈 | 查找访问 `rsp`/`rbp` 附近的 `vmovupd`/`vmovapd` |
| FMA 是否密集 | `vfmadd*pd` 是否能连续稳定出现 |
| broadcast 是否合理穿插 | `vbroadcastsd` 是否集中成一团导致等待 |
| load 是否提前 | A load 是否能提前于使用点 |
| loop 是否过短 | 未展开时分支和地址计算占比可能偏高 |
| unroll 后是否寄存器溢出 | 展开可能提升调度，也可能造成 spill |

如果编译器生成代码已经没有 spill，FMA 排列也较好，那么手写汇编收益会下降。

如果发现明显 spill、load/FMA 调度差、循环地址计算冗余，手写汇编收益就比较明确。

---

## 9. 实践建议

当前不应直接跳到手写汇编，而是按以下顺序推进：

```text
1. 保持 C intrinsics，实现 8x6 p-loop 2x unroll
2. 检查生成汇编，确认是否 spill
3. benchmark 1T/4T 的 n=1024 和 n=2048
4. 若 2x 有收益，再尝试 4x unroll
5. 同步做 Q/R blocking sweep
6. 若 intrinsics 优化后仍明显低于 OpenBLAS，再考虑手写 assembly kernel
```

最终判断标准不是“理论上汇编更快”，而是：

```text
intrinsics 生成代码是否已经成为实测瓶颈
手写汇编预期收益是否足以覆盖维护成本
```
