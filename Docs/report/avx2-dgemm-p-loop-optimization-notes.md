# AVX2 DGEMM p-loop 优化简明笔记

## 0. 版本信息

| 项目 | 值 |
|------|-----|
| 记录时间 | 2026-05-25 23:52:38 +08:00 |
| 当前分支 | main |
| 当前版本 | 2f38072 |
| 关联复盘 | `Docs/report/report2.md` |
| 关联源码 | `src/kernel/avx2/dgemm_kernel.c` |

本文档简要说明 DGEMM micro-kernel 中 `p-loop` 优化通常包含哪些手段、简单案例、优化目标和可能出现的问题。

---

## 1. 什么是 p-loop

在 DGEMM 中，核心计算是：

```text
C[i, j] += A[i, p] * B[p, j]
```

这里的 `p` 对应 K 维。因此 micro-kernel 中遍历 K 维的循环通常称为 `p-loop`。

当前 `8x6` kernel 的结构类似：

```c
for (p = 0; p < k; p++) {
    load A[p]
    broadcast B[p, 0]
    FMA into C column 0
    broadcast B[p, 1]
    FMA into C column 1
    ...
    broadcast B[p, 5]
    FMA into C column 5
}
```

每个 `p` 迭代大致执行：

```text
2 个 A vector load
6 个 B scalar broadcast
12 个 FMA
```

---

## 2. 通常包含哪些优化手段

### 2.1 循环展开 unroll

把一次循环处理 1 个 `p`，改为一次循环处理多个 `p`。

例如 2x unroll：

```c
for (p = 0; p + 1 < k; p += 2) {
    compute p;
    compute p + 1;
}

for (; p < k; p++) {
    compute tail;
}
```

作用：

```text
减少 branch / compare / p++ 开销
扩大编译器和 CPU 可调度的指令窗口
为 load / broadcast / FMA 重排创造空间
```

### 2.2 提前 load

在处理当前 `p` 的 FMA 时，提前加载下一轮需要的 A 或 B。

概念示例：

```text
load A[p]
load A[p+1]          <- 提前准备下一轮
broadcast B[p,0]
FMA p, col0
broadcast B[p,1]
FMA p, col1
broadcast B[p+1,0]   <- 提前准备下一轮 B
FMA p+1, col0
```

作用：

```text
隐藏 load latency
减少 FMA 等待数据的概率
```

### 2.3 load / broadcast / FMA 重排

不要把所有 load、所有 broadcast、所有 FMA 分成大块堆在一起，而是交错排列。

较差的形态：

```text
load load load
broadcast broadcast broadcast
FMA FMA FMA
```

较好的形态：

```text
load
broadcast
FMA
broadcast
FMA
load next
broadcast
FMA
```

作用：

```text
平衡 load port、FMA port、地址生成单元压力
提高指令级并行 ILP
```

### 2.4 使用多个 independent accumulators

不要反复更新同一个 accumulator，而是轮流更新多个 accumulator。

较差：

```text
c0 += a*b
c0 += a*b
c0 += a*b
```

较好：

```text
c00 += a*b
c10 += a*b
c20 += a*b
c30 += a*b
```

作用：

```text
隐藏 FMA latency
让 CPU 更容易持续发射 FMA
```

当前 `8x6` kernel 有 12 个 accumulator，天然具备一定 latency hiding 能力。

### 2.5 减少循环内地址计算

尽量避免循环内重复复杂表达式，例如：

```c
B[p + j * k]
A[p * 8 + i]
```

可以通过指针递增或固定 offset 降低地址计算压力。

作用：

```text
减少整数指令和 AGU 压力
让更多执行资源用于 load 和 FMA
```

---

## 3. 简单案例：1x 到 2x unroll

原始形态：

```c
for (p = 0; p < k; p++) {
    a0 = load A[p];
    b0 = broadcast B[p];
    c0 = fma(a0, b0, c0);
}
```

2x unroll 后：

```c
for (p = 0; p + 1 < k; p += 2) {
    a0 = load A[p];
    a1 = load A[p + 1];

    b0 = broadcast B[p];
    c0 = fma(a0, b0, c0);

    b1 = broadcast B[p + 1];
    c0 = fma(a1, b1, c0);
}

for (; p < k; p++) {
    a0 = load A[p];
    b0 = broadcast B[p];
    c0 = fma(a0, b0, c0);
}
```

实际 DGEMM kernel 会有多个 accumulator，例如 `8x6` 有 12 个 accumulator，所以展开后的真实代码会更长，但思路相同：

```text
一次循环处理 p 和 p+1
同一组 C accumulators 被连续累加两轮 K 维贡献
数学结果不变，只改变指令组织方式
```

---

## 4. 优化目标

p-loop 优化的目标不是单纯减少循环次数，而是让 CPU 更接近理想执行状态：

```text
FMA pipeline 尽量不空
load / broadcast 提前准备好
多个 accumulator 轮流更新，隐藏 FMA latency
branch / compare / 地址计算开销降低
不发生 YMM spill
最终 GFLOPS 提升
```

对于当前 `8x6` kernel，理想方向是：

```text
每周期尽量接近 2 条 256-bit FMA
load / broadcast 不成为 FMA 的等待原因
12 个 accumulator 被均匀更新
```

---

## 5. 可能出现的问题

### 5.1 寄存器压力过大

当前 `8x6` kernel 已经很接近 AVX2 的 16 个 YMM 寄存器上限：

```text
12 个 accumulator
2 个 A vector
1 个 B broadcast / scratch
1 个 alpha / C old scratch
约 16 个 YMM
```

展开后如果同时保留太多临时变量，可能导致编译器 spill：

```asm
vmovupd %ymm?, -0x??(%rsp)
vmovupd -0x??(%rsp), %ymm?
```

一旦 accumulator spill 到栈，性能可能明显下降。

### 5.2 展开过度导致指令体积变大

2x unroll 可能有收益，但 4x 或 8x 不一定更好。

原因：

```text
代码体积增大
指令 cache 压力增加
寄存器压力增加
编译器调度更困难
```

### 5.3 编译器生成代码不符合预期

C intrinsics 不等于最终汇编完全可控。

需要检查：

```bash
gcc -O2 -Wall -Isrc -D__AVX2__ -mavx2 -mfma -S src/kernel/avx2/dgemm_kernel.c
objdump -d src/kernel/avx2/dgemm_kernel.o
```

重点看：

```text
是否出现 YMM spill
FMA 是否密集
broadcast 是否合理穿插
load 是否提前
循环内是否有过多地址计算
```

### 5.4 小 K 或边界场景收益有限

unroll 需要处理 tail：

```c
for (; p < k; p++) {
    compute tail;
}
```

如果 `k` 很小，展开带来的收益可能被额外分支和代码体积抵消。

### 5.5 数值结果应保持一致

p-loop unroll 只改变累加顺序，不改变数学公式。但浮点加法不满足严格结合律，因此极小数值差异是可能的。

验证时应使用合理误差阈值，而不是要求 bitwise identical。

---

## 6. 推荐实验顺序

当前项目建议按以下顺序实验：

```text
1. 实现 8x6 p-loop 2x unroll
2. 检查生成汇编是否出现 YMM spill
3. benchmark n=512 / 1024 / 2048，分别看 1T 和 4T
4. 如果 2x 有收益且无 spill，再尝试 4x unroll
5. 对比不同 blocking 参数，确认收益是否稳定
6. 若 intrinsics 版本受限明显，再考虑手写 assembly kernel
```

记录结果时建议同时保存：

```text
commit id
编译参数
benchmark 输出
pack_a / pack_b / kernel 占比
是否有 spill
```

---

## 7. 总结

`p-loop` 优化可以理解为：

```text
围绕 K 维循环做循环展开和指令调度，让 CPU 看到更多独立指令，从而隐藏延迟、提高 FMA 吞吐。
```

它通常包含：

```text
循环展开
提前 load
load / broadcast / FMA 重排
多个 accumulator 轮流更新
减少循环内地址计算
检查并避免寄存器 spill
```

最终判断标准不是代码看起来更复杂，而是：

```text
生成汇编更合理
没有明显 spill
benchmark GFLOPS 稳定提升
```
