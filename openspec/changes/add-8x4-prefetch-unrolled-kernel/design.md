## Context

当前 AVX2 DGEMM `8x6` micro-kernel 使用 12 个 YMM accumulator、2 个 A vector 和 1 个 B broadcast，主循环最低需要约 15 个 YMM。将 `alpha` broadcast 移到 `p` loop 之后可以避免 `valpha` 常驻主循环，但在 C intrinsics 中同时保留 next-B broadcast 已经观察到 YMM spill。

`8x4` kernel 将 accumulator 数量降到 8 个 YMM，可以释放 4 个 YMM 用于 current/next B broadcast、current/next A preload 和 scratch。它的代价是 `NR` 从 6 降到 4，kernel 调用次数增加约 50%，A 复用下降。因此设计必须同时引入 2x `p` loop unroll 和严格的汇编验证，判断寄存器余量带来的流水化收益是否足以抵消 tile 变小的代价。

## Goals / Non-Goals

**Goals:**

- 新增 `MR=8, NR=4` AVX2 DGEMM fast path，用于实验寄存器更宽松的 micro-kernel。
- 在 `8x4` kernel 中保持 `alpha` broadcast 在主循环之后，避免占用主循环 YMM。
- 通过 2x `p` loop unroll 降低 loop branch 和地址更新开销。
- 在寄存器预算允许时尝试 next-B broadcast 和 next-A preload，提升 FMA pipeline 连续性。
- 通过汇编检查确认主循环无 accumulator spill，并通过 benchmark 与现有 `8x6` 对比。

**Non-Goals:**

- 不在本变更中删除现有 `8x6` fast path。
- 不改变 `my_dgemm` API、矩阵布局语义、packing 函数接口或 driver/kernel 调用约定。
- 不引入手写汇编作为第一实现；本变更优先使用 C intrinsics。
- 不承诺 `8x4` 一定成为默认配置，是否启用由实测结果决定。

## Decisions

### Decision 1: 新增 `8x4` fast path 而不是直接替换 `8x6`

采用 `#if MR == 8 && NR == 4` 新分支实现 `8x4` kernel，保留现有 `8x6` 分支。这样可以通过修改 `src/config/haswell.h` 在 `8x6` 与 `8x4` 之间切换，不破坏当前已知较快的实现。

替代方案是直接把 `8x6` 改写为 `8x4`。该方案风险更高，因为 `8x4` 的 A 复用较低、kernel 调用次数更多，未经 benchmark 不能确认收益。

### Decision 2: `8x4` accumulator 布局保持 8 行、4 列

`8x4` 每列使用两个 YMM accumulator，对应 rows 0-3 和 rows 4-7，共 8 个 accumulator：

```text
c00/c01, c10/c11, c20/c21, c30/c31
```

寄存器预算目标为：

```text
8 accumulators + 2 current A + 2 next A + 1 current B + 1 next B + 1 scratch = 15 YMM
```

这比 `8x6` 同时保留 next-B/next-A 时的 18 YMM 更可行。

### Decision 3: `alpha` broadcast 必须放在 `p` loop 之后

`alpha` 只参与 loop 后的 accumulator 缩放，不应跨主循环占用 YMM。`8x4` 分支必须将：

```c
__m256d valpha = _mm256_set1_pd(alpha);
```

放在 `p` loop 之后，避免主循环寄存器预算被无关变量挤占。

### Decision 4: 先实现保守 2x unroll，再逐步加入 next-B/next-A 流水

2x unroll 的目标是减少分支和地址更新占比，并给编译器更大的调度窗口。实现时应先保证：

```text
无 YMM spill
结果正确
性能不显著退化
```

然后再逐步尝试 next-B broadcast 和 next-A preload。若一次性引入所有预取，难以定位性能变化来自 unroll、B broadcast 提前、A preload 还是 spill。

### Decision 5: 汇编检查是合入前置条件

每个实验版本都必须生成 x86_64 AVX2 汇编，检查主循环：

```text
是否存在 vmovapd/vmovupd 对 rsp/rbp 的 YMM spill/reload
alpha broadcast 是否位于 p loop 后
FMA 是否保持密集且没有明显被多余内存访问打断
```

如果主循环出现 accumulator spill，即使 benchmark 某些小尺寸偶然提升，也不应作为最终方向。

## Risks / Trade-offs

- `8x4` 降低 A 复用 → 通过 benchmark 与 `8x6` 对比 n=512/1024/2048 的 1T/4T 数据，若整体退化则不采用默认配置。
- `NR=4` 使 kernel 调用次数增加约 50% → 使用 2x `p` loop unroll 抵消部分循环开销，同时检查 driver/kernel 占比变化。
- `8x4` 只有 8 条 accumulator 链，FMA latency hiding 余量低于 `8x6` 的 12 条链 → 通过指令顺序和 unroll 保持 FMA 连续发射，必要时回退到 `8x6`。
- C intrinsics 不能完全控制寄存器分配 → 必须以生成汇编为准，若编译器仍引入 spill，则停止该实现方向或考虑手写汇编作为后续变更。
- macOS arm64 本机无法真实运行 AVX2 路径 → 正确性和性能最终需要在 x86_64/WSL/i9-13900K 环境验证。
