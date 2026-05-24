## Context

当前 DGEMM 的 driver 采用三层阻塞循环，但循环层级中 `col1` 位于 `row1` 外层，导致同一个 `k0` 下的 A 微块在处理多个 B 列块时被重复打包。这个问题不会改变计算结果，但会显著放大 packing 开销，并在较大矩阵下压低整体 GFLOPS。

本次修改只涉及 `src/driver/gemm_driver.c` 的调度顺序，不引入新依赖，不修改 kernel 或 pack 函数接口，也不改变公开 API。

## Goals / Non-Goals

**Goals:**
- 让同一 `(row1, k0)` 的 A 子块只打包一次
- 在同一 `k0` 下复用 packed A 给多个 `col1`
- 保持现有 kernel、packing 接口和矩阵计算结果不变
- 将优化范围限制在 driver 层，降低实现风险

**Non-Goals:**
- 不重写 AVX2 kernel
- 不修改 pack 函数实现细节
- 不改变 `gemm_config_t`、`gemm_arg_t` 或 public API
- 不处理现有的 `pack_b_tn` 正确性问题

## Decisions

### D1: 采用“先 B 后 A，再在内层复用 A”的循环顺序

**Decision**: 对每个 `k0`，先遍历并打包所有 `col1` 对应的 B 块；随后遍历 `row0/row1`，对每个 A 微块只执行一次 `pack_A`，并在内层遍历所有 `col1` 调用 kernel。

**Rationale**: 这样可以把 A 的打包结果从“每个 B 列块都重复计算一次”变成“每个 `(row1, k0)` 只计算一次”。对当前工作负载而言，A 的重复打包是主要瓶颈，优先消除它能获得最大的收益。

**Alternative considered**: 保持当前循环顺序，只对 `pack_A` 做缓存或 memoization。拒绝原因是它会引入额外状态管理，且无法从根源上消除多次进入 `pack_A` 的控制流开销。

### D2: 继续使用现有 `sa` / `sb` 缓冲区布局

**Decision**: 不调整 packing buffer 的内存布局和偏移策略，仅改变它们的使用顺序。

**Rationale**: 现有缓冲区布局已经和 kernel 接口匹配，改动最小且可以避免引入新的对齐和索引错误。循环重排本身就能带来主要收益。

**Alternative considered**: 为 A/B 分别引入新的临时缓冲区或双缓冲机制。拒绝原因是复杂度更高，且当前问题的核心并不是缓冲区容量不足，而是重复打包。

### D3: 保持 kernel 调用参数语义不变

**Decision**: 继续向 kernel 传入相同的 `row1_rem`、`col1_rem`、`k_rem`、`sa`、`sb` 和 `C` 偏移指针。

**Rationale**: 这样可以确保计算结果、边界处理和现有 AVX2 路径行为完全一致，避免因为调度重排引入数值差异。

### D4: 不在本次变更中引入新的性能分支

**Decision**: 仅做循环顺序调整，不额外添加新的 fast path、预取、stream store 或 kernel 特化。

**Rationale**: 本次变更的目标是验证调度顺序对性能的影响。保持修改单一，可以更清晰地归因效果，也便于后续单独评估 pack_a cache 优化。

## Risks / Trade-offs

- [Risk] B 的打包会被提前完成，可能增加 `sb` 的驻留时间 → 通过保持 `k0` 内局部性和不改变缓冲区大小来缓解
- [Risk] 循环重排后，调试时更难直接对应原始控制流 → 通过在代码中保留清晰注释说明“先包 B，再复用 A”的意图来缓解
- [Risk] 某些极小矩阵上性能收益不明显，甚至可能持平 → 这是可接受的，因为主要目标是中大矩阵吞吐提升

## Migration Plan

1. 仅修改 `src/driver/gemm_driver.c` 的双精度和单精度 driver 循环顺序
2. 保持 packing 和 kernel 调用接口不变，先完成编译与回归测试
3. 对比修改前后的 benchmark，验证 `pack_A` 占比下降和整体 GFLOPS 提升
4. 如有回归，回退该文件的循环顺序修改即可，回滚路径清晰

## Open Questions

- 是否需要在后续变更中进一步优化 `pack_A` 的读写模式，以叠加本次循环重排的收益？
- 是否需要为 `gemm_driver_float` 与 `gemm_driver_double` 保持完全相同的注释和结构，以便后续维护？
