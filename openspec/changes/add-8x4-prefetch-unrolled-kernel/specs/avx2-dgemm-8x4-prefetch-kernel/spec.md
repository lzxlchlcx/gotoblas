## ADDED Requirements

### Requirement: 8x4 AVX2 DGEMM fast path
系统 SHALL 提供一个 `MR=8, NR=4` 的 AVX2 DGEMM fast path，用于实验寄存器更宽松的 micro-kernel。该 fast path MUST 保持现有 kernel 接口和 `C += alpha * A * B` 语义不变。

#### Scenario: 8x4 配置启用 fast path
- **WHEN** `GEMM_HASWELL_D_MR` 配置为 `8` 且 `GEMM_HASWELL_D_NR` 配置为 `4`
- **THEN** AVX2 DGEMM kernel MUST 编译并使用 `8x4` fast path 处理完整 tile

#### Scenario: 非完整 tile 仍然正确处理
- **WHEN** 输入矩阵维度不能完整覆盖 `8x4` tile
- **THEN** 系统 MUST 通过现有 fallback 路径保持结果正确

### Requirement: 主循环寄存器预算
`8x4` kernel 主循环 SHALL 将 C accumulator 数量控制为 8 个 YMM，并为 current A、next A、current/next B broadcast 和 scratch 保留寄存器余量。`alpha` broadcast MUST 位于 `p` loop 之后，不能作为主循环 live value。

#### Scenario: alpha 不占用主循环 YMM
- **WHEN** 生成 `8x4` kernel 的 x86_64 AVX2 汇编
- **THEN** `alpha` 的 `vbroadcastsd` MUST 出现在 `p` loop 之后

#### Scenario: 主循环无 accumulator spill
- **WHEN** 检查 `8x4` kernel 的 `p` loop 汇编
- **THEN** 主循环内 MUST NOT 出现针对 `%rsp` 或 `%rbp` 栈地址的 YMM `vmovapd`/`vmovupd` spill/reload

### Requirement: 2x p-loop unroll
`8x4` kernel SHALL 实现或实验 2x `p` loop unroll，以降低 loop branch 和地址更新开销，并为 next-B broadcast 与 next-A preload 提供调度窗口。

#### Scenario: k 为偶数时使用展开主体
- **WHEN** `k` 至少为 2 且存在完整的两个 `p` 迭代
- **THEN** kernel MUST 使用 2x unrolled 主体处理成对的 `p` 迭代

#### Scenario: k 为奇数时处理尾部
- **WHEN** `k` 不能被 2 整除
- **THEN** kernel MUST 使用尾部路径处理最后一个 `p` 迭代并保持数值正确

### Requirement: 预取实验可验证
`8x4` kernel SHALL 以可验证方式实验 next-B broadcast 和 next-A preload。实现 MUST 能通过汇编检查确认预取是否实际提前，并通过 benchmark 判断收益。

#### Scenario: next-B broadcast 预取检查
- **WHEN** 实现 next-B broadcast 实验版本并生成汇编
- **THEN** 汇编检查 MUST 记录 next-B broadcast 是否在对应 FMA 使用前提前出现

#### Scenario: next-A preload 检查
- **WHEN** 实现 next-A preload 实验版本并生成汇编
- **THEN** 汇编检查 MUST 记录下一组 A load 是否在下一轮 FMA 使用前提前出现

### Requirement: 性能对比与回退
`8x4` kernel MUST 与现有 `8x6` baseline 进行同机同参数对比。只有在正确性通过、主循环无 spill、且 benchmark 显示收益或明确实验价值时，才能考虑把 `8x4` 作为默认配置。

#### Scenario: 正确性验证
- **WHEN** 完成 `8x4` kernel 实现
- **THEN** `make test` MUST 通过，且在 x86_64 AVX2 环境中 MUST 验证 DGEMM 数值正确性

#### Scenario: benchmark 对比
- **WHEN** 在 i9-13900K 或等价 x86_64 AVX2 环境运行 benchmark
- **THEN** 结果 MUST 至少包含 `8x6` baseline 与 `8x4` 实验版本在 1T/4T、n=512/1024/2048 下的对比

#### Scenario: 性能不达标时回退
- **WHEN** `8x4` 实验版本出现主循环 spill 或 benchmark 明显低于 `8x6`
- **THEN** 默认配置 MUST 保持或回退到 `8x6`
