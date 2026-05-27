## Why

当前 `8x6` AVX2 DGEMM kernel 的 12 个 accumulator 已经把 YMM 寄存器压力推到极限，虽然将 `alpha` broadcast 延后可以释放主循环寄存器，但在 C intrinsics 中同时保留 next-B broadcast 仍会触发 spill。需要探索一个寄存器更宽松的 `8x4` kernel，用释放出的 YMM 支持 B 预取/下一轮 A 预取，并通过 2x `p` loop unroll 降低循环开销。

## What Changes

- 新增 `MR=8, NR=4` 的 AVX2 DGEMM fast path 实验实现。
- 将 `8x4` kernel 设计为 8 个 C accumulator，释放 YMM 给 current/next B broadcast 与 current/next A vector。
- 在 `8x4` kernel 中尝试 2x `p` loop unroll，摊薄 loop branch 和地址更新开销。
- 保持 `alpha` broadcast 在主循环之后，避免 `valpha` 跨 `p` loop 占用 YMM。
- 增加汇编检查与 benchmark 任务，验证无 accumulator spill、FMA 排列、正确性和性能收益。
- 不移除现有 `8x6` kernel；`8x4` 作为可切换实验路径，与 `8x6` 对比后再决定是否成为默认配置。

## Capabilities

### New Capabilities
- `avx2-dgemm-8x4-prefetch-kernel`: 描述 `8x4` AVX2 DGEMM micro-kernel 的寄存器预算、预取/展开策略、正确性与性能验证要求。

### Modified Capabilities

## Impact

- 影响 `src/config/haswell.h` 中双精度 `MR/NR` 配置实验。
- 影响 `src/kernel/avx2/dgemm_kernel.c` 中 AVX2 DGEMM fast path 分支。
- 可能影响 `src/driver/gemm_driver.c` 的 kernel 调用次数和 packed B panel 遍历行为，但不应改变公开 API。
- 需要更新 `Docs/实验日志.md` 和相关性能报告，记录汇编检查、benchmark 数据和是否采用 `8x4` 的结论。
- 不引入新外部依赖，不改变 `my_dgemm` API 或矩阵语义。
