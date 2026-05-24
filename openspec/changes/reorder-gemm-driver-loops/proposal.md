## Why

当前 DGEMM driver 在处理不同的 `col1` 列块时，会重复对同一 `(row1, k0)` 对应的 A 子块做 `pack_A`。这会把 packing 开销放大很多倍，并且让性能随着矩阵规模增长时停滞不前，因此需要重排循环顺序，让 packed A 在多个 `col1` 上复用。

## What Changes

- 调整 `src/driver/gemm_driver.c` 中 `col1` 与 `row1` 的循环层级
- 先完成同一 `k0` 下所有 B 列块的打包，再按 `row1` 打包一次 A，并在内层复用该 A 给多个 `col1`
- 保持现有 kernel、pack 函数和公开 API 不变
- 减少 `pack_A` 的重复调用次数，降低 packing 在总耗时中的占比

## Capabilities

### Modified Capabilities

- `gemm-driver`: 修改三层阻塞驱动的内部循环顺序，使 A 打包结果在同一 `k0` 下可跨多个 `col1` 复用，减少重复 packing 开销

## Impact

- 影响文件：`src/driver/gemm_driver.c`
- 影响范围：DGEMM 阻塞循环、packing 调度顺序、性能特征
- 不影响：`gemm_kernel` 接口、pack 函数签名、公共 API 行为
