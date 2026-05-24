## Why

当前 AVX2 DGEMM 微内核使用 MR=4, NR=8 的配置，仅占用 12/16 个 YMM 寄存器，每 k 次迭代产出 32 FLOPs。16 个 YMM 寄存器中尚有 4 个未被利用，导致计算密度不足，与大矩阵场景下 OpenBLAS 存在 7-30x 性能差距。

更深层的问题：当前微内核将 MR/NR 硬编码在 intrinsics 代码中，切换配置时需要手动重写 kernel。需要建立一套**配置驱动的 kernel 框架**，让 `haswell.h` 中的 MR/NR 宏决定使用哪个 kernel 实现，方便后续实验不同配置。

本次以 MR=8, NR=6 为目标配置（48 FLOPs/k 迭代，+50%），同时建立普适的 `#if` 分发机制。

## What Changes

- 建立配置驱动的 kernel 框架：`dgemm_kernel.c` 包含 `config/haswell.h` 中的 MR/NR 宏，通过 `#if` 选择对应的 intrinsics 实现
- 新增 MR=8×NR=6 的 AVX2 快速路径实现（12 累加器 + 2 A 加载）
- 保留 MR=4×NR=8 的已有实现作为备选路径
- 新增通用 AVX2 fallback 路径（按 4×4 子块循环），支持任意 MR/NR 组合
- Packing / beta 函数已天然泛化（使用 m/n 参数），无需 `#if` 分发
- 更新 Haswell 配置参数为 MR=8, NR=6
- SGEMM 不变，仅影响 DGEMM double 精度路径

## Capabilities

### New Capabilities

- `avx2-dgemm-config-driven-kernel`: 配置驱动的 AVX2 DGEMM 微内核框架，包含 MR=8×NR=6 优化路径、MR=4×NR=8 保留路径、通用 AVX2 fallback 路径，以及适配的 pack/beta/config 模块

### Modified Capabilities

## Impact

- `src/kernel/avx2/dgemm_kernel.c` — 引入 config 宏，`#if` 分发 + MR=8×NR=6 实现
- `src/kernel/avx2/dgemm_pack.c` — packing 函数已泛化，保持不变
- `src/kernel/avx2/dgemm_beta.c` — beta 函数已泛化，保持不变
- `src/config/haswell.h` — DGEMM 配置 MR=8, NR=6
- `src/kernel/avx2/kernel_init.c` — 函数指针表不变（签名兼容）
- `src/driver/gemm_driver.c` — 不变（通过 cfg->MR/cfg->NR 泛化）
- API 层 `src/api/dgemm.c` — 不变
