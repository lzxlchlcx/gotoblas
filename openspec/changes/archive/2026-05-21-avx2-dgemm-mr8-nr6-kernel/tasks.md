## 1. 配置参数更新

- [x] 1.1 更新 `src/config/haswell.h`：将 `GEMM_HASWELL_D_MR` 从 4 改为 8，`GEMM_HASWELL_D_NR` 从 8 改为 6

## 2. 微内核框架搭建与实现

- [x] 2.1 在 `src/kernel/avx2/dgemm_kernel.c` 中引入 `#include "config/haswell.h"`，定义 `MR`/`NR` 宏
- [x] 2.2 保留 MR=4×NR=8 原有快速路径到 `#elif MR == 4 && NR == 8` 分支
- [x] 2.3 实现 MR=8×NR=6 优化快速路径：12 个累加器（6 列 × 2 行组）、2 个 A YMM 加载、6 个 B 广播、scalar fallback 不变
- [x] 2.4 实现通用 AVX2 fallback 路径：`#else` 分支，按 4×4 子块循环处理任意 MR/NR

## 3. Pack/Beta 兼容性验证

- [x] 3.1 确认 `dgemm_pack_a_nn_avx2` / `dgemm_pack_a_tn_avx2` 使用 m 参数驱动，不依赖硬编码 MR
- [x] 3.2 确认 `dgemm_pack_b_nn_avx2` / `dgemm_pack_b_tn_avx2` 使用 n 参数驱动，不依赖硬编码 NR
- [x] 3.3 确认 `dgemm_beta_avx2` 使用 m 参数驱动，不依赖硬编码 MR

## 4. 验证

- [x] 4.1 运行 `make clean && make lib` 确保编译通过（MR=8, NR=6）
- [x] 4.2 运行 `test/test_gemm` 验证数值正确性（注：NT/TT 失败为已有 pack_b_tn bug，MR=4×NR=8 同样失败）
- [x] 4.3 运行 `test/bench_compare` 对比性能，确认 n=1024 1 线程 GFLOPS 高于基线 25.5
- [x] 4.4 临时切换 haswell.h 为 MR=4, NR=8，重新编译验证备选路径仍可工作
