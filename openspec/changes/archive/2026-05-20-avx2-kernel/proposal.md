## Why

当前 MyBLAS 的通用 C 微内核性能约为 6 GFLOPS，而 OpenBLAS 的 AVX2 内核可达 50-100 GFLOPS。性能差距主要来自：通用内核未利用 SIMD 指令、未做循环展开、未使用 FMA 指令。实现 AVX2 内核可将性能提升 5-10 倍。

## What Changes

- 新增 AVX2 微内核：dgemm（4×4，使用 `__m256d` + `vfmadd231pd`）和 sgemm（8×4，使用 `__m256` + `vfmadd231ps`）
- 新增 AVX2 打包函数：利用 `_mm256_loadu_pd` / `_mm256_storeu_pd` 加速 packing
- 新增 AVX2 beta 缩放：利用 SIMD 并行缩放 C 的多列
- 新增 `src/config/haswell.h`：针对 AVX2 CPU 的阻塞参数（P=256, Q=256, R=4096, MR=4, NR=4 for double）
- 修改 `src/api/dgemm.c` 和 `sgemm.c`：运行时 CPU 检测，选择 generic 或 AVX2 内核表
- 修改 `src/driver/gemm_internal.h`：声明 AVX2 内核表

## Capabilities

### New Capabilities

- `avx2-microkernel`: AVX2 SIMD 微内核实现，包括 dgemm（4×4, `__m256d` + FMA）和 sgemm（8×4, `__m256` + FMA），以及边界处理（m<MR 或 n<NR 时回退标量）
- `avx2-pack`: AVX2 加速的打包函数，利用 SIMD load/store 指令加速矩阵元素的搬运和转置
- `avx2-beta`: AVX2 加速的 beta 缩放函数，利用 SIMD 并行处理 C 的多列
- `avx2-config`: AVX2 CPU 的阻塞参数配置（haswell.h），以及运行时 CPU 检测机制（cpuid 检测 AVX2+FMA 支持）

### Modified Capabilities

- `gemm-kernel`: 新增 AVX2 内核表实例（`gemm_kernel_avx2_double` / `gemm_kernel_avx2_float`），kernel_init.c 中初始化
- `gemm-api`: API 层新增运行时内核选择逻辑，检测 CPU 特性后选择对应的 kernel 表和 config

## Impact

- 新增文件：`src/kernel/avx2/` 目录下 8 个文件（dgemm_kernel.c, sgemm_kernel.c, dgemm_pack.c, sgemm_pack.c, dgemm_beta.c, sgemm_beta.c, kernel_init.c, cpuid.c）
- 新增文件：`src/config/haswell.h`
- 修改文件：`src/driver/gemm_internal.h`（新增 extern 声明）
- 修改文件：`src/api/dgemm.c` / `sgemm.c`（新增内核选择逻辑）
- 修改文件：`Makefile`（新增 avx2 编译目标，条件编译 `-mavx2 -mfma`）
- 编译依赖：需要支持 AVX2+FMA 的编译器（GCC ≥ 4.9 或 Clang ≥ 3.8）
