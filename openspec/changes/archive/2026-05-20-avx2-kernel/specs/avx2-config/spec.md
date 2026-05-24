## ADDED Requirements

### Requirement: AVX2 阻塞参数配置

系统 SHALL 提供 `src/config/haswell.h`，定义针对 AVX2 CPU 的阻塞参数：

- dgemm: P=256, Q=256, R=4096, MR=4, NR=4
- sgemm: P=256, Q=256, R=4096, MR=8, NR=4

参数 SHALL 满足 TLB 约束：P×Q×sizeof(element)/page_size ≤ L1 DTLB 条目数。

#### Scenario: 参数初始化

- **WHEN** 调用 `gemm_config_avx2_double()`
- **THEN** cfg 被填充为 AVX2 优化参数

### Requirement: 运行时 CPU 检测

系统 SHALL 在 API 层运行时检测 CPU 是否支持 AVX2+FMA。如果支持，选择 AVX2 内核表和配置；否则回退到通用内核。

检测 SHALL 使用 `__builtin_cpu_supports("avx2")` 和 `__builtin_cpu_supports("fma")`。

#### Scenario: CPU 支持 AVX2+FMA

- **WHEN** 运行在 Haswell 或更新的 CPU 上
- **THEN** 自动选择 `gemm_kernel_avx2_double` / `gemm_kernel_avx2_float`

#### Scenario: CPU 不支持 AVX2

- **WHEN** 运行在 Sandy Bridge 或更旧的 CPU 上
- **THEN** 使用 `gemm_kernel_generic_double` / `gemm_kernel_generic_float`

#### Scenario: 编译器不支持 AVX2

- **WHEN** 编译时未启用 `-mavx2 -mfma`（`__AVX2__` 未定义）
- **THEN** AVX2 内核不编译，运行时只能使用通用内核
