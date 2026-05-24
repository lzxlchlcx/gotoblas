## MODIFIED Requirements

### Requirement: my_dgemm 公共 API

`my_dgemm` SHALL 在分配 packing 缓冲区前，先检测 CPU 特性选择内核表和配置。选择逻辑：

1. 如果编译时定义了 `__AVX2__` 且运行时 `__builtin_cpu_supports("avx2")` 返回真 → 使用 AVX2 内核
2. 否则 → 使用通用内核

选择的内核表和配置 SHALL 传递给 `gemm_driver_double` / `gemm_parallel_double`。

#### Scenario: AVX2 CPU 上自动选择

- **WHEN** 在 Haswell CPU 上调用 my_dgemm
- **THEN** 自动使用 AVX2 内核表，无需用户干预

#### Scenario: 非 AVX2 CPU 上回退

- **WHEN** 在 Sandy Bridge CPU 上调用 my_dgemm
- **THEN** 自动回退到通用内核表

### Requirement: my_sgemm 公共 API

`my_sgemm` SHALL 与 `my_dgemm` 相同的 CPU 检测逻辑，选择对应的 float 内核表。

#### Scenario: AVX2 CPU 上自动选择

- **WHEN** 在 Haswell CPU 上调用 my_sgemm
- **THEN** 自动使用 AVX2 float 内核表
