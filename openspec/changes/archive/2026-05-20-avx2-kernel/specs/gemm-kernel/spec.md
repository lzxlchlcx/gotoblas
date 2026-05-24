## MODIFIED Requirements

### Requirement: Kernel 表

系统 SHALL 在原有 `gemm_kernel_generic_double` / `gemm_kernel_generic_float` 基础上，新增 AVX2 内核表实例：

- `gemm_kernel_avx2_double`：使用 AVX2 dgemm 微内核（MR=4, NR=4）、AVX2 打包函数、AVX2 beta 缩放
- `gemm_kernel_avx2_float`：使用 AVX2 sgemm 微内核（MR=8, NR=4）、AVX2 打包函数、AVX2 beta 缩放

所有内核表 SHALL 保持相同的 `gemm_kernel_table_t` / `sgemm_kernel_table_t` 结构，可互换使用。

#### Scenario: 通用内核表（不变）

- **WHEN** CPU 不支持 AVX2
- **THEN** 使用 `gemm_kernel_generic_double`，行为与之前完全一致

#### Scenario: AVX2 内核表

- **WHEN** CPU 支持 AVX2+FMA
- **THEN** 使用 `gemm_kernel_avx2_double`，函数指针指向 AVX2 实现
