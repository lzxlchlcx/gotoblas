## ADDED Requirements

### Requirement: AVX2 dgemm 微内核

系统 SHALL 提供基于 AVX2 intrinsics 的 dgemm 微内核，MR=4, NR=4。使用 `__m256d` 寄存器和 `_mm256_fmadd_pd` 指令实现。

内核 SHALL 按以下方式计算 C[4×4] += alpha * A[4×k] * B[k×4]：
- 外层循环遍历 k 维度
- 每次迭代：加载 A 的 4 个元素到 `__m256d`（broadcast），加载 B 的 4×4 tile 到 4 个 `__m256d`，执行 FMA 累加到 C 的 4 个 `__m256d` 寄存器
- 最终将 C 寄存器写回内存

#### Scenario: 完整 4×4 tile

- **WHEN** m=4, n=4, k=8, alpha=1.0
- **THEN** 使用 AVX2 intrinsics 计算，结果与通用内核在 double 精度范围内一致

#### Scenario: 部分 tile（m < MR）

- **WHEN** m=2, n=4, k=8
- **THEN** 回退到标量循环计算（不使用 SIMD）

#### Scenario: 部分 tile（n < NR）

- **WHEN** m=4, n=2, k=8
- **THEN** 回退到标量循环计算

### Requirement: AVX2 sgemm 微内核

系统 SHALL 提供基于 AVX2 intrinsics 的 sgemm 微内核，MR=8, NR=4。使用 `__m256` 寄存器和 `_mm256_fmadd_ps` 指令。

#### Scenario: 完整 8×4 tile

- **WHEN** m=8, n=4, k=8, alpha=1.0f
- **THEN** 使用 AVX2 intrinsics 计算，结果与通用内核在 float 精度范围内一致

#### Scenario: 部分 tile

- **WHEN** m < 8 或 n < 4
- **THEN** 回退到标量循环计算
