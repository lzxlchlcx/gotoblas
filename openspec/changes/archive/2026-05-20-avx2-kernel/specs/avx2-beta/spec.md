## ADDED Requirements

### Requirement: AVX2 beta 缩放函数

系统 SHALL 提供使用 AVX2 intrinsics 加速的 beta 缩放函数。当 beta=0 时使用 `_mm256_setzero_pd`/`_mm256_setzero_ps` 批量清零；当 beta!=0 且 beta!=1 时使用 `_mm256_mul_pd`/`_mm256_mul_ps` 批量缩放。

#### Scenario: Beta = 0 批量清零

- **WHEN** beta=0.0, m=16, n=4
- **THEN** 使用 SIMD 批量清零 C 的 16×4 元素

#### Scenario: Beta != 0 且 != 1 批量缩放

- **WHEN** beta=2.0, m=16, n=4
- **THEN** 使用 SIMD 批量缩放 C 的 16×4 元素
