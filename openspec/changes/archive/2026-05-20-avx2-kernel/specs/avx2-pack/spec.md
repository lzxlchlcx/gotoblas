## ADDED Requirements

### Requirement: AVX2 打包函数

系统 SHALL 提供使用 AVX2 intrinsics 加速的打包函数，功能与通用版完全一致，但使用 `_mm256_loadu_pd` / `_mm256_storeu_pd`（double）或 `_mm256_loadu_ps` / `_mm256_storeu_ps`（float）批量加载/存储。

#### Scenario: dgemm pack_a_nn 加速

- **WHEN** 打包 A（m=4, k=256），NoTrans 方式
- **THEN** 使用 AVX2 intrinsics 批量加载/存储，结果与通用版一致

#### Scenario: 尺寸不是 SIMD 宽度倍数

- **WHEN** m=3（不是 4 的倍数），打包 A
- **THEN** 前 m-m%4 个元素用 SIMD，剩余 m%4 个用标量
