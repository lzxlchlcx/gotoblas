## ADDED Requirements

### Requirement: 配置驱动的 AVX2 DGEMM 微内核框架

系统 SHALL 提供 `dgemm_kernel_avx2` 函数，其快速路径通过编译期 `#if` 预处理器宏（源自 `config/haswell.h` 中的 MR/NR 定义）选择对应实现。修改 `haswell.h` 中的 `GEMM_HASWELL_D_MR` / `GEMM_HASWELL_D_NR` 即可切换 kernel 实现。

#### Scenario: 切换配置后自动选择 kernel
- **WHEN** `haswell.h` 定义 MR=8, NR=6 并重新编译
- **THEN** `dgemm_kernel_avx2` 使用 MR=8×NR=6 优化路径（12 累加器 + 2 A 加载）

#### Scenario: 回退到备选路径
- **WHEN** `haswell.h` 定义 MR=4, NR=8 并重新编译
- **THEN** `dgemm_kernel_avx2` 使用 MR=4×NR=8 备选路径（8 累加器 + 1 A 加载）

#### Scenario: 未匹配任何优化路径时使用通用 fallback
- **WHEN** MR/NR 不匹配任何 `#if`/`#elif` 分支（如 MR=6, NR=8）
- **THEN** `dgemm_kernel_avx2` 使用通用 AVX2 路径（按 4×4 子块循环），结果正确

### Requirement: MR=8×NR=6 优化快速路径

当 MR=8, NR=6 时，kernel 的快速路径 SHALL 使用 2 个 YMM 加载 A（rows 0-3 和 4-7），6 个 B 标量广播，12 个 FMA 累加器，产出 48 FLOPs/k 迭代。

#### Scenario: 完整 8×6 微块计算
- **WHEN** m=8, n=6, k 为任意正整数
- **THEN** 使用 AVX2 FMA 指令计算 C[i + j*ldc] += alpha * sum(A[i + p*m] * B[p + j*k])，结果在 1e-10 内正确

#### Scenario: 边界微块 fallback
- **WHEN** m < MR 或 n < NR（如 m=5, n=3）
- **THEN** 使用标量三重循环正确计算，不越界访问

### Requirement: Pack/Beta 函数泛化兼容

`dgemm_pack_a_nn_avx2`、`dgemm_pack_a_tn_avx2`、`dgemm_pack_b_nn_avx2`、`dgemm_pack_b_tn_avx2`、`dgemm_beta_avx2` 函数 SHALL 通过 m/n 参数驱动，不依赖硬编码的 MR/NR 常量，MR/NR 变更时无需修改。

#### Scenario: MR 变更后 pack_a 正确
- **WHEN** config MR 从 4 改为 8，调用 `dgemm_pack_a_nn_avx2(m=8, k=256, A, lda, A_packed)`
- **THEN** 输出 packed A 满足 A_packed[i + p*8] = A[i + p*lda]，与 MR=4 时行为一致（仅 m 参数不同）

#### Scenario: NR 变更后 pack_b 正确
- **WHEN** config NR 从 8 改为 6，调用 `dgemm_pack_b_nn_avx2(k=256, n=6, B, ldb, B_packed)`
- **THEN** 输出 packed B 满足 B_packed[p + j*256] = B[p + j*ldb]

### Requirement: Haswell 配置参数更新

系统 SHALL 在 `gemm_config_avx2_double()` 中将 DGEMM 配置设为 MR=8, NR=6，其余参数（P=256, Q=256, R=4096）保持不变。

#### Scenario: 配置加载
- **WHEN** 调用 `gemm_config_avx2_double(&cfg)`
- **THEN** cfg.MR=8, cfg.NR=6, cfg.P=256, cfg.Q=256, cfg.R=4096

### Requirement: 数值正确性验证

更新后的微内核 SHALL 通过所有现有 DGEMM 正确性测试。

#### Scenario: 正确性测试通过
- **WHEN** 运行 `test/test_gemm` 测试
- **THEN** 所有测试用例通过，误差在 1e-10 内

#### Scenario: Benchmark 性能提升
- **WHEN** 运行 `test/bench_compare`，n=1024，1 线程
- **THEN** MyBLAS DGEMM GFLOPS 高于 MR=4×NR=8 基线（25.5 GFLOPS）
