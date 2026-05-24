## ADDED Requirements

### Requirement: 编译宏控制日志开关

系统 SHALL 提供 `MYBLAS_ENABLE_LOG` 编译宏。定义该宏时启用日志功能，未定义时所有日志宏展开为空语句（`(void)0`），不产生任何运行时开销。

#### Scenario: 启用日志编译
- **WHEN** 编译时添加 `-DMYBLAS_ENABLE_LOG` 标志
- **THEN** 日志系统生效，`myblas_log_print()` 输出统计数据

#### Scenario: 禁用日志编译
- **WHEN** 编译时未定义 `MYBLAS_ENABLE_LOG`
- **THEN** 所有 `MYBLAS_LOG_*` 宏展开为空语句，生成的二进制与无日志版本完全相同（指令级别）

#### Scenario: 运行时无开销
- **WHEN** 未定义 `MYBLAS_ENABLE_LOG` 编译的程序运行 GEMM
- **THEN** GEMM 性能与无日志代码相同（GFLOPS 差异 < 1%）

### Requirement: API 层指标记录

系统 SHALL 在每次 `my_dgemm` 调用时记录：矩阵维度 (m, n, k)、转置标志 (transa, transb)、总耗时（秒）、计算 GFLOPS。

#### Scenario: 单次 DGEMM 调用记录
- **WHEN** 调用 `my_dgemm('N', 'N', 1024, 1024, 1024, ...)` 且日志启用
- **THEN** `myblas_log_print()` 显示 last_m=1024, last_n=1024, last_k=1024，以及对应的 GFLOPS

#### Scenario: 多次调用累积统计
- **WHEN** 连续调用 10 次 `my_dgemm`（不同尺寸）
- **THEN** `myblas_log_print()` 显示 dgemm_calls=10、total_time 为 10 次总和、avg GFLOPS 为平均值

### Requirement: Driver 层分阶段计时

系统 SHALL 在 `gemm_driver_double` 中分别累积 pack_a、pack_b、kernel 三个阶段的耗时和调用次数。

#### Scenario: 分阶段耗时统计
- **WHEN** 对 n=1024 矩阵执行 `my_dgemm` 且日志启用
- **THEN** `myblas_log_print()` 显示 pack_a_time、pack_b_time、kernel_time 的累积值（秒），三者之和接近总 dgemm 耗时

#### Scenario: 调用次数统计
- **WHEN** 对 n=1024, MR=8, NR=6 矩阵执行 `my_dgemm`
- **THEN** `myblas_log_print()` 显示 pack_a_calls、kernel_calls 等于 (m/MR) × (n/NR) × ceil(k/Q)，pack_b_calls 等于 (n/NR) × ceil(k/Q)

### Requirement: 日志公共 API

系统 SHALL 在 `include/myblas.h` 中暴露三个函数：

- `void myblas_log_reset(void)` — 清零所有累积统计
- `void myblas_log_print(void)` — 输出格式化统计报告到 stdout
- `void myblas_log_enable(int enable)` — 运行时开关（仅 `MYBLAS_ENABLE_LOG` 编译时有效）

#### Scenario: reset 清零
- **WHEN** 调用 `myblas_log_reset()` 后再调用 `myblas_log_print()`
- **THEN** 所有计数器和累积时间为零

#### Scenario: print 输出格式
- **WHEN** 执行若干 GEMM 后调用 `myblas_log_print()`
- **THEN** 输出包含：调用次数、总耗时、平均 GFLOPS、pack_a/pack_b/kernel 的耗时/占比/调用次数

### Requirement: Makefile 集成

系统 SHALL 在 Makefile 中提供 `LOG=1` 选项，编译时自动添加 `-DMYBLAS_ENABLE_LOG` 标志。

#### Scenario: make LOG=1
- **WHEN** 运行 `make LOG=1`
- **THEN** 编译产物包含日志功能，运行时可调用 `myblas_log_print()`

#### Scenario: 默认编译无日志
- **WHEN** 运行 `make`（不带 LOG=1）
- **THEN** 编译产物不含任何日志代码
