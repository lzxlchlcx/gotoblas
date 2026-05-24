## Why

当前 MyBLAS 缺少系统化的性能测量手段。性能分析（如 `Docs/avx2-dgemm-performance-report.md` 中 pack_a 占 66% 时间的结论）需要临时编写计时测试代码，无法在日常开发和 benchmark 中自动采集。需要一个轻量的 LOG 系统，能在 GEMM 调用链的关键阶段（API 层总耗时、driver 层的 pack_a/pack_b/kernel 分阶段耗时、调用次数等）自动计算并保存指标数值，且可通过编译宏完全关闭（零开销）。

## What Changes

- 新增 `src/util/myblas_log.h`：定义日志宏接口（`MYBLAS_LOG_*`），通过 `MYBLAS_ENABLE_LOG` 宏开关控制
- 新增 `src/util/myblas_log.c`：日志系统的实现（累积计时、指标聚合、输出格式化）
- 在 `include/myblas.h` 中暴露日志控制 API（`myblas_log_enable()`、`myblas_log_print()`、`myblas_log_reset()`）
- 在 `src/api/dgemm.c` 中埋点：记录每次 `my_dgemm` 调用的总耗时、矩阵尺寸、GFLOPS
- 在 `src/driver/gemm_driver.c` 中埋点：记录 pack_a、pack_b、kernel 各阶段的累积耗时和调用次数
- 当 `MYBLAS_ENABLE_LOG` 未定义时，所有日志宏展开为空，零运行时开销

## Capabilities

### New Capabilities

- `logging-system`: 基于 `clock_gettime` 的轻量级日志系统，支持宏开关、分阶段计时、指标聚合与格式化输出

### Modified Capabilities

## Impact

- `src/util/myblas_log.h` — 新增，日志宏定义
- `src/util/myblas_log.c` — 新增，日志实现
- `include/myblas.h` — 新增 3 个日志 API 声明
- `src/api/dgemm.c` — 添加 API 层计时埋点
- `src/driver/gemm_driver.c` — 添加 driver 层分阶段计时埋点
- `Makefile` — 添加 `src/util/myblas_log.o`，添加 `MYBLAS_ENABLE_LOG` 编译选项
- 不影响任何现有功能（日志关闭时行为完全不变）
