## 1. 日志核心模块

- [x] 1.1 创建 `src/util/myblas_log.h`：定义 `myblas_log_stats_t` 结构体、计时宏 `MYBLAS_LOG_TIMER_START`/`MYBLAS_LOG_TIMER_END`、累积宏 `MYBLAS_LOG_ACCUM`、记录宏 `MYBLAS_LOG_RECORD_CALL`；当 `MYBLAS_ENABLE_LOG` 未定义时全部展开为 `(void)0`
- [x] 1.2 创建 `src/util/myblas_log.c`：实现 `myblas_log_time()`（封装 `clock_gettime`）、`myblas_log_accum()`（累加到全局 stats）、`myblas_log_reset()`（清零）、`myblas_log_print()`（格式化输出到 stdout）、`myblas_log_enable()`（运行时开关）

## 2. 公共 API 暴露

- [x] 2.1 在 `include/myblas.h` 中添加 `myblas_log_reset()`、`myblas_log_print()`、`myblas_log_enable()` 的声明，用 `#ifdef MYBLAS_ENABLE_LOG` 条件编译

## 3. 埋点：API 层

- [x] 3.1 在 `src/api/dgemm.c` 的 `my_dgemm` 函数入口处添加 `MYBLAS_LOG_TIMER_START`，出口处（driver 调用返回后）添加计时结束和 `MYBLAS_LOG_RECORD_CALL` 宏，记录 (m, n, k, transa, transb) 和 GFLOPS

## 4. 埋点：Driver 层

- [x] 4.1 在 `src/driver/gemm_driver.c` 的 `gemm_driver_double` 中，对 `pack_a` 调用前后添加计时宏，累积到 `pack_a_time` 和 `pack_a_calls`
- [x] 4.2 对 `pack_b` 调用前后添加计时宏，累积到 `pack_b_time` 和 `pack_b_calls`
- [x] 4.3 对 `kernel` 调用前后添加计时宏，累积到 `kernel_time` 和 `kernel_calls`

## 5. 构建集成

- [x] 5.1 修改 `Makefile`：添加 `src/util/myblas_log.o` 到 OBJS，添加 `LOG=1` 选项支持（`ifdef LOG` 时追加 `-DMYBLAS_ENABLE_LOG`）

## 6. 验证

- [x] 6.1 默认编译（`make clean && make lib`）确认无日志代码引入（编译通过，运行正常）
- [x] 6.2 `make LOG=1` 编译，运行 benchmark 调用 `myblas_log_print()`，验证输出包含调用次数、分阶段耗时、GFLOPS
