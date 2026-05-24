## Context

MyBLAS 是一个 BLAS 库，核心计算路径为 `my_dgemm` → `gemm_driver_double` → (pack_a, pack_b, kernel)。性能分析需要量化 pack_a / pack_b / kernel 各阶段的时间和调用次数。当前没有内建的测量基础设施，每次分析需临时编写计时代码。

关键约束：
- 日志系统不能影响正常性能（关闭时零开销）
- 使用 `clock_gettime(CLOCK_MONOTONIC)` 计时，已有先例（`test/bench_compare.c`）
- 需支持累积统计（多次 `my_dgemm` 调用的聚合指标）
- 埋点位置：API 层（总耗时）和 driver 层（分阶段耗时）

## Goals / Non-Goals

**Goals:**
- 提供编译期宏 `MYBLAS_ENABLE_LOG` 控制日志开关
- 关闭时所有日志代码完全消除（零开销）
- 记录 API 层指标：每次 `my_dgemm` 的 (m, n, k, transa, transb)、总耗时、GFLOPS
- 记录 Driver 层指标：pack_a 累积耗时、pack_b 累积耗时、kernel 累积耗时、各阶段调用次数
- 提供 `myblas_log_print()` 输出格式化的统计报告
- 提供 `myblas_log_reset()` 清零累积统计

**Non-Goals:**
- 不做运行时级别控制（DEBUG/INFO/ERROR 等级）
- 不写日志到文件（仅输出到 stdout/stderr）
- 不记录逐 kernel 调用的详细 trace（数据量过大）
- 不影响 SGEMM 路径（仅埋点 DGEMM，但架构支持后续扩展）

## Decisions

### 1. 宏开关 + 空结构体方案

**选择**：`MYBLAS_ENABLE_LOG` 编译宏控制。关闭时，日志结构体为空，计时宏展开为空语句。

```c
#ifdef MYBLAS_ENABLE_LOG
  // 实际计时逻辑
  #define MYBLAS_LOG_TIMER_START(var)  double var = myblas_log_time()
  #define MYBLAS_LOG_TIMER_END(var, field) myblas_log_accum(field, myblas_log_time() - var)
#else
  #define MYBLAS_LOG_TIMER_START(var)  ((void)0)
  #define MYBLAS_LOG_TIMER_END(var, field) ((void)0)
#endif
```

**理由**：
- 编译器在 O2 下可完全消除死代码
- 运行时零开销（无分支判断）
- 使用简单，仅需在埋点处加两行宏

**备选方案及拒绝理由**：
- 运行时 `if (log_enabled)` 分支：每次调用都有分支开销，影响热路径性能
- 函数指针回调：灵活但复杂，且引入间接调用开销

### 2. 全局累积统计 + 按调用记录

**选择**：使用全局 `myblas_log_stats_t` 结构体累积统计，同时在 API 层按调用记录最近一次的详细参数。

```c
typedef struct {
    long   dgemm_calls;
    double total_time;
    double pack_a_time;
    double pack_b_time;
    double kernel_time;
    long   pack_a_calls;
    long   pack_b_calls;
    long   kernel_calls;
    int    last_m, last_n, last_k;
    double last_gflops;
} myblas_log_stats_t;
```

**理由**：
- 全局统计支持 benchmark 中多次调用的聚合分析
- 最近一次记录方便单次调用的详细诊断

### 3. 埋点位置

| 层级 | 文件 | 埋点 | 指标 |
|------|------|------|------|
| API | `src/api/dgemm.c` | `my_dgemm` 入口/出口 | 总耗时、(m,n,k)、GFLOPS |
| Driver | `src/driver/gemm_driver.c` | `pack_a` 调用前后 | pack_a 耗时、调用次数 |
| Driver | `src/driver/gemm_driver.c` | `pack_b` 调用前后 | pack_b 耗时、调用次数 |
| Driver | `src/driver/gemm_driver.c` | `kernel` 调用前后 | kernel 耗时、调用次数 |

### 4. 输出格式

`myblas_log_print()` 输出示例：
```
=== MyBLAS Performance Log ===
DGEMM calls: 15
  Total:    2.345 s  (avg: 0.156 s)
  GFLOPS:   avg 28.4, peak 39.2
  Last:     m=1024 n=1024 k=1024  28.4 GFLOPS

Driver breakdown (cumulative):
  pack_a:  1.523 s (64.9%)  262656 calls
  pack_b:  0.031 s ( 1.3%)  2052 calls
  kernel:  0.791 s (33.7%)  262656 calls
```

## Risks / Trade-offs

- **[线程安全]** 全局统计结构体非线程安全，多线程写可能数据竞争 → 单线程累积正确，多线程时数据可能偏差但不会崩溃；后续可加 thread-local
- **[计时开销]** `clock_gettime` 约 50ns/次，driver 层每次 kernel 调用增加 100ns → 对小矩阵有影响，但日志通常仅在分析时开启
- **[代码侵入]** 埋点宏散布在 driver/api 代码中 → 宏名称统一（`MYBLAS_LOG_*`），关闭时完全不可见
