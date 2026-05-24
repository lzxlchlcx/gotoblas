# MyBLAS 日志系统设计文档

## 1. 概述

为 MyBLAS 添加了基于编译宏的轻量级日志系统，可在 GEMM 调用链的关键阶段自动计算并累积性能指标，通过宏开关实现零开销控制。

## 2. 设计目标

- 编译期宏 `MYBLAS_ENABLE_LOG` 控制开关，关闭时所有日志代码完全消除
- 记录 API 层指标：每次 `my_dgemm` 的矩阵维度、总耗时、GFLOPS
- 记录 Driver 层指标：pack_a / pack_b / kernel 各阶段累积耗时与调用次数
- 提供格式化输出函数 `myblas_log_print()`

## 3. 架构

### 3.1 文件结构

```
src/util/myblas_log.h   — 宏定义与类型声明
src/util/myblas_log.c   — 实现函数
src/myblas.h             — 公共 API 暴露（条件编译）
src/api/dgemm.c          — API 层埋点
src/driver/gemm_driver.c — Driver 层埋点
Makefile                 — LOG=1 编译选项
```

### 3.2 核心数据结构

```c
typedef struct {
    long   dgemm_calls;       // my_dgemm 总调用次数
    double total_time;        // 总耗时（秒）
    double gflops_sum;        // GFLOPS 累积（用于计算平均）
    double gflops_peak;       // 峰值 GFLOPS
    double pack_a_time;       // pack_a 累积耗时
    double pack_b_time;       // pack_b 累积耗时
    double kernel_time;       // kernel 累积耗时
    long   pack_a_calls;      // pack_a 调用次数
    long   pack_b_calls;      // pack_b 调用次数
    long   kernel_calls;      // kernel 调用次数
    int    last_m, last_n, last_k;          // 最近一次矩阵维度
    int    last_transa, last_transb;        // 最近一次转置标志
    double last_gflops;       // 最近一次 GFLOPS
    int    enabled;           // 运行时开关
} myblas_log_stats_t;
```

### 3.3 宏接口

| 宏 | 作用 | 关闭时展开 |
|---|------|-----------|
| `MYBLAS_LOG_TIMER_START(var)` | 记录当前时间到 `var` | `((void)0)` |
| `MYBLAS_LOG_TIMER_END(var, field)` | 累加耗时到指定字段 | `((void)0)` |
| `MYBLAS_LOG_RECORD_CALL_ELAPSED(m,n,k,ta,tb,start)` | 记录单次调用信息 | `((void)0)` |
| `MYBLAS_LOG_PRINT()` | 输出统计报告 | `((void)0)` |
| `MYBLAS_LOG_RESET()` | 清零所有统计 | `((void)0)` |
| `MYBLAS_LOG_ENABLE(e)` | 运行时开关 | `((void)0)` |

## 4. 埋点位置

### 4.1 API 层 (`src/api/dgemm.c`)

```
my_dgemm() {
    ...
    if (k==0 || alpha==0) return;   // 跳过无效调用
    MYBLAS_LOG_TIMER_START(log_t0); // ← 计时起点
    ...
    gemm_driver_double(...)          // 实际计算
    MYBLAS_LOG_RECORD_CALL_ELAPSED(m, n, k, ta, tb, log_t0); // ← 记录
    free(sa); free(sb);
}
```

### 4.2 Driver 层 (`src/driver/gemm_driver.c`)

```
for col1:
    MYBLAS_LOG_TIMER_START(_log_pack_b);
    pack_b(...)
    MYBLAS_LOG_TIMER_END(_log_pack_b, "pack_b");

    for row1:
        MYBLAS_LOG_TIMER_START(_log_pack_a);
        pack_a(...)
        MYBLAS_LOG_TIMER_END(_log_pack_a, "pack_a");

        MYBLAS_LOG_TIMER_START(_log_kernel);
        kernel(...)
        MYBLAS_LOG_TIMER_END(_log_kernel, "kernel");
```

## 5. 使用方法

### 5.1 编译

```bash
make LOG=1 lib                          # 库
gcc -O2 -Isrc -DMYBLAS_ENABLE_LOG ...   # 直接编译
```

默认 `make`（不带 `LOG=1`）不包含任何日志代码。

### 5.2 运行时调用

```c
#include "myblas.h"

myblas_log_reset();          // 清零统计（可选，首次自动初始化）
my_dgemm('N','N', 1024, 1024, 1024, 1.0, A, 1024, B, 1024, 0.0, C, 1024);
my_dgemm('N','N', 2048, 2048, 2048, 1.0, A, 2048, B, 2048, 0.0, C, 2048);
myblas_log_print();          // 输出统计报告
```

### 5.3 输出示例

```
=== MyBLAS Performance Log ===
DGEMM calls: 126
  Total:    3.482 s  (avg: 0.0276 s)
  GFLOPS:   avg 19.7, peak 101.9
  Last:     m=2048 n=2048 k=2048 transa=N transb=N  93.0 GFLOPS

Driver breakdown (cumulative):
  pack_a:  2.970 s ( 59.3%)  5244804 calls
  pack_b:  0.054 s (  1.1%)  27289 calls
  kernel:  1.980 s ( 39.6%)  5228482 calls
===============================
```

## 6. 踩坑记录

### 6.1 `enabled` 默认值为 0

初始实现中 `g_stats = {0}` 将 `enabled` 初始化为 0，导致所有 `myblas_log_accum()` 和 `myblas_log_record_call()` 因 `if (!g_stats.enabled) return` 提前返回，数据全部丢弃。

**修复**：`g_stats = {.enabled = 1}`，编译了 `MYBLAS_ENABLE_LOG` 即默认开启。

### 6.2 需要显式调用 `myblas_log_print()`

日志系统只记录不主动输出。用户必须在程序中显式调用 `myblas_log_print()` 才能看到统计。这是有意设计——避免在热路径中插入 `printf`。

### 6.3 include 路径

日志头文件位于 `src/util/myblas_log.h`，其他源文件通过 `#include "util/myblas_log.h"` 引用（`-Isrc` 保证路径解析）。

## 7. 性能影响

`clock_gettime(CLOCK_MONOTONIC)` 约 50ns/次。在 driver 层每个 micro-tile 调用 3 次（pack_a + pack_b + kernel 前后各一次），对 n=2048 的单次 GEMM 约增加 30M 次计时调用 ≈ 1.5s 额外开销。日志仅在分析时通过 `LOG=1` 开启，日常使用无开销。

## 8. 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/util/myblas_log.h` | 新增 | 宏定义、类型声明 |
| `src/util/myblas_log.c` | 新增 | 实现函数 |
| `src/myblas.h` | 修改 | 添加日志 API 声明（条件编译） |
| `src/api/dgemm.c` | 修改 | API 层计时埋点 |
| `src/driver/gemm_driver.c` | 修改 | Driver 层分阶段计时埋点 |
| `Makefile` | 修改 | 添加 `SRCS_UTIL`、`LOG=1` 选项 |
| `test/bench_compare.c` | 修改 | 添加 `myblas_log_print()` 调用 |
