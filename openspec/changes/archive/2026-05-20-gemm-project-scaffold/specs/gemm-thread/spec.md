## ADDED Requirements

### Requirement: 线程分发

系统 SHALL 提供函数 `gemm_parallel`，使用 pthread 将 GEMM 计算分发到 N 个线程。

```c
void gemm_parallel(const gemm_arg_t *arg, const gemm_config_t *cfg,
                   const gemm_kernel_table_t *kernels, int nthreads);
```

#### Scenario: 单线程

- **WHEN** nthreads = 1
- **THEN** 计算在调用线程中执行，不创建任何 pthread

#### Scenario: 四线程

- **WHEN** nthreads = 4, n = 1024
- **THEN** 创建四个 pthread，每个处理约 n/4 = 256 列的 C

### Requirement: N 维度并行

系统 SHALL 沿 N（列）维度划分工作。每个线程处理一个连续的列范围 [n_from, n_to)。

#### Scenario: 均匀划分

- **WHEN** n = 1000, nthreads = 4
- **THEN** 线程 0 处理列 [0, 250)，线程 1 处理 [250, 500)，线程 2 处理 [500, 750)，线程 3 处理 [750, 1000)

#### Scenario: 非均匀划分

- **WHEN** n = 10, nthreads = 3
- **THEN** 线程处理约 [0,4)、[4,7)、[7,10) 列（具体划分由实现定义，但所有列必须被覆盖）

### Requirement: 线程私有打包缓冲区

每个线程 SHALL 拥有自己的打包缓冲区（sa, sb），以避免 false sharing 和数据竞争。

#### Scenario: 缓冲区分配

- **WHEN** nthreads = 4
- **THEN** 分配 4 对 (sa, sb) 缓冲区，每线程一对

### Requirement: 线程阈值

系统 SHALL 仅在总工作量（m * n * k）超过可配置阈值时使用多线程。低于阈值时使用单线程执行。

#### Scenario: 小矩阵

- **WHEN** m=32, n=32, k=32, nthreads=4, threshold=65536
- **THEN** 单线程执行（32*32*32 = 32768 < 65536）

#### Scenario: 大矩阵

- **WHEN** m=256, n=256, k=256, nthreads=4, threshold=65536
- **THEN** 四线程执行（256*256*256 = 16777216 > 65536）

### Requirement: 可配置线程数

线程数 SHALL 通过环境变量 `MYBLAS_NUM_THREADS` 控制，未设置时默认为 1。

#### Scenario: 设置环境变量

- **WHEN** MYBLAS_NUM_THREADS=4
- **THEN** gemm_parallel 使用 4 个线程

#### Scenario: 未设置环境变量

- **WHEN** MYBLAS_NUM_THREADS 未设置
- **THEN** gemm_parallel 使用 1 个线程（单线程）

### Requirement: 线程合并

系统 SHALL 在 gemm_parallel 返回前合并所有工作线程。调用方可以假设 C 在 gemm_parallel 返回时已计算完成。

#### Scenario: 完成保证

- **WHEN** gemm_parallel 返回
- **THEN** 所有线程已完成，C 包含最终结果
