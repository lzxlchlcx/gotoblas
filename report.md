# MyBLAS 项目复现汇报

## 1. GotoBLAS 论文回顾：分块策略与缓存/TLB 约束

### 1.1 问题定义

矩阵乘法计算 `C[m×n] += A[m×k] · B[k×n]`，如果按照教科书的三重循环实现：

```c
for i = 0..m, for j = 0..n, for p = 0..k
    C[i][j] += A[i][p] * B[p][j]
```

内存访问模式是：B 行主序扫描（连续），A 列主序扫描（跨步），C 每个元素写一次。当矩阵超过缓存大小时，缓存行反复失效，性能急剧下降。

### 1.2 GotoBLAS 的核心洞察

Goto & van de Geijn (2008) 将矩阵乘法问题重新定义为 **GEBP（Generalized Block Panel Multiply）** 的分层递归：

```
C = A · B

↓ 沿 N 维度分块

C = [C₀ C₁ ... Cₜ] = A · [B₀ B₁ ... Bₜ] = [A·B₀  A·B₁  ...  A·Bₜ]

↓ 每个 A·Bⱼ 是一个 GEPP（Panel-Panel Multiply）

A·Bⱼ = [A₀ A₁ ... Aₛ]ᵀ · Bⱼ = [A₀·Bⱼ  A₁·Bⱼ  ...  Aₛ·Bⱼ]ᵀ

↓ 每个 Aᵢ·Bⱼ 是一个 GEBP（Block-Panel Multiply），即微内核
```

### 1.3 三级阻塞参数

每个 GEBP 微块内部，还需要进一步分块以适配各级缓存：

| 参数 | 含义 | 适配缓存 | 约束 |
|------|------|----------|------|
| `R` | N 维度阻塞大小 | **L3 缓存** | 每个线程的 C 块驻留 L3，减少对主存的写回 |
| `P` | M 维度阻塞大小 | **L2 缓存** | `P × Q × sizeof(element) ≈ L2 大小` |
| `Q` | K 维度阻塞大小 | **L2 缓存** | A_packed 大小为 `P × Q`，B_packed 为 `Q × R` |
| `MR` | M 微块大小 | **寄存器** | A 微块 `MR × Q` 从 L1 加载 |
| `NR` | N 微块大小 | **寄存器** | B 微块 `Q × NR` 持续广播 |

### 1.4 TLB 约束

TLB（Translation Lookaside Buffer）转换后备缓冲区同样有限。分块还确保了每个工作集映射的虚拟页面数不超过 TLB 容量。阻塞参数设计同时满足：

```
P × Q × sizeof(element) ≤ L2 缓存大小
P × Q × page_size / sizeof(element) ≤ TLB 条目数
```

GotoBLAS 在 `param.h` 中对每个 CPU 型号单独调优这些参数，MyBLAS 采用简化版配置：

| 精度 | P | Q | R | MR | NR |
|------|---|---|---|----|----|
| DGEMM (x86-64) | 256 | 256 | 4096 | 4 | 4 |
| SGEMM (x86-64) | 256 | 256 | 4096 | 8 | 4 |
| DGEMM (generic) | 128 | 128 | 4096 | 4 | 4 |
| SGEMM (generic) | 128 | 256 | 4096 | 8 | 4 |

## 2. 接口设计

### 2.1 BLAS 标准参数语义

以 `my_dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)` 为例：

```
C = α · op(A) · op(B) + β · C

op(X) = X      (若 trans = 'N')
op(X) = Xᵀ     (若 trans = 'T')
```

| 参数 | 含义 |
|------|------|
| `m` | `op(A)` 的行数，也是 C 的行数 |
| `n` | `op(B)` 的列数，也是 C 的列数 |
| `k` | `op(A)` 的列数，`op(B)` 的行数（内积维度） |
| `lda` | A 的 leading dimension（≥ op(A) 的行数） |

矩阵按**列优先**存储，元素 `X(i,j)` 位于 `X + i + j*ldX`。

### 2.2 接口层调度流程

`src/api/dgemm.c` 实现六步调度：

1. **转置映射**：将 `'N'/'T'/'C'` 归一化为 `ta/tb`（0/1）
2. **参数校验**：检查 `m,n,k ≥ 0`，`lda/ldb/ldc` ≥ 对应逻辑行数
3. **退化分支**：`k==0` 或 `alpha==0` 时只做 `C = β·C`
4. **运行时配置选择**：CPU 检测 → 选择阻塞参数（generic/haswell）+ 内核表
5. **缓冲区分配 + beta 预处理**：分配 `sa（P×Q）` 和 `sb（Q×R）` 打包缓冲区；`C *= β`
6. **派发**：单线程 → `gemm_driver_double`，多线程且计算量大 → `gemm_parallel_double`

### 2.3 可插拔内核表

通过函数指针表 `gemm_kernel_table_t` 支持多 CPU：

```c
typedef struct {
    gemm_kernel_func   kernel;     // 微内核
    pack_func          pack_a_nn;  // A pack (NoTrans)
    pack_func          pack_a_tn;  // A pack (Trans)
    pack_func          pack_b_nn;  // B pack (NoTrans)
    pack_func          pack_b_tn;  // B pack (Trans)
    gemm_beta_func     beta;       // C = β·C
} gemm_kernel_table_t;
```

编译时根据 `__AVX2__` 是否定义，选择加载 generic 或 AVX2 内核表。

## 3. Pack 与 Kernel 设计

### 3.1 问题：列主序的不连续性

A、B 都是列主序存储。微内核期望输入**连续**——因为寄存器不可能处理 lda 步长。Pack 函数是列主序与寄存器连续格式之间的桥梁。

### 3.2 Pack A — `pack_a_nn`

对于 `MR × k` 的子块（`transa='N'`），源 A 按列主序存储，每列内 MR 个元素连续，但跨列有 `lda - MR` 的空洞：

```
源 A（列优先, lda > MR）:
  列 c0:    A[row1, c0] ... A[row1+MR-1, c0]    连续
  列 c0+1:  A[row1, c0+1] ... A[row1+MR-1, c0+1] 连续
  跨列:     间隔 lda-MR 个空洞
```

`pack_a_nn` 将每列的 MR 个连续元素逐个拷贝，拼成 `MR × k_rem` 的紧凑 buffer：

```
packed A (sa): [c0的MR个元素 | c1的MR个元素 | ...]  列主序, 步长为 MR
```

`pack_a_tn`（`op(A)=Aᵀ`）：从 `A[p + i*lda]` 读取（即取 A 的行），同样拼成 `[row0..row3]` 连续格式。

### 3.3 Pack B — `pack_b_nn`

B 的 packed 格式是 `[k][NR]` 列主序，步长为 k。`pack_b_nn` 从 `B[p + j*ldb]` 直接拷贝到 `B_packed[p + j*k]`。

`pack_b_tn`（`op(B)=Bᵀ`）：从 `B[j + p*ldb]` 读取（即取 B 的行），写入 `B_packed[p + j*k]`。

### 3.4 微内核

Generic 微内核是最简单的三重循环（`dgemm_kernel_generic`）：

```c
for j = 0..n         // NR 方向
    for i = 0..m     // MR 方向
        sum = 0
        for p = 0..k // K 方向
            sum += A[i + p*m] * B[p + j*k]   // 连续访问！
        C[i + j*ldc] += alpha * sum
```

AVX2 微内核（`dgemm_kernel_avx2`）使用 256-bit SIMD 寄存器计算 4×4 微块：

```
c0 c1 c2 c3  ← __m256d 寄存器各存 4 个累加器

对每个 p (0..k-1):
    a0 = load(A[p*4 .. p*4+3])         // 一次加载 4 个 A 元素
    b0 = broadcast(B[p + 0*k])         // 广播 B 标量
    c0 = fmadd(a0, b0, c0)             // FMA 累加
    b1 = broadcast(B[p + 1*k])
    c1 = fmadd(a0, b1, c1)
    b2 = broadcast(B[p + 2*k])
    c2 = fmadd(a0, b2, c2)
    b3 = broadcast(B[p + 3*k])
    c3 = fmadd(a0, b3, c3)

结果累加到 C[0..3][0..3]
```

每个 p 步骤：**4 次 load + 4 次 broadcast + 4 次 FMA** = 16 次浮点运算 / 8 条指令，计算密度极高。

### 3.5 驱动层阻塞循环

`src/driver/gemm_driver.c` 实现 5 层嵌套循环：

```
for col0 = 0..n step R:           // N 维度外层块 — L3 缓存
    for k0 = 0..k step Q:         // K 维度块 — L2 缓存
        for col1 = col0..col0+col_rem step NR:    // N 微块
            pack_b(k_rem, col1_rem, B[k0+col1*ldb] → sb)
            for row0 = 0..m step P:               // M 维度块
                for row1 = row0..row0+row_rem step MR: // M 微块
                    pack_a(row1_rem, k_rem, A[row1+k0*lda] → sa)
                    kernel(row1_rem, col1_rem, k_rem, sa, sb → C[row1+col1*ldc])
```

B 的 pack 在外层 `col1` 循环中、`row0` 循环之外——**一个 B 微块被多个 A 行块复用**，减少 B 的重复打包。

## 4. 多线程处理

### 4.1 N 维度并行

MyBLAS 采用最简单的 N 维度均分策略。将 C 矩阵按 N 方向切分为连续的列条块，每个线程负责自己的列条块：

```
C[m×n]:
┌──────────────────────────┐
│        线程 0             │  n_0 列
├──────────────────────────┤
│        线程 1             │  n_1 列
├──────────────────────────┤
│        线程 2             │  n_2 列
└──────────────────────────┘
```

每个线程有独立的 `sa/sb` 缓冲区，无锁竞争。

### 4.2 实现

```c
// 线程工作函数
void *gemm_worker_double(void *data) {
    task = (gemm_task_t *)data;
    // 调整 n 和 B/C 指针偏移
    local_arg.n = task->n_to - task->n_from;
    local_arg.B = (double*)arg->B + task->n_from * arg->ldb;
    local_arg.C = (double*)arg->C + task->n_from * arg->ldc;
    gemm_driver_double(&local_arg, ...);
}

// 主函数：拆分 n → 创建线程 → worker在主线程执行 → join
chunk = n / nthreads;
for each thread t:
    n_from = t * chunk + min(t, remainder)
    pthread_create(&threads[t], NULL, worker, &tasks[t]);
gemm_worker_double(&tasks[0]);  // 主线程直接参与计算
for each thread t:
    pthread_join(threads[t], NULL);
```

### 4.3 阈值控制

仅在 `m × n × k > 65536` 且 `nthreads > 1` 时启多线程，避免小矩阵的线程创建开销超过计算收益。

线程数可通过 API `myblas_set_num_threads()` 或环境变量 `MYBLAS_NUM_THREADS` 设置，默认 1。

## 5. 实验结果

### 5.1 测试环境

- CPU: 13th Gen Intel Core i9-13900K（16 核 32 线程，AVX2 支持）
- 编译器: gcc -O2 -Wall
- 精度: double
- 阻塞参数: P=256, Q=256, R=4096, MR=4, NR=4

**AVX2 启用验证（三层确认）：**

| 层级 | 检测方法 | 结果 |
|------|----------|------|
| CPU 硬件 | `grep avx2 /proc/cpuinfo` | 32 处匹配，完整支持 AVX2 |
| 编译期 | `nm libmyblas.a` | AVX2 符号（`dgemm_kernel_avx2`、`gemm_kernel_avx2_double` 等）全部存在 |
| 运行时 | `cpu_supports_avx2()` 函数 | 由 `kernel_init.c` 调用 `cpuid` 指令动态检测，选择 AVX2 内核表 |

### 5.2 正确性测试

20 个测试用例全部通过：

| 测试类别 | 覆盖场景 | 用例数 |
|----------|----------|--------|
| 转置组合 | NN, NT, TN, TT | 4 |
| 边界条件 | α=0, β=0, k=0, m=1, n=1 | 5 |
| 非对齐维度 | 质数/不规则尺寸 | 4 |
| 大矩阵 | 64, 128, 256 方阵 | 3 |
| SGEMM | NN, TT, 非对齐, 大矩阵 | 4 |

测试方法：每个用例用 naive 三重循环参考实现计算结果，与 `my_dgemm`/`my_sgemm` 输出比较，相对误差阈值：
- DGEMM: 1e-10
- SGEMM: 1e-4

### 5.3 性能基准

Single-thread, AVX2 内核：

| 矩阵大小 | GFLOPS | 迭代次数 | 耗时 (s) |
|----------|--------|----------|----------|
| 64×64    | 1.32   | 20       | -        |
| 128×128  | 2.22   | 20       | -        |
| 256×256  | 4.34   | 10       | -        |
| 512×512  | 5.02   | 5        | -        |
| 1024×1024 | 5.90  | 5        | -        |
| 2048×2048 | 5.58  | 3        | -        |

CPU 理论峰值约为 20-40 GFLOPS（取决于具体型号），当前 generic 路径约 3.5 GFLOPS，**达到理论的 10-18%**。瓶颈在于纯 C 标量代码无 SIMD，且打包开销占比显著。

### 5.4 与 OpenBLAS 对比

测试环境：x86-64 Linux（AVX2），gcc -O2，OpenBLAS（pthreads 优化路径）。

**Single-thread：**

| 矩阵大小 | MyBLAS (GFLOPS) | OpenBLAS (GFLOPS) | 差距 |
|----------|-----------------|-------------------|------|
| 64×64    | 1.32            | 5.41              | 4.1x |
| 128×128  | 2.22            | 39.66             | 17.9x |
| 256×256  | 4.34            | 127.84            | 29.4x |
| 512×512  | 5.02            | 138.67            | 27.6x |
| 1024×1024 | 5.90           | 437.74            | 74.2x |
| 2048×2048 | 5.58           | 680.28            | 121.9x |

**4 threads：**

| 矩阵大小 | MyBLAS (GFLOPS) | OpenBLAS (GFLOPS) | 差距 |
|----------|-----------------|-------------------|------|
| 64×64    | 2.36            | 33.16             | 14.1x |
| 128×128  | 3.51            | 151.79            | 43.3x |
| 256×256  | 5.74            | 126.73            | 22.1x |
| 512×512  | 10.91           | 446.65            | 41.0x |
| 1024×1024 | 21.14          | 534.52            | 25.3x |
| 2048×2048 | 21.72          | 587.93            | 27.1x |

**分析：**

- **小矩阵 (64×64)**：差距最小（4-14x），此时打包开销占比大，计算密度低，SIMD 优势不能充分发挥。
- **大矩阵 (2048×2048)**：差距最大（27-122x），OpenBLAS 达到 **680 GFLOPS**（单线程）/ **588 GFLOPS**（4线程），而 MyBLAS 约 5.6 GFLOPS（单线程）/ 21.7 GFLOPS（4线程）。
- **MyBLAS 多线程扩展性**：1→4 线程在 2048 矩阵上获得约 4x 加速（5.6→21.7 GFLOPS），接近线性；但 OpenBLAS 4 线程在 64/128 小矩阵上反而因线程管理开销而性能下降。
- **核心差距**：MyBLAS 当前使用 AVX2 4×4 微内核，每次迭代处理 16 个 FLOP，而 OpenBLAS 使用更优化的内核路径（每次迭代处理更多 FLOP），加上 cache/TLB 参数的深度调优、prefetch 等，累积出 20-120x 的差距。Intel i9-13900K 理论峰值很高，但 MyBLAS 的 AVX2 内核尚未充分发挥其 SIMD 能力。

## 6. 改进空间

### 6.1 微内核优化（最大收益）

- **AVX2 kernel 仅支持固定 4×4**，应扩展到 4×8（OpenBLAS 标准），将计算密度翻倍
- **剩余块处理**：当前 `m<4 || n<4` 直接退回标量，未来可使用 mask load/store 处理尾块
- **SGEMM kernel 固定 8×4**，同理需支持更大 NR

### 6.2 Pack 函数优化

- **AVX2 pack_b_nn/tn 仍为标量** — 应引入 SIMD 向量化，对 k 维度做 4/8 路展开
- **pack_a_nn AVX2 已使用 SIMD**，pack_a_tn 使用 gather 方式，可尝试用 `vinsertf128` 等优化

### 6.3 多架构支持

- 仅 x86-64 AVX2 + generic C fallback，无 **ARM NEON/SVE**、**x86 SSE2** 后备
- 无 **DYNAMIC_ARCH** 运行时自动 CPU 检测切换（当前编译时硬绑定）

### 6.4 功能完整性

- **仅实现 GEMM**，缺 TRSM、SYRK、GEMMT、TRMM 等 Level-3 BLAS
- **无 CBLAS 行主序接口**
- **无复数类型**（CGEMM/ZGEMM）
- **无 LAPACK 上层**

### 6.5 线程层

- N 维度均分策略对 skinny tall 矩阵负载不均
- **无线程池**：每次 GEMM 调用创建/销毁 pthread，开销大
- 无 work-stealing 或动态调度

### 6.6 工程改进

- **内存对齐**：`_mm256_loadu_pd`（非对齐加载）比 `_mm256_load_pd`（对齐加载）有延迟损失，pack 缓冲应用 `aligned_alloc`
- **小矩阵路径**：OpenBLAS 有 `SMALL_MATRIX_OPT` 专用内核，MyBLAS 无
- **参数自适应**：`P/Q/R` 固定值，未按输入规模动态调优
- **Prefetch**：pack/kernel 中均无软预取指令

> 优先级排序：**完善 AVX2 4×8 微内核 → pack_b SIMD 化 → 对齐分配 → 剩余块处理**，这些集中在 kernel 层，代码改动量小，对性能影响最直接。
