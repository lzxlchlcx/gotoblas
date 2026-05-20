# 07 - DGEMM 调用流程

## 文件依赖关系

```
src/api/dgemm.c            → 入口：参数校验 + 分发
  ├── include/myblas.h     → my_dgemm 声明
  ├── driver/gemm_internal.h → 数据结构 (gemm_arg_t, gemm_config_t, gemm_kernel_table_t)
  ├── config/generic.h     → gemm_config_generic_double()
  └── config/haswell.h     → gemm_config_avx2_double()  (AVX2 时)

src/driver/gemm_driver.c  → 六重阻塞循环 + pack + kernel
  └── driver/gemm_internal.h

src/driver/gemm_thread.c  → 并行分发 (pthread)
  └── driver/gemm_internal.h

src/kernel/generic/dgemm_kernel.c  → 4×4 微内核
src/kernel/generic/dgemm_pack.c    → A/B 打包
src/kernel/generic/dgemm_beta.c    → C 缩放/清零
src/kernel/generic/kernel_init.c   → 函数表绑定
```

## 调用序列

```
my_dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
  │
  ├─ 1. 转置解析 ─────────────────────────────────
  │   ta = 0 (不转) / 1 (转)
  │   tb = 0 (不转) / 1 (转)
  │   nrowa = ta ? k : m      // op(A) 逻辑行数
  │   nrowb = tb ? n : k      // op(B) 逻辑行数
  │
  ├─ 2. 参数校验 ─────────────────────────────────
  │   m ≥ 0, n ≥ 0, k ≥ 0
  │   lda ≥ nrowa, ldb ≥ nrowb, ldc ≥ m
  │   m==0 || n==0 → 直接返回
  │
  ├─ 3. 退化分支 ─────────────────────────────────
  │   if (k==0 || alpha==0):
  │     beta==0  → C = 0
  │     beta≠1   → C *= beta
  │     beta==1  → 不做操作
  │     return
  │
  ├─ 4. 运行时配置选择 ───────────────────────────
  │   if AVX2 可用:
  │     cfg = gemm_config_avx2_double()
  │     kernels = &gemm_kernel_avx2_double
  │   else:
  │     cfg = gemm_config_generic_double()
  │     kernels = &gemm_kernel_generic_double
  │
  ├─ 5. 初始化 ───────────────────────────────────
  │   分配 sa (P × Q) 和 sb (Q × R) packing buffer
  │   组装 gemm_arg_t
  │   beta==0 → C = 0
  │   beta≠1  → C *= beta
  │   (beta==1 → 不做操作)
  │
  └─ 6. 派发 ─────────────────────────────────────
      多线程 且 m*n*k > 65536
        → gemm_parallel_double()
           按 n 维度均分 chunk，每个线程拥有独立 sa/sb，
           调用 gemm_worker_double → gemm_driver_double
      否则
        → gemm_driver_double(arg, cfg, kernels, sa, sb)
```

## gemm_driver_double 六重阻塞循环

```
N 外循环 (js, step=R)
  │ B_packed 块大小: Q × R — L3 缓存层
  │
  └── K 循环 (ls, step=Q)
        │ A_packed 块大小: P × Q — L2 缓存层
        │
        └── N 内循环 (jjs, step=NR)
              │ B 的 NR 列子块打包一次
              │
              └── M 循环 (is, step=P)
                    │
                    └── M 内循环 (iis, step=MR)
                          │ A 的 MR 行子块打包一次
                          │
                          └── kernel(MR, NR, min_l, alpha, sa, sb, C[iis][jjs], ldc)
                                C += α · A_packed · B_packed
                                A_packed: MR × min_l (列优先连续)
                                B_packed: min_l × NR (列优先连续)
```

各层对应缓存：

```
循环层          步长        缓存级     数据重用
js (N)          R          L3         B_packed 跨 K 循环复用
ls (K)          Q          L2         A_packed 跨 N 内循环复用
jjs (N inner)   NR         L1         B_packed tile
is (M)          P          —          A_packed 块
iis (M inner)   MR         寄存器      A_packed tile
```

## packing 格式

Generic pack 仅做数据重排，消除源矩阵的 lda 跨步，使 kernel 访问连续：

```
pack_a (NN):  A_packed[i + p*m] = A[i + p*lda]
              → 列优先连续，行最快
pack_a (TN):  A_packed[i + p*m] = A[p + i*lda]
              → 等价于先转置再打包
pack_b (NN):  B_packed[p + j*k] = B[p + j*ldb]
              → 列优先连续，k 维度最快
pack_b (TN):  B_packed[p + j*k] = B[j + p*ldb]
              → 等价于先转置再打包
```

A 和 B 的包循环也在驱动层分开：B 在 `jjs` 层打包（NR 列 × min_l），A 在 `iis` 层打包（MR 行 × min_l），打包后的内存布局与 kernel 的连续访问模式一致。

## generic 4×4 微内核

```c
for (j = 0; j < n; j++)
    for (i = 0; i < m; i++) {
        sum = 0;
        for (p = 0; p < k; p++)
            sum += A[i + p*m] * B[p + j*k];
        C[i + j*ldc] += alpha * sum;
    }
```

- A 布局：`A[MR][k]` 列优先，跨步 m（即 MR）
- B 布局：`B[k][NR]` 列优先，跨步 k
- 内积循环 p 对 A 按列、B 按行访问，均为连续地址

## 并行策略

`gemm_parallel_double` 将 N 维度均分给各线程：

```
线程 0: C[:, 0..chunk)
线程 1: C[:, chunk..2*chunk)
...

每个线程拥有独立的 sa/sb packing buffer，
内部完全走 gemm_driver_double 串行路径。
```

## Generic 配置参数

| 参数 | 值 | 说明 |
|------|----|------|
| P | 128 | M 维阻塞 |
| Q | 128 | K 维阻塞 |
| R | 4096 | N 维阻塞 |
| MR | 4 | 微内核 M 展开 |
| NR | 4 | 微内核 N 展开 |
| offset_a | 0 | A buffer 偏移 |
| offset_b | 0 | B buffer 偏移 |

## 关键设计决策

1. **Beta 在 API 层处理**：`my_dgemm` 在进入 driver 前预乘 beta 到 C，driver 内部仅做 `C += α·A·B`，避免阻塞循环中为 beta 分特例
2. **Packed buffer 调用方分配**：串行路径由 `my_dgemm` 统一分配 sa/sb，并行路径由 `gemm_parallel_double` 为每个线程独立分配
3. **退化路径提前返回**：`k==0 || alpha==0` 时直接处理 C 后返回，避免进入封锁循环
4. **AVX2 运行时检测**：首次调用检查 CPU 特性，后续通过静态变量跳过重复检测
