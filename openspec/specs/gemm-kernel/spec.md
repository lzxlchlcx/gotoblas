## ADDED Requirements

### Requirement: 微内核接口

系统 SHALL 定义微内核函数类型：

```c
typedef int (*gemm_kernel_func)(int m, int n, int k,
    double alpha, const double *A, const double *B,
    double *C, int ldc);
```

其中 A 为 MR×k（已打包），B 为 k×NR（已打包），C 为 m×n（步长 ldc）。

#### Scenario: 完整 tile 计算

- **WHEN** m=MR, n=NR, k=KC, alpha=1.0
- **THEN** C += A_packed * B_packed（对 MR×NR 的 tile 执行矩阵乘累加）

#### Scenario: 部分 tile（边界情况）

- **WHEN** m < MR 或 n < NR
- **THEN** 内核仅计算有效的 m×n 部分，使用掩码或标量回退

### Requirement: 通用 C 微内核

系统 SHALL 提供纯 C 微内核作为基线实现。double 精度：MR=4, NR=4。float 精度：MR=8, NR=4。

#### Scenario: 通用内核正确性

- **WHEN** 使用通用内核计算 4×4 tile，k=8
- **THEN** 结果与参考计算在 double 精度范围内一致（误差 < 1e-12）

### Requirement: Packing 函数

系统 SHALL 为每种精度提供 4 个打包函数：

- `pack_a_nn(m, k, A, lda, A_packed)`：NoTrans 方式打包 A
- `pack_a_tn(m, k, A, lda, A_packed)`：Trans 方式打包 A
- `pack_b_nn(k, n, B, ldb, B_packed)`：NoTrans 方式打包 B
- `pack_b_tn(k, n, B, ldb, B_packed)`：Trans 方式打包 B

所有打包函数 SHALL 生成适合微内核使用的连续内存布局。打包的本质目的是消除原始矩阵的跨页 stride 访问，将工作集压缩到尽可能少的页面中，从而大幅减少 TLB miss。

具体要求：
- 打包后的数据 SHALL 在内存中连续存放，无 stride 跳跃
- 打包 A_packed 的页面足迹 SHALL ≤ L1 DTLB 条目数 × 页面大小
- 打包 B_packed 的页面足迹 SHALL ≤ L2 STLB 条目数 × 页面大小

#### Scenario: pack_a_nn 布局

- **WHEN** 打包 A（m=4, k=3），NoTrans 方式
- **THEN** A_packed 包含 A 的列连续排列：[A_col0(4), A_col1(4), A_col2(4)]

#### Scenario: pack_a_tn 布局

- **WHEN** 打包 A（m=4, k=3），Trans 方式（即需要 A^T）
- **THEN** A_packed 包含 A 的行排列，如同读取 A^T 的列

#### Scenario: pack_b_nn 布局

- **WHEN** 打包 B（k=3, n=4），NoTrans 方式
- **THEN** B_packed 包含 B 的元素，按 NR 宽度的 panel 格式排列

### Requirement: Beta 缩放函数

系统 SHALL 提供 beta 缩放函数：

```c
int gemm_beta(int m, int n, double beta, double *C, int ldc);
```

#### Scenario: Beta = 0

- **WHEN** beta = 0.0, m=4, n=4
- **THEN** C 的所有元素被设为 0.0

#### Scenario: Beta = 2.0

- **WHEN** beta = 2.0, m=4, n=4
- **THEN** C 的所有元素乘以 2.0

### Requirement: Kernel 表

系统 SHALL 提供 `gemm_kernel_t` 结构体，聚合所有内核函数指针，并在编译时至少初始化一个 "generic" 实例。

#### Scenario: 通用 kernel 表初始化

- **WHEN** 库初始化时
- **THEN** 全局 `gemm_kernel_generic` 实例可用，所有函数指针指向通用 C 实现

### Requirement: Float 内核变体

系统 SHALL 为 float 和 double 精度提供独立的 kernel 表。

#### Scenario: Float kernel 表

- **WHEN** 调用 my_sgemm
- **THEN** 使用 float kernel 表（通用内核 MR=8, NR=4）
