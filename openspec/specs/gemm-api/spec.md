## ADDED Requirements

### Requirement: my_dgemm 公共 API

系统 SHALL 提供函数 `my_dgemm`，签名为标准 BLAS DGEMM 格式：

```c
void my_dgemm(char transa, char transb,
              int m, int n, int k,
              double alpha, const double *A, int lda,
                            const double *B, int ldb,
              double beta,        double *C, int ldc);
```

计算：C = alpha * op(A) * op(B) + beta * C，其中 op(X) = X 或 X^T。

#### Scenario: 普通 NN 乘法

- **WHEN** transa='N', transb='N', m=4, n=4, k=4, alpha=1.0, beta=0.0
- **THEN** C[i + j*ldc] = sum_p(A[i + p*lda] * B[p + j*ldb])，对所有 i,j

#### Scenario: A 转置

- **WHEN** transa='T', transb='N'
- **THEN** op(A) 使用 A^T，即 A 按 A[p + i*lda] 方式访问

#### Scenario: B 转置

- **WHEN** transa='N', transb='T'
- **THEN** op(B) 使用 B^T，即 B 按 B[j + p*ldb] 方式访问

### Requirement: my_sgemm 公共 API

系统 SHALL 提供函数 `my_sgemm`，语义与 `my_dgemm` 完全一致，但操作 `float` 数据。

#### Scenario: 浮点乘法

- **WHEN** 调用 my_sgemm 传入 float 数组
- **THEN** 结果与 CBLAS `cblas_sgemm` 在 float 精度范围内一致

### Requirement: 参数校验

系统 SHALL 校验所有参数，并将错误信息输出到 stderr。无效参数包括：m < 0、n < 0、k < 0、lda 不足、ldb 不足、ldc < m、无效的 transa/transb 字符。

#### Scenario: 负维度

- **WHEN** m = -1
- **THEN** 函数向 stderr 打印错误信息后返回，不修改 C

#### Scenario: 无效转置字符

- **WHEN** transa = 'X'
- **THEN** 函数向 stderr 打印错误信息后返回

### Requirement: 特殊情况处理

系统 SHALL 高效处理退化情况，不进入阻塞循环。

#### Scenario: 零维度

- **WHEN** m = 0 或 n = 0
- **THEN** 函数直接返回，不修改 C

#### Scenario: k 为零

- **WHEN** k = 0 且 beta = 1.0
- **THEN** C 保持不变（恒等操作）

#### Scenario: alpha 为零

- **WHEN** alpha = 0.0 且 beta = 0.0
- **THEN** C 被清零

#### Scenario: alpha 为零，beta 非零

- **WHEN** alpha = 0.0 且 beta = 2.0
- **THEN** C = 2.0 * C（不执行矩阵乘法）

### Requirement: 列主序存储

系统 SHALL 对所有矩阵使用列主序（Fortran 风格）存储。

#### Scenario: 列主序布局

- **WHEN** A 是 3×2 矩阵，值为 [[1,2],[3,4],[5,6]]，列主序，lda=3
- **THEN** A[0]=1, A[1]=3, A[2]=5, A[3]=2, A[4]=4, A[5]=6
