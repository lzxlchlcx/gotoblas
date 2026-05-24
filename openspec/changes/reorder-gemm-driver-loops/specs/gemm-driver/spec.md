## MODIFIED Requirements

### Requirement: 三级阻塞循环
驱动层 SHALL 实现 GotoBLAS 三级阻塞循环：
- 外层循环：js 遍历 N 维度，步长 R（适合 L3 缓存）
- 中层循环：ls 遍历 K 维度，步长 Q（适合 L2 缓存）
- 内层循环：is 遍历 M 维度，步长 P（适合 L2 缓存）

在每个 `k0` 阻塞内，驱动层 SHALL 先完成该 `k0` 范围内所有 B 列块的打包，然后对每个 A 微块只打包一次，并复用该 packed A 依次处理多个 B 列块。

#### Scenario: 大矩阵阻塞

- **WHEN** 调用 dgemm，m=1024, n=1024, k=1024
- **THEN** 驱动层按 P×Q（A）和 Q×R（B）的块迭代，并在同一 `k0` 内复用 packed A 处理多个 `col1` 块

### Requirement: Packing 分发
驱动层 SHALL 根据 transa/transb 选择正确的打包函数。

#### Scenario: NN 打包

- **WHEN** transa='N', transb='N'
- **THEN** A 使用 pack_a_tn（转置拷贝，因为内核期望 A 列连续），B 使用 pack_b_nn（直接拷贝）

#### Scenario: TN 打包

- **WHEN** transa='T', transb='N'
- **THEN** A 使用 pack_a_nn（直接拷贝），B 使用 pack_b_nn（直接拷贝）

### Requirement: 正确的内核指针偏移
驱动层 SHALL 向内核传递正确的指针偏移，使内核写入 C 的正确子块。

#### Scenario: 第二个 M 块

- **WHEN** 处理 is=96（第二个 M 块）和 js=0（第一个 N 块）
- **THEN** 内核接收 C + (96 + 0*ldc) 作为输出指针
