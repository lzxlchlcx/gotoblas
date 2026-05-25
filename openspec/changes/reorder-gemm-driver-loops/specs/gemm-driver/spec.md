## MODIFIED Requirements

### Requirement: 三级阻塞循环与 packed 数据复用
驱动层 SHALL 实现 GotoBLAS 风格的三级阻塞循环：
- 外层循环：`col0` 遍历 N 维度，步长 `R`（适合 L3 缓存）
- 中层循环：`k0` 遍历 K 维度，步长 `Q`（适合 L2 缓存）
- M 方向循环：`row0` 遍历 M 维度，步长 `P`（适合 L2 缓存）
- 微块循环：`row1` 按 `MR` 遍历 M 微块，`col1` 按 `NR` 遍历 N 微块

在每个 `(col0, k0)` 阻塞内，驱动层 SHALL 先完成当前 `R` 范围内所有 `NR` B 微块的打包，使 `sb` 形成逻辑上的 `Q x R` packed B panel。

随后，驱动层 SHALL 对每个 `(row1, k0)` 的 `MR x Q` A 微块只打包一次，并复用该 packed A 依次处理当前 `R` 范围内的多个 `Q x NR` packed B 微块。

#### Scenario: 大矩阵阻塞

- **WHEN** 调用 dgemm，m=1024, n=1024, k=1024
- **THEN** 每个 `(col0, k0)` 范围内的 B 微块先被打包到 `sb`
- **AND** 每个 `(row1, k0)` 的 A 微块只打包一次
- **AND** 同一个 packed A 被用于多个 `col1` / `NR` B 微块的 kernel 调用
- **AND** C 的每个 `MR x NR` 子块跨多个 `k0` / `Q` 块累加得到完整结果

### Requirement: Packing 分发
驱动层 SHALL 根据 transa/transb 选择正确的打包函数，并在循环重排后保持该选择语义不变。

#### Scenario: NN 打包

- **WHEN** transa='N', transb='N'
- **THEN** A 使用 pack_a_nn，B 使用 pack_b_nn

#### Scenario: TN 打包

- **WHEN** transa='T', transb='N'
- **THEN** A 使用 pack_a_tn，B 使用 pack_b_nn

#### Scenario: NT 打包

- **WHEN** transa='N', transb='T'
- **THEN** A 使用 pack_a_nn，B 使用 pack_b_tn

#### Scenario: TT 打包

- **WHEN** transa='T', transb='T'
- **THEN** A 使用 pack_a_tn，B 使用 pack_b_tn

### Requirement: 正确的内核指针偏移
驱动层 SHALL 向内核传递正确的指针偏移，使内核写入 C 的正确子块。

#### Scenario: 第二个 M 块

- **WHEN** 处理 is=96（第二个 M 块）和 js=0（第一个 N 块）
- **THEN** 内核接收 C + (96 + 0*ldc) 作为输出指针
