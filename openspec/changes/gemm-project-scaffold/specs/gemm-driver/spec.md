## ADDED Requirements

### Requirement: 三级阻塞循环

驱动层 SHALL 实现 GotoBLAS 三级阻塞循环：
- 外层循环：js 遍历 N 维度，步长 R（适合 L3 缓存）
- 中层循环：ls 遍历 K 维度，步长 Q（适合 L2 缓存）
- 内层循环：is 遍历 M 维度，步长 P（适合 L2 缓存）

每次阻塞迭代前先打包 A 和 B 的对应块，再调用微内核。

#### Scenario: 大矩阵阻塞

- **WHEN** 调用 dgemm，m=1024, n=1024, k=1024
- **THEN** 驱动层按 P×Q（A）和 Q×R（B）的块迭代，每个块打包后再调用微内核

### Requirement: 阻塞参数的 TLB 约束

驱动层的阻塞参数 SHALL 同时满足缓存大小和 TLB reach 约束：
- P × Q × sizeof(element) SHALL ≤ L2 缓存大小
- P × Q × sizeof(element) / page_size SHALL ≤ L1 DTLB 条目数（打包后 A_packed 的页面足迹）
- Q × R × sizeof(element) SHALL ≤ L3 缓存大小
- Q × R × sizeof(element) / page_size SHALL ≤ L2 STLB 条目数

gemm_config_t SHALL 包含 `dtb_entries` 字段，表示 L1 DTLB 条目数（典型值 64）。

#### Scenario: 参数校验

- **WHEN** dtb_entries=64, page_size=4096, sizeof(double)=8
- **THEN** P × Q SHALL ≤ 64 × 4096 / 8 = 32768（即打包后 A_packed 的页面数不超过 64）

#### Scenario: 默认参数

- **WHEN** 目标为通用 x86-64（L1 DTLB=64, L2=256KB, L3=8MB）
- **THEN** 建议参数：P=128, Q=128, R=4096（满足 128×128×8=128KB ≤ 256KB，128×128/512=32 页 ≤ 64）

### Requirement: Packing 分发

驱动层 SHALL 根据 transa/transb 选择正确的打包函数。

#### Scenario: NN 打包

- **WHEN** transa='N', transb='N'
- **THEN** A 使用 pack_a_tn（转置拷贝，因为内核期望 A 列连续），B 使用 pack_b_nn（直接拷贝）

#### Scenario: TN 打包

- **WHEN** transa='T', transb='N'
- **THEN** A 使用 pack_a_nn（直接拷贝），B 使用 pack_b_nn（直接拷贝）

### Requirement: 计算前的 Beta 缩放

驱动层 SHALL 在主计算循环开始前对 C 施加 beta 缩放（当 beta != 1.0 时）。

#### Scenario: Beta = 0

- **WHEN** beta = 0.0
- **THEN** C 在阻塞循环开始前被清零

#### Scenario: Beta = 1

- **WHEN** beta = 1.0
- **THEN** C 在计算前不被修改（累加更新）

### Requirement: 自适应阻塞

驱动层 SHALL 在剩余维度小于完整块大小时自适应调整阻塞因子。具体而言，当剩余 M 大于 P 但小于 2*P 时，SHALL 按 MR 对齐拆分为两个块。

#### Scenario: 非对齐维度

- **WHEN** m = 100, MR = 4, P = 96
- **THEN** 第一个块处理 96 行，第二个块处理 4 行

### Requirement: 打包缓冲区的页面偏移

驱动层 SHALL 为 sa（打包 A 缓冲区）和 sb（打包 B 缓冲区）施加页面对齐偏移，使两者不在同一组 TLB 条目中互相驱逐。

具体策略：sa 使用起始偏移 `GEMM_OFFSET_A`，sb 使用起始偏移 `GEMM_OFFSET_B`，两者 SHALL 至少相差一个页面大小（4096 字节），建议相差 64KB 以上。

#### Scenario: 缓冲区偏移

- **WHEN** 分配打包缓冲区 buffer
- **THEN** sa = buffer + GEMM_OFFSET_A，sb = sa + align(P*Q*sizeof) + GEMM_OFFSET_B，其中 GEMM_OFFSET_A 和 GEMM_OFFSET_B 至少相差一个页面

### Requirement: 正确的内核指针偏移

驱动层 SHALL 向内核传递正确的指针偏移，使内核写入 C 的正确子块。

#### Scenario: 第二个 M 块

- **WHEN** 处理 is=96（第二个 M 块）和 js=0（第一个 N 块）
- **THEN** 内核接收 C + (96 + 0*ldc) 作为输出指针
