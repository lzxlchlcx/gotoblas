## Context

当前 MyBLAS 的 AVX2 DGEMM 微内核采用 MR=4, NR=8 配置，运行在 i9-13900K (Raptor Lake) 上。该 CPU 仅支持 AVX2（16 个 YMM 寄存器），不支持 AVX-512。当前微内核每 k 次迭代产出 32 FLOPs，占用 12/16 个 YMM 寄存器。

核心问题：kernel 中的 intrinsics 代码与 MR/NR 值强绑定（显式声明固定数量的 `__m256d` 累加器），切换配置需要重写。pack/beta 函数已通过 m/n 参数泛化，无需改动。

现有代码结构：
- `src/kernel/avx2/dgemm_kernel.c` — 微内核，硬编码 MR=4, NR=8
- `src/kernel/avx2/dgemm_pack.c` — pack_a_nn, pack_a_tn, pack_b_nn, pack_b_tn（已泛化）
- `src/kernel/avx2/dgemm_beta.c` — beta 缩放（已泛化）
- `src/config/haswell.h` — MR=4, NR=8, P=256, Q=256, R=4096
- `src/driver/gemm_driver.c` — 通过 cfg->MR/cfg->NR 泛化调用

数据布局约定（不变）：
- packed A: `A_packed[i + p * m]`，m 行 k 列，列优先
- packed B: `B_packed[p + j * k]`，k 行 n 列，列优先

## Goals / Non-Goals

**Goals:**
- 建立配置驱动的 kernel 框架，修改 `haswell.h` 宏即可切换 MR/NR
- 实现 MR=8×NR=6 的优化 AVX2 快速路径（48 FLOPs/k 迭代）
- 保留 MR=4×NR=8 路径作为备选
- 提供通用 AVX2 fallback 路径（按 4×4 子块循环），支持任意 MR/NR
- pack/beta 函数无需修改（已泛化）

**Non-Goals:**
- 不修改 SGEMM（float 精度）路径
- 不引入 AVX-512 支持（CPU 不支持）
- 不修改 driver 层或 API 层代码
- 不修改多线程逻辑
- 不做运行时 MR/NR 切换（编译期确定）

## Decisions

### 1. 配置驱动 + `#if` 分发机制

**选择**: kernel 文件 include config 头文件，通过 `#if MR == 8 && NR == 6` 选择优化路径。

```c
// dgemm_kernel.c
#include "config/haswell.h"
#define MR GEMM_HASWELL_D_MR
#define NR GEMM_HASWELL_D_NR

int dgemm_kernel_avx2(int m, int n, int k, ...) {
    if (m < MR || n < NR) {
        // scalar fallback（与 MR/NR 无关）
    }

#if MR == 8 && NR == 6
    // 12 累加器 + 2 A load 优化路径
#elif MR == 4 && NR == 8
    // 8 累加器 + 1 A load 保留路径
#else
    // 通用 AVX2 路径：按 4×4 子块循环
#endif
}
```

**理由**:
- AVX2 intrinsics 的寄存器分配必须在编译期确定，无法用运行时变量控制累加器数量
- `#if` 分发是零开销抽象，不影响运行时性能
- 新增配置只需添加一个 `#elif` 分支
- `#else` 通用路径保证任意配置都有可工作的实现

**备选方案及拒绝理由**:
- 运行时函数指针分发：累加器无法动态分配，不可能实现
- 每个 MR/NR 组合独立文件：维护成本高，scalar fallback 和存储逻辑重复
- 纯宏模板（用宏生成累加器声明）：可读性差，难以调试

### 2. MR=8×NR=6 优化路径设计

**寄存器分配** (15/16 YMM):
| 用途 | 数量 | 说明 |
|------|------|------|
| 累加器 c[j][r] | 6×2=12 | NR=6 列 × ceil(MR/4)=2 行组 |
| A 加载 a0, a1 | 2 | rows 0-3 和 4-7 |
| alpha | 1 | 标量广播 |
| 合计 | 15 | 存储阶段复用 a0/a1 作为 c_old |

**内核主循环体**（每个 k 迭代）:
```
a0 = load(A[p * MR + 0..3])
a1 = load(A[p * MR + 4..7])
for j in 0..5:
    b = set1(B[p + j * k])
    c[j][0] = fmadd(a0, b, c[j][0])
    c[j][1] = fmadd(a1, b, c[j][1])
```

每 k 迭代: 2 loads + 6 broadcasts + 12 FMAs = 20 操作，产出 48 FLOPs。

### 3. 通用 AVX2 Fallback 路径

对不在 `#if`/`#elif` 中的 MR/NR 组合，提供按 4×4 子块循环的通用路径：

```c
for (int jj = 0; jj < n; jj += 4) {
    for (int ii = 0; ii < m; ii += 4) {
        int r = min(4, m - ii);
        // 按 r 元素加载 A，按 4 列加载 B
        // 使用 4+1=5 个 YMM 寄存器
    }
}
```

该路径性能低于优化路径，但保证正确性且适用于任意 MR/NR。

### 4. 配置参数

| 参数 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| MR | 4 | 8 | 微内核行数翻倍 |
| NR | 8 | 6 | 微内核列数 |
| P | 256 | 256 | 行分块不变 |
| Q | 256 | 256 | K 分块不变 |
| R | 4096 | 4096 | 列分块不变 |

### 5. Pack/Beta 不变

`dgemm_pack_a_nn_avx2` 等函数已使用 `m`/`n` 参数驱动打包，不依赖硬编码的 MR/NR。`dgemm_beta_avx2` 同理。这些函数无需修改。

## Risks / Trade-offs

- **[寄存器压力]** MR=8×NR=6 使用 15/16 YMM，编译器可能 spill → 用 `-O2` 编译并检查汇编输出；如有 spill 降至 MR=4×NR=12
- **[维护成本]** 每个 MR/NR 组合需要手写一个 `#if` 分支 → 通用 fallback 保证任意配置可工作，优化路径按需添加
- **[边界处理]** scalar fallback 路径与 MR/NR 无关，使用标量三重循环 → 简单正确，小矩阵性能不敏感
- **[配置一致性]** kernel 的 `#if` 必须与 `haswell.h` 的宏匹配 → kernel 直接 include config 头，消除不一致风险
