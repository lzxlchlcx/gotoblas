## Context

当前 MyBLAS 使用纯 C 标量代码实现微内核，性能约 6 GFLOPS。AVX2 提供 256 位宽 SIMD 寄存器（`__m256d` 存 4 个 double，`__m256` 存 8 个 float），配合 FMA（Fused Multiply-Add）指令可在一个周期内完成 4 次 double 乘加或 8 次 float 乘加。

参考：OpenBLAS Haswell dgemm 内核为 4×8 分块（MR=4, NR=8），使用 4 个 `__m256d` 寄存器存储 C 的 4×8 tile，性能约 50-80 GFLOPS。

## Goals / Non-Goals

**Goals:**
- 使用 AVX2 intrinsics（`__m256d` / `__m256`）+ FMA 实现 dgemm 和 sgemm 微内核
- 保持与通用内核相同的接口（`gemm_kernel_func` 签名不变）
- 运行时 CPU 检测，自动选择最优内核
- 提供 AVX2 优化的 packing 和 beta 缩放函数
- 针对 AVX2 调整阻塞参数

**Non-Goals:**
- 不实现 AVX-512（未来扩展）
- 不实现手写汇编（使用 C intrinsics）
- 不修改驱动层阻塞循环逻辑（只替换内核函数）

## Decisions

### D1: 使用 C intrinsics 而非汇编

**决策**: 使用 `__m256d` / `__m256` intrinsics + `_mm256_fmadd_pd` / `_mm256_fmadd_ps`，不写汇编。

**理由**: intrinsics 可读性远高于汇编，编译器（GCC/Clang -O2）生成的代码质量已接近手写汇编。学习成本低，调试方便。

**替代方案**: 手写 `.S` 汇编文件（如 OpenBLAS）。拒绝原因：可维护性差，且对学习项目收益有限。

### D2: dgemm 微内核采用 4×4 分块

**决策**: MR=4, NR=4。C tile 用 4 个 `__m256d` 寄存器（每列 4 个 double）。

**理由**: 
- 4 个 `__m256d` 刚好覆盖 4×4 tile
- AVX2 的 `vbroadcastsd` 可将 A 的单个元素广播到整个 `__m256d`
- 与 OpenBLAS Haswell dgemm 内核（MR=4, NR=8）相比，NR=4 更简单，适合初版

**替代方案**: MR=4, NR=8（OpenBLAS 风格）。拒绝原因：需要 8 个 `__m256d` 寄存器存储 C，寄存器压力大，编译器可能 spill。先做 4×4 再考虑 4×8。

### D3: sgemm 微内核采用 8×4 分块

**决策**: MR=8, NR=4。C tile 用 4 个 `__m256` 寄存器（每列 8 个 float）。

**理由**: 一个 `__m256` 存 8 个 float，MR=8 可充分利用寄存器宽度。

### D4: 运行时 CPU 检测

**决策**: 使用 `__builtin_cpu_supports("avx2")` 和 `__builtin_cpu_supports("fma")` 检测 CPU 特性。

**理由**: GCC/Clang 内建函数，无需手写 cpuid 汇编。如果编译器不支持，回退到 generic 内核。

**替代方案**: 手写 cpuid 指令。拒绝原因：`__builtin_cpu_supports` 已足够。

### D5: 阻塞参数调优

**决策**: AVX2 内核使用独立的阻塞参数配置 `haswell.h`。

**理由**: AVX2 内核的寄存器使用模式与通用内核不同，需要调整 P/Q/R 以更好利用缓存。参考 OpenBLAS Haswell 参数。

### D6: 边界处理使用标量回退

**决策**: 当 m < MR 或 n < NR 时，微内核回退到标量计算（与通用内核相同的循环结构）。

**理由**: 边界 tile 出现频率低（仅在矩阵维度不是 MR/NR 倍数时），不值得为它写 SIMD 代码。OpenBLAS 也采用类似策略。

## Risks / Trade-offs

**[风险] intrinsics 性能不如手写汇编** → 缓解: 性能差距通常在 10-20% 以内，对学习项目可接受。后续可逐步迁移到汇编。

**[风险] 编译器不支持 AVX2 intrinsics** → 缓解: 条件编译（`#ifdef __AVX2__`），不支持时回退到 generic 内核。

**[风险] 寄存器 spill 导致性能下降** → 缓解: 控制 C tile 大小（4×4 double 或 8×4 float），不超过 16 个 ymm 寄存器。

**[风险] 打包函数成为瓶颈** → 缓解: AVX2 打包函数使用 SIMD load/store 加速，但不改变打包逻辑。打包开销在大矩阵时可忽略。
