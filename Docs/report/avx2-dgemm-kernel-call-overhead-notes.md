# AVX2 DGEMM Kernel 调用与入口分支开销笔记

## 0. 版本信息

| 项目 | 值 |
|------|-----|
| 记录时间 | 2026-05-26 00:12:01 +08:00 |
| 当前分支 | main |
| 当前版本 | 2f38072 |
| 关联复盘 | `Docs/report/report2.md` |
| 关联源码 | `src/driver/gemm_driver.c`, `src/kernel/avx2/dgemm_kernel.c` |

本文档整理 `减少 kernel 调用和入口分支开销` 的相关知识，重点解释函数指针调用和 kernel 入口分支带来的额外开销，以及 full-tile fast path 的优化思路。

---

## 1. 背景

4.1 优化后，`pack_a` 已经不再是主瓶颈，kernel 成为主要耗时来源：

```text
1T n=2048:
pack_a: 1.8%
pack_b: 1.2%
kernel: 97.1%

4T n=2048:
pack_a: 3.0%
pack_b: 1.3%
kernel: 95.6%
```

因此后续优化可以分成两类：

```text
1. 提高单次 kernel 的计算吞吐，例如 p-loop unroll、MR/NR 调整、手写汇编
2. 降低每次 kernel 调用的固定开销，例如减少函数指针调用、入口分支、edge 判断
```

本文档关注第 2 类。

---

## 2. 为什么 kernel 调用次数很多

当前配置：

```text
MR = 8
NR = 6
Q  = 256
```

对于 `m=n=k=2048` 的单次 DGEMM：

```text
M 方向 tile 数 = 2048 / 8 = 256
N 方向 tile 数 = ceil(2048 / 6) = 342
K 方向块数 = 2048 / 256 = 8
```

所以 kernel 调用次数约为：

```text
256 * 342 * 8 = 700416 次 / 单次 DGEMM
```

日志中的 `n=2048` 是多轮 benchmark 累计，因此会看到：

```text
1T n=2048: kernel calls = 2101248
```

也就是 3 次 DGEMM 累计：

```text
700416 * 3 = 2101248
```

每次 full kernel 的计算量是：

```text
8 * 6 * 256 * 2 = 24576 FLOPs
```

单次计算量不算特别小，但几十万到几百万次调用会让固定开销累积。

---

## 3. 当前调用路径

driver 中调用 kernel 的形式类似：

```c
kernels->kernel(row1_rem, col1_rem, k_rem,
                alpha,
                sa,
                sb + k_rem * (col1 - col0),
                &C[row1 + col1 * ldc], ldc);
```

kernel 入口中还有 full/edge 判断：

```c
int dgemm_kernel_avx2(int m, int n, int k,
                      double alpha,
                      const double *A,
                      const double *B,
                      double *C, int ldc)
{
    if (m < MR || n < NR) {
        return dgemm_kernel_scalar(m, n, k, alpha, A, B, C, ldc);
    }
    return dgemm_kernel_fast(m, n, k, alpha, A, B, C, ldc);
}
```

因此 full tile 的实际路径是：

```text
driver
  -> kernels->kernel(...)              // 函数指针间接调用
      -> dgemm_kernel_avx2(...)
          -> if (m < MR || n < NR)     // 每次检查是否 edge
          -> dgemm_kernel_fast(...)    // 真正计算
```

理想 full-tile 主路径是：

```text
driver
  -> kernel_8x6_full(...)
      -> 直接进入 p-loop FMA 计算
```

---

## 4. 函数指针调用开销

### 4.1 什么是函数指针调用

当前 driver 通过 kernel table 调用：

```c
kernels->kernel(...);
```

这不是直接调用：

```c
dgemm_kernel_avx2(...);
```

而是通过函数指针间接调用。

这样做的优点是可以运行时选择不同 kernel：

```text
generic kernel
AVX2 kernel
未来可能的 AVX512 kernel
不同架构 kernel
```

代价是编译器和 CPU 都更难把调用路径优化到最短。

### 4.2 直接调用与函数指针调用

直接调用：

```c
dgemm_kernel_avx2(...);
```

机器码通常类似：

```asm
call dgemm_kernel_avx2
```

函数指针调用：

```c
kernels->kernel(...);
```

机器码通常类似：

```asm
call *%rax
```

区别：

```text
直接调用：目标函数在编译期基本确定
函数指针调用：目标函数地址运行时才从寄存器或内存中取出
```

### 4.3 额外成本来源

函数指针调用的额外开销主要来自以下几方面。

| 来源 | 说明 |
|------|------|
| 间接跳转预测 | CPU 需要预测 `call *%rax` 的目标地址 |
| 难以 inline | 编译器通常无法跨函数指针内联目标函数 |
| 调用目标不透明 | 编译器难以做常量传播和删除无用分支 |
| 参数传递 | 每次都要准备 `m/n/k/alpha/A/B/C/ldc` 等参数 |
| call/ret | 调用和返回本身不是零成本 |

对于当前项目，`kernels->kernel` 大部分时间应稳定指向 `dgemm_kernel_avx2`，所以间接跳转预测通常不会太差。更主要的问题是：

```text
函数指针调用阻断了 inline、常量传播和入口分支消除。
```

下面展开解释这些成本。

#### 4.3.1 间接调用与 CPU 前端预测

直接调用的机器码目标通常是固定的：

```asm
call dgemm_kernel_avx2
```

这种调用的目标地址在编译或链接后基本明确。CPU 前端取指时可以比较容易知道下一段代码在哪里，分支预测器也更容易稳定预测。

函数指针调用通常是：

```asm
call *%rax
```

或者：

```asm
call *(%r??)
```

它的目标地址来自寄存器或内存，CPU 需要使用间接分支预测器来猜测目标地址。

如果预测正确，间接调用的额外成本不一定很大。如果预测错误，CPU 已经取到和部分执行的错误路径指令需要丢弃，流水线会被冲刷，可能损失十几到几十个 cycles。

当前项目中，`kernels->kernel` 在一次运行中大概率长期指向同一个函数，例如 `dgemm_kernel_avx2`，所以预测通常能学会。但它仍然比直接调用多一个不确定性：

```text
直接调用：目标写在指令里
间接调用：目标在运行时数据里
```

另外，现代系统可能启用一些间接分支安全缓解机制，例如 retpoline、IBT/CET 等。是否生效取决于编译选项、系统配置和 CPU 特性。若启用，这类机制可能让间接调用更贵。不过当前项目不应先假设存在该开销，应通过反汇编和 benchmark 判断。

#### 4.3.2 难以 inline 才是更关键的问题

函数指针调用最重要的成本通常不是 `call *%rax` 本身，而是它让编译器很难 inline。

如果 driver 里直接写：

```c
dgemm_kernel_avx2(row1_rem, col1_rem, k_rem, alpha, sa, sb_ptr, C_ptr, ldc);
```

编译器至少知道调用目标是 `dgemm_kernel_avx2`。在某些编译条件下，它可能尝试 inline，或者做跨函数优化。

但当前是：

```c
kernels->kernel(row1_rem, col1_rem, k_rem, alpha, sa, sb_ptr, C_ptr, ldc);
```

编译器通常只能看到：

```text
这里会调用一个函数指针指向的函数。
```

它不知道最终目标是否一定是：

```text
dgemm_kernel_avx2
dgemm_kernel_generic
未来的其他 kernel
```

因此它很难把目标函数展开到 driver 里。

不能 inline 会带来一串连锁影响：

```text
调用边界保留
参数传递保留
入口分支保留
返回路径保留
调用方和被调用方之间不能共享优化信息
```

对于普通业务代码，这通常不是问题。但 DGEMM micro-kernel 是极热路径，调用次数可达几十万到几百万次，这些固定开销就会积累。

#### 4.3.3 调用目标不透明会阻断常量传播

full tile 场景中，`row1_rem` 和 `col1_rem` 大多数时候其实是常量：

```text
row1_rem = MR = 8
col1_rem = NR = 6
```

如果编译器能一路看到这些值传进 kernel，那么它理论上可以做常量传播：

```c
if (m < MR || n < NR) {
    edge_path;
} else {
    fast_path;
}
```

在 `m=8, n=6, MR=8, NR=6` 已知时，这个判断恒为 false：

```text
m < MR -> 8 < 8 -> false
n < NR -> 6 < 6 -> false
```

于是编译器可以把 edge 分支删除，只保留 fast path。

但函数指针调用下，编译器通常看不到目标函数内部，也不敢假设函数指针一定指向 `dgemm_kernel_avx2`。因此它不能把 `row1_rem == 8`、`col1_rem == 6` 传播进 `dgemm_kernel_avx2`，入口判断就保留下来了。

这就是“调用目标不透明”的实际含义：

```text
不是 CPU 算不动 if，而是编译器无法证明这个 if 对主路径没有必要。
```

#### 4.3.4 参数传递与 ABI 成本

当前 kernel 接口是：

```c
kernel(m, n, k, alpha, A, B, C, ldc)
```

在 x86-64 System V ABI 下，整数和指针参数通常通过通用寄存器传递，浮点参数通过 XMM 寄存器传递。

对于当前签名，整数/指针参数大致是：

```text
m   -> rdi
n   -> rsi
k   -> rdx
A   -> rcx
B   -> r8
C   -> r9
ldc -> 栈上传递或由调用约定安排到额外位置
alpha -> xmm0
```

也就是说，当前接口的整数/指针参数数量已经超过常见的 6 个通用参数寄存器数量，`ldc` 可能需要通过栈或额外 move 传递。具体生成方式要以反汇编为准，但参数多一定会增加调用点附近的寄存器准备压力。

对于 full `8x6` kernel，`m` 和 `n` 是冗余信息：

```text
m = 8
n = 6
```

如果改成 full-tile 专用接口：

```c
kernel_8x6_full(k, alpha, A, B, C, ldc)
```

整数/指针参数变成：

```text
k, A, B, C, ldc
```

这些更容易全部放入通用寄存器，调用点也少准备两个参数。更重要的是，接口本身表达了语义：

```text
这个函数只处理完整 8x6 tile，不需要检查 m/n。
```

这对后续手写汇编 kernel 也更友好。

#### 4.3.5 call/ret 本身也不是零成本

一次函数调用通常至少涉及：

```text
call 指令
return address 入栈
跳转到目标函数
ret 返回
```

现代 CPU 有 Return Stack Buffer，可以很好地预测 `ret` 返回地址，所以返回通常不是大问题。但它仍然占用前端、执行资源和调用约定成本。

如果单次 kernel 计算很大，这点成本可以忽略。但当前 `n=2048` 单次 DGEMM 约有：

```text
700416 次 kernel 调用
```

如果每次调用边界多消耗 10 cycles，累计就是：

```text
700416 * 10 = 7004160 cycles
```

这就是“小成本乘以大量调用”后变得可见的原因。

#### 4.3.6 为什么当前更该先消除入口分支，而不是执着于完全消灭函数指针

函数指针有架构分发价值：

```text
generic / AVX2 / 未来 AVX512 可以共用 driver
```

直接去掉函数指针会降低结构通用性。相比之下，拆出 full kernel 和 edge kernel 更稳妥：

```c
kernels->kernel_full_8x6(k, alpha, A, B, C, ldc);
kernels->kernel_edge(m, n, k, alpha, A, B, C, ldc);
```

这样即使仍然通过函数指针调用，full tile 至少不再进入：

```c
if (m < MR || n < NR)
```

也不再传递冗余的 `m/n`。这是低风险、结构清晰的第一步。

后续如果 benchmark 证明函数指针本身仍是可见瓶颈，再考虑 AVX2 专用 driver 路径中的 direct call。

### 4.4 full tile 中的冗余参数

对于完整 `8x6` tile，`m` 和 `n` 实际上固定：

```text
m = 8
n = 6
```

但当前每次仍然传入：

```c
kernel(m, n, k, alpha, A, B, C, ldc)
```

full-tile 专用接口可以简化为：

```c
kernel_8x6_full(k, alpha, A, B, C, ldc);
```

这样可以减少参数语义复杂度，也更适合未来替换为手写汇编 kernel。

---

## 5. Kernel 入口分支开销

### 5.1 入口分支的作用

当前入口分支是：

```c
if (m < MR || n < NR) {
    return dgemm_kernel_scalar(m, n, k, alpha, A, B, C, ldc);
}
return dgemm_kernel_fast(m, n, k, alpha, A, B, C, ldc);
```

它的作用是：

```text
完整 MR x NR tile -> 走 fast AVX2 kernel
边界 tile -> 走 scalar fallback
```

这个判断本身是合理的，因为最后几行或最后几列可能不足 `MR/NR`。

### 5.2 为什么它会污染主路径

对于大矩阵，绝大多数 tile 都是完整 tile。

以 `n=2048, NR=6` 为例：

```text
2048 = 341 * 6 + 2
```

N 方向：

```text
341 个 full 6-column tile
1 个 edge 2-column tile
```

full tile 比例：

```text
341 / 342 = 99.7%
```

M 方向：

```text
2048 / 8 = 256
```

没有 M edge。

也就是说绝大多数 kernel 调用都是：

```text
m == 8
n == 6
```

但每次仍然要执行：

```c
if (m < MR || n < NR)
```

这就是主路径污染：为了极少数 edge tile，99%+ 的 full tile 都要走通用入口。

### 5.3 分支预测准确也不是免费

大矩阵下，这个分支通常容易预测，因为大多数情况都走 fast path。

但预测准确不代表没有成本：

```text
需要比较 m 和 MR
需要比较 n 和 NR
需要条件跳转
多一层 wrapper
消耗 front-end 解码和 uop 资源
```

因此这里主要不是 branch mispredict 成本，而是：

```text
full tile 明明已知是完整 tile，却每次仍然检查 edge 条件。
```

### 5.4 edge tile 的分支变化

调用模式通常类似：

```text
full, full, full, ..., full, edge
full, full, full, ..., full, edge
```

CPU 可以学到大部分是 full path，但最后一个 edge tile 仍可能带来分支方向变化。

不过 edge 数量很少，所以 mispredict 不是主要瓶颈。主要问题仍是通用入口在 full 主路径上的固定开销。

---

## 6. Full-tile fast path

full-tile fast path 的目标是：

```text
让完整 MR x NR tile 直接进入专用 fast kernel
让边界 tile 单独进入 edge kernel
```

当前形式：

```c
kernel(row1_rem, col1_rem, ...);
```

kernel 内部再判断：

```c
if (m < MR || n < NR)
    scalar;
else
    fast;
```

优化后可以变为：

```c
if (row1_rem == MR && col1_rem == NR) {
    kernel_full_8x6(k_rem, alpha, sa, sb_ptr, C_ptr, ldc);
} else {
    kernel_edge(row1_rem, col1_rem, k_rem, alpha, sa, sb_ptr, C_ptr, ldc);
}
```

更进一步，可以把 full 区域和 edge 区域拆开：

```text
先处理所有完整 8x6 tile
最后单独处理剩余边界 tile
```

这样 full 主循环里连 `row1_rem == MR && col1_rem == NR` 判断也可以减少。

---

## 7. 优化层次

### 7.1 层次一：拆 full kernel 和 edge kernel

提供两个入口：

```c
kernel_full_8x6(k, alpha, A, B, C, ldc);
kernel_edge(m, n, k, alpha, A, B, C, ldc);
```

收益：

```text
full kernel 内部不再判断 m/n
edge 逻辑更清晰
接口更适合未来手写汇编 kernel
```

风险：

```text
需要改 kernel table、init 和 driver
需要确保 edge 正确性
```

### 7.2 层次二：driver 对 full tile 直接走 full kernel

如果仍然使用：

```c
kernels->kernel_full(...);
```

仍然是函数指针调用，但已经去掉了 kernel 入口分支。

如果架构路径已经确定，也可以直接调用：

```c
dgemm_kernel_8x6_full_avx2(...);
```

这样可以进一步减少函数指针开销，但会降低通用性。

### 7.3 层次三：full 区域和 edge 区域分离

主循环只处理完整 tile：

```c
for (row1 in full rows) {
    pack_a(MR);
    for (col1 in full cols) {
        kernel_full_8x6(...);
    }
}
```

边界单独处理：

```c
for (remaining rows/cols) {
    kernel_edge(...);
}
```

收益：

```text
主路径没有 row1_rem/col1_rem 判断
主路径没有 edge 分支
代码结构更适合专用 kernel
```

风险：

```text
driver 代码更复杂
边界处理更容易出错
```

---

## 8. 减少调用次数和减少每次调用开销的区别

`减少 kernel 调用和入口分支开销` 可以拆成两个方向：

```text
1. 让每次 kernel 调用更便宜
2. 让 kernel 调用次数更少
```

本文档主要讨论第 1 类。

第 2 类可以通过增大 kernel tile 实现，例如从 `8x6` 变成 `8x12`。但在 AVX2 下寄存器数量受限：

```text
8x6  -> 12 accumulators
8x12 -> 24 accumulators
```

AVX2 x86-64 只有 16 个 YMM，因此直接做完整 `8x12` accumulator 不现实。

也可以做伪 `8x12`，内部先算前 6 列再算后 6 列，但本质上仍接近两个 `8x6`，收益需要实测。

因此当前更实际的路线是：

```text
保持 8x6 tile 不变
先减少 full tile 每次调用的固定开销
```

---

## 9. 预期收益

这个优化一般不会像 4.1 那样带来 1.5x-2x 的提升。

原因是每次 full kernel 的计算量不小：

```text
8 * 6 * 256 * 2 = 24576 FLOPs
```

假设单核 50 GFLOPS，单次 kernel 计算时间约：

```text
24576 / 50e9 = 491 ns
```

函数指针调用、参数准备、入口分支可能是几 ns 到十几 ns 级别，因此比例大约可能是：

```text
1% - 5%
```

但由于单次 `n=2048` DGEMM 约有：

```text
700416 次 kernel 调用
```

即使每次只节省 10 cycles，也会累积为：

```text
700416 * 10 = 7004160 cycles
```

按 5GHz 粗略估算约为：

```text
1.4 ms
```

收益不是最大，但它风险低，且为后续 full kernel / assembly kernel 打基础。

---

## 10. 如何验证

### 10.1 看是否存在函数指针调用

反汇编 driver：

```bash
objdump -d src/driver/gemm_driver.o
```

如果看到类似：

```asm
call *%rax
call *%r??
```

说明是间接调用。

如果看到类似：

```asm
call dgemm_kernel_avx2
```

说明是直接调用。

### 10.2 看 kernel 入口分支

反汇编 kernel：

```bash
objdump -d src/kernel/avx2/dgemm_kernel.o
```

在 `dgemm_kernel_avx2` 入口附近查找：

```text
cmp
test
jl / jle / jb / jne / je
```

这类指令通常对应入口判断。

### 10.3 做版本对比

建议对比三个版本：

```text
A. 当前版本：函数指针 + kernel 入口判断
B. full/edge 分离：函数指针 + full kernel 无入口判断
C. direct full call：直接调用 full kernel
```

测试指标：

```text
LOG=0 GFLOPS
LOG=1 kernel calls 和 breakdown
正确性测试
边界尺寸表现
```

建议尺寸：

```text
64, 128, 256, 512, 1024, 2048
```

边界尺寸也要测：

```text
65, 127, 255, 513, 1023, 2047
```

因为 full-tile fast path 最容易在边界处理上出错。

---

## 11. 可能出现的问题

### 11.1 接口复杂度上升

从一个通用入口：

```c
kernel(m, n, k, alpha, A, B, C, ldc)
```

拆成：

```c
kernel_full_8x6(k, alpha, A, B, C, ldc)
kernel_edge(m, n, k, alpha, A, B, C, ldc)
```

会影响 kernel table、init 和 driver。

### 11.2 edge 处理容易出错

full tile 简单，edge tile 容易涉及：

```text
最后几行
最后几列
k_rem < Q
transa/transb
lda/ldb/ldc offset
```

必须用边界尺寸测试验证。

### 11.3 LOG 模式可能扭曲收益

如果开启 `MYBLAS_ENABLE_LOG`，每次 kernel 调用周围可能有计时逻辑：

```c
MYBLAS_LOG_TIMER_START(_log_kernel);
kernels->kernel(...);
MYBLAS_LOG_TIMER_END(_log_kernel, "kernel");
```

大量小调用下，计时成本可能放大固定开销。

建议区分：

```text
LOG=1：用于 breakdown 分析
LOG=0：用于真实性能比较
```

### 11.4 收益可能有限

如果主要瓶颈仍在 p-loop FMA 吞吐，减少调用开销只能带来小幅收益。

该优化更像是：

```text
清理主路径固定成本
为后续专用 kernel 铺路
```

不是替代 micro-kernel 本身优化。

---

## 12. 推荐实施路线

建议按低风险顺序推进：

```text
1. 暴露 full 8x6 kernel 入口，保留现有通用 edge 入口
2. driver 中对 full tile 调用 full kernel，对 edge tile 走原通用 kernel
3. benchmark 对比 LOG=0 性能
4. 用 LOG=1 确认 kernel calls 和 breakdown
5. 再考虑是否拆 full 区域和 edge 区域
6. 最后再评估是否需要 direct call 替代函数指针
```

当前最实用的第一步：

```text
新增 kernel_full_8x6 入口，去掉 full tile 上的 m/n 判断。
```

不建议第一步就大改 driver 循环结构，因为 full/edge 区域拆分更容易引入边界 bug。

---

## 13. 总结

函数指针调用的额外开销主要来自：

```text
间接 call
不能 inline
调用目标不透明
参数传递和 call/ret
可能的间接跳转预测成本
```

kernel 入口分支的额外开销主要来自：

```text
每次 full tile 都要判断 m/n 是否为 edge
多一层 wrapper
多一次条件跳转
极少数 edge 逻辑污染主路径
```

最直观的理解：

```text
函数指针调用：driver 不知道自己具体在调谁，所以编译器优化受限。
kernel 入口分支：kernel 每次都问“我是边界吗？”，但 99%+ 的答案都是“不是”。
```

优化目标：

```text
让 99%+ 的 full tile 走最短路径，直接进入计算；
让少数 edge tile 单独走通用路径。
```
