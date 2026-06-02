# MyBLAS CUDA DGEMM / SGEMM 实现与兼容性报告

## 0. 版本信息

| 项目 | 值 |
|------|-----|
| 记录时间 | 2026-06-02 12:17:10 +08:00 |
| 工作目录 | `/home/l2316/program/gotoblas` |
| 目标 GPU | NVIDIA RTX 3060 Laptop GPU |
| Compute Capability | SM 8.6 |
| 显存 | 6 GiB |
| CUDA | 12.6 |
| 关联计划 | `.kilo/plans/1780363657014-witty-planet.md`, `.kilo/plans/1780364390211-silent-moon.md` |
| 关联源码 | `src/kernel/cuda/`, `src/api/gemm_gpu_dispatch.c`, `src/api/dgemm.c`, `src/api/sgemm.c`, `Makefile` |

本文档记录当前 MyBLAS CUDA DGEMM / SGEMM 后端的实现状态、代码架构、与原 CPU 路径的兼容方式、测试结果和性能现状。

---

## 1. 当前实现概览

当前 MyBLAS 已新增 CUDA 后端，支持：

```text
DGEMM: double precision GEMM
SGEMM: single precision GEMM
```

DGEMM 后端提供 4 个版本的 kernel，SGEMM 后端提供 5 个版本的 kernel：

| 版本 | 名称 | 核心思路 | 当前用途 |
|------|------|----------|----------|
| `v0` | Naive | 每个 thread 计算一个 C 元素 | 正确性基线 |
| `v1` | Shared Memory Tiling | 16x16 tile，A/B 子块放入 shared memory 复用 | DGEMM 大矩阵表现较好 |
| `v2` | Register Blocking | 每个 thread 计算 8x8 输出子块，寄存器累加 | SGEMM 当前最快版本之一 |
| `v3` | Double Buffer + Padding | 在 v2 基础上使用双 shared buffer 和 padding | DGEMM 默认版本 |
| `v4` | Tensor Core / TF32 | SGEMM 使用 WMMA Tensor Core，TF32 multiply + FP32 accumulate | 显式 fast SGEMM 版本 |

GPU 后端当前已经接入 `my_dgemm()` 和 `my_sgemm()`：

```text
大矩阵：尝试走 GPU
小矩阵：保留原 CPU 路径
GPU 不可用或 GPU 调用失败：回退 CPU 路径
```

当前实现重点是正确性、架构接入和公平 benchmark。性能上 DGEMM 已经比较接近 cuBLAS 的 FP64 性能上限；SGEMM 已新增 Tensor Core / TF32 v4 fast path，并通过 device-resident benchmark 与 cuBLAS 使用相同计时口径对比。

---

## 2. 新增文件结构

当前新增的 GPU 相关文件如下：

```text
src/
  api/
    gemm_gpu_dispatch.c
  config/
    ampere.h
  kernel/
    cuda/
      gemm_gpu.h
      gpu_common.cuh
      gpu_init.cu
      dgemm_kernel.cu
      sgemm_kernel.cu

test/
  test_gemm_gpu.c
  bench_gemm_gpu.c
```

各文件职责如下：

| 文件 | 职责 |
|------|------|
| `src/config/ampere.h` | Ampere GPU tile 参数和 GPU dispatch 阈值 |
| `src/kernel/cuda/gpu_common.cuh` | CUDA 公共宏、错误检查、辅助函数 |
| `src/kernel/cuda/gemm_gpu.h` | C/CUDA 共享的 GPU 后端接口声明 |
| `src/kernel/cuda/gpu_init.cu` | GPU 检测、handle 创建销毁、默认 GPU GEMM 包装入口 |
| `src/kernel/cuda/dgemm_kernel.cu` | DGEMM v0-v3 CUDA kernel、device-resident API 和版本化 host wrapper |
| `src/kernel/cuda/sgemm_kernel.cu` | SGEMM v0-v4 CUDA kernel、device-resident API 和版本化 host wrapper |
| `src/api/gemm_gpu_dispatch.c` | CPU API 层使用的 GPU 自动分发状态管理 |
| `test/test_gemm_gpu.c` | GPU correctness tests，覆盖版本、转置、非对齐尺寸和 device API smoke test |
| `test/bench_gemm_gpu.c` | GPU benchmark，包含 DGEMM v0-v3、SGEMM v0-v4 和 cuBLAS 对比 |

---

## 3. 构建系统架构

### 3.1 CUDA 自动启用

`Makefile` 中新增：

```makefile
NVCC    ?= nvcc
CUDA_FLAGS ?= -O2 -arch=sm_86 -Isrc

HAS_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo yes || echo no)
USE_CUDA ?= $(if $(filter yes,$(HAS_NVCC)),1,0)
```

含义：

```text
如果系统能找到 nvcc，默认 USE_CUDA=1
如果找不到 nvcc，默认 USE_CUDA=0
用户也可以显式指定 USE_CUDA=0 强制 CPU-only 构建
```

CUDA 启用时：

```makefile
CFLAGS += -DUSE_CUDA -I$(CUDA_HOME)/include
OBJS += $(SRCS_CUDA:.cu=.o)
LDFLAGS += -lcudart -lcublas -L$(CUDA_HOME)/lib64
```

这样 C 文件可以通过 `#ifdef USE_CUDA` 判断是否启用 GPU 代码。

### 3.2 CPU-only 构建兼容

纯 CPU 构建命令：

```bash
make clean && make lib USE_CUDA=0
```

该路径已经验证通过。CUDA 源文件不会参与编译，最终库仍然只包含原 CPU 后端和一个无操作的 GPU dispatch stub。

---

## 4. API 集成方式

### 4.1 原有 CPU API 保持不变

用户侧 API 没有变化：

```c
void my_dgemm(char transa, char transb,
              int m, int n, int k,
              double alpha, const double *A, int lda,
                            const double *B, int ldb,
              double beta,        double *C, int ldc);

void my_sgemm(char transa, char transb,
              int m, int n, int k,
              float alpha,  const float *A, int lda,
                            const float *B, int ldb,
              float beta,         float *C, int ldc);
```

也就是说，外部调用方不需要知道 GPU 后端是否存在。GPU 分发是内部自动行为。

### 4.2 DGEMM 分发路径

`src/api/dgemm.c` 在参数检查和 `alpha/k/beta` 快速返回之后，进入原 CPU config 选择之前，新增 GPU 尝试路径：

```c
#ifdef USE_CUDA
if (gpu_should_dispatch(m, n, k, 0) && gpu_dispatch_init() == 0) {
    if (gpu_dgemm(gpu_dispatch_handle(), transa, transb, m, n, k,
                  alpha, A, lda, B, ldb, beta, C, ldc) == 0) {
        return;
    }
}
#endif
```

含义：

```text
1. 先判断矩阵尺寸是否达到 GPU 阈值
2. 再初始化 GPU dispatch 状态
3. GPU 计算成功则直接返回
4. GPU 不可用或调用失败则继续走原 CPU 路径
```

### 4.3 SGEMM 分发路径

`src/api/sgemm.c` 同样在 CPU config 选择之前新增 GPU 尝试路径：

```c
#ifdef USE_CUDA
if (gpu_should_dispatch(m, n, k, 1) && gpu_dispatch_init() == 0) {
    if (gpu_sgemm(gpu_dispatch_handle(), transa, transb, m, n, k,
                  alpha, A, lda, B, ldb, beta, C, ldc) == 0) {
        return;
    }
}
#endif
```

其中 precision 参数约定为：

```text
0 -> DGEMM
1 -> SGEMM
```

---

## 5. GPU Dispatch 设计

### 5.1 Dispatch 状态

`src/api/gemm_gpu_dispatch.c` 中维护进程级 GPU 状态：

```c
static gpu_gemm_handle_t *gpu_handle = 0;
static int gpu_available = -1;
```

状态含义：

```text
gpu_available = -1  尚未检测
gpu_available = 0   GPU 不可用
gpu_available = 1   GPU 可用
```

### 5.2 分发阈值

阈值定义在 `src/config/ampere.h`：

```c
#define GPU_DGEMM_DISPATCH_MIN 512
#define GPU_SGEMM_DISPATCH_MIN 256
```

当前策略：

```text
DGEMM: m >= 512 && n >= 512 && k >= 512 时才尝试 GPU
SGEMM: m >= 256 && n >= 256 && k >= 256 时才尝试 GPU
```

这样做是为了避免小矩阵在 GPU 上被 PCIe 拷贝、kernel launch 和 cudaMalloc/cudaFree 开销吞噬收益。

### 5.3 CPU fallback

GPU fallback 逻辑非常保守：

```text
如果 GPU 不可用 -> CPU
如果 GPU init 失败 -> CPU
如果 GPU GEMM 返回错误 -> CPU
如果 USE_CUDA=0 -> CPU
如果矩阵尺寸低于阈值 -> CPU
```

这保证了新增 GPU 后端不会破坏原 CPU 后端的可用性。

---

## 6. CUDA Host Wrapper 设计

### 6.1 GPU 检测接口

`src/kernel/cuda/gpu_init.cu` 暴露 C 接口：

```c
int gpu_is_available(void);
int gpu_get_compute_capability(void);
size_t gpu_get_memory(void);
```

这些接口用于：

```text
dispatch 初始化
测试输出 GPU 信息
benchmark 输出设备信息
```

### 6.2 Handle 抽象

当前定义了：

```c
typedef struct gpu_gemm_handle_t gpu_gemm_handle_t;
```

内部当前只包含 CUDA stream：

```c
struct gpu_gemm_handle_t {
    cudaStream_t stream;
};
```

需要注意：当前 host wrapper 尚未真正复用 device buffer。版本化 host-wrapper 调用仍然每次执行：

```text
cudaMalloc dA/dB/dC
cudaMemcpy H2D
launch kernel
cudaMemcpy D2H
cudaFree dA/dB/dC
```

因此 host-wrapper API 仍然是端到端路径，适合保持兼容性和 correctness 测试。

### 6.3 Device-resident API

为了让 MyBLAS benchmark 与 cuBLAS 使用相同计时口径，CUDA 后端新增只接收 device pointer 的版本化 API：

```c
int gpu_dgemm_device_version(int version,
                             char transa, char transb,
                             int m, int n, int k,
                             double alpha, const double *dA, int lda,
                             const double *dB, int ldb,
                             double beta, double *dC, int ldc);

int gpu_sgemm_device_version(int version,
                             char transa, char transb,
                             int m, int n, int k,
                             float alpha, const float *dA, int lda,
                             const float *dB, int ldb,
                             float beta, float *dC, int ldc);
```

语义：

```text
1. dA/dB/dC 必须已经是当前 CUDA device 上的有效 device pointer
2. API 只负责 version dispatch、kernel launch 和 cudaGetLastError
3. API 不执行 cudaMalloc/cudaFree
4. API 不执行 H2D/D2H cudaMemcpy
5. API 不调用 cudaDeviceSynchronize
```

`gpu_dgemm_version()` 和 `gpu_sgemm_version()` 仍保留 host-wrapper 行为，内部改为分配/copy 后调用对应 device API。这样外部兼容性不变，而 benchmark 可以绕过重复分配和拷贝开销。

### 6.4 默认版本选择

当前 API 默认调用：

```c
gpu_dgemm(...) -> gpu_dgemm_version(3, ...)
gpu_sgemm(...) -> gpu_sgemm_version(2, ...)
```

SGEMM 默认选择 v2，而不是 v4。原因是 v4 使用 Tensor Core / TF32，不应静默替代默认 FP32 语义。v4 通过 `gpu_sgemm_version(4, ...)` 或 `gpu_sgemm_device_version(4, ...)` 显式调用。

从实测结果看，`v3` 不是所有场景下最快：

```text
DGEMM 大矩阵：v1/v2 往往不低于 v3
SGEMM 大矩阵：v2 明显好于 v3
```

后续可继续考虑按精度和尺寸选择默认版本，但 SGEMM Tensor Core / TF32 fast path 应继续保持显式版本语义。

---

## 7. CUDA Kernel 架构

### 7.1 v0 Naive Kernel

v0 的设计是：

```text
一个 CUDA thread 计算一个 C[row, col]
每个 thread 独立遍历 K 维
不使用 shared memory
不做 register blocking
```

伪代码：

```c
sum = 0;
for (p = 0; p < k; p++) {
    sum += A[row, p] * B[p, col];
}
C[row, col] = beta * C[row, col] + alpha * sum;
```

用途：

```text
正确性基线
性能下限参考
验证 transpose / leading dimension / edge 逻辑
```

### 7.2 v1 Shared Memory Tiling

v1 的设计是：

```text
每个 block 计算一个 16x16 C tile
每轮从 global memory 加载 A/B tile 到 shared memory
block 内线程复用 shared memory 中的 A/B 数据
```

伪代码：

```c
for (kk = 0; kk < k; kk += TILE) {
    sA[ty][tx] = A[row, kk + tx];
    sB[ty][tx] = B[kk + ty, col];
    __syncthreads();

    for (p = 0; p < TILE; p++) {
        sum += sA[ty][p] * sB[p][tx];
    }
    __syncthreads();
}
```

v1 对 DGEMM 大矩阵效果较好，因为 RTX 3060 Laptop 的 FP64 峰值较低，普通 CUDA core shared-memory tiling 已经能接近 cuBLAS 的 FP64 上限。

### 7.3 v2 Register Blocking

v2 的设计是：

```text
每个 block 计算 128x128 C tile
每个 thread 计算 8x8 输出子块
C 子块在寄存器中累加
A/B tile 放入 shared memory
```

参数来自 `src/config/ampere.h`：

```c
#define GPU_DGEMM_TILE_M    128
#define GPU_DGEMM_TILE_N    128
#define GPU_DGEMM_TILE_K    8
#define GPU_DGEMM_THREAD_M  8
#define GPU_DGEMM_THREAD_N  8

#define GPU_SGEMM_TILE_M    128
#define GPU_SGEMM_TILE_N    128
#define GPU_SGEMM_TILE_K    16
#define GPU_SGEMM_THREAD_M  8
#define GPU_SGEMM_THREAD_N  8
```

block 维度：

```text
blockDim.x = TILE_N / THREAD_N = 16
blockDim.y = TILE_M / THREAD_M = 16
threads/block = 256
```

每个 thread 维护：

```text
acc[8][8]
```

也就是每个 thread 计算 64 个 C 元素。

### 7.4 v3 Double Buffer + Padding

v3 当前在 v2 基础上使用：

```text
两组 shared memory buffer
shared memory padding，降低 bank conflict 风险
```

当前 shared memory 形态类似：

```c
__shared__ double sA[2][TM][TK + 1];
__shared__ double sB[2][TK][TN + 1];
```

需要注意：当前 v3 尚未真正使用 Ampere `cp.async`。它是结构上的 double buffer + padding 版本，而不是完整异步流水版本。因此当前 v3 不一定比 v2 快。

### 7.5 SGEMM v4 Tensor Core / TF32

SGEMM v4 是显式 fast path，使用 Ampere Tensor Core 的 TF32 模式：

```text
输入: float
乘法: TF32 precision
累加: FP32 accumulator
输出: float
```

当前 kernel 设计：

```text
block tile: 128x128
K tile: 32
threads/block: 256
warps/block: 8
每个 warp 计算 64x32 输出区域
每个 warp 维护 4x2 个 16x16 WMMA accumulator fragment
```

实现要点：

```text
1. shared memory 中保存经过 __float_to_tf32 rounding 的 A/B tile
2. A/B WMMA fragment 均按 row_major 读取当前 shared memory layout
3. 边界元素在 cooperative load 阶段填 0，支持任意 m/n/k
4. 支持 NN/NT/TN/TT 转置组合
5. alpha=1、beta=0 且 ldc 4 对齐时使用 direct-store fast path
```

精度语义：

```text
v0-v3: 普通 FP32 CUDA-core 路径
v4: Tensor Core / TF32 multiply + FP32 accumulate
```

因此 v4 correctness 使用独立 TF32 容差，不与严格 FP32 容差混用。

---

## 8. 转置和 Leading Dimension 支持

GPU kernel 内部通过 helper 函数处理 BLAS column-major 下的转置：

```c
load_a(A, lda, transa, row, p)
load_b(B, ldb, transb, p, col)
```

逻辑：

```text
transa == N: A[row + p * lda]
transa == T: A[p + row * lda]

transb == N: B[p + col * ldb]
transb == T: B[col + p * ldb]
```

因此当前 GPU 后端支持：

```text
NN
NT
TN
TT
```

也支持非紧致 leading dimension，例如：

```text
lda > logical rows of A
ldb > logical rows of B
ldc > m
```

正确性测试中已经覆盖非对齐尺寸和转置组合。

---

## 9. 与原 CPU 架构的兼容方式

### 9.1 原 CPU driver 和 kernel table 不变

新增 GPU 后端没有改变原 CPU 计算链路：

```text
my_dgemm / my_sgemm
  -> 参数检查
  -> alpha/k/beta 快速路径
  -> GPU 尝试路径（仅 USE_CUDA）
  -> 原 AVX2/generic config 选择
  -> 原 packing buffer 分配
  -> 原 gemm_driver_double / gemm_driver_float
  -> 原 kernel table dispatch
```

也就是说，只要 GPU 不接管，后面的 CPU 路径与原架构保持一致。

### 9.2 编译期兼容

当 `USE_CUDA=0` 时：

```text
src/kernel/cuda/*.cu 不参与编译
src/api/dgemm.c / sgemm.c 中的 GPU 代码被 #ifdef USE_CUDA 排除
src/api/gemm_gpu_dispatch.c 编译为 stub
```

stub 行为：

```c
int gpu_dispatch_init(void) { return -1; }
void gpu_dispatch_cleanup(void) { }
gpu_gemm_handle_t *gpu_dispatch_handle(void) { return 0; }
int gpu_should_dispatch(...) { return 0; }
```

因此 CPU-only 构建完全不会依赖 CUDA runtime、cuBLAS 或 CUDA headers。

### 9.3 运行时兼容

当 `USE_CUDA=1` 但运行时 GPU 不可用：

```text
gpu_is_available() 返回 0
gpu_should_dispatch() 返回 0
my_dgemm / my_sgemm 直接继续 CPU 路径
```

当 GPU 初始化失败或 GPU kernel 调用失败：

```text
GPU wrapper 返回非 0
API 继续执行原 CPU 路径
```

该设计保证了 GPU 后端是增强能力，不是原 CPU 路径的硬依赖。

---

## 10. 测试覆盖

### 10.1 原 CPU 回归测试

命令：

```bash
make test
```

当前结果：

```text
20 passed, 0 failed
```

该测试覆盖：

```text
DGEMM 转置组合
DGEMM alpha/beta/k=0 边界
DGEMM 非对齐尺寸
DGEMM 64/128/256 大小
SGEMM 基础与非对齐尺寸
```

### 10.2 GPU 正确性测试

命令：

```bash
make test_gpu
```

当前结果：

```text
CUDA GPU detected: SM 86, 6.0 GiB
GPU GEMM tests: 123 passed, 0 failed
```

覆盖范围：

```text
DGEMM v0/v1/v2/v3
SGEMM v0/v1/v2/v3/v4
NN/NT/TN/TT
非对齐尺寸 5x7x11、17x13x19
64x64x64
非紧致 lda/ldb/ldc
alpha/beta 组合
SGEMM v4 TF32 独立容差
device-resident API smoke test
```

### 10.3 CPU-only 构建验证

命令：

```bash
make clean && make lib USE_CUDA=0 && make test USE_CUDA=0
```

当前结果：

```text
20 passed, 0 failed
```

---

## 11. 当前性能结果

GPU benchmark 命令：

```bash
make bench_gpu
```

测试设备：

```text
GPU: SM 86, 6.0 GiB
```

### 11.1 DGEMM 性能

当前 MyBLAS 数字为 device-resident benchmark：host 侧提前分配并拷贝 `dA/dB/dC`，计时段只包含 repeated kernel launch 和最终同步，与 cuBLAS 计时口径一致。

| n | cuBLAS GFLOPS | v0 | v1 | v2 | v3 |
|---:|--------------:|---:|---:|---:|---:|
| 128 | 62.08 | 40.50 | 92.51 | 4.56 | 4.48 |
| 256 | 110.29 | 54.82 | 156.18 | 20.45 | 20.65 |
| 512 | 99.24 | 66.89 | 137.19 | 74.73 | 80.73 |
| 1024 | 162.65 | 88.58 | 171.22 | 132.66 | 131.31 |
| 2048 | 159.16 | 86.74 | 162.25 | 168.37 | 170.95 |

DGEMM 结论：

```text
n=2048 时，v2/v3 约 168-171 GFLOPS，略高于本次 cuBLAS measured throughput。
考虑 RTX 3060 Laptop FP64 吞吐较低，这个结果已经比较接近实际上限。
当前默认 v3 在大矩阵下表现合理。
```

### 11.2 SGEMM 性能

SGEMM v4 为 Tensor Core / TF32 fast path，使用显式 version 调用。cuBLAS SGEMM 默认允许 TF32 Tensor Core，因此 v4 与 cuBLAS 默认路径精度口径更接近；v0-v3 仍是普通 FP32 CUDA-core 路径。

| n | cuBLAS GFLOPS | v0 | v1 | v2 | v3 | v4 TC/TF32 |
|---:|--------------:|---:|---:|---:|---:|------------:|
| 128 | 123.87 | 41.54 | 199.92 | 50.72 | 51.85 | 101.51 |
| 256 | 1224.61 | 63.16 | 211.99 | 249.92 | 251.34 | 513.85 |
| 512 | 3352.65 | 68.38 | 623.06 | 1044.50 | 1011.69 | 1868.46 |
| 1024 | 5541.41 | 89.12 | 628.32 | 1691.55 | 1669.81 | 3131.81 |
| 2048 | 4585.21 | 87.18 | 575.16 | 2422.12 | 2202.29 | 3314.08 |

SGEMM 结论：

```text
n=2048 时，v4 Tensor Core / TF32 达到 3314 GFLOPS，约为 cuBLAS 的 72.3%。
v4 明显快于普通 FP32 CUDA-core 路径 v0-v3。
device-resident 口径去除了 host-wrapper repeated cudaMalloc/cudaMemcpy/cudaFree 开销，更能反映 kernel 本身吞吐。
```

---

## 12. 已修复的兼容性问题

### 12.1 AVX2 DGEMM transb 打包 bug

在测试过程中发现原有 AVX2 DGEMM `B^T` 打包存在 bug：

```text
dgemm_pack_b_tn_avx2 使用 4 列向量化写入
但 packed B 的内存布局是按列连续，每列间 stride 为 k
原向量化写入把跨列数据当成连续地址写入，破坏 NT/TT 结果
```

修复方式：

```text
将 dgemm_pack_b_tn_avx2 改回标量正确写回
```

影响：

```text
修复后 make test 中 DGEMM NT/TT 全部通过
```

### 12.2 FP32 测试近零误差判定

原 `test/test_gemm.c` 和新增 `test/test_gemm_gpu.c` 中，FP32 近零结果只按相对误差判断，容易把 `1e-7` 量级的正常浮点误差误判为失败。

修复方式：

```text
使用绝对误差或相对误差任一满足即通过
```

形式：

```c
if (diff > eps && diff / denom > eps) fail;
```

---

## 13. 当前限制

### 13.1 Host wrapper 没有复用 GPU buffer

当前版本化 wrapper 每次调用都执行：

```text
cudaMalloc
cudaMemcpy H2D
kernel launch
cudaMemcpy D2H
cudaFree
```

这会明显影响端到端性能，尤其是小矩阵。当前已经新增 device-resident API，并将 benchmark 切换到该口径；但普通 host-wrapper 调用仍然没有缓存 buffer。

后续应把 `gpu_gemm_handle_t` 扩展为：

```c
typedef struct {
    double *d_A, *d_B, *d_C;
    float *s_A, *s_B, *s_C;
    size_t capacity_a, capacity_b, capacity_c;
    cudaStream_t stream;
} gpu_gemm_handle_t;
```

然后按容量复用 device buffer。

### 13.2 v3 尚未真正使用 cp.async

当前 v3 是 double-buffer shared memory 结构，但没有真正使用 Ampere `cp.async` 异步拷贝。实际效果不稳定，甚至可能慢于 v2。

后续如果要让 v3 成为真正优化版，需要实现：

```text
cp.async global -> shared
pipeline commit / wait
计算当前 tile 时异步加载下一 tile
减少 __syncthreads 成本
```

### 13.3 SGEMM v4 仍不是极限 Tensor Core kernel

SGEMM 已有 Tensor Core / TF32 v4，但当前仍是 WMMA 第一版实现，尚未达到 cuBLAS/CUTLASS 级别的极限吞吐。

后续如果目标是进一步追 cuBLAS SGEMM，需要考虑：

```text
NN 专用 Tensor Core kernel
inline PTX mma.sync
更高效的 warp-level tiling
vectorized global/shared load
更精细的 shared memory layout
cp.async double buffering
```

### 13.4 默认版本选择需要调整

当前自动 dispatch 默认使用：

```text
gpu_dgemm -> v3
gpu_sgemm -> v2
```

SGEMM v4 不作为默认路径，因为 TF32 不是严格 FP32 乘法语义。后续如果用户明确希望默认追求最大吞吐，可以增加显式配置开关，例如：

```text
MYBLAS_SGEMM_DEFAULT_TF32=1
或运行时环境变量 GPU_SGEMM_USE_TENSOR_CORES=1
```

---

## 14. 后续优化建议

### 14.1 短期

| 优先级 | 任务 | 目标 |
|-------:|------|------|
| 1 | 使用 CUDA event timing | 降低 CPU wall-time 噪声 |
| 2 | 实现 host-wrapper buffer cache | 降低 `my_dgemm/my_sgemm` 端到端调用开销 |
| 3 | 增加自动 dispatch benchmark | 验证 `my_dgemm/my_sgemm` 端到端收益 |
| 4 | 增加 cuBLAS pedantic FP32 对照 | 区分 TF32 与严格 FP32 性能口径 |

### 14.2 中期

| 优先级 | 任务 | 目标 |
|-------:|------|------|
| 1 | SGEMM v4 NN 专用 kernel | 减少转置分支和地址计算 |
| 2 | v4 vectorized load/store | 改善 global/shared memory 带宽利用 |
| 3 | v4 cp.async double buffering | 提高 Tensor Core 数据供给效率 |
| 4 | v3 引入真正 cp.async | 提升普通 CUDA-core tiled kernel 效率 |

### 14.3 长期

| 优先级 | 任务 | 目标 |
|-------:|------|------|
| 1 | inline PTX mma.sync / CUTLASS-style kernel | 进一步接近 cuBLAS SGEMM 吞吐 |
| 2 | CUDA Graph 或 persistent handle | 降低重复调用开销 |
| 3 | 更完整的 GPU memory pool | 支持高频 GEMM 调用 |
| 4 | 多 GPU 或 stream 并发策略 | 扩展更复杂 workload |

---

## 15. 总结

当前 CUDA 后端已经完成从无到有的架构接入：

```text
1. CUDA 编译系统已接入，支持自动 nvcc 检测和 USE_CUDA=0 CPU-only 构建
2. DGEMM 有 v0-v3 四级 GPU kernel，SGEMM 有 v0-v4 五级 GPU kernel
3. my_dgemm / my_sgemm 已支持大矩阵自动 GPU dispatch
4. GPU 不可用或失败时可自动回退 CPU
5. 原 CPU driver、packing、kernel table 架构保持兼容
6. CPU 和 GPU correctness tests 均通过
7. GPU benchmark 已切换为 device-resident 口径，可与 cuBLAS 公平对比
8. SGEMM v4 已实现 Tensor Core / TF32 fast path
```

当前性能判断：

```text
DGEMM: 大矩阵已经较好，n=2048 时 v2/v3 接近或略高于本次 cuBLAS measured throughput
SGEMM: v4 Tensor Core / TF32 在 n=2048 达到 3314 GFLOPS，约为 cuBLAS 默认路径 72.3%
```

当前最重要的工程结论是：

```text
GPU 后端已以保守 fallback 的方式兼容原 CPU 架构；
SGEMM v4 已证明 Tensor Core 路径显著优于普通 FP32 CUDA-core 版本；
后续优化可以集中在 host-wrapper buffer 复用、CUDA event timing、v4 NN 专用 kernel、cp.async 和 mma.sync，
而不需要重构原有 CPU GEMM 主线。
```
