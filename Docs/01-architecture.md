# 01 - 整体架构

OpenBLAS 采用三层分层架构，通过宏和函数指针实现多精度、多转置、多 CPU 的统一调度。

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Public Interface (interface/)                          │
│  dgemm_, sgemm_, cblas_dgemm ...                                │
│  职责: 参数校验 → 转置映射 → GEMV 优化转发 → 线程数决策 → 调度   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Driver (driver/level3/)                               │
│  level3.c / level3_thread.c                                     │
│  职责: 三级阻塞循环 (js/ls/is) → packing → 调用 kernel          │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Kernel (kernel/{arch}/)                               │
│  dgemm_kernel_4x8_haswell.S, gemm_ncopy_8.S ...                 │
│  职责: 微内核 GEBP 计算、矩阵 packing、beta 缩放                │
└─────────────────────────────────────────────────────────────────┘
```

## 编译时多态机制

OpenBLAS 通过编译时宏实现同一份代码支持多种精度：

- 每个精度单独编译一次（如 `dgemm.c` 编译时定义 `DOUBLE`）
- `common_macro.h` 将通用宏（如 `GEMM_KERNEL_N`）映射到具体函数（如 `DGEMM_KERNEL`）
- `common_d.h` 定义 `DGEMM_KERNEL` → `dgemm_kernel`（BLASFUNC 包装）

验证来源：`common_macro.h:175-191`, `common_d.h`

## DYNAMIC_ARCH 机制

当定义 `DYNAMIC_ARCH` 时，所有内核函数通过 `gotoblas_t` 结构体的函数指针调用，运行时自动检测 CPU 并选择最优内核。

验证来源：`common_param.h`（gotoblas_t 结构体定义），`driver/others/dynamic.c`
