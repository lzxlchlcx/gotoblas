# OpenBLAS/GotoBLAS 接口设计文档

基于 OpenBLAS 源码（GotoBLAS2 1.13 BSD 分支）的实际代码验证。

## 文档结构

| 文件 | 内容 |
|------|------|
| `01-architecture.md` | 整体架构概览 |
| `02-interface-layer.md` | 公共接口层（interface/） |
| `03-driver-layer.md` | 驱动层（driver/level3/） |
| `04-kernel-layer.md` | 内核层（kernel/） |
| `05-data-structures.md` | 核心数据结构 |
| `06-gemm-parameters.md` | GEMM 阻塞参数 |
| `07-dgemm-flow.md` | DGEMM 调用流程详解 |
| `08-myblas-blocking-tutorial.md` | MyBLAS 分块算法新手教学 |

## 源码目录结构

```
OpenBLAS/
├── interface/          # Layer 1: 公共 BLAS/CBLAS 接口
├── driver/             # Layer 2: 调度与阻塞逻辑
│   ├── level3/         #   Level-3 BLAS 驱动
│   └── others/         #   线程管理、内存管理
├── kernel/             # Layer 3: 计算内核
│   ├── x86_64/         #   x86-64 架构内核
│   ├── arm64/          #   ARM64 架构内核
│   ├── generic/        #   通用 C 实现
│   └── ...
├── common.h            # 核心头文件
├── common_macro.h      # 宏定义（精度类型映射）
├── common_param.h      # DYNAMIC_ARCH 函数指针表
├── common_level3.h     # Level-3 内核声明
├── param.h             # 各 CPU 的阻塞参数
├── cblas.h             # CBLAS 公共头文件
└── lapack/             # LAPACK 实现
```
