# 06 - GEMM 阻塞参数

## 参数定义

GotoBLAS 的核心优化在于将矩阵乘法分解为适合各级缓存的小块。三个关键阻塞参数：

| 参数 | 含义 | 适配缓存 |
|------|------|----------|
| `GEMM_P` | M 维度阻塞（行块大小） | L2 |
| `GEMM_Q` | K 维度阻塞（深度块大小） | L2 |
| `GEMM_R` | N 维度阻塞（列块大小） | L3 |

两个微内核阻塞参数：

| 参数 | 含义 | 适配 |
|------|------|------|
| `GEMM_UNROLL_M` | 寄存器行阻塞 | 寄存器 |
| `GEMM_UNROLL_N` | 寄存器列阻塞 | 寄存器 |

验证来源：`param.h`, `driver/level3/level3.c:289`

## Haswell 参数（x86-64）

### SGEMM (float)

```c
SGEMM_DEFAULT_P = 320       // M 阻塞
SGEMM_DEFAULT_Q = 320       // K 阻塞
SGEMM_DEFAULT_R = sgemm_r   // N 阻塞（动态）
SGEMM_DEFAULT_UNROLL_M = 8  // 微内核行数
SGEMM_DEFAULT_UNROLL_N = 4  // 微内核列数
// 内核: sgemm_kernel_8x4_haswell_2.c
```

验证来源：`param.h:1608,1617,1666-1673`

### DGEMM (double)

```c
DGEMM_DEFAULT_P = 512       // M 阻塞
DGEMM_DEFAULT_Q = 256       // K 阻塞
DGEMM_DEFAULT_R = 13824     // N 阻塞
DGEMM_DEFAULT_UNROLL_M = 4  // 微内核行数
DGEMM_DEFAULT_UNROLL_N = 8  // 微内核列数
// 内核: dgemm_kernel_4x8_haswell.S
```

验证来源：`param.h:1609,1618,1624,1567,1574`

### ZGEMM (complex double)

```c
ZGEMM_DEFAULT_P = 192       // M 阻塞
ZGEMM_DEFAULT_Q = 192       // K 阻塞
ZGEMM_DEFAULT_R = zgemm_r   // N 阻塞（动态）
ZGEMM_DEFAULT_UNROLL_M = 4  // 微内核行数
ZGEMM_DEFAULT_UNROLL_N = 2  // 微内核列数
// 内核: zgemm_kernel_4x2_haswell.c
```

验证来源：`param.h:1611,1621,1626,1570,1576`

## 缓存适配原理

```
P × Q × sizeof(element) ≈ L2 缓存大小

DGEMM 为例:
  P = 512, Q = 256, sizeof(double) = 8
  512 × 256 × 8 = 1MB ... 但这是 A 的大小
  实际上 P × Q 是 A_packed 的大小，需要放入 L2

Haswell L2 = 256KB:
  实际 A_packed = P × Q × 8 = 512 × 256 × 8 = 1MB
  这里 P 和 Q 不是同时全部放入 L2
  而是每次放入 P × min_l（min_l ≤ Q）的 A 块
  加上 min_l × min_jj 的 B 块
  总共约 P × Q × 8 的工作集放入 L2
```

## 阻塞循环与缓存层次

```
L3 缓存 ←── GEMM_R（N 维度）
  │
  └── L2 缓存 ←── GEMM_P × GEMM_Q（M × K 维度）
        │
        └── L1 缓存 ←── GEMM_UNROLL_M × GEMM_UNROLL_N × K
              │
              └── 寄存器 ←── GEMM_UNROLL_M × GEMM_UNROLL_N
```

## CPU 特定参数一览

| CPU | DGEMM_P | DGEMM_Q | DGEMM_R | UNROLL_M | UNROLL_N | 内核 |
|-----|---------|---------|---------|----------|----------|------|
| Haswell | 512 | 256 | 13824 | 4 | 8 | 4x8_haswell |
| Skylake-X | 512 | 256 | 13824 | 4 | 8 | 4x8_skylakex |
| Zen | 512 | 256 | 13824 | 4 | 8 | (同 Haswell) |
| Sandy Bridge | 504 | 256 | 8192 | 4 | 8 | 4x8_sandy |
| Nehalem | 504 | 128 | 8192 | 2 | 8 | 2x8_nehalem |

验证来源：`param.h` 各 CPU 段

## GEMM_OFFSET_A / GEMM_OFFSET_B

用于避免缓存行冲突的偏移量：

```c
#define GEMM_DEFAULT_OFFSET_A     0
#define GEMM_DEFAULT_OFFSET_B     0
#define GEMM_DEFAULT_ALIGN (BLASLONG)0x03fffUL  // 16KB 对齐
```

某些 CPU（如 LoongArch64）需要特殊偏移：

```c
// driver/level3/interface/gemm.c 中
#if defined(ARCH_LOONGARCH64) && !defined(NO_AFFINITY)
  sa = (XFLOAT *)((BLASLONG)buffer + (WhereAmI() & 0xf) * GEMM_OFFSET_A);
#else
  sa = (XFLOAT *)((BLASLONG)buffer + GEMM_OFFSET_A);
#endif
```

验证来源：`param.h:1534-1536`, `interface/gemm.c:234-240`
