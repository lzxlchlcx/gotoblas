## 1. 阻塞参数配置

- [x] 1.1 创建 src/config/haswell.h：定义 AVX2 阻塞参数（dgemm: P=256, Q=256, R=4096, MR=4, NR=4; sgemm: MR=8, NR=4）和 gemm_config_avx2_double/float 函数
- [x] 1.2 修改 src/driver/gemm_internal.h：新增 extern 声明 gemm_kernel_avx2_double 和 gemm_kernel_avx2_float

## 2. AVX2 微内核实现

- [x] 2.1 实现 src/kernel/avx2/dgemm_kernel.c：AVX2 dgemm 微内核（MR=4, NR=4），使用 __m256d + _mm256_fmadd_pd，边界回退标量
- [x] 2.2 实现 src/kernel/avx2/sgemm_kernel.c：AVX2 sgemm 微内核（MR=8, NR=4），使用 __m256 + _mm256_fmadd_ps，边界回退标量

## 3. AVX2 打包函数

- [x] 3.1 实现 src/kernel/avx2/dgemm_pack.c：4 个打包函数的 AVX2 版本，使用 _mm256_loadu_pd/_mm256_storeu_pd 加速
- [x] 3.2 实现 src/kernel/avx2/sgemm_pack.c：4 个打包函数的 AVX2 版本，使用 _mm256_loadu_ps/_mm256_storeu_ps 加速

## 4. AVX2 Beta 缩放

- [x] 4.1 实现 src/kernel/avx2/dgemm_beta.c：AVX2 beta 缩放，使用 SIMD 批量清零/缩放
- [x] 4.2 实现 src/kernel/avx2/sgemm_beta.c：AVX2 beta 缩放（float 版本）

## 5. 内核表初始化

- [x] 5.1 创建 src/kernel/avx2/kernel_init.c：初始化 gemm_kernel_avx2_double 和 gemm_kernel_avx2_float 表
- [x] 5.2 创建 src/kernel/avx2/cpuid.c：实现 cpu_supports_avx2() 检测函数

## 6. API 层集成

- [x] 6.1 修改 src/api/dgemm.c：新增运行时 CPU 检测，根据结果选择 AVX2 或 generic 内核表和 config
- [x] 6.2 修改 src/api/sgemm.c：同上（float 版本）

## 7. 构建系统

- [x] 7.1 修改 Makefile：新增 avx2 源文件，使用条件编译（-mavx2 -mfma），仅在支持时编译 AVX2 目标
- [x] 7.2 验证 make lib 在支持 AVX2 的机器上成功编译
- [x] 7.3 验证 make test 全部通过（AVX2 内核）
- [x] 7.4 验证 make bench 报告 GFLOPS 提升
