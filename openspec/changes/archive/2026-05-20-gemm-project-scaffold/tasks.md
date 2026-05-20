## 1. Project Setup

- [x] 1.1 Create directory structure: src/api/, src/driver/, src/kernel/generic/, src/kernel/avx2/, src/config/, include/, test/
- [x] 1.2 Create include/myblas.h with public API declarations (my_dgemm, my_sgemm)
- [x] 1.3 Create Makefile with targets: lib, test, bench, clean

## 2. Data Structures and Config

- [x] 2.1 Define gemm_arg_t struct in src/driver/gemm_internal.h (m, n, k, A, B, C, lda, ldb, ldc, alpha, beta, transa, transb, nthreads)
- [x] 2.2 Define gemm_config_t struct (P, Q, R, MR, NR)
- [x] 2.3 Define gemm_kernel_t struct with function pointers (kernel, pack_a_nn, pack_a_tn, pack_b_nn, pack_b_tn, beta)
- [x] 2.4 Create src/config/generic.h with default blocking parameters for generic kernel (double: P=256, Q=128, R=4096, MR=4, NR=4; float: MR=8, NR=4)

## 3. Kernel Layer — Generic C Implementation

- [x] 3.1 Implement src/kernel/generic/dgemm_kernel.c: pure-C micro-kernel (MR=4, NR=4, k-loop with multiply-accumulate)
- [x] 3.2 Implement src/kernel/generic/sgemm_kernel.c: pure-C micro-kernel (MR=8, NR=4)
- [x] 3.3 Implement src/kernel/generic/dgemm_pack.c: pack_a_nn, pack_a_tn, pack_b_nn, pack_b_tn for double
- [x] 3.4 Implement src/kernel/generic/sgemm_pack.c: packing functions for float
- [x] 3.5 Implement src/kernel/generic/dgemm_beta.c: C = beta * C for double
- [x] 3.6 Implement src/kernel/generic/sgemm_beta.c: C = beta * C for float
- [x] 3.7 Create src/kernel/generic/kernel_init.c: initialize gemm_kernel_generic_double and gemm_kernel_generic_float tables

## 4. Driver Layer — Blocking Loop

- [x] 4.1 Implement src/driver/gemm_driver.c: three-level blocking loop (js/ls/is) with packing dispatch and kernel invocation
- [x] 4.2 Implement adaptive blocking: when remaining M > P but < 2*P, split into MR-aligned chunks
- [x] 4.3 Implement beta scaling call before main loop (skip when beta == 1.0)
- [x] 4.4 Implement correct pointer offset calculation for C sub-blocks

## 5. API Layer — Parameter Handling

- [x] 5.1 Implement src/api/dgemm.c: my_dgemm with parameter validation, special-case handling, and call to gemm_driver
- [x] 5.2 Implement src/api/sgemm.c: my_sgemm (same logic, float precision)
- [x] 5.3 Implement special cases: m=0/n=0 return, k=0 skip, alpha=0 skip, beta=0/1 fast path
- [x] 5.4 Implement transpose dispatch: select correct pack_a/pack_b based on transa/transb

## 6. Thread Layer — Multi-threading

- [x] 6.1 Implement src/driver/gemm_thread.c: gemm_parallel with N-dimension split using pthreads
- [x] 6.2 Implement thread-private packing buffer allocation (sa/sb per thread)
- [x] 6.3 Implement thread threshold check: single-thread when m*n*k < MYBLAS_THREAD_THRESHOLD
- [x] 6.4 Implement MYBLAS_NUM_THREADS environment variable reading

## 7. Testing

- [x] 7.1 Create test/test_gemm.c: correctness test for dgemm NN against naive reference (small matrices)
- [x] 7.2 Add tests for all 4 transpose combinations (NN, NT, TN, TT)
- [x] 7.3 Add tests for edge cases: alpha=0, beta=0, beta=1, m=1, n=1, k=0
- [x] 7.4 Add tests for non-aligned dimensions (m=5, n=7, k=11)
- [x] 7.5 Add tests for sgemm (float precision)
- [x] 7.6 Create test/bench_gemm.c: GFLOPS benchmark for dgemm (128, 256, 512, 1024)

## 8. Build and Integration

- [x] 8.1 Write Makefile with proper dependencies, -O2 -mavx2 flags for AVX2 targets
- [x] 8.2 Verify build produces libmyblas.a (static library)
- [x] 8.3 Verify `make test` runs all correctness tests and passes
- [x] 8.4 Verify `make bench` runs benchmark and reports GFLOPS
