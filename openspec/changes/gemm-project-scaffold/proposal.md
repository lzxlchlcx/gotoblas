## Why

We need a from-scratch implementation of the GotoBLAS GEMM algorithm (matrix multiplication C = αAB + βC) as a standalone library. The existing OpenBLAS codebase is massive (~10k files) and supports dozens of CPU architectures, dozens of BLAS functions, and complex build systems. For learning, experimentation, and targeted optimization, we need a minimal, clean implementation that focuses solely on GEMM (sgemm/dgemm) with single-thread and multi-thread support, following GotoBLAS's cache-oblivious blocking and micro-kernel architecture.

## What Changes

- Create a new `src/` directory with a clean, modular C project structure
- Implement the four-layer architecture: API → Driver → Kernel → Thread
- Provide `my_dgemm()` and `my_sgemm()` as the public API
- Implement GotoBLAS-style three-level blocking loop (R/Q/P) with explicit packing
- Provide a pure-C generic kernel as baseline, with placeholder for AVX2 assembly
- Implement pthread-based multi-threading with N-dimension parallelism
- Include correctness tests (vs CBLAS reference) and a GFLOPS benchmark
- Support runtime kernel selection via function pointer table

## Capabilities

### New Capabilities

- `gemm-api`: Public BLAS-style API for dgemm/sgemm with parameter validation and special-case handling (alpha=0, beta=0/1, m/n=0, GEMV forward)
- `gemm-driver`: GotoBLAS-style three-level blocking driver with R/Q/P parameters, packing dispatch, and micro-kernel invocation
- `gemm-kernel`: Micro-kernel interface (GEBP), matrix packing functions (4 transpose variants), beta scaling, with generic C baseline and pluggable kernel table
- `gemm-thread`: pthread-based parallel dispatch along N dimension, thread-private packing buffers, configurable thread count

### Modified Capabilities

(none - this is a greenfield project)

## Impact

- New `src/` directory tree (~20 files)
- New `test/` directory for correctness and performance tests
- New `Makefile` for build system
- Depends on: pthreads (for threading), optionally CBLAS (for test reference only)
- No changes to existing codebase (OpenBLAS directory is reference only)
