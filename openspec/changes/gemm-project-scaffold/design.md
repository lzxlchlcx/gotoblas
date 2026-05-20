## Context

We are building a standalone GEMM library from scratch, inspired by GotoBLAS/OpenBLAS. The reference OpenBLAS codebase (`OpenBLAS/`) is available for study but we are not modifying it. The project starts from an empty `src/` directory.

The core algorithm is GotoBLAS's three-level blocking with explicit matrix packing and a register-tiled micro-kernel (GEBP). The library must support both single-thread and multi-thread execution.

Key constraints:
- Only GEMM (sgemm/dgemm) — no other BLAS functions
- Must be self-contained (no dependency on OpenBLAS internals)
- Must be testable against standard CBLAS for correctness
- Target platform: Linux x86-64 (primary), with generic C fallback

## Goals / Non-Goals

**Goals:**
- Clean four-layer architecture: API → Driver → Kernel → Thread
- Correct implementation of GotoBLAS blocking algorithm (R/Q/P)
- Pluggable kernel table for different CPU targets
- pthread-based multi-threading with N-dimension parallelism
- Correctness tests and GFLOPS benchmark
- Minimal, readable code (~20 files total)

**Non-Goals:**
- No other BLAS functions (gemv, trsm, syrk, etc.)
- No complex arithmetic (no cgemm/zgemm)
- No LAPACK
- No DYNAMIC_ARCH runtime CPU detection (compile-time kernel selection is sufficient for now)
- No Windows support
- No CBLAS interface (just the Fortran-style `my_dgemm` / `my_sgemm`)

## Decisions

### D1: Four-layer architecture with explicit separation

**Decision**: Separate API, Driver, Kernel, and Thread into distinct directories/files.

**Rationale**: OpenBLAS mixes all layers in a single file via macro templates (`level3.c` is included by `gemm.c`). This is efficient for code generation but terrible for readability. Our project prioritizes clarity.

**Alternative considered**: Single-file implementation (like a textbook GEMM). Rejected because it doesn't scale to multi-threading and kernel swapping.

### D2: Function pointer table for kernel selection

**Decision**: Define a `gemm_kernel_t` struct containing function pointers for kernel, packing, and beta. Select at init time.

```c
typedef struct {
    gemm_kernel_func kernel;     // micro-kernel
    pack_func pack_a_nn;         // A packing (NoTrans)
    pack_func pack_a_tn;         // A packing (Trans)
    pack_func pack_b_nn;         // B packing (NoTrans)
    pack_func pack_b_tn;         // B packing (Trans)
    gemm_beta_func beta;         // C = beta * C
} gemm_kernel_t;
```

**Rationale**: Mirrors OpenBLAS's `gotoblas_t` but drastically simplified. Allows runtime kernel selection without #ifdef spaghetti.

**Alternative considered**: Compile-time kernel selection via `#include` of different `.c` files (like OpenBLAS's `KERNEL.HASWELL`). Rejected for now — function pointers are simpler and the performance cost is negligible for a learning project.

### D3: Blocking parameters as runtime configuration

**Decision**: Pass P/Q/R/MR/NR/dtb_entries via a `gemm_config_t` struct, not compile-time #define.

**Rationale**: Makes it easy to experiment with different blocking strategies without recompiling. OpenBLAS uses compile-time `param.h` which is faster but harder to iterate on.

**TLB constraint**: Blocking parameters MUST satisfy both cache size and TLB reach constraints:
- P × Q × sizeof(element) ≤ min(L2_cache, L1_DTLB_entries × page_size)
- Q × R × sizeof(element) ≤ min(L3_cache, L2_STLB_entries × page_size)

This is a key insight from the GotoBLAS paper: the blocking parameters are not just cache-blocking, they are also TLB-blocking. Without this, TLB misses can dominate even when cache hits are good.

### D4: pthread-based threading with N-dimension split

**Decision**: Split the N dimension (outermost js loop) across threads. Each thread gets its own packing buffers.

**Rationale**: This is the simplest parallelization strategy that avoids false sharing. Each thread writes to a disjoint slice of C. OpenBLAS supports M/MN split as well, but N-split is sufficient for a first implementation.

**Alternative considered**: OpenMP `#pragma omp parallel for`. Rejected because it hides thread management details that we want to understand and control.

### D5: Separate packing functions per transpose combination

**Decision**: Provide 4 packing functions (pack_a_nn, pack_a_tn, pack_b_nn, pack_b_tn) instead of using macros to select.

**Rationale**: OpenBLAS uses `ICOPY_OPERATION` / `OCOPY_OPERATION` macros that expand differently based on `#define NN/NT/TN/TT`. This is efficient but hard to follow. Explicit functions are clearer.

### D6: Buffer page-offset to avoid TLB eviction

**Decision**: Apply configurable page offsets (GEMM_OFFSET_A, GEMM_OFFSET_B) to sa and sb buffers so they don't share the same TLB set.

**Rationale**: When sa and sb are adjacent in memory, accessing them alternately in the kernel causes TLB set conflicts. A 64KB+ offset ensures they map to different TLB entries. This is a direct optimization from the GotoBLAS paper, reflected in OpenBLAS's `param.h` (e.g., Loongson: `GEMM_OFFSET_A1=0x10000`, `GEMM_OFFSET_B1=0x100000`).

### D7: Pure C generic kernel as baseline

**Decision**: Implement a portable C micro-kernel first, with MR=4, NR=4 (double) or MR=8, NR=4 (float).

**Rationale**: Establishes correctness before optimization. Assembly kernels (AVX2/AVX-512) are added later as alternative kernel table entries.

## Risks / Trade-offs

**[Risk] Performance gap vs OpenBLAS** → Mitigation: This is a learning project, not a production library. The generic C kernel will be ~10-50x slower than OpenBLAS's AVX2 kernel. Assembly optimization is a later phase.

**[Risk] Packing overhead dominates for small matrices** → Mitigation: Add a small-matrix fast path that bypasses packing (direct naive multiplication for m*n*k < threshold).

**[Risk] Thread overhead for small workloads** → Mitigation: Only use threads when MNK > threshold (same approach as OpenBLAS's `SMP_THRESHOLD_MIN`).

**[Risk] Blocking parameters not well-tuned** → Mitigation: Make them runtime-configurable so we can benchmark different values easily.

## Open Questions

- Should we support both RowMajor and ColMajor, or ColMajor only? (Current plan: ColMajor only, like BLAS standard)
- Should the generic kernel use compiler intrinsics (`__m256d`) or pure scalar C? (Current plan: scalar C first, then add intrinsics as a separate kernel)
