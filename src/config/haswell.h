#ifndef MYBLAS_CONFIG_HASWELL_H
#define MYBLAS_CONFIG_HASWELL_H

#include "driver/gemm_internal.h"

#define GEMM_HASWELL_DTB_ENTRIES  64
#define GEMM_HASWELL_PAGE_SIZE    4096

#define GEMM_HASWELL_D_P  256
#define GEMM_HASWELL_D_Q  256
#define GEMM_HASWELL_D_R  4096
#define GEMM_HASWELL_D_MR 4
#define GEMM_HASWELL_D_NR 8

#define GEMM_HASWELL_S_P  256
#define GEMM_HASWELL_S_Q  256
#define GEMM_HASWELL_S_R  4096
#define GEMM_HASWELL_S_MR 8
#define GEMM_HASWELL_S_NR 4

#define GEMM_HASWELL_OFFSET_A  (size_t)0
#define GEMM_HASWELL_OFFSET_B  (size_t)0

static inline void gemm_config_avx2_double(gemm_config_t *cfg) {
    cfg->P           = GEMM_HASWELL_D_P;
    cfg->Q           = GEMM_HASWELL_D_Q;
    cfg->R           = GEMM_HASWELL_D_R;
    cfg->MR          = GEMM_HASWELL_D_MR;
    cfg->NR          = GEMM_HASWELL_D_NR;
    cfg->dtb_entries = GEMM_HASWELL_DTB_ENTRIES;
    cfg->offset_a    = GEMM_HASWELL_OFFSET_A;
    cfg->offset_b    = GEMM_HASWELL_OFFSET_B;
}

static inline void gemm_config_avx2_float(gemm_config_t *cfg) {
    cfg->P           = GEMM_HASWELL_S_P;
    cfg->Q           = GEMM_HASWELL_S_Q;
    cfg->R           = GEMM_HASWELL_S_R;
    cfg->MR          = GEMM_HASWELL_S_MR;
    cfg->NR          = GEMM_HASWELL_S_NR;
    cfg->dtb_entries = GEMM_HASWELL_DTB_ENTRIES;
    cfg->offset_a    = GEMM_HASWELL_OFFSET_A;
    cfg->offset_b    = GEMM_HASWELL_OFFSET_B;
}

#endif
