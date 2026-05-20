#ifndef MYBLAS_CONFIG_GENERIC_H
#define MYBLAS_CONFIG_GENERIC_H

#include "driver/gemm_internal.h"

#define GEMM_GENERIC_DTB_ENTRIES  64
#define GEMM_GENERIC_PAGE_SIZE    4096

#define GEMM_GENERIC_D_P  128
#define GEMM_GENERIC_D_Q  128
#define GEMM_GENERIC_D_R  4096
#define GEMM_GENERIC_D_MR 4
#define GEMM_GENERIC_D_NR 4

#define GEMM_GENERIC_S_P  128
#define GEMM_GENERIC_S_Q  256
#define GEMM_GENERIC_S_R  4096
#define GEMM_GENERIC_S_MR 8
#define GEMM_GENERIC_S_NR 4

#define GEMM_OFFSET_A  (size_t)0
#define GEMM_OFFSET_B  (size_t)0

static inline void gemm_config_generic_double(gemm_config_t *cfg) {
    cfg->P           = GEMM_GENERIC_D_P;
    cfg->Q           = GEMM_GENERIC_D_Q;
    cfg->R           = GEMM_GENERIC_D_R;
    cfg->MR          = GEMM_GENERIC_D_MR;
    cfg->NR          = GEMM_GENERIC_D_NR;
    cfg->dtb_entries = GEMM_GENERIC_DTB_ENTRIES;
    cfg->offset_a    = GEMM_OFFSET_A;
    cfg->offset_b    = GEMM_OFFSET_B;
}

static inline void gemm_config_generic_float(gemm_config_t *cfg) {
    cfg->P           = GEMM_GENERIC_S_P;
    cfg->Q           = GEMM_GENERIC_S_Q;
    cfg->R           = GEMM_GENERIC_S_R;
    cfg->MR          = GEMM_GENERIC_S_MR;
    cfg->NR          = GEMM_GENERIC_S_NR;
    cfg->dtb_entries = GEMM_GENERIC_DTB_ENTRIES;
    cfg->offset_a    = GEMM_OFFSET_A;
    cfg->offset_b    = GEMM_OFFSET_B;
}

#endif
