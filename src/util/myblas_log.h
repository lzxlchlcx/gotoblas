#ifndef MYBLAS_LOG_H
#define MYBLAS_LOG_H

#ifdef MYBLAS_ENABLE_LOG

#include <time.h>

typedef struct {
    long   dgemm_calls;
    double total_time;
    double gflops_sum;
    double gflops_peak;
    double pack_a_time;
    double pack_b_time;
    double kernel_time;
    long   pack_a_calls;
    long   pack_b_calls;
    long   kernel_calls;
    int    last_m, last_n, last_k;
    int    last_transa, last_transb;
    double last_gflops;
    int    enabled;
} myblas_log_stats_t;

double myblas_log_time(void);
void   myblas_log_accum(const char *field, double dt);
void   myblas_log_record_call(int m, int n, int k, int transa, int transb, double elapsed);
void   myblas_log_reset(void);
void   myblas_log_print(void);
void   myblas_log_enable(int enable);
myblas_log_stats_t *myblas_log_get_stats(void);

#define MYBLAS_LOG_TIMER_START(var) \
    double var = myblas_log_time()

#define MYBLAS_LOG_TIMER_END(var, field) \
    do { myblas_log_accum(field, myblas_log_time() - (var)); } while(0)

#define MYBLAS_LOG_RECORD_CALL(m, n, k, ta, tb, elapsed) \
    myblas_log_record_call(m, n, k, ta, tb, elapsed)

#define MYBLAS_LOG_RECORD_CALL_ELAPSED(m, n, k, ta, tb, start_var) \
    myblas_log_record_call(m, n, k, ta, tb, myblas_log_time() - (start_var))

#define MYBLAS_LOG_RESET()      myblas_log_reset()
#define MYBLAS_LOG_PRINT()      myblas_log_print()
#define MYBLAS_LOG_ENABLE(e)    myblas_log_enable(e)

#else

#define MYBLAS_LOG_TIMER_START(var)       ((void)0)
#define MYBLAS_LOG_TIMER_END(var, field)  ((void)0)
#define MYBLAS_LOG_RECORD_CALL(m, n, k, ta, tb, elapsed) ((void)0)
#define MYBLAS_LOG_RECORD_CALL_ELAPSED(m, n, k, ta, tb, start_var) ((void)0)
#define MYBLAS_LOG_RESET()                ((void)0)
#define MYBLAS_LOG_PRINT()                ((void)0)
#define MYBLAS_LOG_ENABLE(e)              ((void)0)

#endif

#endif
