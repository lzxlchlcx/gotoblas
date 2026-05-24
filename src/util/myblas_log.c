#include "myblas_log.h"

#ifdef MYBLAS_ENABLE_LOG

#include <stdio.h>
#include <string.h>

static myblas_log_stats_t g_stats = {.enabled = 1};

double myblas_log_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

myblas_log_stats_t *myblas_log_get_stats(void)
{
    return &g_stats;
}

void myblas_log_accum(const char *field, double dt)
{
    if (!g_stats.enabled) return;
    if (strcmp(field, "pack_a") == 0) {
        g_stats.pack_a_time += dt;
        g_stats.pack_a_calls++;
    } else if (strcmp(field, "pack_b") == 0) {
        g_stats.pack_b_time += dt;
        g_stats.pack_b_calls++;
    } else if (strcmp(field, "kernel") == 0) {
        g_stats.kernel_time += dt;
        g_stats.kernel_calls++;
    }
}

void myblas_log_record_call(int m, int n, int k, int transa, int transb, double elapsed)
{
    if (!g_stats.enabled) return;
    double gflops = 2.0 * m * n * k / (elapsed * 1e9);
    g_stats.dgemm_calls++;
    g_stats.total_time += elapsed;
    g_stats.gflops_sum += gflops;
    if (gflops > g_stats.gflops_peak) g_stats.gflops_peak = gflops;
    g_stats.last_m = m;
    g_stats.last_n = n;
    g_stats.last_k = k;
    g_stats.last_transa = transa;
    g_stats.last_transb = transb;
    g_stats.last_gflops = gflops;
}

void myblas_log_reset(void)
{
    memset(&g_stats, 0, sizeof(g_stats));
    g_stats.enabled = 1;
}

void myblas_log_enable(int enable)
{
    g_stats.enabled = enable;
}

void myblas_log_print(void)
{
    myblas_log_stats_t *s = &g_stats;
    printf("\n=== MyBLAS Performance Log ===\n");

    if (s->dgemm_calls == 0) {
        printf("  (no DGEMM calls recorded)\n");
        return;
    }

    double avg_gflops = s->gflops_sum / s->dgemm_calls;
    printf("DGEMM calls: %ld\n", s->dgemm_calls);
    printf("  Total:    %.3f s  (avg: %.4f s)\n", s->total_time, s->total_time / s->dgemm_calls);
    printf("  GFLOPS:   avg %.1f, peak %.1f\n", avg_gflops, s->gflops_peak);
    printf("  Last:     m=%d n=%d k=%d transa=%c transb=%c  %.1f GFLOPS\n",
           s->last_m, s->last_n, s->last_k,
           s->last_transa ? 'T' : 'N',
           s->last_transb ? 'T' : 'N',
           s->last_gflops);

    double driver_total = s->pack_a_time + s->pack_b_time + s->kernel_time;
    if (driver_total > 0) {
        printf("\nDriver breakdown (cumulative):\n");
        printf("  pack_a:  %.3f s (%5.1f%%)  %ld calls\n",
               s->pack_a_time, s->pack_a_time / driver_total * 100, s->pack_a_calls);
        printf("  pack_b:  %.3f s (%5.1f%%)  %ld calls\n",
               s->pack_b_time, s->pack_b_time / driver_total * 100, s->pack_b_calls);
        printf("  kernel:  %.3f s (%5.1f%%)  %ld calls\n",
               s->kernel_time, s->kernel_time / driver_total * 100, s->kernel_calls);
    }
    printf("===============================\n\n");
}

#endif
