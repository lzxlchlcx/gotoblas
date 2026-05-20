#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "driver/gemm_internal.h"
#include "config/generic.h"

static int myblas_num_threads = 1;

void myblas_set_num_threads(int num_threads) {
    if (num_threads > 0) myblas_num_threads = num_threads;
}

int myblas_get_num_threads(void) {
    const char *env = getenv("MYBLAS_NUM_THREADS");
    if (env) {
        int n = atoi(env);
        if (n > 0) return n;
    }
    return myblas_num_threads;
}

typedef struct {
    const gemm_arg_t *arg;
    const gemm_config_t *cfg;
    const void *kernels;
    int n_from, n_to;
    void *sa, *sb;
} gemm_task_t;

static void *gemm_worker_double(void *data)
{
    gemm_task_t *task = (gemm_task_t *)data;
    gemm_arg_t local_arg = *task->arg;
    local_arg.n = task->n_to - task->n_from;
    local_arg.B = (const double *)task->arg->B + task->n_from * task->arg->ldb;
    local_arg.C = (double *)task->arg->C + task->n_from * task->arg->ldc;

    gemm_driver_double(&local_arg, task->cfg,
                       (const gemm_kernel_table_t *)task->kernels,
                       (double *)task->sa, (double *)task->sb);
    return NULL;
}

static void *gemm_worker_float(void *data)
{
    gemm_task_t *task = (gemm_task_t *)data;
    gemm_arg_t local_arg = *task->arg;
    local_arg.n = task->n_to - task->n_from;
    local_arg.B = (const float *)task->arg->B + task->n_from * task->arg->ldb;
    local_arg.C = (float *)task->arg->C + task->n_from * task->arg->ldc;

    gemm_driver_float(&local_arg, task->cfg,
                      (const sgemm_kernel_table_t *)task->kernels,
                      (float *)task->sa, (float *)task->sb);
    return NULL;
}

void gemm_parallel_double(const gemm_arg_t *arg, const gemm_config_t *cfg,
                          const gemm_kernel_table_t *kernels)
{
    int nthreads = arg->nthreads;
    if (nthreads <= 1) {
        size_t sa_size = (size_t)cfg->P * cfg->Q * sizeof(double) + cfg->offset_a;
        size_t sb_size = (size_t)cfg->Q * cfg->R * sizeof(double) + cfg->offset_b;
        double *sa = (double *)malloc(sa_size);
        double *sb = (double *)malloc(sb_size);
        if (sa && sb)
            gemm_driver_double(arg, cfg, kernels, sa, sb);
        free(sa); free(sb);
        return;
    }

    pthread_t *threads = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
    gemm_task_t *tasks = (gemm_task_t *)malloc(nthreads * sizeof(gemm_task_t));
    if (!threads || !tasks) {
        free(threads); free(tasks);
        size_t sa_size = (size_t)cfg->P * cfg->Q * sizeof(double) + cfg->offset_a;
        size_t sb_size = (size_t)cfg->Q * cfg->R * sizeof(double) + cfg->offset_b;
        double *sa = (double *)malloc(sa_size);
        double *sb = (double *)malloc(sb_size);
        if (sa && sb)
            gemm_driver_double(arg, cfg, kernels, sa, sb);
        free(sa); free(sb);
        return;
    }

    int n = arg->n;
    int chunk = n / nthreads;
    int remainder = n % nthreads;

    int n_from = 0;
    for (int t = 0; t < nthreads; t++) {
        int n_to = n_from + chunk + (t < remainder ? 1 : 0);

        tasks[t].arg = arg;
        tasks[t].cfg = cfg;
        tasks[t].kernels = kernels;
        tasks[t].n_from = n_from;
        tasks[t].n_to = n_to;

        size_t sa_size = (size_t)cfg->P * cfg->Q * sizeof(double) + cfg->offset_a;
        size_t sb_size = (size_t)cfg->Q * cfg->R * sizeof(double) + cfg->offset_b;
        tasks[t].sa = malloc(sa_size);
        tasks[t].sb = malloc(sb_size);

        if (!tasks[t].sa || !tasks[t].sb) {
            free(tasks[t].sa); free(tasks[t].sb);
            for (int j = 0; j < t; j++) {
                free(tasks[j].sa); free(tasks[j].sb);
            }
            free(threads); free(tasks);
            return;
        }

        n_from = n_to;
    }

    for (int t = 1; t < nthreads; t++) {
        pthread_create(&threads[t], NULL, gemm_worker_double, &tasks[t]);
    }
    gemm_worker_double(&tasks[0]);

    for (int t = 1; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
    }

    for (int t = 0; t < nthreads; t++) {
        free(tasks[t].sa);
        free(tasks[t].sb);
    }
    free(threads);
    free(tasks);
}

void gemm_parallel_float(const gemm_arg_t *arg, const gemm_config_t *cfg,
                         const sgemm_kernel_table_t *kernels)
{
    int nthreads = arg->nthreads;
    if (nthreads <= 1) {
        size_t sa_size = (size_t)cfg->P * cfg->Q * sizeof(float) + cfg->offset_a;
        size_t sb_size = (size_t)cfg->Q * cfg->R * sizeof(float) + cfg->offset_b;
        float *sa = (float *)malloc(sa_size);
        float *sb = (float *)malloc(sb_size);
        if (sa && sb)
            gemm_driver_float(arg, cfg, kernels, sa, sb);
        free(sa); free(sb);
        return;
    }

    pthread_t *threads = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
    gemm_task_t *tasks = (gemm_task_t *)malloc(nthreads * sizeof(gemm_task_t));
    if (!threads || !tasks) {
        free(threads); free(tasks);
        size_t sa_size = (size_t)cfg->P * cfg->Q * sizeof(float) + cfg->offset_a;
        size_t sb_size = (size_t)cfg->Q * cfg->R * sizeof(float) + cfg->offset_b;
        float *sa = (float *)malloc(sa_size);
        float *sb = (float *)malloc(sb_size);
        if (sa && sb)
            gemm_driver_float(arg, cfg, kernels, sa, sb);
        free(sa); free(sb);
        return;
    }

    int n = arg->n;
    int chunk = n / nthreads;
    int remainder = n % nthreads;

    int n_from = 0;
    for (int t = 0; t < nthreads; t++) {
        int n_to = n_from + chunk + (t < remainder ? 1 : 0);

        tasks[t].arg = arg;
        tasks[t].cfg = cfg;
        tasks[t].kernels = kernels;
        tasks[t].n_from = n_from;
        tasks[t].n_to = n_to;

        size_t sa_size = (size_t)cfg->P * cfg->Q * sizeof(float) + cfg->offset_a;
        size_t sb_size = (size_t)cfg->Q * cfg->R * sizeof(float) + cfg->offset_b;
        tasks[t].sa = malloc(sa_size);
        tasks[t].sb = malloc(sb_size);

        if (!tasks[t].sa || !tasks[t].sb) {
            free(tasks[t].sa); free(tasks[t].sb);
            for (int j = 0; j < t; j++) {
                free(tasks[j].sa); free(tasks[j].sb);
            }
            free(threads); free(tasks);
            return;
        }

        n_from = n_to;
    }

    for (int t = 1; t < nthreads; t++) {
        pthread_create(&threads[t], NULL, gemm_worker_float, &tasks[t]);
    }
    gemm_worker_float(&tasks[0]);

    for (int t = 1; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
    }

    for (int t = 0; t < nthreads; t++) {
        free(tasks[t].sa);
        free(tasks[t].sb);
    }
    free(threads);
    free(tasks);
}
