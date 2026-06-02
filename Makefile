CC      = gcc
CFLAGS  = -O2 -Wall -Isrc
LDFLAGS = -lpthread
AR      = ar
NVCC    ?= nvcc
CUDA_FLAGS ?= -O2 -arch=sm_86 -Isrc

UNAME_S := $(shell uname -s)
IS_WSL  := $(shell uname -r | grep -qi microsoft && echo 1 || echo 0)

# OpenBLAS defaults for macOS / WSL (can be overridden by env or command line)
ifeq ($(UNAME_S),Darwin)
  OPENBLAS_INC ?= /opt/homebrew/opt/openblas/include
  OPENBLAS_LIB ?= /opt/homebrew/opt/openblas/lib
else ifeq ($(IS_WSL),1)
  OPENBLAS_INC ?= /home/lzx/miniconda3/envs/main/include
  OPENBLAS_LIB ?= /home/lzx/miniconda3/envs/main/lib
endif

$(info OPENBLAS_INC=$(OPENBLAS_INC))
$(info OPENBLAS_LIB=$(OPENBLAS_LIB))

SRCS_API    = src/api/dgemm.c src/api/sgemm.c src/api/gemm_gpu_dispatch.c
SRCS_DRIVER = src/driver/gemm_driver.c src/driver/gemm_thread.c
SRCS_KERNEL_GENERIC = src/kernel/generic/dgemm_kernel.c \
              src/kernel/generic/sgemm_kernel.c \
              src/kernel/generic/dgemm_pack.c \
              src/kernel/generic/sgemm_pack.c \
              src/kernel/generic/dgemm_beta.c \
              src/kernel/generic/sgemm_beta.c \
              src/kernel/generic/kernel_init.c

# AVX2 sources (compiled with -mavx2 -mfma)
SRCS_AVX2   = src/kernel/avx2/dgemm_kernel.c \
              src/kernel/avx2/sgemm_kernel.c \
              src/kernel/avx2/dgemm_pack.c \
              src/kernel/avx2/sgemm_pack.c \
              src/kernel/avx2/dgemm_beta.c \
              src/kernel/avx2/sgemm_beta.c \
              src/kernel/avx2/kernel_init.c \
              src/kernel/avx2/cpuid.c

SRCS = $(SRCS_API) $(SRCS_DRIVER) $(SRCS_KERNEL_GENERIC)
SRCS_UTIL = src/util/myblas_log.c
OBJS = $(SRCS:.c=.o) $(SRCS_UTIL:.c=.o)

HAS_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo yes || echo no)
USE_CUDA ?= $(if $(filter yes,$(HAS_NVCC)),1,0)

SRCS_CUDA = src/kernel/cuda/gpu_init.cu \
            src/kernel/cuda/dgemm_kernel.cu \
            src/kernel/cuda/sgemm_kernel.cu

ifeq ($(USE_CUDA),1)
  CUDA_HOME ?= /usr/local/cuda
  CFLAGS += -DUSE_CUDA -I$(CUDA_HOME)/include
  OBJS += $(SRCS_CUDA:.cu=.o)
  LDFLAGS += -lcudart -lcublas -L$(CUDA_HOME)/lib64
endif

ifdef LOG
  CFLAGS += -DMYBLAS_ENABLE_LOG
endif

# Check if compiler supports AVX2
HAS_AVX2 := $(shell echo 'int main(){}' | $(CC) -mavx2 -mfma -x c - -o /dev/null 2>/dev/null && echo yes || echo no)

ifeq ($(HAS_AVX2),yes)
  CFLAGS += -D__AVX2__
  OBJS += $(SRCS_AVX2:.c=.o)
endif

LIB     = libmyblas.a
TEST    = test/test_gemm
BENCH   = test/bench_gemm
COMPARE = test/bench_compare
TEST_GPU = test/test_gemm_gpu
BENCH_GPU = test/bench_gemm_gpu

.PHONY: all lib test bench test_gpu bench_gpu clean

all: lib

lib: $(LIB)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c -o $@ $<

# AVX2 objects need -mavx2 -mfma flags
src/kernel/avx2/%.o: src/kernel/avx2/%.c
	$(CC) $(CFLAGS) -mavx2 -mfma -c -o $@ $<

test: $(TEST)
	./$(TEST)

bench: $(BENCH)
	./$(BENCH)

compare: $(COMPARE)
	./$(COMPARE)

test_gpu: $(TEST_GPU)
	./$(TEST_GPU)

bench_gpu: $(BENCH_GPU)
	./$(BENCH_GPU)

$(TEST): test/test_gemm.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lmyblas $(LDFLAGS) -lm

$(BENCH): test/bench_gemm.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lmyblas $(LDFLAGS) -lm

$(COMPARE): test/bench_compare.c
	make lib LOG=1
	$(CC) $(CFLAGS) -DMYBLAS_ENABLE_LOG -I$(OPENBLAS_INC) -o $@ $< -L. -L$(OPENBLAS_LIB) -lmyblas -lopenblas $(LDFLAGS) -lm

$(TEST_GPU): test/test_gemm_gpu.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lmyblas $(LDFLAGS) -lm

$(BENCH_GPU): test/bench_gemm_gpu.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lmyblas $(LDFLAGS) -lm

clean:
	rm -f $(OBJS) $(SRCS_AVX2:.c=.o) $(SRCS_CUDA:.cu=.o) $(LIB) $(TEST) $(BENCH) $(COMPARE) $(TEST_GPU) $(BENCH_GPU)
