CC      = gcc
CFLAGS  = -O2 -Wall -Isrc
LDFLAGS = -lpthread
AR      = ar

SRCS_API    = src/api/dgemm.c src/api/sgemm.c
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

.PHONY: all lib test bench clean

all: lib

lib: $(LIB)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# AVX2 objects need -mavx2 -mfma flags
src/kernel/avx2/%.o: src/kernel/avx2/%.c
	$(CC) $(CFLAGS) -mavx2 -mfma -c -o $@ $<

test: $(TEST)
	./$(TEST)

bench: $(BENCH)
	./$(BENCH)

compare: $(COMPARE)
	./$(COMPARE)

$(TEST): test/test_gemm.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lmyblas $(LDFLAGS) -lm

$(BENCH): test/bench_gemm.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lmyblas $(LDFLAGS) -lm

OPENBLAS_INC = /home/lzx/miniconda3/envs/main/include
OPENBLAS_LIB = /home/lzx/miniconda3/envs/main/lib

$(COMPARE): test/bench_compare.c $(LIB)
	$(CC) $(CFLAGS) -DMYBLAS_ENABLE_LOG -I$(OPENBLAS_INC) -o $@ $< -L. -L$(OPENBLAS_LIB) -lmyblas -lopenblas $(LDFLAGS) -lm

clean:
	rm -f $(OBJS) $(SRCS_AVX2:.c=.o) $(LIB) $(TEST) $(BENCH) $(COMPARE)
