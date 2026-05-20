CC      = gcc
CFLAGS  = -O2 -Wall -Iinclude -Isrc
LDFLAGS = -lpthread
AR      = ar

SRCS_API    = src/api/dgemm.c src/api/sgemm.c
SRCS_DRIVER = src/driver/gemm_driver.c src/driver/gemm_thread.c
SRCS_KERNEL = src/kernel/generic/dgemm_kernel.c \
              src/kernel/generic/sgemm_kernel.c \
              src/kernel/generic/dgemm_pack.c \
              src/kernel/generic/sgemm_pack.c \
              src/kernel/generic/dgemm_beta.c \
              src/kernel/generic/sgemm_beta.c \
              src/kernel/generic/kernel_init.c
SRCS = $(SRCS_API) $(SRCS_DRIVER) $(SRCS_KERNEL)
OBJS = $(SRCS:.c=.o)

LIB     = libmyblas.a
TEST    = test/test_gemm
BENCH   = test/bench_gemm

.PHONY: all lib test bench clean

all: lib

lib: $(LIB)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(TEST)
	./$(TEST)

bench: $(BENCH)
	./$(BENCH)

$(TEST): test/test_gemm.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lmyblas $(LDFLAGS) -lm

$(BENCH): test/bench_gemm.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lmyblas $(LDFLAGS) -lm

clean:
	rm -f $(OBJS) $(LIB) $(TEST) $(BENCH)
