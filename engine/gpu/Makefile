# Standalone opt-in GPU experiment; the shipping engine is unchanged.
CC = aarch64-linux-gnu-gcc
PYTHON ?= python3
BUILD ?= ../../build/gpu-c
CFLAGS = -O3 -std=c11 -ffp-contract=off -Wall -Wextra -Werror -I../src -I../include

.PHONY: all emit
all: $(BUILD)/v3d_q4_test

$(BUILD)/v3d_q4_test: v3d.c v3d.h v3d_q4_test.c q4gpu_kernels.h ../src/kern_quant.c ../src/kern_act.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) v3d.c v3d_q4_test.c ../src/kern_quant.c ../src/kern_act.c -static -lm -o $@

emit:
	$(PYTHON) emit_q4.py --assembler-src "$(ASSEMBLER_SRC)" --out q4gpu_kernels.h
