# Standalone V3D tests and the opt-in engine split oracle.
CC = aarch64-linux-gnu-gcc
PYTHON ?= python3
BUILD ?= ../../build/gpu-c
CFLAGS = -O3 -std=c11 -ffp-contract=off -Wall -Wextra -Werror -I../src -I../include

.PHONY: all emit split-test engine-fast fallback-tools
all: $(BUILD)/v3d_q4_test

$(BUILD)/v3d_q4_test: v3d.c v3d.h v3d_q4_test.c q4gpu_kernels.h ../src/kern_quant.c ../src/kern_act.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) v3d.c v3d_q4_test.c ../src/kern_quant.c ../src/kern_act.c -static -lm -o $@

emit:
	$(PYTHON) emit_q4.py --assembler-src "$(ASSEMBLER_SRC)" --out q4gpu_kernels.h

fallback-tools: $(BUILD)/fault_ioctl.so

$(BUILD)/fault_ioctl.so: fault_ioctl.c
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -shared -fPIC $< -ldl -o $@

engine-fast:
	$(MAKE) -C .. pi MATVEC=fast

COREOBJ = $(filter-out ../../build/aarch64/engine/obj-fast/pinfer_run.o ../../build/aarch64/engine/obj-fast/pinfer_ppl.o,$(wildcard ../../build/aarch64/engine/obj-fast/*.o))
split-test: $(BUILD)/gpu_split_test $(BUILD)/gpu_split_test_down $(BUILD)/gpu_split_test_qkv

$(BUILD)/gpu_split_test: v3d_q4_test.c engine-fast
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -mcpu=cortex-a72 -DPF_TEST_ENGINE $< $(COREOBJ) -static -pthread -lm -o $@

$(BUILD)/gpu_split_test_down: v3d_q4_test.c engine-fast
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -mcpu=cortex-a72 -DPF_TEST_ENGINE -DTEST_ROWS=2560 -DTEST_COLS=9216 $< $(COREOBJ) -static -pthread -lm -o $@

$(BUILD)/gpu_split_test_qkv: v3d_q4_test.c engine-fast
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -mcpu=cortex-a72 -DPF_TEST_ENGINE -DTEST_ROWS=8192 $< $(COREOBJ) -static -pthread -lm -o $@
