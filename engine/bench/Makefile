# kbench: kernel microbenchmark + unit tests (cross-compiled static for the Pi 4).
# Run inside WSL from the project root:
#   make -C engine/bench            -> build/aarch64/kbench
#   make -C engine/bench test-qemu  -> golden + correctness checks under qemu (not timing)
#   make -C engine/bench asm        -> disassembly of the kernels for inspection
CC      := aarch64-linux-gnu-gcc
CFLAGS  := -O3 -mcpu=cortex-a72 -std=c11 -Wall -Wextra -I../src $(EXTRA)
LDFLAGS := -static -pthread -lm
ROOT    := ../..
OUT     := $(ROOT)/build/aarch64
SRC     := kbench.c kbench_util.c ../src/matvec_fast.c ../src/kern_float.c \
           ../src/kern_quant.c ../src/kern_act.c
HDR     := kbench.h ../src/kern.h ../src/ops.h ../src/pq_format.h ../src/pq_quant.h
PYTHON  ?= python.exe  # Windows python (has numpy); WSL python3 has none
GOLDEN  := $(ROOT)/build/kbench_golden.bin

all: $(OUT)/kbench

$(OUT)/kbench: $(SRC) $(HDR) | $(OUT)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)

$(OUT):
	mkdir -p $@

$(GOLDEN): $(ROOT)/tools/quant.py
	cd $(ROOT) && $(PYTHON) tools/quant.py golden build/kbench_golden.bin

test-qemu: $(OUT)/kbench $(GOLDEN)
	qemu-aarch64 -cpu cortex-a72 $(OUT)/kbench --golden $(GOLDEN) --check \
	    --shapes kv,ab,down --threads 1,3 --batch 1,3,8

asm: $(OUT)/kbench
	aarch64-linux-gnu-objdump -d --no-show-raw-insn $(OUT)/kbench > $(ROOT)/build/kbench.asm

.PHONY: all test-qemu asm
