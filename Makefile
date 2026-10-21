# Cross-compile Pi 4 binaries. Run inside WSL Ubuntu from the project root:
#   make            -> build/aarch64/*
#   make test-qemu  -> run them under qemu-aarch64 (correctness only, not speed)
CC      := aarch64-linux-gnu-gcc
CFLAGS  := -O3 -mcpu=cortex-a72 -std=c11 -Wall -Wextra
LDFLAGS := -static -pthread
OUT     := build/aarch64

all: $(OUT)/memband

$(OUT)/memband: bench/memband/memband.c | $(OUT)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(OUT):
	mkdir -p $@

test-qemu: all
	qemu-aarch64 -cpu cortex-a72 $(OUT)/memband

clean:
	rm -rf build

.PHONY: all test-qemu clean
