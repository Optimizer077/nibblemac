ROOT := $(abspath ../..)
OUT := $(ROOT)/build/metal
CC := xcrun clang
CFLAGS := -O3 -g -std=c11 -Wall -Wextra -Werror -ffp-contract=off -mmacosx-version-min=13.0
OBJCFLAGS := $(CFLAGS) -fobjc-arc
FRAMEWORKS := -framework Foundation -framework Metal

.PHONY: all test bench test-a8 test-graphs graph-tools
all: $(OUT)/nibblemac_metal $(OUT)/kernels.metal

$(OUT):
	mkdir -p $@

$(OUT)/metal.o: metal.m metal.h Makefile | $(OUT)
	$(CC) $(OBJCFLAGS) -c $< -o $@

$(OUT)/check.o: check.c metal.h Makefile | $(OUT)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUT)/nibblemac_metal: $(OUT)/metal.o $(OUT)/check.o
	$(CC) $(OBJCFLAGS) $^ $(FRAMEWORKS) -o $@

# Runtime compilation requires only Command Line Tools, not the offline Metal toolchain.
$(OUT)/kernels.metal: kernels.metal | $(OUT)
	cp $< $@

test: all
	$(OUT)/nibblemac_metal test $(OUT)/kernels.metal

bench: all
	$(OUT)/nibblemac_metal bench $(OUT)/kernels.metal

$(OUT)/check_a8.o: check_a8.c metal.h ../src/kern.h | $(OUT)
	$(CC) $(CFLAGS) -c $< -o $@
$(OUT)/kern_quant.o: ../src/kern_quant.c ../src/kern.h | $(OUT)
	$(CC) $(CFLAGS) -c $< -o $@
$(OUT)/kern_act.o: ../src/kern_act.c ../src/kern.h | $(OUT)
	$(CC) $(CFLAGS) -c $< -o $@
$(OUT)/check_a8: $(OUT)/check_a8.o $(OUT)/kern_quant.o $(OUT)/kern_act.o $(OUT)/metal.o
	$(CC) $(OBJCFLAGS) $^ $(FRAMEWORKS) -o $@
test-a8: all $(OUT)/check_a8
	$(OUT)/check_a8 $(OUT)/kernels.metal

$(OUT)/check_graphs.o: check_graphs.c metal.h ../src/kern.h ../src/forward.h | $(OUT)
	$(CC) $(CFLAGS) -I../include -c $< -o $@
$(OUT)/kern_lin.o: ../src/kern_lin.c ../src/kern.h | $(OUT)
	$(CC) $(CFLAGS) -c $< -o $@
$(OUT)/forward_ops.o: ../src/forward_ops.c ../src/forward.h | $(OUT)
	$(CC) $(CFLAGS) -I../include -c $< -o $@
$(OUT)/check_graphs: $(OUT)/check_graphs.o $(OUT)/kern_lin.o $(OUT)/forward_ops.o $(OUT)/kern_quant.o $(OUT)/kern_act.o $(OUT)/metal.o
	$(CC) $(OBJCFLAGS) $^ $(FRAMEWORKS) -o $@
test-graphs: all $(OUT)/check_graphs
	$(OUT)/check_graphs $(OUT)/kernels.metal

graph-tools: all $(OUT)/check_graphs $(OUT)/check_a8
