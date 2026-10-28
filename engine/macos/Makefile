ROOT := $(abspath ../..)
BACKEND ?= metal
ifeq ($(filter $(BACKEND),metal cpu cpu-baseline cpu-i8mm),)
$(error BACKEND must be metal, cpu, cpu-baseline, or cpu-i8mm)
endif
OUT := $(ROOT)/build/macos-$(BACKEND)/engine
CC := xcrun clang
CPU ?= native
CFLAGS := -O3 -g -std=c11 -ffp-contract=off -Wall -Wextra -Werror -mcpu=$(CPU) -mmacosx-version-min=13.0
INCS := -I$(ROOT)/engine/include -I$(ROOT)/engine/src
DEFS := -DPF_MATVEC_NAME=\"macos-$(BACKEND)\"
SRC := $(filter-out $(ROOT)/engine/src/matvec_ref.c,$(wildcard $(ROOT)/engine/src/*.c))
OBJ := $(addprefix $(OUT)/,$(notdir $(SRC:.c=.o)))
HEADERS := $(wildcard $(ROOT)/engine/src/*.h) $(ROOT)/engine/include/pinfer.h \
           $(ROOT)/engine/metal/inference.h $(ROOT)/engine/metal/metal.h
LIBS := -lm -pthread
ifeq ($(BACKEND),cpu-baseline)
DEFS += -DNM_NO_DOTPROD
endif
ifeq ($(BACKEND),cpu-i8mm)
DEFS += -DNM_USE_I8MM
endif
ifeq ($(BACKEND),metal)
DEFS += -DNM_INFERENCE_METAL -DNM_SHADER_PATH=\"$(ROOT)/engine/metal/kernels.metal\"
OBJ += $(OUT)/metal.o $(OUT)/inference.o
LIBS += -framework Foundation -framework Metal
endif

.PHONY: all test-i8mm i8mm-tools
all: $(OUT)/pinfer_run $(OUT)/pinfer_ppl

$(OUT):
	mkdir -p $@
$(OUT)/%.o: $(ROOT)/engine/src/%.c $(HEADERS) Makefile | $(OUT)
	$(CC) $(CFLAGS) $(INCS) $(DEFS) -c $< -o $@
$(OUT)/%.o: $(ROOT)/engine/tools/%.c $(HEADERS) Makefile | $(OUT)
	$(CC) $(CFLAGS) $(INCS) $(DEFS) -c $< -o $@
$(OUT)/metal.o: $(ROOT)/engine/metal/metal.m $(ROOT)/engine/metal/metal.h Makefile | $(OUT)
	$(CC) $(CFLAGS) -fobjc-arc -c $< -o $@
$(OUT)/inference.o: $(ROOT)/engine/metal/inference.c $(ROOT)/engine/metal/inference.h $(ROOT)/engine/metal/metal.h Makefile | $(OUT)
	$(CC) $(CFLAGS) $(INCS) $(DEFS) -c $< -o $@
$(OUT)/pinfer_run: $(OBJ) $(OUT)/pinfer_run.o
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@
$(OUT)/pinfer_ppl: $(OBJ) $(OUT)/pinfer_ppl.o
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

$(OUT)/oracle_quant.o: $(ROOT)/engine/src/kern_quant.c $(HEADERS) Makefile | $(OUT)
	$(CC) $(CFLAGS) $(INCS) -DNM_NO_DOTPROD -Dkern_q4=oracle_q4 -Dkern_q4k=oracle_q4k -Dkern_q8=oracle_q8 -c $< -o $@
$(OUT)/check_i8mm.o: check_i8mm.c $(HEADERS) Makefile | $(OUT)
	$(CC) $(CFLAGS) $(INCS) $(DEFS) -c $< -o $@
$(OUT)/check_i8mm: $(OUT)/check_i8mm.o $(OUT)/kern_quant.o $(OUT)/kern_act.o $(OUT)/oracle_quant.o
	$(CC) $(CFLAGS) $^ -lm -o $@
i8mm-tools: $(OUT)/check_i8mm
test-i8mm: i8mm-tools
	$(OUT)/check_i8mm
