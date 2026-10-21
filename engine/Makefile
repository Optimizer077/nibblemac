# pinfer engine. Run inside WSL Ubuntu, from anywhere:
#   make -C engine                 -> build/host/engine/pinfer_run     (x86, tests)
#   make -C engine pi              -> build/aarch64/engine/pinfer_run  (Pi 4, static)
#   make -C engine pi MATVEC=fast  -> NEON kernels (matvec_fast.c) instead of the scalar ref
# Host and Pi objects of different MATVEC choices go to separate directories, so switching
# MATVEC never links a stale kernel object.
ROOT    := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
MATVEC  ?= ref
ifeq ($(filter $(MATVEC),ref fast),)
$(error MATVEC must be ref or fast)
endif

SRC_DIR := $(ROOT)/engine/src
# Every src/*.c except the kernel variants is core, so new modules need no Makefile edit.
CORE    := $(filter-out matvec_%.c kern_%.c,$(notdir $(wildcard $(SRC_DIR)/*.c))) \
           matvec_$(MATVEC).c
# The fast kernels are split over kern_*.c (NEON, aarch64 only); link them only with fast.
ifeq ($(MATVEC),fast)
CORE    += $(notdir $(wildcard $(SRC_DIR)/kern_*.c))
endif
INCS    := -I$(ROOT)/engine/include -I$(SRC_DIR)
DEFS    := -DPF_MATVEC_NAME=\"$(MATVEC)\"
HDRS    := $(wildcard $(SRC_DIR)/*.h) $(ROOT)/engine/include/pinfer.h

HOST_CC     := gcc
HOST_CFLAGS := -O2 -std=c11 -Wall -Wextra -g
HOST_LIBS   := -lm -pthread
HOST_OUT    := $(ROOT)/build/host/engine
HOST_OBJ    := $(HOST_OUT)/obj-$(MATVEC)

PI_CC     := aarch64-linux-gnu-gcc
PI_CFLAGS := -O3 -mcpu=cortex-a72 -std=c11 -Wall -Wextra
PI_LIBS   := -static -pthread -lm
PI_OUT    := $(ROOT)/build/aarch64/engine
PI_OBJ    := $(PI_OUT)/obj-$(MATVEC)

host: $(HOST_OUT)/pinfer_run
pi: $(PI_OUT)/pinfer_run
all: host pi

$(HOST_OUT)/pinfer_run: $(addprefix $(HOST_OBJ)/,$(CORE:.c=.o)) $(HOST_OBJ)/pinfer_run.o
	$(HOST_CC) $(filter %.o,$^) -o $@ $(HOST_LIBS)

$(HOST_OBJ)/%.o: $(SRC_DIR)/%.c $(HDRS) | $(HOST_OBJ)
	$(HOST_CC) $(HOST_CFLAGS) $(INCS) $(DEFS) -c $< -o $@

$(HOST_OBJ)/pinfer_run.o: $(ROOT)/engine/tools/pinfer_run.c $(HDRS) | $(HOST_OBJ)
	$(HOST_CC) $(HOST_CFLAGS) $(INCS) $(DEFS) -c $< -o $@

$(PI_OUT)/pinfer_run: $(addprefix $(PI_OBJ)/,$(CORE:.c=.o)) $(PI_OBJ)/pinfer_run.o
	$(PI_CC) $(filter %.o,$^) -o $@ $(PI_LIBS)

$(PI_OBJ)/%.o: $(SRC_DIR)/%.c $(HDRS) | $(PI_OBJ)
	$(PI_CC) $(PI_CFLAGS) $(INCS) $(DEFS) -c $< -o $@

$(PI_OBJ)/pinfer_run.o: $(ROOT)/engine/tools/pinfer_run.c $(HDRS) | $(PI_OBJ)
	$(PI_CC) $(PI_CFLAGS) $(INCS) $(DEFS) -c $< -o $@

$(HOST_OBJ) $(PI_OBJ):
	mkdir -p $@

# The binary is relinked with whichever MATVEC was asked for last.
.PHONY: host pi all clean test FORCE
FORCE:
$(HOST_OUT)/pinfer_run $(PI_OUT)/pinfer_run: FORCE

test: host
	cd $(ROOT) && uv run --no-project --with numpy python tests/test_golden.py

clean:
	rm -rf $(HOST_OUT) $(PI_OUT)
