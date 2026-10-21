# pinfer engine. Run inside WSL Ubuntu, from anywhere:
#   make -C engine                 -> build/host/engine/pinfer_run     (x86, tests)
#   make -C engine pi              -> build/aarch64/engine/pinfer_run  (Pi 4, static)
#   make -C engine pi MATVEC=fast  -> NEON kernels (matvec_fast.c) instead of the scalar ref
#   make -C engine lib             -> libpinfer.{a,so} for host and Pi (see "Library" below)
#   make -C engine install PREFIX=/opt/pinfer [TARGET=host|pi] [DESTDIR=...]
#   make -C engine examples        -> build/{host,aarch64}/examples/pinfer_example
#   make -C engine wheel-host | wheel-pi   -> build/dist/pinfer-<ver>-py3-none-<plat>.whl
#   make -C engine host SAN=address,undefined -> build/host-san/engine/ (sanitizer build)
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

# -ffp-contract=off is already implied by -std=c11, stated explicitly: the forward pass and
# the kernels rely on separate multiply and add roundings for bit-exactness (ops.h, SPEC.md).
HOST_CC     := gcc
HOST_CFLAGS := -O2 -std=c11 -ffp-contract=off -Wall -Wextra -g
HOST_LIBS   := -lm -pthread
HOST_OUT    := $(ROOT)/build/host/engine
# Sanitizer build of the host tools: make -C engine host SAN=address,undefined
# -> build/host-san/engine/ (separate objects; the tests take --bin). Host only.
SAN ?=
ifneq ($(SAN),)
HOST_CFLAGS += -fsanitize=$(SAN) -fno-omit-frame-pointer
HOST_LIBS   += -fsanitize=$(SAN)
HOST_OUT    := $(ROOT)/build/host-san/engine
endif
HOST_OBJ    := $(HOST_OUT)/obj-$(MATVEC)

PI_CC     := aarch64-linux-gnu-gcc
PI_CFLAGS := -O3 -mcpu=cortex-a72 -std=c11 -ffp-contract=off -Wall -Wextra
PI_LIBS   := -static -pthread -lm
PI_OUT    := $(ROOT)/build/aarch64/engine
PI_OBJ    := $(PI_OUT)/obj-$(MATVEC)

# Command-line tools: engine/tools/<name>.c, each linked against CORE.
TOOLS := pinfer_run pinfer_ppl
TOOL_DIR := $(ROOT)/engine/tools

host: $(addprefix $(HOST_OUT)/,$(TOOLS))
pi: $(addprefix $(PI_OUT)/,$(TOOLS))
all: host pi

$(addprefix $(HOST_OUT)/,$(TOOLS)): $(HOST_OUT)/%: $(addprefix $(HOST_OBJ)/,$(CORE:.c=.o)) $(HOST_OBJ)/%.o
	$(HOST_CC) $(filter %.o,$^) -o $@ $(HOST_LIBS)

$(HOST_OBJ)/%.o: $(SRC_DIR)/%.c $(HDRS) | $(HOST_OBJ)
	$(HOST_CC) $(HOST_CFLAGS) $(INCS) $(DEFS) -c $< -o $@

$(HOST_OBJ)/%.o: $(TOOL_DIR)/%.c $(HDRS) | $(HOST_OBJ)
	$(HOST_CC) $(HOST_CFLAGS) $(INCS) $(DEFS) -c $< -o $@

$(addprefix $(PI_OUT)/,$(TOOLS)): $(PI_OUT)/%: $(addprefix $(PI_OBJ)/,$(CORE:.c=.o)) $(PI_OBJ)/%.o
	$(PI_CC) $(filter %.o,$^) -o $@ $(PI_LIBS)

$(PI_OBJ)/%.o: $(SRC_DIR)/%.c $(HDRS) | $(PI_OBJ)
	$(PI_CC) $(PI_CFLAGS) $(INCS) $(DEFS) -c $< -o $@

$(PI_OBJ)/%.o: $(TOOL_DIR)/%.c $(HDRS) | $(PI_OBJ)
	$(PI_CC) $(PI_CFLAGS) $(INCS) $(DEFS) -c $< -o $@

$(HOST_OBJ) $(PI_OBJ):
	mkdir -p $@

# The binary is relinked with whichever MATVEC was asked for last.
.PHONY: host pi all clean test FORCE host-lib pi-lib lib examples install \
        check-lib wheel-host wheel-pi
FORCE:
$(addprefix $(HOST_OUT)/,$(TOOLS)) $(addprefix $(PI_OUT)/,$(TOOLS)): FORCE

# From Windows: uv run --no-project --with numpy python tests/test_golden.py
PY ?= /root/venv-ref/bin/python
test: host
	cd $(ROOT) && $(PY) tests/test_golden.py

# ------------------------------------------------------------------------------- Library
# libpinfer.a / libpinfer.so hold the same CORE objects as pinfer_run, compiled with -fPIC
# into pic-$(MATVEC)/. Only pinfer_* is visible to users:
#   .so  linker version script pkg/pinfer.map (every other symbol is local)
#   .a   one relocatable object (ld -r) whose other globals objcopy makes local, so the
#        engine's pf_*/kern_* names can never clash with an application's symbols.
# Version, SONAME and pinfer.pc come from PINFER_VERSION_* in include/pinfer.h.
PUBHDR    := $(ROOT)/engine/include/pinfer.h
PKG_DIR   := $(ROOT)/engine/pkg
MAP       := $(PKG_DIR)/pinfer.map
ver_field  = $(shell sed -n 's/.*define PINFER_VERSION_$(1)  *\([0-9][0-9]*\).*/\1/p' $(PUBHDR))
VER_MAJOR := $(call ver_field,MAJOR)
VER_MINOR := $(call ver_field,MINOR)
VER_PATCH := $(call ver_field,PATCH)
VERSION   := $(VER_MAJOR).$(VER_MINOR).$(VER_PATCH)
# 0.x: any minor release may change the ABI, so the SONAME carries major.minor.
SONAME    := libpinfer.so.$(VER_MAJOR).$(VER_MINOR)

HOST_PIC  := $(HOST_OUT)/pic-$(MATVEC)
HOST_LIB  := $(HOST_OUT)/lib
PI_PIC    := $(PI_OUT)/pic-$(MATVEC)
PI_LIB    := $(PI_OUT)/lib
PI_TOOL   := aarch64-linux-gnu-
# -fno-math-errno: without it GCC keeps a libm call next to the inlined sqrt instruction only
# to set errno, and WSL's glibc 2.43 versions that call sqrtf@GLIBC_2.43, which the Pi
# (trixie, glibc 2.41) cannot load. The engine never reads errno; results are bit-identical.
LIB_CFLAGS := -fPIC -fno-math-errno
SO_FLAGS  := -shared -Wl,-soname,$(SONAME) -Wl,--version-script=$(MAP) -Wl,-z,defs \
             -Wl,--as-needed -Wl,-z,noexecstack

host-lib: $(HOST_LIB)/libpinfer.a $(HOST_LIB)/libpinfer.so
pi-lib: $(PI_LIB)/libpinfer.a $(PI_LIB)/libpinfer.so
lib: host-lib pi-lib

$(HOST_PIC)/%.o: $(SRC_DIR)/%.c $(HDRS) | $(HOST_PIC)
	$(HOST_CC) $(HOST_CFLAGS) $(LIB_CFLAGS) $(INCS) $(DEFS) -c $< -o $@

$(PI_PIC)/%.o: $(SRC_DIR)/%.c $(HDRS) | $(PI_PIC)
	$(PI_CC) $(PI_CFLAGS) $(LIB_CFLAGS) $(INCS) $(DEFS) -c $< -o $@

# $(1) tool prefix, $(2) objects, $(3) archive. pinfer_all.o lives next to the objects.
define build_static
	$(1)gcc -r -nostdlib $(2) -o $(dir $(firstword $(2)))pinfer_all.o
	$(1)objcopy -w --keep-global-symbol='pinfer_*' $(dir $(firstword $(2)))pinfer_all.o
	rm -f $(3) && $(1)ar rcs $(3) $(dir $(firstword $(2)))pinfer_all.o
endef

# $(1) cc, $(2) arch label, $(3) output dir
define build_info
	printf '{"version": "%s", "matvec": "%s", "arch": "%s", "cc": "%s", "soname": "%s"}\n' \
	  '$(VERSION)' '$(MATVEC)' '$(2)' "$$($(1) -dumpmachine) $$($(1) -dumpfullversion)" \
	  '$(SONAME)' > $(3)/BUILD_INFO.json
endef

$(HOST_LIB)/libpinfer.a: $(addprefix $(HOST_PIC)/,$(CORE:.c=.o)) | $(HOST_LIB)
	$(call build_static,,$(filter %.o,$^),$@)

$(HOST_LIB)/libpinfer.so: $(addprefix $(HOST_PIC)/,$(CORE:.c=.o)) $(MAP) | $(HOST_LIB)
	$(HOST_CC) $(SO_FLAGS) $(filter %.o,$^) -o $@ -lm -pthread $(if $(SAN),-fsanitize=$(SAN))
	$(call build_info,$(HOST_CC),x86_64,$(HOST_LIB))
	ln -sf libpinfer.so $(HOST_LIB)/$(SONAME)

$(PI_LIB)/libpinfer.a: $(addprefix $(PI_PIC)/,$(CORE:.c=.o)) | $(PI_LIB)
	$(call build_static,$(PI_TOOL),$(filter %.o,$^),$@)

# Dynamic against the Pi's glibc (only libc/libm); `make check-lib` prints the newest GLIBC_x.y
# symbol version it needs, which decides the wheel's manylinux tag.
$(PI_LIB)/libpinfer.so: $(addprefix $(PI_PIC)/,$(CORE:.c=.o)) $(MAP) | $(PI_LIB)
	$(PI_CC) $(SO_FLAGS) $(filter %.o,$^) -o $@ -lm -pthread
	$(call build_info,$(PI_CC),aarch64,$(PI_LIB))
	ln -sf libpinfer.so $(PI_LIB)/$(SONAME)

$(HOST_PIC) $(PI_PIC) $(HOST_LIB) $(PI_LIB):
	mkdir -p $@

# Libraries follow the last MATVEC choice, like pinfer_run.
$(HOST_LIB)/libpinfer.a $(HOST_LIB)/libpinfer.so $(PI_LIB)/libpinfer.a $(PI_LIB)/libpinfer.so: FORCE

# Exported symbols, NEEDED libraries and glibc symbol versions of both builds.
# PI_GLIBC_MAX: the Pi's glibc (Raspberry Pi OS trixie = 2.41); the Pi .so must not need newer.
PI_GLIBC_MAX ?= 2.41
check-lib: lib
	sh $(PKG_DIR)/check_lib.sh $(HOST_LIB) "" 99.99
	sh $(PKG_DIR)/check_lib.sh $(PI_LIB) $(PI_TOOL) $(PI_GLIBC_MAX)

# ------------------------------------------------------------------------------ Examples
# Host: dynamic link against libpinfer.so (rpath to build/host/engine/lib).
# Pi:   static link against libpinfer.a, like every other Pi binary here.
EX_SRC  := $(ROOT)/examples/c/simple.c
HOST_EX := $(ROOT)/build/host/examples
PI_EX   := $(ROOT)/build/aarch64/examples
examples: $(HOST_EX)/pinfer_example $(PI_EX)/pinfer_example

$(HOST_EX)/pinfer_example: $(EX_SRC) $(PUBHDR) $(HOST_LIB)/libpinfer.so
	mkdir -p $(dir $@)
	$(HOST_CC) -O2 -std=c11 -Wall -Wextra -I$(ROOT)/engine/include $< -o $@ \
	  -L$(HOST_LIB) -l:libpinfer.so -Wl,-rpath,$(HOST_LIB)

$(PI_EX)/pinfer_example: $(EX_SRC) $(PUBHDR) $(PI_LIB)/libpinfer.a
	mkdir -p $(dir $@)
	$(PI_CC) -O2 -std=c11 -Wall -Wextra -I$(ROOT)/engine/include $< -o $@ \
	  $(PI_LIB)/libpinfer.a -static -pthread -lm

# ------------------------------------------------------------------------------- Install
#   make -C engine install PREFIX=/opt/pinfer               (host libraries)
#   make -C engine install TARGET=pi DESTDIR=/tmp/stage PREFIX=/usr/local   (Pi staging tree)
PREFIX  ?= /usr/local
LIBDIR  ?= $(PREFIX)/lib
INCDIR  ?= $(PREFIX)/include
TARGET  ?= host
ifeq ($(filter $(TARGET),host pi),)
$(error TARGET must be host or pi)
endif
INST_LIB := $(if $(filter pi,$(TARGET)),$(PI_LIB),$(HOST_LIB))

install: $(TARGET)-lib
	install -d $(DESTDIR)$(INCDIR) $(DESTDIR)$(LIBDIR)/pkgconfig
	install -m 644 $(PUBHDR) $(DESTDIR)$(INCDIR)/pinfer.h
	install -m 644 $(INST_LIB)/libpinfer.a $(DESTDIR)$(LIBDIR)/libpinfer.a
	install -m 755 $(INST_LIB)/libpinfer.so $(DESTDIR)$(LIBDIR)/libpinfer.so.$(VERSION)
	ln -sf libpinfer.so.$(VERSION) $(DESTDIR)$(LIBDIR)/$(SONAME)
	ln -sf $(SONAME) $(DESTDIR)$(LIBDIR)/libpinfer.so
	sed -e 's|@PREFIX@|$(PREFIX)|' -e 's|@LIBDIR@|$(LIBDIR)|' -e 's|@INCDIR@|$(INCDIR)|' \
	    -e 's|@VERSION@|$(VERSION)|' $(PKG_DIR)/pinfer.pc.in > $(DESTDIR)$(LIBDIR)/pkgconfig/pinfer.pc

# --------------------------------------------------------------------------------- Wheels
# Platform wheels bundling libpinfer.so (python/hatch_build.py picks the library and the tag).
# WHEEL_PY needs hatchling: python3 -m venv /root/venv-pinfer-build && .../pip install hatchling
WHEEL_PY ?= /root/venv-pinfer-build/bin/python
DIST     ?= $(ROOT)/build/dist
wheel-host: host-lib
	cd $(ROOT)/python && PINFER_WHEEL_ARCH=host $(WHEEL_PY) -m pip wheel --no-deps \
	  --no-build-isolation -w $(DIST) .

wheel-pi: pi-lib
	cd $(ROOT)/python && PINFER_WHEEL_ARCH=aarch64 $(WHEEL_PY) -m pip wheel --no-deps \
	  --no-build-isolation -w $(DIST) .

clean:
	rm -rf $(HOST_OUT) $(PI_OUT) $(HOST_EX) $(PI_EX)
