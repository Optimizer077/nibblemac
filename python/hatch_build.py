"""Hatch build hook: bundle libpinfer.so into a platform wheel.

Environment:
  PINFER_WHEEL_ARCH  host | aarch64 — which build to bundle:
                     host    -> build/host/engine/lib/libpinfer.so
                     aarch64 -> build/aarch64/engine/lib/libpinfer.so
  PINFER_ROOT        repository root (default: the parent of this directory)

The wheel tag is derived from the library itself, not from the build machine:
  - architecture from the ELF header (x86_64 / aarch64),
  - manylinux_2_<N> where 2.<N> is the newest GLIBC_x.y symbol version the .so references
    (at least 2.17), so pip refuses the wheel on a system whose glibc is too old.
Editable installs (``pip install -e python``) bundle nothing; the package then finds the
library in the source tree (see pinfer/_lib.py).
"""
from __future__ import annotations

import os
import re
import struct
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_ELF_MACHINES = {62: "x86_64", 183: "aarch64"}
_GLIBC_RE = re.compile(rb"GLIBC_2\.(\d+)(?:\.\d+)?\x00")


def elf_arch(data: bytes) -> str:
    """Architecture name of a little-endian 64-bit ELF file."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not a little-endian 64-bit ELF file")
    machine = struct.unpack_from("<H", data, 18)[0]
    if machine not in _ELF_MACHINES:
        raise ValueError(f"unsupported ELF machine {machine}")
    return _ELF_MACHINES[machine]


def glibc_minor(data: bytes) -> int:
    """Newest glibc 2.x symbol version referenced (version names live in .dynstr)."""
    found = [int(m) for m in _GLIBC_RE.findall(data)]
    return max([17, *found])


class PinferBuildHook(BuildHookInterface):
    """Adds libpinfer.so and the converter to the wheel and sets its platform tag."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel" or version == "editable":
            return
        root = Path(os.environ.get("PINFER_ROOT", Path(self.root).resolve().parent))
        arch = os.environ.get("PINFER_WHEEL_ARCH")
        sub = {"host": "host", "x86_64": "host", "aarch64": "aarch64", "pi": "aarch64"}.get(
            arch or "")
        if sub is None:
            raise RuntimeError("set PINFER_WHEEL_ARCH=host|aarch64 (use `make -C engine "
                               "wheel-host` or `wheel-pi`); a wheel without libpinfer.so "
                               "would not work")
        lib_dir = root / "build" / sub / "engine" / "lib"
        so = lib_dir / "libpinfer.so"
        if not so.is_file():
            target = "pi-lib" if sub == "aarch64" else "host-lib"
            raise RuntimeError(f"{so} missing: run `make -C engine {target}`")
        data = so.read_bytes()
        tag = f"py3-none-manylinux_2_{glibc_minor(data)}_{elf_arch(data)}"
        build_data["pure_python"] = False
        build_data["tag"] = tag
        inc = build_data.setdefault("force_include", {})
        inc[str(so)] = "pinfer/libpinfer.so"
        if (lib_dir / "BUILD_INFO.json").is_file():
            inc[str(lib_dir / "BUILD_INFO.json")] = "pinfer/BUILD_INFO.json"
        for script in sorted((root / "tools").glob("*.py")):
            inc[str(script)] = f"pinfer/_tools/{script.name}"
        self.app.display_info(f"pinfer wheel: {so} -> {tag}")
