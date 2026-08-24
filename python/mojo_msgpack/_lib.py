from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "msgpack.mojo")
LIB = os.path.join(ROOT, "dist", "libmojo-msgpack.so")

I = ctypes.c_int64
B = ctypes.c_bool

_SIGNATURES = {
    "mmp_initialize_runtime": ([], None),
    "mmp_pack_size": ([I, I, I, I], I),
    "mmp_pack": ([I, I, I, I, I, I, I, I, I], I),
    "mmp_pack_int_array": ([I, I, B, I, I], I),
    "mmp_pack_payload": ([I, I, I, I, I], I),
    "mmp_unpack_uint_array": ([I, I, I, I], I),
    "mmp_token_count": ([I, I], I),
    "mmp_tokenize": ([I, I, I, I, I, I, I, I], I),
}


class BuildError(RuntimeError):
    pass


def _mojo_command():
    override = os.environ.get("MOJO_MSGPACK_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_MSGPACK_MOJO=/path/to/mojo")


def build(force=False):
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    command = _mojo_command() + ["build", "--emit", "shared-lib", SRC, "-o", LIB]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode or not os.path.exists(LIB):
        raise BuildError((result.stderr or result.stdout).strip()[:4000])
    return LIB


_library = None


def lib():
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
        _library.mmp_initialize_runtime()
    return _library


def addr(array, dtype):
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype != np.dtype(dtype):
        raise TypeError(f"FFI buffer has dtype {array.dtype}, expected {np.dtype(dtype)}")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if array.ctypes.data == 0:
        raise ValueError("FFI buffers must have a non-null address")
    return array.ctypes.data


if __name__ == "__main__":
    print(build(force="--force" in sys.argv))
