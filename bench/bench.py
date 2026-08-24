"""Honest end-to-end benchmarks against the upstream C extension."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import msgpack  # noqa: E402
import mojo_msgpack  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as source:
            for line in source:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    integers = list(range(250_000))
    records = [
        {"id": i, "name": f"item-{i}", "active": i % 3 == 0, "scores": [i, i + 1, i + 2]}
        for i in range(30_000)
    ]
    binary = bytes(range(256)) * 32_768
    large_binary = bytes(range(256)) * 131_072

    integer_wire = msgpack.packb(integers)
    record_wire = msgpack.packb(records)
    binary_wire = msgpack.packb(binary)
    assert mojo_msgpack.packb(records) == record_wire
    assert mojo_msgpack.unpackb(record_wire) == records

    cases = [
        (
            "packb 250k integers",
            lambda: mojo_msgpack.packb(integers),
            lambda: msgpack.packb(integers),
        ),
        (
            "unpackb 250k integers",
            lambda: mojo_msgpack.unpackb(integer_wire),
            lambda: msgpack.unpackb(integer_wire),
        ),
        (
            "packb 30k nested records",
            lambda: mojo_msgpack.packb(records),
            lambda: msgpack.packb(records),
        ),
        (
            "unpackb 30k nested records",
            lambda: mojo_msgpack.unpackb(record_wire),
            lambda: msgpack.unpackb(record_wire),
        ),
        (
            "packb 8 MiB binary",
            lambda: mojo_msgpack.packb(binary),
            lambda: msgpack.packb(binary),
        ),
        (
            "unpackb 8 MiB binary",
            lambda: mojo_msgpack.unpackb(binary_wire),
            lambda: msgpack.unpackb(binary_wire),
        ),
        (
            "packb 32 MiB binary",
            lambda: mojo_msgpack.packb(large_binary),
            lambda: msgpack.packb(large_binary),
        ),
    ]

    mojo_msgpack.packb([1, 2, 3])
    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| case | mojo-msgpack | msgpack 1.2.1 | relative | result |")
    print("|---|---:|---:|---:|---|")
    for name, ours, reference in cases:
        mojo_time = timeit(ours)
        reference_time = timeit(reference)
        ratio = reference_time / mojo_time
        result = "faster" if ratio > 1 else "slower"
        print(
            f"| {name} | {mojo_time * 1000:.2f} ms | "
            f"{reference_time * 1000:.2f} ms | {ratio:.3f}x | {result} |"
        )


if __name__ == "__main__":
    main()
