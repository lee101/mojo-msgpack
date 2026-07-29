# mojo-msgpack

`mojo-msgpack` is a standalone MessagePack encoder and decoder whose wire-format
kernels are implemented in Mojo and exposed to Python through `ctypes`. Its Python
API follows the names and signatures of the upstream
[`msgpack`](https://pypi.org/project/msgpack/) package for the covered subset.

The project is useful as a correct MessagePack implementation, a Mojo 1.0 C-ABI
example, and a measured baseline for moving a recursive binary codec across the
Python/Mojo boundary. It is not currently a speed replacement for upstream's
mature C extension; the benchmark results below show that plainly.

## Coverage

Implemented:

- `pack`, `packb`, `dump`, `dumps`, `Packer`
- `unpack`, `unpackb`, `load`, `loads`, streaming `Unpacker`
- nil, booleans, signed and unsigned 64-bit integers
- float32 and float64 encodings
- UTF-8 strings, binary values, arrays, maps, and nested combinations
- fixed and variable-length extension types
- 32-, 64-, and 96-bit standard timestamp extensions
- `datetime=True` packing and timestamp decode modes 0 through 3
- `default`, `strict_types`, `use_single_float`, `use_bin_type`, `raw`,
  `use_list`, all object/list/ext hooks, strict map keys, Unicode error policy,
  and decode length limits
- incremental feeds, file-like input, `skip`, `tell`, `read_bytes`,
  `read_array_header`, and `read_map_header`

Not covered:

- PyPy, Windows, macOS, or architectures other than Linux x86-64
- a prebuilt wheel; installation currently builds the Mojo shared library locally
- zero-copy decode views; decoded strings and binary values are Python-owned
- upstream C extension's exact internal buffer-allocation behavior
- every deprecated compatibility quirk and exact wording of every exception
- Python package-name shadowing; use `import mojo_msgpack as msgpack` to substitute
  it explicitly

The parity suite compares exact encoded bytes and decoded behavior against
upstream `msgpack` 1.2.1, including non-canonical published wire encodings and
malformed or truncated inputs.

## Install

From a source checkout on Linux x86-64:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-msgpack.so`. The Pixi environment sets
`PYTHONPATH=python`, so repository commands find the Python package directly.

## Usage

```python
import mojo_msgpack as msgpack

document = {
    "sensor": "north",
    "samples": [12, 15, 18],
    "valid": True,
    "payload": b"\x00\x01\x02",
}

wire = msgpack.packb(document)
assert msgpack.unpackb(wire) == document

unpacker = msgpack.Unpacker()
unpacker.feed(msgpack.packb(1) + msgpack.packb("two"))
assert list(unpacker) == [1, "two"]
```

Run it with `pixi run python example.py`, or use the same imports interactively
with `pixi run python`.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64. Times are the best of three end-to-end runs and include Python
object traversal and materialization. Relative values are
`upstream time / mojo-msgpack time`; values below 1 mean mojo-msgpack is slower.

| case | mojo-msgpack | msgpack 1.2.1 | relative | result |
|---|---:|---:|---:|---|
| packb 250k integers | 40.79 ms | 13.69 ms | 0.336x | slower |
| unpackb 250k integers | 23.18 ms | 17.47 ms | 0.754x | slower |
| packb 30k nested records | 283.13 ms | 14.87 ms | 0.053x | slower |
| unpackb 30k nested records | 384.92 ms | 31.77 ms | 0.083x | slower |
| packb 8 MiB binary | 1.52 ms | 1.92 ms | 1.261x | faster |
| unpackb 8 MiB binary | 0.67 ms | 0.58 ms | 0.858x | slower |

The result is expected: upstream traverses Python objects inside one highly
optimized C extension. This implementation's general path builds or consumes a
NumPy token tape and then materializes Python objects, so Python work still
dominates nested documents. Homogeneous integer lists bypass recursive tape
construction during packing, and large unsigned-integer arrays have a direct
materialization path during unpacking. Top-level payload inputs cross the FFI
boundary as zero-copy NumPy views. Large payload copies use unaligned-safe SIMD
with a scalar tail; payloads of at least 16 MiB are divided among four CPU workers.

There is intentionally no GPU path. MessagePack marker handling, byte swapping,
and payload copying have low arithmetic intensity, well below the roughly two
operations per byte needed to amortize device transfer and launch costs. A GPU
path would lose to the CPU, so the project does not add the `max` dependency or
compete for shared device memory.

## How it works

General packing flattens a Python object into four contiguous arrays: an 8-bit
type tape and 64-bit value, payload-offset, and payload-length tapes. Variable
string, binary, and extension data share one contiguous byte arena. The 64-bit
tapes are buffer-backed and remain zero-copy when viewed by NumPy. A single
`ctypes` call gives those buffers to Mojo, which calculates the exact encoded
size and writes canonical MessagePack headers, big-endian numbers, and payloads
into caller-owned memory. Homogeneous integer lists and top-level string or
binary payloads use specialized direct kernels.

Unpacking makes two Mojo passes over the wire bytes. The first validates marker
and payload widths and counts tokens; the second emits a compact token tape with
source offsets. Python then reconstructs nested lists and maps and applies the
requested hooks. Streaming decode retains incomplete bytes and consumes exactly
one complete root object at a time.

All FFI buffers cross as integer addresses. Mojo reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside non-parametric
`@export` functions using the C ABI. Python/NumPy owns every allocation, so the
shared library does not expose an allocator or require a matching free call.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```
