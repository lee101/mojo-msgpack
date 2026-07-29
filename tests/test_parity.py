from __future__ import annotations

import datetime
import io
import math
import random
import struct
from array import array

import msgpack
import numpy as np
import pytest

import mojo_msgpack as mojo


SCALARS = [
    None,
    False,
    True,
    0,
    127,
    128,
    255,
    256,
    65535,
    65536,
    2**32 - 1,
    2**32,
    2**64 - 1,
    -1,
    -32,
    -33,
    -128,
    -129,
    -32768,
    -32769,
    -(2**31),
    -(2**31) - 1,
    -(2**63),
    0.0,
    -0.0,
    1.5,
    math.inf,
    -math.inf,
]


@pytest.mark.parametrize("value", SCALARS)
def test_scalar_encoding_is_byte_exact(value):
    assert mojo.packb(value) == msgpack.packb(value)
    assert mojo.unpackb(msgpack.packb(value)) == value


@pytest.mark.parametrize("length", [0, 1, 31, 32, 255, 256, 65535, 65536])
def test_string_boundaries_are_byte_exact(length):
    value = "x" * length
    packed = msgpack.packb(value)
    assert mojo.packb(value) == packed
    assert mojo.unpackb(packed) == value


@pytest.mark.parametrize("length", [0, 1, 255, 256, 65535, 65536])
def test_binary_boundaries_are_byte_exact(length):
    value = b"x" * length
    packed = msgpack.packb(value)
    assert mojo.packb(value) == packed
    assert mojo.unpackb(packed) == value


@pytest.mark.parametrize("length", [(16 << 20) - 1, (16 << 20) + 17])
def test_large_binary_serial_and_parallel_copy_paths(length):
    value = bytes(range(251)) * (length // 251) + bytes(range(length % 251))
    packed = msgpack.packb(value)
    assert mojo.packb(value) == packed
    assert mojo.unpackb(packed) == value


def test_direct_payload_simd_tails():
    for length in range(130):
        value = bytes((index * 17) % 251 for index in range(length))
        assert mojo.packb(value) == msgpack.packb(value)


def test_flat_integer_array_fast_path_boundaries_and_fallback():
    value = [
        -(1 << 63),
        -(1 << 31) - 1,
        -33,
        -32,
        -1,
        0,
        127,
        128,
        255,
        256,
        (1 << 32) - 1,
        1 << 32,
        (1 << 63) - 1,
    ]
    assert mojo.packb(value) == msgpack.packb(value)
    assert mojo.unpackb(msgpack.packb(value * 32)) == value * 32
    mixed = [1, True, 2]
    assert mojo.packb(mixed) == msgpack.packb(mixed)


def test_multibyte_and_noncontiguous_memoryviews():
    values = memoryview(array("I", [0x01020304, 0xA0B0C0D0]))
    assert mojo.packb(values) == msgpack.packb(values)
    sliced = memoryview(b"abcdef")[::2]
    assert mojo.packb(sliced) == msgpack.packb(bytes(sliced))


@pytest.mark.parametrize("length", [0, 1, 15, 16, 65535, 65536])
def test_array_boundaries_are_byte_exact(length):
    value = [None] * length
    packed = msgpack.packb(value)
    assert mojo.packb(value) == packed
    assert mojo.unpackb(packed) == value


@pytest.mark.parametrize("length", [0, 1, 15, 16, 1000])
def test_map_boundaries_are_byte_exact(length):
    value = {str(i): i for i in range(length)}
    packed = msgpack.packb(value)
    assert mojo.packb(value) == packed
    assert mojo.unpackb(packed) == value


def test_nested_mixed_document_parity():
    value = {
        "project": "mojo-msgpack",
        "ok": True,
        "values": [None, -40000, 2**40, 3.25, b"\x00\xff"],
        "nested": [{"i": i, "even": i % 2 == 0} for i in range(100)],
    }
    packed = msgpack.packb(value)
    assert mojo.packb(value) == packed
    assert mojo.unpackb(packed) == value


def test_unicode_and_error_policy_parity():
    value = "Mojo \N{GREEK SMALL LETTER LAMDA} \N{HIRAGANA LETTER A}"
    assert mojo.packb(value) == msgpack.packb(value)
    invalid = b"\xa1\xff"
    assert mojo.unpackb(invalid, unicode_errors="replace") == msgpack.unpackb(
        invalid, unicode_errors="replace"
    )
    with pytest.raises(UnicodeDecodeError):
        mojo.unpackb(invalid)


def test_single_float_parity():
    for value in [0.1, -123.5, math.inf]:
        packed = msgpack.packb(value, use_single_float=True)
        assert mojo.packb(value, use_single_float=True) == packed
        assert mojo.unpackb(packed) == msgpack.unpackb(packed)


def test_legacy_raw_encoding_and_decoding():
    value = b"x" * 40
    packed = msgpack.packb(value, use_bin_type=False)
    assert mojo.packb(value, use_bin_type=False) == packed
    assert mojo.unpackb(packed, raw=True) == value


def test_tuple_and_strict_types():
    assert mojo.packb((1, 2)) == msgpack.packb((1, 2))
    with pytest.raises(TypeError):
        mojo.packb((1, 2), strict_types=True)


def test_default_converter():
    class Point:
        def __init__(self, x, y):
            self.x, self.y = x, y

    default = lambda point: [point.x, point.y]
    value = Point(2, 3)
    assert mojo.packb(value, default=default) == msgpack.packb(value, default=default)


@pytest.mark.parametrize("length", [1, 2, 3, 4, 8, 16, 17, 255, 256])
def test_extension_encoding_parity(length):
    ours = mojo.ExtType(42, b"x" * length)
    theirs = msgpack.ExtType(42, b"x" * length)
    packed = msgpack.packb(theirs)
    assert mojo.packb(ours) == packed
    assert mojo.unpackb(packed) == ours


def test_ext_hook_parity():
    packed = msgpack.packb(msgpack.ExtType(7, b"payload"))
    hook = lambda code, data: {"code": code, "data": data}
    assert mojo.unpackb(packed, ext_hook=hook) == msgpack.unpackb(packed, ext_hook=hook)


@pytest.mark.parametrize(
    "stamp",
    [
        mojo.Timestamp(1),
        mojo.Timestamp(2**33, 123456789),
        mojo.Timestamp(-1, 999999999),
    ],
)
def test_timestamp_wire_formats(stamp):
    reference = msgpack.Timestamp(stamp.seconds, stamp.nanoseconds)
    assert stamp.to_bytes() == reference.to_bytes()
    assert mojo.packb(stamp) == msgpack.packb(reference)


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_timestamp_unpack_modes(mode):
    stamp = msgpack.Timestamp(1_700_000_000, 123_456_000)
    packed = msgpack.packb(stamp)
    got = mojo.unpackb(packed, timestamp=mode)
    expected = msgpack.unpackb(packed, timestamp=mode)
    if mode == 0:
        assert (got.seconds, got.nanoseconds) == (expected.seconds, expected.nanoseconds)
    else:
        assert got == expected


def test_datetime_option_parity():
    value = datetime.datetime(
        2025, 4, 3, 2, 1, 0, 123456, tzinfo=datetime.timezone.utc
    )
    assert mojo.packb(value, datetime=True) == msgpack.packb(value, datetime=True)
    with pytest.raises(ValueError):
        mojo.packb(value.replace(tzinfo=None), datetime=True)


def test_raw_use_list_and_hooks():
    packed = msgpack.packb({"a": [1, 2]})
    assert mojo.unpackb(packed, raw=True) == {b"a": [1, 2]}
    assert mojo.unpackb(packed, use_list=False) == {"a": (1, 2)}
    assert mojo.unpackb(packed, list_hook=sum) == {"a": 3}
    assert mojo.unpackb(packed, object_hook=lambda d: ("object", d)) == (
        "object",
        {"a": [1, 2]},
    )
    assert mojo.unpackb(packed, object_pairs_hook=tuple) == (("a", [1, 2]),)


def test_strict_map_key_parity():
    packed = msgpack.packb({1: "one"})
    with pytest.raises(ValueError):
        mojo.unpackb(packed)
    assert mojo.unpackb(packed, strict_map_key=False) == {1: "one"}


@pytest.mark.parametrize(
    ("value", "option"),
    [
        ("abc", "max_str_len"),
        (b"abc", "max_bin_len"),
        ([1, 2, 3], "max_array_len"),
        ({"a": 1, "b": 2}, "max_map_len"),
        (mojo.ExtType(1, b"abc"), "max_ext_len"),
    ],
)
def test_length_limits(value, option):
    packed = mojo.packb(value)
    with pytest.raises(ValueError):
        mojo.unpackb(packed, **{option: 1})


@pytest.mark.parametrize("data", [b"", b"\x92\x01", b"\xa3a", b"\xc4\x05x"])
def test_incomplete_input_matches_exception_family(data):
    with pytest.raises(ValueError):
        mojo.unpackb(data)
    with pytest.raises(ValueError):
        msgpack.unpackb(data)


def test_reserved_marker_is_format_error():
    with pytest.raises(mojo.FormatError):
        mojo.unpackb(b"\xc1")


def test_extra_data_exposes_object_and_tail():
    with pytest.raises(mojo.ExtraData) as raised:
        mojo.unpackb(b"\x01\x02")
    assert raised.value.unpacked == 1
    assert raised.value.extra == b"\x02"


def test_integer_overflow_parity():
    for value in [2**64, -(2**63) - 1]:
        with pytest.raises(OverflowError):
            mojo.packb(value)


def test_packer_buffer_and_special_methods():
    packer = mojo.Packer(autoreset=False)
    assert packer.pack(1) is None
    assert packer.pack("x") is None
    assert packer.bytes() == msgpack.packb(1) + msgpack.packb("x")
    packer.reset()
    packer.pack_array_header(3)
    assert packer.bytes() == b"\x93"
    packer.reset()
    packer.pack_map_pairs([("a", 1), ("b", 2)])
    assert packer.bytes() == msgpack.packb({"a": 1, "b": 2})
    packer.reset()
    packer.pack_ext_type(3, b"x")
    assert bytes(packer.getbuffer()) == msgpack.packb(msgpack.ExtType(3, b"x"))


def test_stream_functions_and_aliases():
    stream = io.BytesIO()
    mojo.dump({"x": 1}, stream)
    stream.seek(0)
    assert mojo.load(stream) == {"x": 1}
    assert mojo.dumps([1, 2]) == mojo.packb([1, 2])
    assert mojo.loads(mojo.dumps([1, 2])) == [1, 2]


def test_unpacker_incremental_chunks_and_tell():
    values = [1, "two", [3, 4], {"five": 5}, b"six"]
    wire = b"".join(msgpack.packb(value) for value in values)
    unpacker = mojo.Unpacker()
    decoded = []
    for byte in wire:
        unpacker.feed(bytes([byte]))
        decoded.extend(unpacker)
    assert decoded == values
    assert unpacker.tell() == len(wire)


def test_unpacker_handles_complete_then_partial_object():
    unpacker = mojo.Unpacker()
    unpacker.feed(b"\x01\x92\x02")
    assert unpacker.unpack() == 1
    with pytest.raises(mojo.OutOfData):
        unpacker.unpack()
    unpacker.feed(b"\x03")
    assert unpacker.unpack() == [2, 3]


def test_unpacker_file_and_header_methods():
    wire = msgpack.packb([1, 2]) + msgpack.packb({"a": 3})
    unpacker = mojo.Unpacker(io.BytesIO(wire), read_size=1)
    assert unpacker.read_array_header() == 2
    assert unpacker.unpack() == 1
    assert unpacker.unpack() == 2
    assert unpacker.read_map_header() == 1
    assert unpacker.unpack() == "a"
    assert unpacker.unpack() == 3


def test_unpacker_skip_read_bytes_and_buffer_limit():
    wire = msgpack.packb("skip") + msgpack.packb(42)
    unpacker = mojo.Unpacker()
    unpacker.feed(wire)
    assert unpacker.skip() is None
    assert unpacker.unpack() == 42

    unpacker = mojo.Unpacker()
    unpacker.feed(b"abcdef")
    assert unpacker.read_bytes(3) == b"abc"
    assert unpacker.tell() == 3

    with pytest.raises(mojo.BufferFull):
        mojo.Unpacker(max_buffer_size=2).feed(b"abc")


def test_exported_ffi_rejects_invalid_addresses_and_capacities():
    from mojo_msgpack._lib import lib

    library = lib()
    assert library.mmp_pack_size(0, 0, 0, 1) == -1
    assert library.mmp_token_count(0, 1) == -1

    source = np.frombuffer(b"\x01", dtype=np.uint8)
    output = np.empty(1, dtype=np.uint8)
    assert (
        library.mmp_pack_payload(
            8, source.ctypes.data, 1, output.ctypes.data, output.size
        )
        == -1
    )


def test_noncanonical_published_markers_decode_like_upstream():
    vectors = [
        b"\xcc\x01",
        b"\xcd\x00\x01",
        b"\xce\x00\x00\x00\x01",
        b"\xcf\x00\x00\x00\x00\x00\x00\x00\x01",
        b"\xd0\xff",
        b"\xd1\xff\xff",
        b"\xd2\xff\xff\xff\xff",
        b"\xd3\xff\xff\xff\xff\xff\xff\xff\xff",
        b"\xd9\x01x",
        b"\xda\x00\x01x",
        b"\xdb\x00\x00\x00\x01x",
    ]
    for packed in vectors:
        assert mojo.unpackb(packed) == msgpack.unpackb(packed)


def test_deterministic_random_tree_parity():
    rng = random.Random(7)

    def make(depth):
        if depth == 0:
            return rng.choice([None, rng.randrange(-100000, 100000), rng.random(), "text", b"bin"])
        if rng.randrange(2):
            return [make(depth - 1) for _ in range(rng.randrange(5))]
        return {f"k{i}": make(depth - 1) for i in range(rng.randrange(5))}

    for _ in range(100):
        value = make(3)
        packed = msgpack.packb(value)
        assert mojo.packb(value) == packed
        assert mojo.unpackb(packed) == value
