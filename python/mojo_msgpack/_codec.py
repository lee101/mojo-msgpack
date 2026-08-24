from __future__ import annotations

import datetime
import struct
from array import array

import numpy as np

from ._lib import addr, lib
from .exceptions import ExtraData, FormatError, OutOfData, StackError
from .ext import ExtType, Timestamp

NIL, FALSE, TRUE, UINT, SINT, FLOAT32, FLOAT64 = range(7)
STRING, BINARY, ARRAY, MAP, EXT, RAW = range(7, 13)
_MASK64 = (1 << 64) - 1
_FLAT_ARRAY_THRESHOLD = 256
_DIRECT_UINT_ARRAY_THRESHOLD = 256
_NO_DIRECT_DECODE = object()


class _Tape:
    def __init__(
        self,
        *,
        default=None,
        use_single_float=False,
        use_bin_type=True,
        strict_types=False,
        datetime_enabled=False,
        unicode_errors="strict",
    ):
        self.default = default
        self.use_single_float = use_single_float
        self.use_bin_type = use_bin_type
        self.strict_types = strict_types
        self.datetime_enabled = datetime_enabled
        self.unicode_errors = unicode_errors
        self.types = bytearray()
        self.values = array("Q")
        self.offsets = array("Q")
        self.lengths = array("Q")
        self.payload = bytearray()

    def add(self, kind, value=0, data=None):
        self.types.append(kind)
        self.values.append(value & _MASK64)
        if data is None:
            self.offsets.append(0)
            self.lengths.append(0)
        else:
            if len(data) > 0xFFFFFFFF:
                raise OverflowError("payload is too large")
            self.offsets.append(len(self.payload))
            self.lengths.append(len(data))
            self.payload.extend(data)

    def flatten(self, value, depth=0):
        if depth > 511:
            raise ValueError("recursion limit exceeded")
        exact = type(value)
        if value is None:
            self.add(NIL)
        elif exact is bool:
            self.add(TRUE if value else FALSE)
        elif exact is int or (not self.strict_types and isinstance(value, int)):
            if not -(1 << 63) <= value <= (1 << 64) - 1:
                raise OverflowError("Integer value out of range")
            self.add(SINT if value < 0 else UINT, value)
        elif exact is float or (not self.strict_types and isinstance(value, float)):
            if self.use_single_float:
                bits = struct.unpack("!I", struct.pack("!f", value))[0]
                self.add(FLOAT32, bits)
            else:
                bits = struct.unpack("!Q", struct.pack("!d", value))[0]
                self.add(FLOAT64, bits)
        elif exact is str or (not self.strict_types and isinstance(value, str)):
            data = value.encode("utf-8", self.unicode_errors)
            self.add(STRING, data=data)
        elif exact in (bytes, bytearray, memoryview) or (
            not self.strict_types and isinstance(value, (bytes, bytearray, memoryview))
        ):
            data = bytes(value)
            self.add(BINARY if self.use_bin_type else RAW, data=data)
        elif exact is Timestamp:
            self.add(EXT, 255, value.to_bytes())
        elif exact is ExtType:
            self.add(EXT, value.code, value.data)
        elif exact is datetime.datetime and self.datetime_enabled:
            if value.tzinfo is None:
                raise ValueError("Cannot convert naive datetime to timestamp")
            stamp = Timestamp.from_datetime(value)
            self.add(EXT, 255, stamp.to_bytes())
        elif exact is list or (not self.strict_types and isinstance(value, list)):
            if len(value) > 0xFFFFFFFF:
                raise OverflowError("array is too large")
            self.add(ARRAY, len(value))
            for item in value:
                self.flatten(item, depth + 1)
        elif (exact is tuple and not self.strict_types) or (
            not self.strict_types and isinstance(value, tuple)
        ):
            self.add(ARRAY, len(value))
            for item in value:
                self.flatten(item, depth + 1)
        elif exact is dict or (not self.strict_types and isinstance(value, dict)):
            if len(value) > 0xFFFFFFFF:
                raise OverflowError("map is too large")
            self.add(MAP, len(value))
            for key, item in value.items():
                self.flatten(key, depth + 1)
                self.flatten(item, depth + 1)
        elif self.default is not None:
            self.flatten(self.default(value), depth + 1)
        else:
            raise TypeError(f"can not serialize {exact.__name__!r} object")


def _payload_source(data):
    try:
        view = memoryview(data)
        if not view.c_contiguous:
            raise BufferError
        return np.frombuffer(view.cast("B"), dtype=np.uint8)
    except (TypeError, BufferError):
        return np.frombuffer(bytes(data), dtype=np.uint8)


def _encode_payload(kind, data):
    source = _payload_source(data)
    length = source.size
    if length > 0xFFFFFFFF:
        raise OverflowError("payload is too large")
    if not length:
        source = np.frombuffer(b"\0", dtype=np.uint8)
    destination = np.empty(length + 5, dtype=np.uint8)
    written = lib().mmp_pack_payload(
        kind, addr(source, np.uint8), length, addr(destination, np.uint8), destination.size
    )
    if written < 0:
        raise RuntimeError("invalid direct payload kind")
    return destination[:written].tobytes()


def _encode_int_list(value):
    length = len(value)
    if length > 0xFFFFFFFF:
        raise OverflowError("array is too large")
    if not length:
        return b"\x90"
    try:
        storage = array("Q", value)
        dtype = np.uint64
        signed = False
    except (OverflowError, TypeError):
        try:
            storage = array("q", value)
            dtype = np.int64
            signed = True
        except (OverflowError, TypeError):
            return None
    if not all(type(item) is int for item in value):
        return None
    values = np.frombuffer(storage, dtype=dtype)
    destination = np.empty(5 + 9 * length, dtype=np.uint8)
    written = lib().mmp_pack_int_array(
        addr(values, values.dtype),
        length,
        signed,
        addr(destination, np.uint8),
        destination.size,
    )
    if written < 0:
        raise RuntimeError("Mojo integer-array encoder rejected valid input")
    return destination[:written].tobytes()


def encode(value, **options):
    strict_types = options.get("strict_types", False)
    exact = type(value)
    if exact is str or (not strict_types and isinstance(value, str)):
        return _encode_payload(
            STRING,
            value.encode("utf-8", options.get("unicode_errors", "strict")),
        )
    if exact in (bytes, bytearray, memoryview) or (
        not strict_types and isinstance(value, (bytes, bytearray, memoryview))
    ):
        return _encode_payload(
            BINARY if options.get("use_bin_type", True) else RAW,
            value,
        )
    if exact is list:
        packed = _encode_int_list(value)
        if packed is not None:
            return packed

    tape = _Tape(**options)
    tape.flatten(value)
    types = np.frombuffer(tape.types, dtype=np.uint8)
    values = np.frombuffer(tape.values, dtype=np.uint64)
    offsets = np.frombuffer(tape.offsets, dtype=np.uint64)
    lengths = np.frombuffer(tape.lengths, dtype=np.uint64)
    payload = np.frombuffer(tape.payload or b"\0", dtype=np.uint8)
    library = lib()
    size = library.mmp_pack_size(
        addr(types, np.uint8),
        addr(values, np.uint64),
        addr(lengths, np.uint64),
        len(types),
    )
    if size < 0:
        raise RuntimeError("invalid internal token tape")
    destination = np.empty(max(size, 1), dtype=np.uint8)
    written = library.mmp_pack(
        addr(types, np.uint8),
        addr(values, np.uint64),
        addr(offsets, np.uint64),
        addr(lengths, np.uint64),
        addr(payload, np.uint8),
        payload.size,
        len(types),
        addr(destination, np.uint8),
        destination.size,
    )
    if written != size:
        raise RuntimeError("Mojo encoder produced an unexpected size")
    return destination[:size].tobytes()


class _Decoder:
    def __init__(self, packed, **options):
        self.data = bytes(packed)
        self.use_list = options.get("use_list", True)
        self.raw = options.get("raw", False)
        self.timestamp = options.get("timestamp", 0)
        self.strict_map_key = options.get("strict_map_key", True)
        self.object_hook = options.get("object_hook")
        self.object_pairs_hook = options.get("object_pairs_hook")
        self.list_hook = options.get("list_hook")
        self.ext_hook = options.get("ext_hook", ExtType)
        self.unicode_errors = options.get("unicode_errors") or "strict"
        self.limits = {
            STRING: options.get("max_str_len", -1),
            BINARY: options.get("max_bin_len", -1),
            ARRAY: options.get("max_array_len", -1),
            MAP: options.get("max_map_len", -1),
            EXT: options.get("max_ext_len", -1),
        }
        self._tokenize()

    def _tokenize(self):
        source = np.frombuffer(self.data or b"\0", dtype=np.uint8)
        library = lib()
        count = library.mmp_token_count(addr(source, np.uint8), len(self.data))
        self.error_offset = None
        valid_size = len(self.data)
        if count < 0:
            self.error_offset = -count - 1
            valid_size = self.error_offset
            count = library.mmp_token_count(addr(source, np.uint8), valid_size)
            if count < 0:
                raise FormatError("invalid MessagePack input")
        capacity = max(count, 1)
        self.types = np.empty(capacity, dtype=np.uint8)
        self.values = np.empty(capacity, dtype=np.uint64)
        self.offsets = np.empty(capacity, dtype=np.uint64)
        self.lengths = np.empty(capacity, dtype=np.uint64)
        self.ends = np.empty(capacity, dtype=np.uint64)
        got = library.mmp_tokenize(
            addr(source, np.uint8),
            valid_size,
            addr(self.types, np.uint8),
            addr(self.values, np.uint64),
            addr(self.offsets, np.uint64),
            addr(self.lengths, np.uint64),
            addr(self.ends, np.uint64),
            capacity,
        )
        if got != count:
            raise RuntimeError("Mojo tokenizer produced an unexpected token count")
        self.count = count
        if count >= _FLAT_ARRAY_THRESHOLD and not np.all(self.types[1:count] == UINT):
            self.types = self.types.tolist()

    def _check_limit(self, kind, length):
        limit = self.limits.get(kind, -1)
        if limit is not None and limit >= 0 and length > limit:
            raise ValueError(f"{length} exceeds configured limit")

    def parse(self, index=0, depth=0):
        if index >= self.count:
            if self.error_offset is not None and self.data[self.error_offset] == 0xC1:
                raise FormatError("invalid MessagePack marker 0xc1")
            raise OutOfData("No more data to unpack")
        if depth > 511:
            raise StackError("too deeply nested")
        kind = int(self.types[index])
        next_index = index + 1
        if kind == NIL:
            result = None
        elif kind == FALSE:
            result = False
        elif kind == TRUE:
            result = True
        elif kind == UINT:
            result = int(self.values[index])
        elif kind == SINT:
            value = int(self.values[index])
            result = value - (1 << 64) if value >= 1 << 63 else value
        elif kind == FLOAT32:
            value = int(self.values[index])
            result = struct.unpack("!f", struct.pack("!I", value))[0]
        elif kind == FLOAT64:
            value = int(self.values[index])
            result = struct.unpack("!d", struct.pack("!Q", value))[0]
        elif kind in (STRING, BINARY):
            length = int(self.lengths[index])
            self._check_limit(kind, length)
            start = int(self.offsets[index])
            payload = self.data[start : start + length]
            result = (
                payload.decode("utf-8", self.unicode_errors)
                if kind == STRING and not self.raw
                else payload
            )
        elif kind == EXT:
            value = int(self.values[index])
            length = int(self.lengths[index])
            self._check_limit(kind, length)
            start = int(self.offsets[index])
            payload = self.data[start : start + length]
            code = value if value < 128 else value - 256
            if code == -1:
                stamp = Timestamp.from_bytes(payload)
                if self.timestamp == 0:
                    result = stamp
                elif self.timestamp == 1:
                    result = stamp.to_unix()
                elif self.timestamp == 2:
                    result = stamp.to_unix_nano()
                elif self.timestamp == 3:
                    result = stamp.to_datetime()
                else:
                    raise ValueError("timestamp must be 0, 1, 2, or 3")
            else:
                result = self.ext_hook(code, payload)
        elif kind == ARRAY:
            value = int(self.values[index])
            self._check_limit(kind, value)
            if (
                value >= _FLAT_ARRAY_THRESHOLD
                and next_index + value <= self.count
                and (
                    np.all(self.types[next_index : next_index + value] == UINT)
                    if isinstance(self.types, np.ndarray)
                    else all(
                        kind == UINT
                        for kind in self.types[next_index : next_index + value]
                    )
                )
            ):
                items = self.values[next_index : next_index + value].tolist()
                next_index += value
            else:
                flat_end = next_index + value
                flat_uint = 3 <= value <= 16 and flat_end <= self.count
                if flat_uint:
                    for token_index in range(next_index, flat_end):
                        if self.types[token_index] != UINT:
                            flat_uint = False
                            break
                if flat_uint:
                    items = self.values[next_index:flat_end].tolist()
                    next_index = flat_end
                else:
                    items = []
                    for _ in range(value):
                        item, next_index = self.parse(next_index, depth + 1)
                        items.append(item)
            result = items if self.use_list else tuple(items)
            if self.list_hook is not None:
                result = self.list_hook(result)
        elif kind == MAP:
            value = int(self.values[index])
            self._check_limit(kind, value)
            pairs = []
            for _ in range(value):
                key, next_index = self.parse(next_index, depth + 1)
                if self.strict_map_key and not isinstance(key, (str, bytes)):
                    raise ValueError(f"{type(key).__name__} is not allowed for map key")
                item, next_index = self.parse(next_index, depth + 1)
                pairs.append((key, item))
            if self.object_pairs_hook is not None:
                result = self.object_pairs_hook(pairs)
            else:
                result = dict(pairs)
                if self.object_hook is not None:
                    result = self.object_hook(result)
        else:
            raise FormatError("unknown internal token")
        return result, next_index

    def one(self):
        result, next_index = self.parse()
        end = int(self.ends[next_index - 1])
        if next_index != self.count or self.error_offset is not None:
            raise ExtraData(result, self.data[end:])
        return result

    def first(self):
        result, next_index = self.parse()
        return result, int(self.ends[next_index - 1])


def decode_uint_array(packed, *, use_list=True, max_array_len=-1):
    data = packed if type(packed) is bytes else bytes(packed)
    size = len(data)
    if not size:
        return None
    marker = data[0]
    if 0x90 <= marker <= 0x9F:
        count = marker & 15
    elif marker == 0xDC and size >= 3:
        count = int.from_bytes(data[1:3], "big")
    elif marker == 0xDD and size >= 5:
        count = int.from_bytes(data[1:5], "big")
    else:
        return None
    if count < _DIRECT_UINT_ARRAY_THRESHOLD:
        return None
    if max_array_len is not None and max_array_len >= 0 and count > max_array_len:
        raise ValueError(f"{count} exceeds configured limit")
    source = np.frombuffer(data, dtype=np.uint8)
    values = np.empty(count, dtype=np.uint64)
    got = lib().mmp_unpack_uint_array(
        addr(source, np.uint8), size, addr(values, np.uint64), count
    )
    if got != count:
        return None
    result = values.tolist()
    return result if use_list else tuple(result)


def decode_binary(packed, *, max_bin_len=-1):
    data = packed if type(packed) is bytes else bytes(packed)
    size = len(data)
    if size < 2:
        return _NO_DIRECT_DECODE
    marker = data[0]
    if marker == 0xC4:
        payload_start = 2
        length = data[1]
    elif marker == 0xC5 and size >= 3:
        payload_start = 3
        length = int.from_bytes(data[1:3], "big")
    elif marker == 0xC6 and size >= 5:
        payload_start = 5
        length = int.from_bytes(data[1:5], "big")
    else:
        return _NO_DIRECT_DECODE
    if payload_start + length != size:
        return _NO_DIRECT_DECODE
    if max_bin_len is not None and max_bin_len >= 0 and length > max_bin_len:
        raise ValueError(f"{length} exceeds configured limit")
    return data[payload_start:]
