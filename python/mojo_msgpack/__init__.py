from __future__ import annotations

from ._codec import (
    _Decoder,
    _NO_DIRECT_DECODE,
    decode_binary,
    decode_uint_array,
    encode,
)
from .exceptions import *
from .ext import ExtType, Timestamp

version = (0, 1, 0)
__version__ = "0.1.0"


class Packer:
    def __init__(
        self,
        default=None,
        *,
        use_single_float=False,
        autoreset=True,
        use_bin_type=True,
        strict_types=False,
        datetime=False,
        unicode_errors=None,
        buf_size=0x40000,
    ):
        self.default = default
        self.use_single_float = use_single_float
        self.autoreset = autoreset
        self.use_bin_type = use_bin_type
        self.strict_types = strict_types
        self.datetime = datetime
        self.unicode_errors = unicode_errors or "strict"
        self._buffer = bytearray()

    def _finish(self, data):
        if self.autoreset:
            return data
        self._buffer.extend(data)
        return None

    def pack(self, obj):
        data = encode(
            obj,
            default=self.default,
            use_single_float=self.use_single_float,
            use_bin_type=self.use_bin_type,
            strict_types=self.strict_types,
            datetime_enabled=self.datetime,
            unicode_errors=self.unicode_errors,
        )
        return self._finish(data)

    @staticmethod
    def _container_header(length, fixed, code16, code32):
        if not 0 <= length <= 0xFFFFFFFF:
            raise ValueError("length out of range")
        if length <= 15:
            return bytes([fixed | length])
        if length <= 0xFFFF:
            return bytes([code16]) + length.to_bytes(2, "big")
        return bytes([code32]) + length.to_bytes(4, "big")

    def pack_array_header(self, size):
        return self._finish(self._container_header(size, 0x90, 0xDC, 0xDD))

    def pack_map_header(self, size):
        return self._finish(self._container_header(size, 0x80, 0xDE, 0xDF))

    def pack_ext_type(self, typecode, data):
        return self._finish(
            encode(
                ExtType(typecode, data),
                use_single_float=self.use_single_float,
                use_bin_type=self.use_bin_type,
                strict_types=self.strict_types,
                unicode_errors=self.unicode_errors,
            )
        )

    def pack_map_pairs(self, pairs):
        pairs = list(pairs)
        chunks = [self._container_header(len(pairs), 0x80, 0xDE, 0xDF)]
        temporary = Packer(
            default=self.default,
            use_single_float=self.use_single_float,
            use_bin_type=self.use_bin_type,
            strict_types=self.strict_types,
            datetime=self.datetime,
            unicode_errors=self.unicode_errors,
        )
        for key, value in pairs:
            chunks.extend((temporary.pack(key), temporary.pack(value)))
        return self._finish(b"".join(chunks))

    def bytes(self):
        return bytes(self._buffer)

    def getbuffer(self):
        return memoryview(self._buffer)

    def reset(self):
        self._buffer.clear()


def packb(o, **kwargs):
    return Packer(**kwargs).pack(o)


def pack(o, stream, **kwargs):
    stream.write(packb(o, **kwargs))


def unpackb(
    packed,
    *,
    object_hook=None,
    list_hook=None,
    use_list=True,
    raw=False,
    timestamp=0,
    strict_map_key=True,
    unicode_errors=None,
    object_pairs_hook=None,
    ext_hook=ExtType,
    max_str_len=-1,
    max_bin_len=-1,
    max_array_len=-1,
    max_map_len=-1,
    max_ext_len=-1,
):
    try:
        direct_binary = decode_binary(packed, max_bin_len=max_bin_len)
        if direct_binary is not _NO_DIRECT_DECODE:
            return direct_binary
        if list_hook is None:
            direct = decode_uint_array(
                packed, use_list=use_list, max_array_len=max_array_len
            )
            if direct is not None:
                return direct
        return _Decoder(
            packed,
            object_hook=object_hook,
            list_hook=list_hook,
            use_list=use_list,
            raw=raw,
            timestamp=timestamp,
            strict_map_key=strict_map_key,
            unicode_errors=unicode_errors,
            object_pairs_hook=object_pairs_hook,
            ext_hook=ext_hook,
            max_str_len=max_str_len,
            max_bin_len=max_bin_len,
            max_array_len=max_array_len,
            max_map_len=max_map_len,
            max_ext_len=max_ext_len,
        ).one()
    except OutOfData:
        raise ValueError("Unpack failed: incomplete input") from None


def unpack(stream, **kwargs):
    return unpackb(stream.read(), **kwargs)


class Unpacker:
    def __init__(
        self,
        file_like=None,
        read_size=0,
        *,
        use_list=True,
        raw=False,
        timestamp=0,
        strict_map_key=True,
        object_hook=None,
        object_pairs_hook=None,
        list_hook=None,
        unicode_errors=None,
        max_buffer_size=0x6400000,
        ext_hook=ExtType,
        max_str_len=-1,
        max_bin_len=-1,
        max_array_len=-1,
        max_map_len=-1,
        max_ext_len=-1,
    ):
        self.file_like = file_like
        self.read_size = read_size or min(16 * 1024, max_buffer_size)
        self.max_buffer_size = (1 << 32) - 1 if max_buffer_size == 0 else max_buffer_size
        self.options = dict(
            use_list=use_list,
            raw=raw,
            timestamp=timestamp,
            strict_map_key=strict_map_key,
            object_hook=object_hook,
            object_pairs_hook=object_pairs_hook,
            list_hook=list_hook,
            unicode_errors=unicode_errors,
            ext_hook=ext_hook,
            max_str_len=max_str_len,
            max_bin_len=max_bin_len,
            max_array_len=max_array_len,
            max_map_len=max_map_len,
            max_ext_len=max_ext_len,
        )
        self._buffer = bytearray()
        self._offset = 0
        self._last_read = 0

    def feed(self, data):
        if self.file_like is not None:
            raise AssertionError("unpacker is attached to a file")
        if len(self._buffer) + len(data) > self.max_buffer_size:
            raise BufferFull()
        self._buffer.extend(data)

    def _fill(self):
        if self.file_like is None:
            return False
        chunk = self.file_like.read(self.read_size)
        if not chunk:
            return False
        if len(self._buffer) + len(chunk) > self.max_buffer_size:
            raise BufferFull()
        self._buffer.extend(chunk)
        return True

    def unpack(self):
        while True:
            try:
                decoder = _Decoder(self._buffer, **self.options)
                result, used = decoder.first()
                del self._buffer[:used]
                self._offset += used
                self._last_read = used
                return result
            except OutOfData:
                if not self._fill():
                    raise

    def skip(self):
        self.unpack()

    def _read_header(self, array):
        while not self._buffer and self._fill():
            pass
        if not self._buffer:
            raise OutOfData()
        marker = self._buffer[0]
        fixed = 0x90 if array else 0x80
        code16 = 0xDC if array else 0xDE
        code32 = 0xDD if array else 0xDF
        if fixed <= marker <= fixed + 15:
            size, width = marker & 15, 1
        elif marker == code16:
            width = 3
            while len(self._buffer) < width and self._fill():
                pass
            if len(self._buffer) < width:
                raise OutOfData()
            size = int.from_bytes(self._buffer[1:3], "big")
        elif marker == code32:
            width = 5
            while len(self._buffer) < width and self._fill():
                pass
            if len(self._buffer) < width:
                raise OutOfData()
            size = int.from_bytes(self._buffer[1:5], "big")
        else:
            raise ValueError("Unexpected type header on stream")
        del self._buffer[:width]
        self._offset += width
        return size

    def read_array_header(self):
        return self._read_header(True)

    def read_map_header(self):
        return self._read_header(False)

    def tell(self):
        return self._offset

    def read_bytes(self, n):
        while len(self._buffer) < n and self._fill():
            pass
        if len(self._buffer) < n:
            raise OutOfData()
        result = bytes(self._buffer[:n])
        del self._buffer[:n]
        self._offset += n
        return result

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.unpack()
        except OutOfData:
            raise StopIteration from None


load = unpack
loads = unpackb
dump = pack
dumps = packb
