"""MessagePack wire encoder and tokenizer exposed through a small C ABI."""

from std.sys.info import simd_width_of as simdwidthof

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime UPtr = UnsafePointer[UInt64, AnyOrigin[mut=True]]

comptime NIL = 0
comptime FALSE = 1
comptime TRUE = 2
comptime UINT = 3
comptime SINT = 4
comptime FLOAT32 = 5
comptime FLOAT64 = 6
comptime STRING = 7
comptime BINARY = 8
comptime ARRAY = 9
comptime MAP = 10
comptime EXT = 11
comptime RAW = 12
comptime PARALLEL_COPY_THRESHOLD = 16 << 20
comptime COPY_CHUNK_SIZE = 1 << 18


def _put_u16(dst: BPtr, pos: Int, value: UInt64):
    dst[pos] = UInt8(value >> 8)
    dst[pos + 1] = UInt8(value)


def _put_u32(dst: BPtr, pos: Int, value: UInt64):
    dst[pos] = UInt8(value >> 24)
    dst[pos + 1] = UInt8(value >> 16)
    dst[pos + 2] = UInt8(value >> 8)
    dst[pos + 3] = UInt8(value)


def _put_u64(dst: BPtr, pos: Int, value: UInt64):
    dst[pos] = UInt8(value >> 56)
    dst[pos + 1] = UInt8(value >> 48)
    dst[pos + 2] = UInt8(value >> 40)
    dst[pos + 3] = UInt8(value >> 32)
    dst[pos + 4] = UInt8(value >> 24)
    dst[pos + 5] = UInt8(value >> 16)
    dst[pos + 6] = UInt8(value >> 8)
    dst[pos + 7] = UInt8(value)


@always_inline
def _copy_bytes_serial(
    dst: BPtr,
    dst_pos: Int,
    src: BPtr,
    src_pos: Int,
    length: Int,
):
    comptime W = simdwidthof[DType.float64]()
    comptime BYTE_W = W * 8
    var i = Int(0)
    if length < BYTE_W:
        while i < length:
            dst[dst_pos + i] = src[src_pos + i]
            i += 1
        return
    while i + BYTE_W <= length:
        var values = src.load[width=BYTE_W, alignment=1](src_pos + i)
        dst.store[alignment=1](dst_pos + i, values)
        i += BYTE_W
    while i < length:
        dst[dst_pos + i] = src[src_pos + i]
        i += 1


@always_inline
def _copy_bytes(
    dst: BPtr,
    dst_pos: Int,
    src: BPtr,
    src_pos: Int,
    length: Int,
):
    if length < PARALLEL_COPY_THRESHOLD:
        _copy_bytes_serial(dst, dst_pos, src, src_pos, length)
        return

    var chunks = (length + COPY_CHUNK_SIZE - 1) // COPY_CHUNK_SIZE

    for chunk in range(chunks):
        var start = chunk * COPY_CHUNK_SIZE
        var amount = min(COPY_CHUNK_SIZE, length - start)
        _copy_bytes_serial(dst, dst_pos + start, src, src_pos + start, amount)


def _get_u16(src: BPtr, pos: Int) -> UInt64:
    return (UInt64(src[pos]) << 8) | UInt64(src[pos + 1])


def _get_u32(src: BPtr, pos: Int) -> UInt64:
    return (
        (UInt64(src[pos]) << 24)
        | (UInt64(src[pos + 1]) << 16)
        | (UInt64(src[pos + 2]) << 8)
        | UInt64(src[pos + 3])
    )


def _get_u64(src: BPtr, pos: Int) -> UInt64:
    return (
        (UInt64(src[pos]) << 56)
        | (UInt64(src[pos + 1]) << 48)
        | (UInt64(src[pos + 2]) << 40)
        | (UInt64(src[pos + 3]) << 32)
        | (UInt64(src[pos + 4]) << 24)
        | (UInt64(src[pos + 5]) << 16)
        | (UInt64(src[pos + 6]) << 8)
        | UInt64(src[pos + 7])
    )


def _uint_size(value: UInt64) -> Int:
    if value <= 0x7f:
        return 1
    if value <= 0xff:
        return 2
    if value <= 0xffff:
        return 3
    if value <= 0xffffffff:
        return 5
    return 9


def _sint_size(value: UInt64) -> Int:
    if value >= 0xffffffffffffffe0:
        return 1
    if value >= 0xffffffffffffff80:
        return 2
    if value >= 0xffffffffffff8000:
        return 3
    if value >= 0xffffffff80000000:
        return 5
    return 9


def _str_header(length: UInt64, allow_str8: Bool) -> Int:
    if length <= 31:
        return 1
    if allow_str8 and length <= 0xff:
        return 2
    if length <= 0xffff:
        return 3
    return 5


def _bin_header(length: UInt64) -> Int:
    if length <= 0xff:
        return 2
    if length <= 0xffff:
        return 3
    return 5


def _container_header(length: UInt64) -> Int:
    if length <= 15:
        return 1
    if length <= 0xffff:
        return 3
    return 5


def _ext_header(length: UInt64) -> Int:
    if length == 1 or length == 2 or length == 4 or length == 8 or length == 16:
        return 2
    if length <= 0xff:
        return 3
    if length <= 0xffff:
        return 4
    return 6


@export("mmp_pack_size")
def mmp_pack_size(types_addr: Int, values_addr: Int, lengths_addr: Int, count: Int) abi("C") -> Int:
    if count < 0 or (count > 0 and (types_addr == 0 or values_addr == 0 or lengths_addr == 0)):
        return -1
    if count == 0:
        return 0
    var types = BPtr(unsafe_from_address=types_addr)
    var values = UPtr(unsafe_from_address=values_addr)
    var lengths = UPtr(unsafe_from_address=lengths_addr)
    var total = Int(0)
    for i in range(count):
        var kind = Int(types[i])
        var value = values[i]
        var length = lengths[i]
        if kind == NIL or kind == FALSE or kind == TRUE:
            total += 1
        elif kind == UINT:
            total += _uint_size(value)
        elif kind == SINT:
            total += _sint_size(value)
        elif kind == FLOAT32:
            total += 5
        elif kind == FLOAT64:
            total += 9
        elif kind == STRING:
            total += _str_header(length, True) + Int(length)
        elif kind == RAW:
            total += _str_header(length, False) + Int(length)
        elif kind == BINARY:
            total += _bin_header(length) + Int(length)
        elif kind == ARRAY or kind == MAP:
            total += _container_header(value)
        elif kind == EXT:
            total += _ext_header(length) + Int(length)
        else:
            return -1
    return total


@export("mmp_pack")
def mmp_pack(
    types_addr: Int,
    values_addr: Int,
    offsets_addr: Int,
    lengths_addr: Int,
    payload_addr: Int,
    payload_size: Int,
    count: Int,
    dst_addr: Int,
    dst_size: Int,
) abi("C") -> Int:
    if count < 0 or payload_size < 0 or dst_size < 0:
        return -1
    if count == 0:
        return 0
    if (
        types_addr == 0
        or values_addr == 0
        or offsets_addr == 0
        or lengths_addr == 0
        or payload_addr == 0
        or dst_addr == 0
    ):
        return -1
    var required = mmp_pack_size(types_addr, values_addr, lengths_addr, count)
    if required < 0 or required > dst_size:
        return -1
    var types = BPtr(unsafe_from_address=types_addr)
    var values = UPtr(unsafe_from_address=values_addr)
    var offsets = UPtr(unsafe_from_address=offsets_addr)
    var lengths = UPtr(unsafe_from_address=lengths_addr)
    var payload = BPtr(unsafe_from_address=payload_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    for i in range(count):
        var kind = Int(types[i])
        if kind == STRING or kind == RAW or kind == BINARY or kind == EXT:
            if (
                offsets[i] > UInt64(payload_size)
                or lengths[i] > UInt64(payload_size) - offsets[i]
            ):
                return -1
    var pos = Int(0)
    for i in range(count):
        var kind = Int(types[i])
        var value = values[i]
        var length = lengths[i]
        if kind == NIL:
            dst[pos] = 0xc0
            pos += 1
        elif kind == FALSE:
            dst[pos] = 0xc2
            pos += 1
        elif kind == TRUE:
            dst[pos] = 0xc3
            pos += 1
        elif kind == UINT:
            if value <= 0x7f:
                dst[pos] = UInt8(value)
                pos += 1
            elif value <= 0xff:
                dst[pos] = 0xcc
                dst[pos + 1] = UInt8(value)
                pos += 2
            elif value <= 0xffff:
                dst[pos] = 0xcd
                _put_u16(dst, pos + 1, value)
                pos += 3
            elif value <= 0xffffffff:
                dst[pos] = 0xce
                _put_u32(dst, pos + 1, value)
                pos += 5
            else:
                dst[pos] = 0xcf
                _put_u64(dst, pos + 1, value)
                pos += 9
        elif kind == SINT:
            if value >= 0xffffffffffffffe0:
                dst[pos] = UInt8(value)
                pos += 1
            elif value >= 0xffffffffffffff80:
                dst[pos] = 0xd0
                dst[pos + 1] = UInt8(value)
                pos += 2
            elif value >= 0xffffffffffff8000:
                dst[pos] = 0xd1
                _put_u16(dst, pos + 1, value)
                pos += 3
            elif value >= 0xffffffff80000000:
                dst[pos] = 0xd2
                _put_u32(dst, pos + 1, value)
                pos += 5
            else:
                dst[pos] = 0xd3
                _put_u64(dst, pos + 1, value)
                pos += 9
        elif kind == FLOAT32:
            dst[pos] = 0xca
            _put_u32(dst, pos + 1, value)
            pos += 5
        elif kind == FLOAT64:
            dst[pos] = 0xcb
            _put_u64(dst, pos + 1, value)
            pos += 9
        elif kind == STRING or kind == RAW:
            if length <= 31:
                dst[pos] = UInt8(0xa0 | length)
                pos += 1
            elif kind == STRING and length <= 0xff:
                dst[pos] = 0xd9
                dst[pos + 1] = UInt8(length)
                pos += 2
            elif length <= 0xffff:
                dst[pos] = 0xda
                _put_u16(dst, pos + 1, length)
                pos += 3
            else:
                dst[pos] = 0xdb
                _put_u32(dst, pos + 1, length)
                pos += 5
            for j in range(Int(length)):
                dst[pos + j] = payload[Int(offsets[i]) + j]
            pos += Int(length)
        elif kind == BINARY:
            if length <= 0xff:
                dst[pos] = 0xc4
                dst[pos + 1] = UInt8(length)
                pos += 2
            elif length <= 0xffff:
                dst[pos] = 0xc5
                _put_u16(dst, pos + 1, length)
                pos += 3
            else:
                dst[pos] = 0xc6
                _put_u32(dst, pos + 1, length)
                pos += 5
            for j in range(Int(length)):
                dst[pos + j] = payload[Int(offsets[i]) + j]
            pos += Int(length)
        elif kind == ARRAY or kind == MAP:
            if value <= 15:
                dst[pos] = UInt8((UInt64(0x90) if kind == ARRAY else UInt64(0x80)) | value)
                pos += 1
            elif value <= 0xffff:
                dst[pos] = 0xdc if kind == ARRAY else 0xde
                _put_u16(dst, pos + 1, value)
                pos += 3
            else:
                dst[pos] = 0xdd if kind == ARRAY else 0xdf
                _put_u32(dst, pos + 1, value)
                pos += 5
        elif kind == EXT:
            if length == 1:
                dst[pos] = 0xd4
                pos += 1
            elif length == 2:
                dst[pos] = 0xd5
                pos += 1
            elif length == 4:
                dst[pos] = 0xd6
                pos += 1
            elif length == 8:
                dst[pos] = 0xd7
                pos += 1
            elif length == 16:
                dst[pos] = 0xd8
                pos += 1
            elif length <= 0xff:
                dst[pos] = 0xc7
                dst[pos + 1] = UInt8(length)
                pos += 2
            elif length <= 0xffff:
                dst[pos] = 0xc8
                _put_u16(dst, pos + 1, length)
                pos += 3
            else:
                dst[pos] = 0xc9
                _put_u32(dst, pos + 1, length)
                pos += 5
            dst[pos] = UInt8(value)
            pos += 1
            for j in range(Int(length)):
                dst[pos + j] = payload[Int(offsets[i]) + j]
            pos += Int(length)
        else:
            return -1
    return pos


@export("mmp_pack_int_array")
def mmp_pack_int_array(
    values_addr: Int,
    count: Int,
    signed_input: Bool,
    dst_addr: Int,
    dst_size: Int,
) abi("C") -> Int:
    if (
        count < 0
        or dst_size < 0
        or values_addr == 0
        or dst_addr == 0
        or count > (dst_size - 5) // 9
    ):
        return -1
    var values = UPtr(unsafe_from_address=values_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var pos = Int(0)
    var array_length = UInt64(count)
    if array_length <= 15:
        dst[pos] = UInt8(0x90 | array_length)
        pos += 1
    elif array_length <= 0xffff:
        dst[pos] = 0xdc
        _put_u16(dst, pos + 1, array_length)
        pos += 3
    else:
        dst[pos] = 0xdd
        _put_u32(dst, pos + 1, array_length)
        pos += 5

    for i in range(count):
        var value = values[i]
        if signed_input and value >= 0x8000000000000000:
            if value >= 0xffffffffffffffe0:
                dst[pos] = UInt8(value)
                pos += 1
            elif value >= 0xffffffffffffff80:
                dst[pos] = 0xd0
                dst[pos + 1] = UInt8(value)
                pos += 2
            elif value >= 0xffffffffffff8000:
                dst[pos] = 0xd1
                _put_u16(dst, pos + 1, value)
                pos += 3
            elif value >= 0xffffffff80000000:
                dst[pos] = 0xd2
                _put_u32(dst, pos + 1, value)
                pos += 5
            else:
                dst[pos] = 0xd3
                _put_u64(dst, pos + 1, value)
                pos += 9
        elif value <= 0x7f:
            dst[pos] = UInt8(value)
            pos += 1
        elif value <= 0xff:
            dst[pos] = 0xcc
            dst[pos + 1] = UInt8(value)
            pos += 2
        elif value <= 0xffff:
            dst[pos] = 0xcd
            _put_u16(dst, pos + 1, value)
            pos += 3
        elif value <= 0xffffffff:
            dst[pos] = 0xce
            _put_u32(dst, pos + 1, value)
            pos += 5
        else:
            dst[pos] = 0xcf
            _put_u64(dst, pos + 1, value)
            pos += 9
    return pos


@export("mmp_pack_payload")
def mmp_pack_payload(
    kind: Int,
    src_addr: Int,
    length: Int,
    dst_addr: Int,
    dst_size: Int,
) abi("C") -> Int:
    if length < 0 or src_addr == 0 or dst_addr == 0 or dst_size < length + 5:
        return -1
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var pos = Int(0)
    var size = UInt64(length)
    if kind == STRING or kind == RAW:
        if size <= 31:
            dst[pos] = UInt8(0xa0 | size)
            pos += 1
        elif kind == STRING and size <= 0xff:
            dst[pos] = 0xd9
            dst[pos + 1] = UInt8(size)
            pos += 2
        elif size <= 0xffff:
            dst[pos] = 0xda
            _put_u16(dst, pos + 1, size)
            pos += 3
        else:
            dst[pos] = 0xdb
            _put_u32(dst, pos + 1, size)
            pos += 5
    elif kind == BINARY:
        if size <= 0xff:
            dst[pos] = 0xc4
            dst[pos + 1] = UInt8(size)
            pos += 2
        elif size <= 0xffff:
            dst[pos] = 0xc5
            _put_u16(dst, pos + 1, size)
            pos += 3
        else:
            dst[pos] = 0xc6
            _put_u32(dst, pos + 1, size)
            pos += 5
    else:
        return -1
    _copy_bytes(dst, pos, src, 0, length)
    return pos + length


def _need(pos: Int, amount: Int, size: Int) -> Bool:
    return amount >= 0 and pos <= size and amount <= size - pos


@export("mmp_token_count")
def mmp_token_count(src_addr: Int, size: Int) abi("C") -> Int:
    if size < 0 or src_addr == 0:
        return -1
    var src = BPtr(unsafe_from_address=src_addr)
    var pos = Int(0)
    var count = Int(0)
    while pos < size:
        var marker_pos = pos
        var marker = Int(src[pos])
        pos += 1
        var payload_len = UInt64(0)
        var header_extra = Int(0)
        if marker <= 0x7f or marker >= 0xe0 or (marker >= 0x80 and marker <= 0x9f):
            pass
        elif marker >= 0xa0 and marker <= 0xbf:
            payload_len = UInt64(marker & 31)
        elif marker == 0xc0 or marker == 0xc2 or marker == 0xc3:
            pass
        elif marker == 0xcc or marker == 0xd0:
            header_extra = 1
        elif marker == 0xcd or marker == 0xd1:
            header_extra = 2
        elif marker == 0xce or marker == 0xca or marker == 0xd2:
            header_extra = 4
        elif marker == 0xcf or marker == 0xcb or marker == 0xd3:
            header_extra = 8
        elif marker == 0xd9 or marker == 0xc4:
            if not _need(pos, 1, size):
                return -(marker_pos + 1)
            payload_len = UInt64(src[pos])
            header_extra = 1
        elif marker == 0xda or marker == 0xc5:
            if not _need(pos, 2, size):
                return -(marker_pos + 1)
            payload_len = _get_u16(src, pos)
            header_extra = 2
        elif marker == 0xdb or marker == 0xc6:
            if not _need(pos, 4, size):
                return -(marker_pos + 1)
            payload_len = _get_u32(src, pos)
            header_extra = 4
        elif marker == 0xdc or marker == 0xde:
            header_extra = 2
        elif marker == 0xdd or marker == 0xdf:
            header_extra = 4
        elif marker >= 0xd4 and marker <= 0xd8:
            payload_len = UInt64(1 << (marker - 0xd4))
            header_extra = 1
        elif marker == 0xc7:
            if not _need(pos, 1, size):
                return -(marker_pos + 1)
            payload_len = UInt64(src[pos])
            header_extra = 2
        elif marker == 0xc8:
            if not _need(pos, 2, size):
                return -(marker_pos + 1)
            payload_len = _get_u16(src, pos)
            header_extra = 3
        elif marker == 0xc9:
            if not _need(pos, 4, size):
                return -(marker_pos + 1)
            payload_len = _get_u32(src, pos)
            header_extra = 5
        elif marker == 0xc1:
            return -(marker_pos + 1)
        else:
            return -(marker_pos + 1)
        if not _need(pos, header_extra, size):
            return -(marker_pos + 1)
        pos += header_extra
        if not _need(pos, Int(payload_len), size):
            return -(marker_pos + 1)
        pos += Int(payload_len)
        count += 1
    return count


@export("mmp_tokenize")
def mmp_tokenize(
    src_addr: Int,
    size: Int,
    types_addr: Int,
    values_addr: Int,
    offsets_addr: Int,
    lengths_addr: Int,
    ends_addr: Int,
    capacity: Int,
) abi("C") -> Int:
    if (
        size < 0
        or capacity < 0
        or src_addr == 0
        or types_addr == 0
        or values_addr == 0
        or offsets_addr == 0
        or lengths_addr == 0
        or ends_addr == 0
    ):
        return -1
    var expected = mmp_token_count(src_addr, size)
    if expected < 0 or expected > capacity:
        return -1
    var src = BPtr(unsafe_from_address=src_addr)
    var types = BPtr(unsafe_from_address=types_addr)
    var values = UPtr(unsafe_from_address=values_addr)
    var offsets = UPtr(unsafe_from_address=offsets_addr)
    var lengths = UPtr(unsafe_from_address=lengths_addr)
    var ends = UPtr(unsafe_from_address=ends_addr)
    var pos = Int(0)
    var index = Int(0)
    while pos < size:
        if index >= capacity:
            return -1
        var marker = Int(src[pos])
        pos += 1
        values[index] = 0
        offsets[index] = 0
        lengths[index] = 0
        if marker <= 0x7f:
            types[index] = UINT
            values[index] = UInt64(marker)
        elif marker >= 0xe0:
            types[index] = SINT
            values[index] = UInt64(marker) | 0xffffffffffffff00
        elif marker >= 0x80 and marker <= 0x8f:
            types[index] = MAP
            values[index] = UInt64(marker & 15)
        elif marker >= 0x90 and marker <= 0x9f:
            types[index] = ARRAY
            values[index] = UInt64(marker & 15)
        elif marker >= 0xa0 and marker <= 0xbf:
            types[index] = STRING
            lengths[index] = UInt64(marker & 31)
            offsets[index] = UInt64(pos)
            pos += marker & 31
        elif marker == 0xc0:
            types[index] = NIL
        elif marker == 0xc2:
            types[index] = FALSE
        elif marker == 0xc3:
            types[index] = TRUE
        elif marker == 0xcc:
            types[index] = UINT
            values[index] = UInt64(src[pos])
            pos += 1
        elif marker == 0xcd:
            types[index] = UINT
            values[index] = _get_u16(src, pos)
            pos += 2
        elif marker == 0xce:
            types[index] = UINT
            values[index] = _get_u32(src, pos)
            pos += 4
        elif marker == 0xcf:
            types[index] = UINT
            values[index] = _get_u64(src, pos)
            pos += 8
        elif marker == 0xd0:
            types[index] = SINT
            values[index] = UInt64(src[pos])
            if src[pos] >= 0x80:
                values[index] |= 0xffffffffffffff00
            pos += 1
        elif marker == 0xd1:
            types[index] = SINT
            values[index] = _get_u16(src, pos)
            if src[pos] >= 0x80:
                values[index] |= 0xffffffffffff0000
            pos += 2
        elif marker == 0xd2:
            types[index] = SINT
            values[index] = _get_u32(src, pos)
            if src[pos] >= 0x80:
                values[index] |= 0xffffffff00000000
            pos += 4
        elif marker == 0xd3:
            types[index] = SINT
            values[index] = _get_u64(src, pos)
            pos += 8
        elif marker == 0xca:
            types[index] = FLOAT32
            values[index] = _get_u32(src, pos)
            pos += 4
        elif marker == 0xcb:
            types[index] = FLOAT64
            values[index] = _get_u64(src, pos)
            pos += 8
        elif marker == 0xd9:
            types[index] = STRING
            lengths[index] = UInt64(src[pos])
            pos += 1
            offsets[index] = UInt64(pos)
            pos += Int(lengths[index])
        elif marker == 0xda:
            types[index] = STRING
            lengths[index] = _get_u16(src, pos)
            pos += 2
            offsets[index] = UInt64(pos)
            pos += Int(lengths[index])
        elif marker == 0xdb:
            types[index] = STRING
            lengths[index] = _get_u32(src, pos)
            pos += 4
            offsets[index] = UInt64(pos)
            pos += Int(lengths[index])
        elif marker == 0xc4:
            types[index] = BINARY
            lengths[index] = UInt64(src[pos])
            pos += 1
            offsets[index] = UInt64(pos)
            pos += Int(lengths[index])
        elif marker == 0xc5:
            types[index] = BINARY
            lengths[index] = _get_u16(src, pos)
            pos += 2
            offsets[index] = UInt64(pos)
            pos += Int(lengths[index])
        elif marker == 0xc6:
            types[index] = BINARY
            lengths[index] = _get_u32(src, pos)
            pos += 4
            offsets[index] = UInt64(pos)
            pos += Int(lengths[index])
        elif marker == 0xdc or marker == 0xdd:
            types[index] = ARRAY
            if marker == 0xdc:
                values[index] = _get_u16(src, pos)
                pos += 2
            else:
                values[index] = _get_u32(src, pos)
                pos += 4
        elif marker == 0xde or marker == 0xdf:
            types[index] = MAP
            if marker == 0xde:
                values[index] = _get_u16(src, pos)
                pos += 2
            else:
                values[index] = _get_u32(src, pos)
                pos += 4
        elif marker >= 0xd4 and marker <= 0xd8:
            types[index] = EXT
            lengths[index] = UInt64(1 << (marker - 0xd4))
            values[index] = UInt64(src[pos])
            pos += 1
            offsets[index] = UInt64(pos)
            pos += Int(lengths[index])
        elif marker == 0xc7 or marker == 0xc8 or marker == 0xc9:
            types[index] = EXT
            if marker == 0xc7:
                lengths[index] = UInt64(src[pos])
                pos += 1
            elif marker == 0xc8:
                lengths[index] = _get_u16(src, pos)
                pos += 2
            else:
                lengths[index] = _get_u32(src, pos)
                pos += 4
            values[index] = UInt64(src[pos])
            pos += 1
            offsets[index] = UInt64(pos)
            pos += Int(lengths[index])
        ends[index] = UInt64(pos)
        index += 1
    return index
