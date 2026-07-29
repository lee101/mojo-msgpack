from __future__ import annotations

import datetime as _datetime
import struct
from collections import namedtuple


class ExtType(namedtuple("ExtTypeBase", "code data")):
    """A MessagePack application-defined extension value."""

    __slots__ = ()

    def __new__(cls, code, data):
        if not isinstance(code, int):
            raise TypeError("code must be int")
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if not 0 <= code <= 127:
            raise ValueError("code must be 0~127")
        return super().__new__(cls, code, data)


class Timestamp:
    """The standard MessagePack timestamp extension."""

    __slots__ = ("seconds", "nanoseconds")

    def __init__(self, seconds, nanoseconds=0):
        if not isinstance(seconds, int):
            raise TypeError("seconds must be an integer")
        if not isinstance(nanoseconds, int):
            raise TypeError("nanoseconds must be an integer")
        if not 0 <= nanoseconds < 1_000_000_000:
            raise ValueError("nanoseconds must be in range(0, 1000000000)")
        self.seconds = seconds
        self.nanoseconds = nanoseconds

    def __repr__(self):
        return f"Timestamp(seconds={self.seconds}, nanoseconds={self.nanoseconds})"

    def __eq__(self, other):
        return (
            type(other) is type(self)
            and self.seconds == other.seconds
            and self.nanoseconds == other.nanoseconds
        )

    def __hash__(self):
        return hash((self.seconds, self.nanoseconds))

    def to_bytes(self):
        if self.seconds >> 34 == 0:
            value = self.nanoseconds << 34 | self.seconds
            return struct.pack("!I", value) if value <= 0xFFFFFFFF else struct.pack("!Q", value)
        return struct.pack("!Iq", self.nanoseconds, self.seconds)

    @staticmethod
    def from_bytes(data):
        if len(data) == 4:
            return Timestamp(struct.unpack("!I", data)[0])
        if len(data) == 8:
            value = struct.unpack("!Q", data)[0]
            return Timestamp(value & 0x3FFFFFFFF, value >> 34)
        if len(data) == 12:
            nanoseconds, seconds = struct.unpack("!Iq", data)
            return Timestamp(seconds, nanoseconds)
        raise ValueError("timestamp payload must be 4, 8, or 12 bytes")

    @staticmethod
    def from_unix(value):
        seconds = int(value // 1)
        return Timestamp(seconds, int((value % 1) * 1_000_000_000))

    def to_unix(self):
        return self.seconds + self.nanoseconds / 1_000_000_000

    @staticmethod
    def from_unix_nano(value):
        return Timestamp(*divmod(value, 1_000_000_000))

    def to_unix_nano(self):
        return self.seconds * 1_000_000_000 + self.nanoseconds

    @staticmethod
    def from_datetime(value):
        return Timestamp(int(value.timestamp() // 1), value.microsecond * 1000)

    def to_datetime(self):
        epoch = _datetime.datetime.fromtimestamp(0, _datetime.timezone.utc)
        return epoch + _datetime.timedelta(
            seconds=self.seconds, microseconds=self.nanoseconds // 1000
        )
