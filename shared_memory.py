import mmap
import os
import random
import struct
import sys
try:
    from posix_ipc import SharedMemory as _PosixSharedMemory, Error, ExistentialError, O_CREX
except ImportError as ie:
    if os.name != "nt":
        # On Windows, posix_ipc is not required to be available.
        raise ie
    else:
        _PosixSharedMemory = object
        class ExistentialError(BaseException): pass
        class Error(BaseException): pass
        O_CREX = -1


class WindowsNamedSharedMemory:

    def __init__(self, name, flags=None, mode=None, size=None, read_only=False):
        if name is None:
            name = f'wnsm_{os.getpid()}_{random.randrange(100000)}'

        self._mmap = mmap.mmap(-1, size, tagname=name)
        self.buf = memoryview(self._mmap)
        self.name = name
        self.size = size

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name!r}, size={self.size})'

    def close(self):
        self.buf.release()
        self._mmap.close()

    def unlink(self):
        """Windows ensures that destruction of the last reference to this
        named shared memory block will result in the release of this memory."""
        pass


class PosixSharedMemory(_PosixSharedMemory):

    def __init__(self, name, flags=None, mode=None, size=None, read_only=False):
        if name and (flags is None):
            _PosixSharedMemory.__init__(self, name)
        else:
            if name is None:
                name = f'psm_{os.getpid()}_{random.randrange(100000)}'
            _PosixSharedMemory.__init__(self, name, flags=O_CREX, size=size)

        self._mmap = mmap.mmap(self.fd, self.size)
        self.buf = memoryview(self._mmap)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name!r}, size={self.size})'

    def close(self):
        self.buf.release()
        self._mmap.close()
        self.close_fd()


class SharedMemory:

    def __new__(self, *args, **kwargs):
        if os.name == 'nt':
            cls = WindowsNamedSharedMemory
        else:
            cls = PosixSharedMemory
        return cls(*args, **kwargs)


encoding = "utf8"

class ShareableList:
    """Pattern for a mutable list-like object shareable via a shared
    memory block.  It differs from the built-in list type in that these
    lists can not change their overall length (i.e. no append, insert,
    etc.)

    Because values are packed into a memoryview as bytes, the struct
    packing format for any storable value must require no more than 8
    characters to describe its format."""

    # TODO: Adjust for discovered word size of machine.
    types_mapping = {
        int: "q",
        float: "d",
        bool: "xxxxxxx?",
        str: "%ds",
        bytes: "%ds",
        None.__class__: "xxxxxx?x",
    }
    alignment = 8
    back_transform_codes = {
        0: lambda value: value,                   # int, float, bool
        1: lambda value: value.rstrip(b'\x00').decode(encoding),  # str
        2: lambda value: value.rstrip(b'\x00'),   # bytes
        3: lambda _value: None,                   # None
    }

    @staticmethod
    def _extract_recreation_code(value):
        if not isinstance(value, (str, bytes, None.__class__)):
            return 0
        elif isinstance(value, str):
            return 1
        elif isinstance(value, bytes):
            return 2
        else:
            return 3  # NoneType

    def __init__(self, iterable=None, name=None):
        if iterable is not None:
            _formats = [
                self.types_mapping[type(item)] if not isinstance(item, (str, bytes))
                    else self.types_mapping[type(item)] % (
                        self.alignment * (len(item) // self.alignment + 1),
                    )
                for item in iterable
            ]
            self._list_len = len(_formats)
            assert sum(len(fmt) <= 8 for fmt in _formats) == self._list_len
            self._allocated_bytes = tuple(
                    self.alignment if fmt[-1] != "s" else int(fmt[:-1])
                    for fmt in _formats
            )
            _back_transform_codes = [
                self._extract_recreation_code(item) for item in iterable
            ]
            requested_size = struct.calcsize(
                "q" + self._format_size_metainfo + "".join(_formats)
            )

        else:
            requested_size = 1  # Some platforms require > 0.

        self.shm = SharedMemory(name, size=requested_size)

        if iterable is not None:
            _enc = encoding
            struct.pack_into(
                "q" + self._format_size_metainfo,
                self.shm.buf,
                0,
                self._list_len,
                *(self._allocated_bytes)
            )
            struct.pack_into(
                "".join(_formats),
                self.shm.buf,
                self._offset_data_start,
                *(v.encode(_enc) if isinstance(v, str) else v for v in iterable)
            )
            struct.pack_into(
                self._format_packing_metainfo,
                self.shm.buf,
                self._offset_packing_formats,
                *(v.encode(_enc) for v in _formats)
            )
            struct.pack_into(
                self._format_back_transform_codes,
                self.shm.buf,
                self._offset_back_transform_codes,
                *(_back_transform_codes)
            )

        else:
            self._list_len = len(self)  # Obtains size from offset 0 in buffer.
            self._allocated_bytes = struct.unpack_from(
                self._format_size_metainfo,
                self.shm.buf,
                1 * 8
            )

    def get_packing_format(self, position):
        position = position if position >= 0 else position + self._list_len
        if (position >= self._list_len) or (self._list_len < 0):
            raise IndexError("Requested position out of range.")

        v = struct.unpack_from(
            "8s",
            self.shm.buf,
            self._offset_packing_formats + position * 8
        )[0]
        fmt = v.rstrip(b'\x00')
        fmt_as_str = fmt.decode(encoding)

        return fmt_as_str

    def get_back_transform(self, position):
        position = position if position >= 0 else position + self._list_len
        if (position >= self._list_len) or (self._list_len < 0):
            raise IndexError("Requested position out of range.")

        transform_code = struct.unpack_from(
            "b",
            self.shm.buf,
            self._offset_back_transform_codes + position
        )[0]
        transform_function = self.back_transform_codes[transform_code]

        return transform_function

    def set_packing_format_and_transform(self, position, fmt_as_str, value):
        position = position if position >= 0 else position + self._list_len
        if (position >= self._list_len) or (self._list_len < 0):
            raise IndexError("Requested position out of range.")

        struct.pack_into(
            "8s",
            self.shm.buf,
            self._offset_packing_formats + position * 8,
            fmt_as_str.encode(encoding)
        )

        transform_code = self._extract_recreation_code(value)
        struct.pack_into(
            "b",
            self.shm.buf,
            self._offset_back_transform_codes + position,
            transform_code
        )

    def __getitem__(self, position):
        try:
            offset = self._offset_data_start + sum(self._allocated_bytes[:position])
            (v,) = struct.unpack_from(
                self.get_packing_format(position),
                self.shm.buf,
                offset
            )
        except IndexError:
            raise IndexError("index out of range")

        back_transform = self.get_back_transform(position)
        v = back_transform(v)

        return v

    def __setitem__(self, position, value):
        try:
            offset = self._offset_data_start + sum(self._allocated_bytes[:position])
            current_format = self.get_packing_format(position)
        except IndexError:
            raise IndexError("assignment index out of range")

        if not isinstance(value, (str, bytes)):
            new_format = self.types_mapping[type(item)]
        elif current_format[-1] == "s":
            if int(current_format[:-1]) < len(value):
                raise ValueError("exceeds available storage for existing str")
            new_format = current_format
        else:
            if len(value) > self.alignment:
                raise ValueError("str exceeds available storage")
            new_format = current_format

        self.set_packing_format_and_transform(
            position,
            new_format,
            value
        )
        value = value.encode(encoding) if isinstance(value, str) else value
        struct.pack_into(new_format, self.shm.buf, offset, value)

    def __len__(self):
        return struct.unpack_from("q", self.shm.buf, 0)[0]

    @property
    def format(self):
        "The struct packing format used by all currently stored values."
        return "".join(self.get_packing_format(i) for i in range(self._list_len))

    @property
    def _format_size_metainfo(self):
        "The struct packing format used for metainfo on storage sizes."
        return f"{self._list_len}q"

    @property
    def _format_packing_metainfo(self):
        return "8s" * self._list_len

    @property
    def _format_back_transform_codes(self):
        return "b" * self._list_len

    @property
    def _offset_data_start(self):
        return (self._list_len + 1) * 8  # 8 bytes per "q"

    @property
    def _offset_packing_formats(self):
        return self._offset_data_start + sum(self._allocated_bytes)

    @property
    def _offset_back_transform_codes(self):
        return self._offset_packing_formats + self._list_len * 8

    @classmethod
    def copy(cls, self):
        "L.copy() -> ShareableList -- a shallow copy of L."

        return cls(self)

    def count(self, value):
        "L.count(value) -> integer -- return number of occurrences of value."

        return sum(value == entry for entry in self)

    def index(self, value):
        """L.index(value) -> integer -- return first index of value.
        Raises ValueError if the value is not present."""

        for position, entry in enumerate(self):
            if value == entry:
                return position
        else:
            raise ValueError(f"{value!r} not in this container")
