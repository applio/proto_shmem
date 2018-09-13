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


class ShareableList:
    """Pattern for a mutable list-like object shareable via a shared
    memory block.  It differs from the built-in list type in that these
    lists can not change their overall length (i.e. no append, insert,
    etc.)"""

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
    encoding = "utf8"

    @staticmethod
    def _extract_recreation_function(value):
        if isinstance(value, None.__class__):
            recreation_function = lambda x: None
        else:
            recreation_function = type(value)
        return recreation_function

    def __init__(self, iterable):
        self._formats = [
            self.types_mapping[type(item)] if not isinstance(item, (str, bytes))
                else self.types_mapping[type(item)] % (
                    self.alignment * (len(item) // self.alignment + 1),
                )
            for item in iterable
        ]
        self._allocated_bytes = tuple(
                self.alignment if fmt[-1] != "s" else int(fmt[:-1])
                for fmt in self._formats
        )
        self._back_transforms = [
            (self._extract_recreation_function(item), len(item))
                if isinstance(item, (str, bytes))
                else (self._extract_recreation_function(item), None)
            for item in iterable
        ]
        self._len = len(self._formats)
        self.shm = SharedMemory(None, size=struct.calcsize(self.format))
        _enc = self.encoding
        struct.pack_into(
            self.format,
            self.shm.buf,
            0,
            *(v.encode(_enc) if isinstance(v, str) else v for v in iterable),
        )

    def __getitem__(self, position):
        try:
            offset = sum(self._allocated_bytes[:position])
        except IndexError:
            raise IndexError("index out of range")

        (v,) = struct.unpack_from(self._formats[position], self.shm.buf, offset)

        if isinstance(v, (bytes, bool)):
            transform_type, transform_len = self._back_transforms[position]
            if transform_type == str:
                v = transform_type(v[:transform_len], encoding=self.encoding)
            elif transform_len is not None:
                v = transform_type(v[:transform_len])
            else:
                v = transform_type(v)
        return v

    def __setitem__(self, position, value):
        try:
            offset = sum(self._allocated_bytes[:position])
            current_format = self._formats[position]
        except IndexError:
            raise IndexError("assignment index out of range")

        if not isinstance(value, (str, bytes)):
            new_format = self.types_mapping[type(item)]
            new_back_transform = (self._extract_recreation_function(value), None)
        elif current_format[-1] == "s":
            if int(current_format[:-1]) < len(value):
                raise ValueError("exceeds available storage for existing str")
            new_format = current_format
            new_back_transform = (self._extract_recreation_function(value), len(value))
            if isinstance(value, str):
                value = value.encode(self.encoding)
        else:
            if len(value) > self.alignment:
                raise ValueError("str exceeds available storage")
            new_format = current_format
            new_back_transform = (self._extract_recreation_function(value), len(value))
            if isinstance(value, str):
                value = value.encode(self.encoding)

        struct.pack_into(new_format, self.shm.buf, offset, value)
        self._formats[position] = new_format
        self._back_transforms[position] = new_back_transform

    @property
    def format(self):
        "The struct packing format used by all currently stored values."
        return "".join(self._formats)

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
