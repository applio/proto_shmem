import mmap
import os
import random
import sys
from posix_ipc import SharedMemory as _PosixSharedMemory, Error, ExistentialError, O_CREX


class WindowsNamedSharedMemory:

    def __init__(self, name, flags=None, mode=None, size=None, read_only=False):
        if name is None:
            name = f'wnsm_{os.getpid()}_{random.randrange(100000)}'

        self.buf = mmap.mmap(-1, size, tagname=name)

    def close(self):
        self.buf.close()

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

        self.buf = mmap.mmap(self.fd, self.size)

    def close(self):
        self.buf.close()
        self.close_fd()


class SharedMemory:

    def __new__(self, *args, **kwargs):
        if os.name == 'nt':
            cls = WindowsNamedSharedMemory
        else:
            cls = PosixSharedMemory
        return cls(*args, **kwargs)