"""A pickleable wrapper for sharing NumPy ndarrays between processes using POSIX shared memory."""

import mmap

import numpy as np
import posix_ipc


class SharedNDArray:
    """Creates a new SharedNDArray, a pickleable wrapper for sharing NumPy ndarrays between
    processes using POSIX shared memory.

    SharedNDArrays are designed to be sent over multiprocessing.Pipe and Queue without serializing
    or transmitting the underlying ndarray or buffer. While the associated file descriptor is
    closed when the SharedNDArray is garbage collected, the underlying buffer is not released when
    the process ends: you must manually call the unlink() method from the last process to use it.

    Attributes:
        array: The wrapped NumPy ndarray, backed by POSIX shared memory.
    """

    def __init__(self, shape, dtype=np.float64, name=None):
        """Creates a new SharedNDArray.

        If name is left blank, a new POSIX shared memory segment is created using a random name.

        Args:
            shape: Shape of the wrapped ndarray.
            dtype: Data type of the wrapped ndarray.
            name: Optional; the filesystem path of the underlying POSIX shared memory.

        Returns:
            A new SharedNDArray of the given shape and dtype and backed by the given optional name.

        Raises:
            SharedNDArrayError: if an error occurs.
        """
        size = 96 + int(np.prod(shape)) * np.dtype(dtype).itemsize
        if name:
            try:
                self._shm = posix_ipc.SharedMemory(name)
            except posix_ipc.ExistentialError as ee:
                raise ee.__class__(f"{ee.args[0]}; requested name: {name}")
        else:
            self._shm = posix_ipc.SharedMemory(None, posix_ipc.O_CREX, size=size)
        self._buf = mmap.mmap(self._shm.fd, size)
        self.array = np.ndarray(shape, dtype, self._buf, order='C')

    def flush(self):
        # Why is this necessary, if at all?!?
        # TODO: Should this use np.memmap() instead?  What would impact to performance be?
        self._buf.seek(0)
        self._buf.write(self.array.tobytes())

    def __getattr__(self, name):
        return getattr(self.array, name)

    @classmethod
    def copy(cls, arr):
        """Creates a new SharedNDArray that is a copy of the given ndarray.

        Args:
            arr: The ndarray to copy.

        Returns:
            A new SharedNDArray object with the given ndarray's shape and data type and a copy of
            its data.

        Raises:
            SharedNDArrayError: if an error occurs.
        """
        new_shm = cls.zeros_like(arr)
        new_shm.array[:] = arr
        new_shm.flush()
        return new_shm

    @classmethod
    def zeros_like(cls, arr):
        """Creates a new zero-filled SharedNDArray with the shape and dtype of the given ndarray.

        Raises:
            SharedNDArrayError: if an error occurs.
        """
        return cls(arr.shape, arr.dtype)

    def unlink(self):
        """Marks the underlying shared for deletion.

        This method should be called exactly once from one process. Failure to call it before all
        processes exit will result in a memory leak! It will raise SharedNDArrayError if the
        underlying shared memory was already marked for deletion from any process.

        Raises:
            SharedNDArrayError: if an error occurs.
        """
        self._shm.unlink()

    def __del__(self):
        self._buf.close()
        self._shm.close_fd()

    def __getstate__(self):
        return self.array.shape, self.array.dtype, self._shm.name

    def __setstate__(self, state):
        self.__init__(*state)
