### Creating a larger shared memory segment and then managing the
### incremental use of that block for multiple objects as blocks
### within appears to add complexity and add to the cost (management).


import mmap
import multiprocessing as mp
import posix_ipc


def attach_or_create_shmem(shared_memory_name, size=1024**2):
    return SharedMemorySegment(shared_memory_name, size)


class SharedMemorySegment:

    def __init__(self, name, size=1024**2):
        try:
            # Create
            self.shared_memory = posix_ipc.SharedMemory(name,
                                                        posix_ipc.O_CREX,
                                                        size=size)
            self.shared_memory.remaining_size = size  # monkey patch

        except posix_ipc.ExistentialError:
            # ... or attach
            self.shared_memory = posix_ipc.SharedMemory(name)


    def close(self):
        self.shared_memory.close_fd()
        self.shared_memory.unlink()


    def array(self, *args, **kwargs):
        return self.ndarray(*args, **kwargs)


class SharedNDArray:

    def __init__(self, shmem, arr, dtype=None):
        try:
            shape = arr.shape
            dtype = arr.dtype
        except AttributeError:
            shape = arr    # assume is a tuple
            dtype = dtype  # assume contains numpy.dtype

        self._shmem = shmem
        size = np.dtype(dtype).itemsize * int(np.prod(shape))

        self.mmapfile = mmap.mmap(self._shmem.shared_memory.fd, size)




    def close(self):
        self.mmapfile.close()

