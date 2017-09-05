import os
from multiprocessing import managers
import multiprocessing as mp
import numpy as np
from shared_ndarray import SharedNDArray, posix_ipc


class SharedMemoryContext:

    def __init__(self, name, segment_names=[]):
        self.name = name
        self.segment_names = segment_names

    def register_segment(self, segment):
        print("Registering segment:", segment.name)
        self.segment_names.append(segment.name)

    def close(self):
        for segment_name in self.segment_names:
            print("Unlinking segment:", segment_name)
            segment = posix_ipc.SharedMemory(segment_name)
            segment.close_fd()
            segment.unlink()
        self.segment_names[:] = []

    def __del__(self):
        self.close()

    def __getstate__(self):
        return (self.name, self.segment_names)

    def __setstate__(self, state):
        self.__init__(*state)

    def ndarray(self, arr, dtype=None):
        "To be removed -- needs to be register-able to intro numpy support."
        try:
            shape = arr.shape
            dtype = arr.dtype
            shared_array = SharedNDArray.copy(arr)
        except AttributeError:
            shape = arr    # assume is a tuple
            dtype = dtype  # assume contains numpy.dtype
            shared_array = SharedNDArray(shape, dtype)
        self.register_segment(shared_array._shm)
        return shared_array


class SharedMemoryManager(managers.SyncManager, SharedMemoryContext):
    pass


def block_multiply(block_arr_tuple, block_size=1000):
    block, arr = block_arr_tuple
    arr[block_size*block:block_size*(block+1)] = \
        arr[block_size*block:block_size*(block+1)] * 4
    print(arr[:2], block, block_size*block, block_size*(block+1), os.getpid())


def block_multiply_2(data_tuple, block_size=1000):
    block, result_arr, arr = data_tuple
    result_arr[:] = arr[block_size*block:block_size*(block+1)] * 4
    print(result_arr[:2], block, block_size*block, block_size*(block+1), os.getpid(), id(result_arr))
    print(arr[:2], block, block_size*block, block_size*(block+1), os.getpid(), id(arr))


def block_multiply_3(data_tuple):
    block, block_size, arr, shm = data_tuple
    result_arr = arr[block_size*block:block_size*(block+1)] * 4000
    return shm.ndarray(result_arr)


def block_exponential_4(data_tuple):
    block, block_size, shm_result_arr, shm_arr = data_tuple
    shm_result_arr.array[:] = shm_arr.array[block_size*block:block_size*(block+1)] ** 0.5
    shm_result_arr.flush()
    #print(shm_result_arr.array[:2], block, block_size*block, block_size*(block+1), os.getpid(), id(shm_result_arr))
    #print(shm_arr.array[:2], block, block_size*block, block_size*(block+1), os.getpid(), id(shm_arr))


def main01():
    shm = SharedMemoryContext("unique_id_001")
    try:
        #local_r = np.random.random_sample((4000,))
        local_r = np.ones((4000,))
        shared_r = shm.ndarray(local_r)
        shared_results = [ shm.ndarray(1000, local_r.dtype) for i in range(4) ]
        #shared_results = [ shm.ndarray(np.ones((1000,))) for i in range(4) ]
        #block_multiply((0, shared_r.array), block_size=4000)
        #with mp.Pool(processes=4) as p:
        #    _results = p.map(block_multiply, enumerate([shared_r.array] * 4))
        with mp.Pool(processes=4) as p:
            _results = p.map(block_multiply_2, ((i, shared_results[i].array, shared_r.array) for i in range(4)))
        print(shared_results[0].array[:10], "first 10 of", len(shared_results[0].array))
        print(np.all(np.isclose(shared_results[0].array, local_r[:1000] * 4)))
        print(np.any(shared_results[0].array == 4))
        print(id(local_r), id(shared_r.array), [id(x.array) for x in shared_results])
    finally:
        shm.close()

def main02():
    shm = SharedMemoryContext("unique_id_002")
    try:
        local_r = np.ones((4000,))
        shared_r = shm.ndarray(local_r)
        with mp.Pool(processes=1) as p:
            results = p.map(block_multiply_3, ((i, 4000, shared_r.array, shm) for i in range(4)))
        #with mp.Pool(processes=4) as p:
        #    results = p.map(block_multiply_3, ((i, 1000, shared_r.array, shm) for i in range(4)))
        combined_results = np.concatenate([shared_array.array for shared_array in results])
        print(combined_results[:10], "first 10 of", len(combined_results))
        print(np.all(np.isclose(combined_results, local_r * 4)))
    finally:
        shm.close()

def main04_shmem_parallel(scale=1000, iterations=400000):
    shm = SharedMemoryContext("unique_id_001")
    try:
        local_r = np.random.random_sample((4 * scale,))
        shared_r = shm.ndarray(local_r)
        shared_results = [ shm.ndarray(scale, local_r.dtype) for i in range(4) ]
        with mp.Pool(processes=2) as p:
            _results = p.map(block_exponential_4, ((i % 4, scale, shared_results[i % 4], shared_r) for i in range(iterations)))
        print(shared_results[0].array[:10], "first 10 of", len(shared_results[0].array))
        print(np.all(np.isclose(shared_results[0].array, np.ones((scale,)))))
        print(id(local_r), id(shared_r.array), [id(x.array) for x in shared_results])
    finally:
        shm.close()

def main04_single(scale=1000, iterations=400000):
    shm = SharedMemoryContext("unique_id_001")
    try:
        local_r = np.random.random_sample((4 * scale,))
        shared_r = shm.ndarray(local_r)
        shared_results = [ shm.ndarray(scale, local_r.dtype) for i in range(4) ]
        _results = list(map(block_exponential_4, ((i % 4, scale, shared_results[i % 4], shared_r) for i in range(iterations))))
        print(shared_results[0].array[:10], "first 10 of", len(shared_results[0].array))
        print(np.all(np.isclose(shared_results[0].array, np.ones((scale,)))))
        print(id(local_r), id(shared_r.array), [id(x.array) for x in shared_results])
    finally:
        shm.close()


if __name__ == '__main__':
    main04_shmem_parallel(scale=10000)
    #main04_single(scale=10000)


