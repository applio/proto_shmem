import os
from multiprocessing.managers import BaseProxy, MakeProxyType, DictProxy
import multiprocessing as mp
import numpy as np
from shared_ndarray.shared_ndarray import SharedNDArray
import shared_memory


class SharedList(SharedNDArray):

    def __init__(self, iterable, *, shared_memory_context):
        self.shared_memory_context = shared_memory_context
        if isinstance(iterable, (list, tuple)):
            # Skip creation of short-lived duplicate list.
            shared_array = SharedNDArray.copy(np.array(iterable))
        else:
            shared_array = SharedNDArray.copy(np.array(list(iterable)))
        self.allocated_size = -1  # TODO: overallocate more than needed
        self.replace_held_shared_array(shared_array, destroy_old=False)

    def replace_held_shared_array(self, shared_array, destroy_old=True):
        if destroy_old:
            self.shared_memory_context.destroy_segment(self._shm.name)
        self.array = shared_array.array
        self._shm = shared_array._shm
        self.shared_memory_context.register_segment(self._shm)
        # Preserve it or the SharedNDArray kills it and all these mutation
        # operations will fail catastrophically.
        # But if it's preserved, then its refcount won't drop to 0 even
        # after the last Proxy reference is killed.
        # Ultimately, replace SharedNDArray's functionality.
        self._shared_array = shared_array

    def __del__(self):
        self._shm.close()

    def __contains__(self, value):
        return self.array.__contains__(value)

    def __getitem__(self, position):
        return self.array.__getitem__(position)

    def __setitem__(self, position, value):
        return self.array.__setitem__(position, value)

    def _getstate(self):
        return self.array.shape, self.array.dtype, self._shm.name

    def append(self, value):
        arr_of_one = np.array([value], dtype=self.array.dtype)
        shared_array = SharedNDArray.copy(
            np.concatenate([self.array, arr_of_one]))
        self.allocated_size = -1  # TODO: overallocate more than needed
        self.replace_held_shared_array(shared_array)

    def clear(self):
        shared_array = SharedNDArray.copy(np.array([], dtype=self.array.dtype))
        self.allocated_size = -1  # TODO: overallocate more than needed
        self.replace_held_shared_array(shared_array)

    def copy(self):
        raise NotImplementedError

    def count(self, value):
        return len(np.argwhere(self.array == value))

    def extend(self, values):
        arr_of_multiple = np.array(values, dtype=self.array.dtype)
        shared_array = SharedNDArray.copy(
            np.concatenate([self.array, arr_of_multiple]))
        self.allocated_size = -1  # TODO: overallocate more than needed
        self.replace_held_shared_array(shared_array)

    def index(self, value):
        try:
            return np.argwhere(self.array == value)[0][0]
        except IndexError:
            raise ValueError(f"{value!r} is not in list")

    def insert(self, position, value):
        arr_of_one = np.array([value], dtype=self.array.dtype)
        shared_array = SharedNDArray.copy(
            np.concatenate([self.array[:position],
                            arr_of_one,
                            self.array[position:]]))
        self.allocated_size = -1  # TODO: overallocate more than needed
        self.replace_held_shared_array(shared_array)

    def pop(self, position=None):
        # TODO: implement use of position
        retval = self.array[-1]
        # TODO: fix problem when shrinking to length of 0
        shared_array = SharedNDArray.copy(self.array[:-1])
        self.allocated_size = -1  # TODO: overallocate more than needed
        self.replace_held_shared_array(shared_array)
        return retval

    def remove(self, value):
        position = self.index(value)
        shared_array = SharedNDArray.copy(
            np.concatenate([self.array[:position], self.array[position+1:]]))
        self.allocated_size = -1  # TODO: overallocate more than needed
        self.replace_held_shared_array(shared_array)        

    def reverse(self):
        self.array[:] = self.array[::-1]

    def sort(self):
        self.array.sort()


BaseSharedSequenceProxy = MakeProxyType(
    'BaseSharedSequenceProxy',
    ('__contains__', '__getitem__', '__len__', 'count', 'index'))

# These operations must be performed by process with ownership.
BaseSharedListProxy = MakeProxyType(
    'BaseSharedListProxy',
    ('__setitem__', 'reverse', 'sort'))


class SharedListProxy(BaseSharedListProxy):
    # Really no point in deriving from BaseSharedSequenceProxy because most
    # of those methods can be performed better in the local process rather
    # than asking the process with ownership to have to perform them all.

    _shared_memory_proxy = True

    _exposed_ = ('_getstate',
                 '__contains__', '__getitem__', '__iter__', '__len__',
                 '__str__', 'append', 'clear', 'count', 'extend', 'index',
                 'insert', 'pop',
                 'remove',) + BaseSharedListProxy._exposed_

    def __init__(self, *args, **kwargs):
        BaseProxy.__init__(self, *args, **kwargs)
        self.attach_object()

    def attach_object(self):
        "Attach to existing object in shared memory segment."
        self.shared_array = SharedNDArray(*(self._callmethod('_getstate')))

    def __contains__(self, value):
        return self.shared_array.array.__contains__(value)

    def __getitem__(self, position):
        return self.shared_array.array.__getitem__(position)

    def __iter__(self):
        return iter(self.shared_array.array)

    def __len__(self):
        return len(self.shared_array.array)

    def __str__(self):
        return str(list(self))

    def append(self, value):
        retval = self._callmethod('append', (value,))
        self.attach_object()
        return retval

    def clear(self):
        self._callmethod('clear', ())
        self.attach_object()

    def count(self, value):
        return len(np.argwhere(self.shared_array.array == value))

    def extend(self, value):
        retval = self._callmethod('extend', (value,))
        self.attach_object()
        return retval

    def index(self, value):
        try:
            return np.argwhere(self.shared_array.array == value)[0][0]
        except IndexError:
            raise ValueError(f"{value} is not in list")

    def insert(self, position, value):
        retval = self._callmethod('insert', (position, value))
        self.attach_object()
        return retval

    def pop(self, position=None):
        retval = self._callmethod('pop', (position,))
        self.attach_object()
        return retval

    def remove(self, value=None):
        retval = self._callmethod('remove', (value,))
        self.attach_object()
        return retval

    def __dir__(self):
        return sorted(self._exposed_[1:])


shared_memory.SharedMemoryManager.register('list', SharedList, SharedListProxy)


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
    block, block_size, arr, result_arr = data_tuple
    result_arr[:] = arr[block_size*block:block_size*(block+1)] * 4000
    return result_arr


def block_exponential_4(data_tuple):
    block, block_size, shm_result_arr, shm_arr = data_tuple
    shm_result_arr[:] = shm_arr[block_size*block:block_size*(block+1)] ** 0.001
    #shm_result_arr.flush()
    #print(shm_result_arr[:2], block, block_size*block, block_size*(block+1), os.getpid(), id(shm_result_arr))
    #print(shm_arr[:2], block, block_size*block, block_size*(block+1), os.getpid(), id(shm_arr))


def main01():
    shm = shared_memory.SharedMemoryTracker("unique_id_001")
    try:
        #local_r = np.random.random_sample((4000,))
        local_r = np.ones((4000,))
        shared_r = shm.wrap(local_r)
        shared_results = [ shm.wrap(np.ndarray(1000, local_r.dtype)) for i in range(4) ]
        #shared_results = [ shm.wrap(np.ones((1000,))) for i in range(4) ]
        #block_multiply((0, shared_r), block_size=4000)
        #with mp.Pool(processes=4) as p:
        #    _results = p.map(block_multiply, enumerate([shared_r] * 4))
        with mp.Pool(processes=4) as p:
            _results = p.map(block_multiply_2, ((i, shared_results[i], shared_r) for i in range(4)))
        print(shared_results[0][:10], "first 10 of", len(shared_results[0]))
        print(np.all(np.isclose(shared_results[0], local_r[:1000] * 4)))  # Should be True
        print(np.any(shared_results[0] == 4))                             # Should be True
        print(id(local_r), id(shared_r), [id(x) for x in shared_results])
    finally:
        shm.unlink()

def main02():
    smt = shared_memory.SharedMemoryTracker("unique_id_002")
    try:
        local_r = np.ones((4000,))
        shared_r = smt.wrap(local_r)
        with mp.Pool(processes=1) as p:
            results = p.map(block_multiply_3, ((i, 1000, shared_r, smt.wrap(np.ndarray((1000,), local_r.dtype))) for i in range(4)))
        #with mp.Pool(processes=4) as p:
        #    results = p.map(block_multiply_3, ((i, 1000, shared_r, smt) for i in range(4)))
        combined_results = np.concatenate([shared_array for shared_array in results])
        print(combined_results[:10], "first 10 of", len(combined_results))
        print(np.all(np.isclose(combined_results, local_r * 4000)))  # Should be True
    finally:
        smt.unlink()

def main04_parallel(scale=1000, iterations=400000, nprocs=2, blocks=8):
    shm = shared_memory.SharedMemoryTracker("unique_id_004")
    try:
        local_r = np.random.random_sample((blocks * scale,))
        shared_r = shm.wrap(local_r)
        shared_results = [ shm.wrap(np.ndarray(scale, local_r.dtype)) for i in range(blocks) ]
        if nprocs > 0:
            with mp.Pool(processes=nprocs) as p:
                _results = p.map(block_exponential_4, ((i % blocks, scale, shared_results[i % blocks], shared_r) for i in range(iterations)))
        else:
            _results = list(map(block_exponential_4, ((i % blocks, scale, shared_results[i % blocks], shared_r) for i in range(iterations))))
        print(shared_results[0][:10], "first 10 of", len(shared_results[0]))
        print(np.all(np.isclose(shared_results[0], np.ones((scale,)))))  # Likely False
        print(np.all(shared_results[0] > 0.99))  # Should be True
        print(id(local_r), id(shared_r), [id(x) for x in shared_results])
    finally:
        shm.unlink()


class C:
    def __init__(self, a, b):
        self.a = a
        self.b = b


def main05():
    import sys
    lookup_table = {}
    authkey = b'howdy'
    mp.current_process().authkey = authkey  # TODO: add this to multiprocessing docs regarding authkey
    shared_memory.SharedMemoryManager.register("get_lookup_table", callable=lambda: lookup_table, proxytype=DictProxy)

    if sys.argv[-1] == 'master':
        m = shared_memory.SharedMemoryManager(address=('127.0.0.1', 5005), authkey=authkey)
        m.start()
        print(f"master running as pid {os.getpid()}")

        w = m.list([3, 4, 5])
        lookup_table = m.get_lookup_table()
        lookup_table['w'] = w

    else:
        m = shared_memory.SharedMemoryManager(address=('127.0.0.1', 5005), authkey=authkey)
        m.connect()
        print(f"remote running as pid {os.getpid()}")

        lookup_table = m.get_lookup_table()
        w = lookup_table['w']

    return m, lookup_table, w



if __name__ == '__main__':
    #main04_parallel(scale=10000, nprocs=2)

    m, lookup_table, w = main05()


