import os
from multiprocessing.managers import BaseProxy, MakeProxyType, AutoProxy, \
                                     DictProxy, SyncManager, Server
import multiprocessing as mp
import numpy as np
from shared_ndarray.shared_ndarray import SharedNDArray
import shared_memory


class SharedMemoryTracker:
    "Manages one or more shared memory segments."

    def __init__(self, name, segment_names=[]):
        self.shared_memory_context_name = name
        self.segment_names = segment_names

    def register_segment(self, segment):
        print(f"DBG Registering segment {segment.name!r} in pid {os.getpid()}")
        self.segment_names.append(segment.name)

    def destroy_segment(self, segment_name):
        print(f"DBG Destroying segment {segment_name!r} in pid {os.getpid()}")
        self.segment_names.remove(segment_name)
        segment = shared_memory.SharedMemory(segment_name, size=1)
        segment.close()
        segment.unlink()

    def unlink(self):
        for segment_name in self.segment_names:
            self.destroy_segment(segment_name)
        self.segment_names[:] = []

    def __del__(self):
        print(f"DBG somebody called {self.__class__.__name__}.__del__: {os.getpid()}")
        self.unlink()

    def __getstate__(self):
        return (self.shared_memory_context_name, self.segment_names)

    def __setstate__(self, state):
        self.__init__(*state)

    def wrap(self, obj_exposing_buffer_protocol):
        wrapped_obj = shared_memory.shareable_wrap(obj_exposing_buffer_protocol)
        self.register_segment(wrapped_obj._shm)
        return wrapped_obj


class AugmentedServer(Server):
    def __init__(self, *args, **kwargs):
        Server.__init__(self, *args, **kwargs)
        self.shared_memory_context = \
            SharedMemoryTracker(f"shmm_{self.address}_{os.getpid()}")
        print(f"DBG AugmentedServer started by pid {os.getpid()}")

    def create(self, c, typeid, *args, **kwargs):
        # Unless set up as a shared proxy, don't make shared_memory_context
        # a standard part of kwargs.  This makes things easier for supplying
        # simple functions.
        if hasattr(self.registry[typeid][-1], "_shared_memory_proxy"):
            kwargs['shared_memory_context'] = self.shared_memory_context
        return Server.create(self, c, typeid, *args, **kwargs)

    def shutdown(self, c):
        self.shared_memory_context.unlink()
        return Server.shutdown(self, c)


class SharedMemoryManager(SyncManager):
    """Like SyncManager but uses AugmentedServer instead of Server.

    TODO: relocate/merge into managers submodule."""

    _Server = AugmentedServer

    def __init__(self, *args, **kwargs):
        # TODO: Remove after debugging satisfied
        SyncManager.__init__(self, *args, **kwargs)
        print(f"{self.__class__.__name__} created by pid {os.getpid()}")

    def __del__(self):
        # TODO: Remove after debugging satisfied
        print(f"{self.__class__.__name__} told to die by pid {os.getpid()}")
        pass

    def get_server(self):
        'Better than monkeypatching for now; merge into Server ultimately'
        if self._state.value != State.INITIAL:
            if self._state.value == State.STARTED:
                raise ProcessError("Already started server")
            elif self._state.value == State.SHUTDOWN:
                raise ProcessError("Manager has shut down")
            else:
                raise ProcessError(
                    "Unknown state {!r}".format(self._state.value))
        return AugmentedServer(self._registry, self._address,
                               self._authkey, self._serializer)


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


SharedMemoryManager.register('list', SharedList, SharedListProxy)


def block_multiply(block_arr_tuple, block_size=1000):
    block, arr = block_arr_tuple
    arr[block_size*block:block_size*(block+1)] = \
        arr[block_size*block:block_size*(block+1)] * 4
    print(arr[:2], block, block_size*block, block_size*(block+1), os.getpid())


def block_multiply_2(data_tuple, block_size=1000):
    block, result_arr, arr = data_tuple
    result_arr.array[:] = arr.array[block_size*block:block_size*(block+1)] * 4
    print(result_arr.array[:2], block, block_size*block, block_size*(block+1), os.getpid(), id(result_arr.array))
    print(arr.array[:2], block, block_size*block, block_size*(block+1), os.getpid(), id(arr.array))


def block_multiply_3(data_tuple):
    block, block_size, arr, result_arr = data_tuple
    result_arr.array[:] = arr.array[block_size*block:block_size*(block+1)] * 4000
    return result_arr


def block_exponential_4(data_tuple):
    block, block_size, shm_result_arr, shm_arr = data_tuple
    shm_result_arr.array[:] = shm_arr.array[block_size*block:block_size*(block+1)] ** 0.001
    #shm_result_arr.flush()
    #print(shm_result_arr.array[:2], block, block_size*block, block_size*(block+1), os.getpid(), id(shm_result_arr))
    #print(shm_arr.array[:2], block, block_size*block, block_size*(block+1), os.getpid(), id(shm_arr))


def main01():
    shm = SharedMemoryTracker("unique_id_001")
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
            _results = p.map(block_multiply_2, ((i, shared_results[i], shared_r) for i in range(4)))
        print(shared_results[0].array[:10], "first 10 of", len(shared_results[0].array))
        print(np.all(np.isclose(shared_results[0].array, local_r[:1000] * 4)))  # Should be True
        print(np.any(shared_results[0].array == 4))                             # Should be True
        print(id(local_r), id(shared_r.array), [id(x.array) for x in shared_results])
    finally:
        shm.unlink()

def main02():
    smt = SharedMemoryTracker("unique_id_002")
    try:
        local_r = np.ones((4000,))
        shared_r = smt.ndarray(local_r)
        with mp.Pool(processes=1) as p:
            results = p.map(block_multiply_3, ((i, 1000, shared_r, smt.ndarray((1000,), local_r.dtype)) for i in range(4)))
        #with mp.Pool(processes=4) as p:
        #    results = p.map(block_multiply_3, ((i, 1000, shared_r.array, smt) for i in range(4)))
        combined_results = np.concatenate([shared_array.array for shared_array in results])
        print(combined_results[:10], "first 10 of", len(combined_results))
        print(np.all(np.isclose(combined_results, local_r * 4000)))  # Should be True
    finally:
        smt.unlink()

def main04_parallel(scale=1000, iterations=400000, nprocs=2, blocks=8):
    shm = SharedMemoryTracker("unique_id_004")
    try:
        local_r = np.random.random_sample((blocks * scale,))
        shared_r = shm.ndarray(local_r)
        shared_results = [ shm.ndarray(scale, local_r.dtype) for i in range(blocks) ]
        if nprocs > 0:
            with mp.Pool(processes=nprocs) as p:
                _results = p.map(block_exponential_4, ((i % blocks, scale, shared_results[i % blocks], shared_r) for i in range(iterations)))
        else:
            _results = list(map(block_exponential_4, ((i % blocks, scale, shared_results[i % blocks], shared_r) for i in range(iterations))))
        print(shared_results[0].array[:10], "first 10 of", len(shared_results[0].array))
        print(np.all(np.isclose(shared_results[0].array, np.ones((scale,)))))  # Likely False
        print(np.all(shared_results[0].array > 0.99))  # Should be True
        print(id(local_r), id(shared_r.array), [id(x.array) for x in shared_results])
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
    SharedMemoryManager.register("get_lookup_table", callable=lambda: lookup_table, proxytype=DictProxy)

    if sys.argv[-1] == 'master':
        m = SharedMemoryManager(address=('127.0.0.1', 5005), authkey=authkey)
        m.start()
        print(f"master running as pid {os.getpid()}")

        w = m.list([3, 4, 5])
        lookup_table = m.get_lookup_table()
        lookup_table['w'] = w

    else:
        m = SharedMemoryManager(address=('127.0.0.1', 5005), authkey=authkey)
        m.connect()
        print(f"remote running as pid {os.getpid()}")

        lookup_table = m.get_lookup_table()
        w = lookup_table['w']

    return m, lookup_table, w



if __name__ == '__main__':
    #main04_parallel(scale=10000, nprocs=2)

    m, lookup_table, w = main05()


