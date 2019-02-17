from shared_memory import SharedMemory


def shareable_wrap(
    existing_obj=None,
    shmem_name=None,
    cls=None,
    shape=(0,),
    strides=None,
    dtype=None,
    format=None,
    **kwargs
):
    """Provides a fast, convenient way to encapsulate objects that support
    the buffer protocol as both producer and consumer, duplicating the
    original object's data in shared memory and returning a new wrapped
    object that when serialized via pickle does not serialize its data.

    The function has been written in a general way to potentially work with
    any object supporting the buffer protocol as producer and consumer.  It
    is known to work well with NumPy ndarrays.  Among the Python core data
    types and standard library, there are a number of objects supporting
    the buffer protocol as a producer but not as a consumer.

    Without an example of a producer+consumer of the buffer protocol in
    the Python core to demonstrate the use of this function, this function
    was removed from multiprocessing.shared_memory."""

    augmented_kwargs = dict(kwargs)
    extras = dict(shape=shape, strides=strides, dtype=dtype, format=format)
    for key, value in extras.items():
        if value is not None:
            augmented_kwargs[key] = value

    if existing_obj is not None:
        existing_type = getattr(
            existing_obj,
            "_proxied_type",
            type(existing_obj)
        )

        #agg = existing_obj.itemsize
        #size = [ agg := i * agg for i in existing_obj.shape ][-1]
        # TODO: replace use of reduce below with above 2 lines once available
        size = reduce(
            lambda x, y: x * y,
            existing_obj.shape,
            existing_obj.itemsize
        )

    else:
        assert shmem_name is not None
        existing_type = cls
        size = 1

    shm = SharedMemory(shmem_name, size=size)

    class CustomShareableProxy(existing_type):

        def __init__(self, *args, buffer=None, **kwargs):
            # If copy method called, prevent recursion from replacing _shm.
            if not hasattr(self, "_shm"):
                self._shm = shm
                self._proxied_type = existing_type
            else:
                # _proxied_type only used in pickling.
                assert hasattr(self, "_proxied_type")
            try:
                existing_type.__init__(self, *args, **kwargs)
            except Exception:
                pass

        def __repr__(self):
            if not hasattr(self, "_shm"):
                return existing_type.__repr__(self)
            formatted_pairs = (
                "%s=%r" % kv for kv in self._build_state(self).items()
            )
            return f"{self.__class__.__name__}({', '.join(formatted_pairs)})"

        #def __getstate__(self):
        #    if not hasattr(self, "_shm"):
        #        return existing_type.__getstate__(self)
        #    state = self._build_state(self)
        #    return state

        #def __setstate__(self, state):
        #    self.__init__(**state)

        def __reduce__(self):
            return (
                shareable_wrap,
                (
                    None,
                    self._shm.name,
                    self._proxied_type,
                    self.shape,
                    self.strides,
                    self.dtype.str if hasattr(self, "dtype") else None,
                    getattr(self, "format", None),
                ),
            )

        def copy(self):
            dupe = existing_type.copy(self)
            if not hasattr(dupe, "_shm"):
                dupe = shareable_wrap(dupe)
            return dupe

        @staticmethod
        def _build_state(existing_obj, generics_only=False):
            state = {
                "shape": existing_obj.shape,
                "strides": existing_obj.strides,
            }
            try:
                state["dtype"] = existing_obj.dtype
            except AttributeError:
                try:
                    state["format"] = existing_obj.format
                except AttributeError:
                    pass
            if not generics_only:
                try:
                    state["shmem_name"] = existing_obj._shm.name
                    state["cls"] = existing_type
                except AttributeError:
                    pass
            return state

    proxy_type = type(
        f"{existing_type.__name__}Shareable",
        CustomShareableProxy.__bases__,
        dict(CustomShareableProxy.__dict__),
    )

    if existing_obj is not None:
        try:
            proxy_obj = proxy_type(
                buffer=shm.buf,
                **proxy_type._build_state(existing_obj)
            )
        except Exception:
            proxy_obj = proxy_type(
                buffer=shm.buf,
                **proxy_type._build_state(existing_obj, True)
            )

        mveo = memoryview(existing_obj)
        proxy_obj._shm.buf[:mveo.nbytes] = mveo.tobytes()

    else:
        proxy_obj = proxy_type(buffer=shm.buf, **augmented_kwargs)

    return proxy_obj


def share_DataFrame_column(df, column):
    """Because pandas uses a BlockManager to oversee the numpy.ndarray
    objects behind each column of a DataFrame, getting or setting columns
    in a DataFrame often results in the creation of new numpy arrays from
    existing ones, making it non-trivial to place any array using shared
    memory into a DataFrame without it being replaced along the way.  This
    function takes an already existing column on a DataFrame and copies
    its numpy.ndarray-backed contents into shared memory for subsequent
    and continued use in the DataFrame.

    If an operation is performed on a DataFrame such that yet another
    copy of the existing column's ndarray data is constructed, that
    column may need to be placed into shared memory again.

    Be aware that because panda's BlockManager prefers to combine the
    storage of two or more columns of the same dtype into a single
    numpy array (with ndim>=2), requesting that one column be moved
    to shared memory may cause all columns of that same dtype to be
    moved as well."""

    # A DataFrame's BlockManager is currently held in `_data`.
    blkmgr = df._data

    # A BlockManager holds onto instances of Block (or subtypes such
    # as IntBlock) in a tuple named `blocks`.
    column_NDFrame_position = blkmgr.items.get_loc(column)
    column_blocks_position = blkmgr._blknos[column_NDFrame_position]
    insitu_block = blkmgr.blocks[column_blocks_position]

    # Sharing an ndarray of objects is not supported.
    assert not insitu_block.dtype.str.endswith('O')

    # Take the existing ndarray and copy it into shared memory,
    # replacing it in the block with its "shared" equivalent which
    # should also be conveniently pickle-able without needing to
    # serialize the data it holds.
    shared_values = shareable_wrap(insitu_block.values)
    insitu_block.values = shared_values

    # Return a handle on the object in shared memory for tracking
    # purposes so that its shared memory segment may be released later.
    return shared_values


def share_Series(series):
    "Analog to share_DataFrame_column() but for a pandas Series object."

    singleblkmgr = series._data

    insitu_block = singleblkmgr.blocks[0]

    assert not insitu_block.dtype.str.endswith('O')

    shared_values = shareable_wrap(insitu_block.values)
    insitu_block.values = shared_values

    return shared_values
