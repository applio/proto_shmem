from shared_memory import shareable_wrap


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
