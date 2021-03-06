"""A pickleable wrapper for sharing NumPy ndarrays between processes using POSIX shared memory."""

from shared_memory import Error as SharedNDArrayError

from .shared_ndarray import SharedNDArray

__all__ = ['SharedNDArray', 'SharedNDArrayError']

__author__ = 'Katherine Crowson'
__copyright__ = 'Copyright 2017, Katherine Crowson'
__credits__ = ['Katherine Crowson']
__license__ = 'MIT'
__maintainer__ = 'Katherine Crowson'
__email__ = 'crowsonkb@gmail.com'
