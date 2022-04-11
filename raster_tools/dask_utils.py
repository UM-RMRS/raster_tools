import dask.array as da
import numpy as np


def _handle_empty(func):
    def wrapped(x, axis=None, keepdims=False):
        if x.size > 0 or np.isnan(x.size):
            try:
                return func(x, axis=axis, keepdims=keepdims)
            except ValueError:
                pass
        return np.array([], dtype=x.dtype)

    return wrapped


# np.nan{min, max} both throw errors for empty chumks. dask.array.nan{min, max}
# handles empty chunks but requires that the chunk sizes be known at runtime.
# This safely handles empty chunks. There may still be corner cases that have
# not been found but for now it works.
_nanmin_empty_safe = _handle_empty(np.nanmin)
_nanmax_empty_safe = _handle_empty(np.nanmax)


def dask_nanmin(x):
    """
    Retrieves the minimum value, ignoring nan values and handling empty blocks.
    """
    return da.reduction(
        x,
        _nanmin_empty_safe,
        _nanmin_empty_safe,
        axis=None,
        keepdims=False,
        dtype=x.dtype,
    )


def dask_nanmax(x):
    """
    Retrieves the maximum value, ignoring nan values and handling empty blocks.
    """
    return da.reduction(
        x,
        _nanmax_empty_safe,
        _nanmax_empty_safe,
        axis=None,
        keepdims=False,
        dtype=x.dtype,
    )
