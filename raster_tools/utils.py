import os
from functools import wraps

import numpy as np
import xarray as xr


def validate_file(path):
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"Could not find file: '{path}'")


def validate_path(path):
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Path does not exist: '{path}'")


def is_strictly_increasing(x):
    x = np.atleast_1d(np.asarray(x).squeeze())
    assert x.ndim == 1

    diff = np.diff(x)
    return len(diff) > 0 and (diff > 0).all()


def is_strictly_decreasing(x):
    x = np.atleast_1d(np.asarray(x).squeeze())
    assert x.ndim == 1

    diff = np.diff(x)
    return len(diff) > 0 and (diff < 0).all()


def can_broadcast(*shapes, max_dim=3, min_dim=3):
    try:
        result = np.broadcast_shapes(*shapes)
        return len(result) >= min_dim and len(result) <= max_dim
    except ValueError:
        return False


def make_raster_ds(raster_dataarray, mask_dataarray):
    return xr.Dataset({"raster": raster_dataarray, "mask": mask_dataarray})


def merge_masks(masks):
    mask = None
    for m in masks:
        if mask is None:
            mask = m.copy()
        else:
            mask |= m
    return mask


def single_band_mappable(
    func_=None, *, no_input_chunk=False, pass_block_info=False
):
    """
    Decorator allowing functions that operate on 2D chunks to handle single
    band 3D chunks.

    This allows functions to be mapped to (1, y, x) chunks in a dask map_blocks
    operation.

    """

    if not no_input_chunk:

        def decorator(func):
            @wraps(func)
            def wrapper(chunk, *args, block_info=None, **kwargs):
                has_band_dim = chunk.ndim == 3
                if has_band_dim:
                    if chunk.shape[0] > 1:
                        raise TypeError(
                            f"Expexted 0 or 1 bands. Got {chunk.shape[0]}."
                        )
                    chunk = chunk[0]
                if not pass_block_info:
                    result = func(chunk, *args, **kwargs)
                else:
                    result = func(
                        chunk, *args, block_info=block_info, **kwargs
                    )
                if has_band_dim:
                    if result.ndim > 2:
                        raise TypeError(
                            "Expected result to have 2 dimensions. "
                            f"Got {result.ndim}."
                        )
                    result = np.expand_dims(result, axis=0)
                return result

            return wrapper

    else:

        def decorator(func):
            @wraps(func)
            def wrapper(*args, block_info=None, **kwargs):
                result = func(*args, block_info=block_info, **kwargs)
                if (
                    block_info is not None
                    and result.ndim < 3
                    and len(block_info[None]["chunk-shape"]) == 3
                ):
                    result = np.expand_dims(result, axis=0)
                return result

            return wrapper

    if func_ is None:
        return decorator
    return decorator(func_)
