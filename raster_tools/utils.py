import os

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
            mask = m
        else:
            mask |= m
    return mask
