import dask.array as da
import numba as nb
import numpy as np
from dask_image import ndfilters
from functools import partial

from ._types import promote_data_dtype
from ._utils import is_int


ngjit = nb.jit(nopython=True, nogil=True)


@ngjit
def _agg_nan_min(x):
    return np.nanmin(x)


@ngjit
def _agg_nan_max(x):
    return np.nanmax(x)


@ngjit
def _agg_nan_mean(x):
    return np.nanmean(x)


@ngjit
def _agg_nan_median(x):
    return np.nanmedian(x)


@ngjit
def _agg_nan_sum(x):
    return np.nansum(x)


@ngjit
def _agg_nan_var(x):
    return np.nanvar(x)


@ngjit
def _agg_nan_std(x):
    return np.nanstd(x)


@ngjit
def _agg_nan_unique(x):
    # Create set of floats. {1.0} is a hack to tell numba.jit what type the set
    # contains
    s = {1.0}
    s.clear()
    for v in x.ravel():
        if not np.isnan(v):
            s.add(v)
    if len(s):
        return len(s)
    return np.nan


@ngjit
def _agg_nan_mode(x):
    one: nb.types.uint16 = 1
    c = {}
    x = x.ravel()
    j: nb.types.uint16 = 0
    for i in range(x.size):
        v = x[i]
        if not np.isnan(v):
            if v in c:
                c[v] += one
            else:
                c[v] = one
            x[j] = v
            j += one
    vals = x[:j]
    if len(vals) == 0:
        return np.nan
    vals.sort()
    cnts = np.empty(len(vals), dtype=nb.types.uint16)
    for i in range(len(vals)):
        cnts[i] = c[vals[i]]
    return vals[np.argmax(cnts)]


@ngjit
def _agg_nan_entropy(x):
    c = {}
    one: nb.types.uint16 = 1
    n: nb.types.uint16 = 0
    for v in x.ravel():
        if not np.isnan(v):
            if v in c:
                c[v] += one
            else:
                c[v] = one
            n += one
    if len(c) == 0:
        return np.nan
    entr = 0.0
    frac = one / n
    for cnt in c.values():
        p = cnt * frac
        entr += -p * np.log(p)
    return entr


@ngjit
def _agg_nan_asm(x):
    c = {}
    one: nb.types.uint16 = 1
    n: nb.types.uint16 = 0
    for v in x.ravel():
        if not np.isnan(v):
            if v in c:
                c[v] += one
            else:
                c[v] = one
            n += one
    if len(c) == 0:
        return np.nan
    asm = 0.0
    frac = one / n
    for cnt in c.values():
        p = cnt * frac
        asm += p * p
    return asm


@ngjit
def _apply_filter_func(chunk, func, kernel):
    out = np.empty_like(chunk)
    rows, cols = chunk.shape
    krows, kcols = kernel.shape
    kr_top = (krows - 1) // 2
    kr_bot = krows // 2
    kc_left = (kcols - 1) // 2
    kc_right = kcols // 2
    r_ovlp = max(kr_top, kr_bot)
    c_ovlp = max(kc_left, kc_right)
    kernel_values = np.empty(kernel.size, dtype=chunk.dtype)
    kernel = kernel.ravel()

    # iterate over rows, skipping outer overlap rows
    for r in range(r_ovlp, rows - r_ovlp):
        # iterate over cols, skipping outer overlap cols
        for c in range(c_ovlp, cols - c_ovlp):
            kernel_values.fill(np.nan)
            # Kernel flat index
            ki = 0
            # iterate over kernel footprint, this extends into overlap regions
            # at edges
            for kr in range(r - kr_top, r + kr_bot + 1):
                for kc in range(c - kc_left, c + kc_right + 1):
                    if kernel[ki]:
                        kernel_values[ki] = chunk[kr, kc]
                    ki += 1
            out[r, c] = func(kernel_values)
    return out


def _get_offsets(kernel):
    """
    Returns the number of cells on either side of kernel center in both
    directions.
    """
    krows, kcols = kernel.shape
    kr_top = (krows - 1) // 2
    kr_bot = krows // 2
    kc_left = (kcols - 1) // 2
    kc_right = kcols // 2
    return ((kr_top, kr_bot), (kc_left, kc_right))


def _focal_dask(data, kernel, func):
    chunk_func = partial(_apply_filter_func, func=func, kernel=kernel)
    offsets = _get_offsets(kernel)
    # map_overlap does not support asymmetrical padding so take max. This adds
    # at most one extra pixel to each dim.
    rpad = max(offsets[0])
    cpad = max(offsets[1])
    return data.map_overlap(
        chunk_func,
        depth={0: rpad, 1: cpad},
        boundary=np.nan,
        dtype=data.dtype,
        meta=np.array((), dtype=data.dtype),
    )


def _focal(data, kernel, func):
    # TODO: check for cupy eventually
    return _focal_dask(data, kernel, func)


# Focal ops that promote dtype to float
FOCAL_PROMOTING_OPS = frozenset(
    (
        "asm",
        "entropy",
        "mean",
        "median",
        "std",
        "var",
    )
)


FOCAL_STATS = frozenset(
    (
        "asm",
        "entropy",
        "max",
        "mean",
        "median",
        "mode",
        "min",
        "std",
        "sum",
        "var",
        "unique",
    )
)


def focal(data, kernel, stat, nan_aware=False):
    """
    Applies the `stat` function to the `data` using `kernel` to determine the
    neighborhood for each pixel. `nan_aware` indicates whether the filter
    should handle NaN values. If `nan_aware` is False, optimizations may be
    made.
    """
    if stat not in FOCAL_STATS:
        raise ValueError(f"Unknown focal stat: '{stat}'")
    if len(kernel.shape) != 2:
        raise ValueError("Kernel must be 2D")

    if isinstance(data, np.ndarray):
        data = da.from_array(data)
    if nan_aware or stat in FOCAL_PROMOTING_OPS:
        data = promote_data_dtype(data)

    kernel = kernel.astype(bool)
    if stat == "asm":
        return _focal(data, kernel, _agg_nan_asm)
    elif stat == "entropy":
        return _focal(data, kernel, _agg_nan_entropy)
    elif stat == "min":
        if not nan_aware:
            return ndfilters.minimum_filter(
                data, footprint=kernel, mode="nearest"
            )
        else:
            return _focal(data, kernel, _agg_nan_min)
    elif stat == "max":
        if not nan_aware:
            return ndfilters.maximum_filter(
                data, footprint=kernel, mode="nearest"
            )
        else:
            return _focal(data, kernel, _agg_nan_max)
    elif stat == "mode":
        return _focal(data, kernel, _agg_nan_mode)
    elif stat == "mean":
        return _focal(data, kernel, _agg_nan_mean)
    elif stat == "median":
        return _focal(data, kernel, _agg_nan_median)
    elif stat == "std":
        return _focal(data, kernel, _agg_nan_std)
    elif stat == "var":
        return _focal(data, kernel, _agg_nan_var)
    elif stat == "sum":
        if not nan_aware:
            return ndfilters.correlate(data, kernel, mode="constant")
        else:
            return _focal(data, kernel, _agg_nan_sum)
    elif stat == "unique":
        return _focal(data, kernel, _agg_nan_unique)


def get_focal_window(width_or_radius, height=None):
    """
    Get a rectangle, circle, or annulus focal window.

    A rectangle window is simply a NxM grid of `True` values. A circle window
    is a grid with a centered circle of `True` values surrounded by `False`
    values. The circle extends to the edge of the grid. An annulus window is
    the same as a circle window but  has a nested circle of `False` values
    inside the main circle.

    Parameters
    ----------
    width_or_radius : int or 2-tuple of ints
        If an int and `height` is `None`, specifies the radius of a circle
        window. If an int and `height` is also an int, specifies the width of
        a rectangle window. If a 2-tuple of ints, the values specify the inner
        and outer radii of an annulus window.
    height : int or None
        If `None` (default), `width_or_radius` will be used to construct a
        circle or annulus window. If an int, specifies the height of a
        rectangle window.

    Returns
    -------
    window : ndarray
        A focal window containing bools with the specified pattern.
    """
    if isinstance(width_or_radius, (list, tuple)):
        if len(width_or_radius) != 2:
            raise ValueError(
                "If width_or_radius is a sequence, it must be size 2"
            )
        if width_or_radius[0] >= width_or_radius[1]:
            raise ValueError(
                "First radius value must be less than or equal to the second."
            )
    else:
        width_or_radius = [width_or_radius]
    for value in width_or_radius:
        if not is_int(value):
            raise TypeError(
                f"width_or_radius values must be integers: {value}"
            )
        elif value <= 0:
            raise ValueError(
                "Window width or radius values must be greater than 0."
                f" Got {value}"
            )
    if height is not None:
        if len(width_or_radius) == 2:
            raise ValueError(
                "height must be None if width_or_radius indicates annulus"
            )
        if not is_int(height):
            raise TypeError(f"height must be an integer or None: {height}")
        elif height <= 0:
            raise ValueError(
                f"Window height must be greater than 0. Got {height}"
            )

    window_out = None
    windows = []
    if height is None:
        for rvalue in width_or_radius:
            width = ((rvalue - 1) * 2) + 1
            height = width
            r = (width - 1) // 2
            window = np.zeros((height, width), dtype=bool)
            for x in range(width):
                for y in range(height):
                    rxy = np.sqrt((x - r) ** 2 + (y - r) ** 2)
                    if rxy <= r:
                        window[x, y] = True
            windows.append(window)
        if len(windows) != 2:
            window_out = windows[0]
        else:
            w1, w2 = windows
            padding = (np.array(w2.shape) - np.array(w1.shape)) // 2
            w1 = np.pad(w1, padding, mode="constant", constant_values=False)
            w2[w1] = False
            window_out = w2
    else:
        width = width_or_radius[0]
        window_out = np.ones((width, height), dtype=bool)
    return window_out
