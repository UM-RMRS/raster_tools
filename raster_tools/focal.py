from functools import partial

import dask.array as da
import numba as nb
import numpy as np
import xarray as xr
from dask_image import ndfilters

from raster_tools.dtypes import (
    F64,
    U8,
    U16,
    U32,
    U64,
    is_bool,
    is_float,
    is_int,
    promote_data_dtype,
    promote_dtype_to_float,
)
from raster_tools.raster import Raster, get_raster
from raster_tools.stat_common import (
    nan_unique_count_jit,
    nanasm_jit,
    nanentropy_jit,
    nanmax_jit,
    nanmean_jit,
    nanmedian_jit,
    nanmin_jit,
    nanmode_jit,
    nanstd_jit,
    nansum_jit,
    nanvar_jit,
)
from raster_tools.utils import make_raster_ds, single_band_mappable

__all__ = [
    "check_kernel",
    "convolve",
    "correlate",
    "focal",
    "get_focal_window",
]


ngjit = nb.jit(nopython=True, nogil=True)


@single_band_mappable
@ngjit
def _focal_chunk(chunk, kernel, kernel_func):
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
            out[r, c] = kernel_func(kernel_values)
    return out


@single_band_mappable
@nb.jit(nopython=True, nogil=True, parallel=True)
def _correlate2d_chunk(chunk, kernel):
    out = np.empty_like(chunk)
    rows, cols = chunk.shape
    krows, kcols = kernel.shape
    kr_top = (krows - 1) // 2
    kr_bot = krows // 2
    kc_left = (kcols - 1) // 2
    kc_right = kcols // 2
    r_ovlp = max(kr_top, kr_bot)
    c_ovlp = max(kc_left, kc_right)
    kernel = kernel.ravel()

    # iterate over rows, skipping outer overlap rows
    for r in nb.prange(r_ovlp, rows - r_ovlp):
        # iterate over cols, skipping outer overlap cols
        for c in nb.prange(c_ovlp, cols - c_ovlp):
            # Kernel flat index
            ki = 0
            # iterate over kernel footprint, this extends into overlap regions
            # at edges
            v = 0.0
            for kr in range(r - kr_top, r + kr_bot + 1):
                for kc in range(c - kc_left, c + kc_right + 1):
                    val = chunk[kr, kc]
                    if not np.isnan(val):
                        v += kernel[ki] * chunk[kr, kc]
                    ki += 1
            out[r, c] = v
    return out


@nb.jit(
    "UniTuple(UniTuple(int64, 2), 2)(UniTuple(int64, 2))",
    nopython=True,
    nogil=True,
)
def _get_offsets(kernel_shape):
    """
    Returns the number of cells on either side of kernel center in both
    directions.
    """
    krows, kcols = kernel_shape
    kr_top = (krows - 1) // 2
    kr_bot = krows // 2
    kc_left = (kcols - 1) // 2
    kc_right = kcols // 2
    return ((kr_top, kr_bot), (kc_left, kc_right))


def check_kernel(kernel):
    """Validate and return the focal kernel.

    Parameters
    ----------
    kernel : 2D numpy.ndarray
        The kernel to validate. Errors are raised if it is not a numpy array,
        contains nan values, or is not 2D.

    Returns
    -------
    numpy.ndarray
        The input `kernel`, if nothing was wrong with it.

    """
    if not isinstance(kernel, np.ndarray):
        raise TypeError("Kernel must be numpy.ndarray")
    if len(kernel.shape) != 2:
        raise ValueError("Kernel must be 2D")
    if np.isnan(kernel).any():
        raise ValueError("Kernel can't contain NaN values")
    return kernel


def _check_data(data):
    if not isinstance(data, (np.ndarray, da.Array)):
        raise TypeError("Kernel must be numpy.ndarray or dask.array.Array")
    if len(data.shape) != 2:
        raise ValueError("Data must be 2D")


_MODE_TO_DASK_BOUNDARY = {
    "reflect": "reflect",
    "nearest": "nearest",
    "wrap": "periodic",
    "constant": "constant",
}
_VALID_CORRELATE_MODES = frozenset(_MODE_TO_DASK_BOUNDARY.keys())


def _correlate(data, kernel, mode="constant", cval=0.0, nan_aware=False):
    """Cross-correlates a `kernel` with `data`.

    This function can be used for convolution as well; just rotate the kernel
    180 degress (e.g. ``kernel = kernel[::-1, ::-1])`` before calling this
    function.

    Parameters
    ----------
    data : 2D ndarray or dask array
        Data to cross-correlate the kernel with.
    kernel : 2D ndarray
        Kernel to apply to the data through cross-correlation.
    mode : {'reflect', 'nearest', 'wrap', 'constant'}, optional
        The mode to use for the edges of the data. Default is 'constant'.
    cval : scalar, optional
        The value to use when `mode` is 'constant'. Default is ``0.0``.
    nan_aware : bool, optional
        If ``True``, NaN values are ignored during correlation. If ``False``,
        a faster correlation algorithm can be used. Default is ``False``.

    Returns
    -------
    correlated : 2D dask array
        The cross-correlation result as a lazy dask array.

    """
    if isinstance(data, np.ndarray):
        data = da.from_array(data)
    if nan_aware:
        data = promote_data_dtype(data)
    if is_bool(kernel.dtype):
        kernel = kernel.astype(int)
    if is_float(data.dtype) and is_int(kernel.dtype):
        kernel = kernel.astype(data.dtype)
    if is_int(data.dtype) and is_float(kernel.dtype):
        data = promote_data_dtype(data)

    if nan_aware:
        boundary = _MODE_TO_DASK_BOUNDARY[mode]
        if boundary == "constant":
            boundary = cval

        offsets = _get_offsets(kernel.shape)
        # map_overlap does not support asymmetrical padding so take max. This
        # adds at most one extra pixel to each dim.
        rpad, cpad = [max(o) for o in offsets]
        data = data.map_overlap(
            _correlate2d_chunk,
            kernel=kernel,
            depth={0: 0, 1: rpad, 2: cpad},
            boundary=boundary,
            dtype=data.dtype,
            meta=np.array((), dtype=data.dtype),
        )
    else:
        # Shift pixel origins to match ESRI behavior for even shaped kernels
        shift_origin = [d % 2 == 0 for d in kernel.shape]
        origin = [-1 if shift else 0 for shift in shift_origin]
        data = da.stack(
            [
                ndfilters.correlate(
                    band, kernel, mode=mode, cval=cval, origin=origin
                )
                for band in data
            ]
        )
    return data


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
_STAT_TO_FUNC = {
    "asm": nanasm_jit,
    "entropy": nanentropy_jit,
    "max": nanmax_jit,
    "mean": nanmean_jit,
    "median": nanmedian_jit,
    "mode": nanmode_jit,
    "min": nanmin_jit,
    "std": nanstd_jit,
    "sum": nansum_jit,
    "var": nanvar_jit,
    "unique": nan_unique_count_jit,
}


def _focal(data, kernel, stat, nan_aware=False):
    """Apply a focal stat function.

    Applies the `stat` function to the `data` using `kernel` to determine the
    neighborhood for each pixel. `nan_aware` indicates whether the filter
    should handle NaN values. If `nan_aware` is False, optimizations may be
    made.

    """
    if isinstance(data, np.ndarray):
        data = da.from_array(data)

    kernel = kernel.astype(bool)
    if not nan_aware and stat in ("min", "max", "sum"):
        # Use ndfilters which is faster but dosn't handle nan values
        if stat == "min":
            func = partial(
                ndfilters.minimum_filter, footprint=kernel, mode="nearest"
            )
        elif stat == "max":
            func = partial(
                ndfilters.maximum_filter, footprint=kernel, mode="nearest"
            )
        elif stat == "sum":
            func = partial(
                ndfilters.correlate, weights=kernel, mode="constant"
            )
        data = da.stack([func(d) for d in data])
    else:
        # Use _focal_chunk which is slower but handles nan values.
        # Promote to allow for nan boundary fill values.
        new_dtype = promote_dtype_to_float(data.dtype)
        if new_dtype != data.dtype:
            data = data.astype(new_dtype)

        offsets = _get_offsets(kernel.shape)
        # map_overlap does not support asymmetrical padding so take max. This
        # adds at most one extra pixel to each dim.
        rpad = max(offsets[0])
        cpad = max(offsets[1])
        data = data.map_overlap(
            _focal_chunk,
            kernel=kernel,
            kernel_func=_STAT_TO_FUNC[stat],
            # dask
            depth={0: 0, 1: rpad, 2: cpad},
            boundary=np.nan,
            dtype=data.dtype,
            meta=np.array((), dtype=data.dtype),
        )
    return data


def focal(raster, focal_type, width_or_radius, height=None, ignore_null=False):
    """Applies a focal filter to raster bands individually.

    The filter uses a window/footprint that is created using the
    `width_or_radius` and `height` parameters. The window can be a
    rectangle, circle or annulus.

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the focal operation on.
    focal_type : str
        Specifies the aggregation function to apply to the focal
        neighborhood at each pixel. Can be one of the following string
        values:

        'min'
            Finds the minimum value in the neighborhood.
        'max'
            Finds the maximum value in the neighborhood.
        'mean'
            Finds the mean of the neighborhood.
        'median'
            Finds the median of the neighborhood.
        'mode'
            Finds the mode of the neighborhood.
        'sum'
            Finds the sum of the neighborhood.
        'std'
            Finds the standard deviation of the neighborhood.
        'var'
            Finds the variance of the neighborhood.
        'asm'
            Angular second moment. Applies -sum(P(g)**2) where P(g) gives
            the probability of g within the neighborhood.
        'entropy'
            Calculates the entropy. Applies -sum(P(g) * log(P(g))). See
            'asm' above.
        'unique'
            Calculates the number of unique values in the neighborhood.
    width_or_radius : int or 2-tuple of ints
        If an int and `height` is `None`, specifies the radius of a circle
        window. If an int and `height` is also an int, specifies the width
        of a rectangle window. If a 2-tuple of ints, the values specify the
        inner and outer radii of an annulus window.
    height : int or None
        If `None` (default), `width_or_radius` will be used to construct a
        circle or annulus window. If an int, specifies the height of a
        rectangle window.
    ignore_null : bool
        If `False`, cells marked as null remain null. If `True`, cells marked
        as null may receive a data value, provided there are valid cells in the
        neighborhood. The default is `False`.

    Returns
    -------
    Raster
        The resulting raster with focal filter applied to each band. The
        bands will have the same shape as the original Raster.

    """
    raster = get_raster(raster)
    if focal_type not in FOCAL_STATS:
        raise ValueError(f"Unknown focal operation: '{focal_type}'")

    window = get_focal_window(width_or_radius, height)
    data = raster.data

    # Convert to float and fill nulls with nan, if needed
    if raster._masked:
        new_dtype = promote_dtype_to_float(raster.dtype)
        if new_dtype != data.dtype:
            data = data.astype(new_dtype)
        data = da.where(raster.mask, np.nan, data)

    data = _focal(data, window, focal_type, raster._masked)

    if raster._masked:
        if focal_type == "unique":
            # Values of 0 mean that all values in window were nan (AKA null) so
            # replace them with nan
            data = da.where(data == 0, np.nan, data)
    else:
        if focal_type == "mode" and is_int(raster.dtype):
            data = data.astype(raster.dtype)
        elif focal_type == "unique":
            n = window.size
            unq_dtype = None
            for dt in (U8, U16, U32, U64):
                if np.can_cast(n, dt):
                    unq_dtype = dt
                    break
            data = data.astype(unq_dtype)

    xdata = xr.DataArray(
        data, coords=raster.xdata.coords, dims=raster.xdata.dims
    )
    xmask = raster.xmask.copy()
    # Nan values will only appear in the result data if there were null values
    # present in the input. Thus we only need to worry about updating the mask
    # if the input was masked.
    if raster._masked:
        nan_mask = np.isnan(xdata)
        if ignore_null:
            xmask = nan_mask
        else:
            xmask |= nan_mask
        xdata = xdata.rio.write_nodata(raster.null_value)
    ds = make_raster_ds(xdata, xmask)
    if raster.crs is not None:
        ds = ds.rio.write_crs(raster.crs)
    return Raster(ds, _fast_path=True).burn_mask()


def get_focal_window(width_or_radius, height=None):
    """Get a rectangle, circle, or annulus focal window.

    A rectangle window is simply a NxM grid of ``True`` values. A circle window
    is a grid with a centered circle of ``True`` values surrounded by `False`
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


def correlate(raster, kernel, mode="constant", cval=0.0):
    """Cross-correlate `kernel` with each band individually. Returns a new
    Raster.

    The kernel is applied to each band in isolation so returned raster has
    the same shape as the original.

    Parameters
    ----------
    raster : Raster or path str
        The raster to cross-correlate `kernel` with. Can be multibanded.
    kernel : array_like
        2D array of kernel weights
    mode : {'reflect', 'constant', 'nearest', 'wrap'}, optional
        Determines how the data is extended beyond its boundaries. The
        default is 'constant'.

        'reflect' (d c b a | a b c d | d c b a)
            The data pixels are reflected at the boundaries.
        'constant' (k k k k | a b c d | k k k k)
            A constant value determined by `cval` is used to extend the
            data pixels.
        'nearest' (a a a a | a b c d | d d d d)
            The data is extended using the boundary pixels.
        'wrap' (a b c d | a b c d | a b c d)
            The data is extended by wrapping to the opposite side of the
            grid.
    cval : scalar, optional
        Value used to fill when `mode` is 'constant'. Default is 0.0.

    Returns
    -------
    Raster
        The resulting new Raster.
    """
    raster = get_raster(raster)
    kernel = np.asarray(kernel)
    check_kernel(kernel)
    if mode not in _VALID_CORRELATE_MODES:
        raise ValueError(f"Invalid mode: '{mode}'")
    if is_float(kernel.dtype) and is_int(raster.dtype):
        raster = raster.astype(F64)
    data = raster.data
    final_dtype = data.dtype

    # Convert to float and fill nulls with nan, if needed
    upcast = False
    if raster._masked:
        new_dtype = promote_dtype_to_float(data.dtype)
        upcast = new_dtype != data.dtype
        if upcast:
            data = data.astype(new_dtype)
        data = da.where(raster.mask, np.nan, data)

    data = _correlate(
        data, kernel, mode=mode, cval=cval, nan_aware=raster._masked
    )

    # Cast back to int, if needed
    if upcast:
        data = data.astype(final_dtype)

    xdata = xr.DataArray(
        data, coords=raster.xdata.coords, dims=raster.xdata.dims
    ).rio.write_nodata(raster.null_value)
    xmask = raster.xmask.copy()
    ds = make_raster_ds(xdata, xmask)
    if raster.crs is not None:
        ds = ds.rio.write_crs(raster.crs)
    return Raster(ds, _fast_path=True).burn_mask()


def convolve(raster, kernel, mode="constant", cval=0.0):
    """Convolve `kernel` with each band individually. Returns a new Raster.

    This is the same as correlation but the kernel is rotated 180 degrees,
    e.g. ``kernel = kernel[::-1, ::-1]``.  The kernel is applied to each
    band in isolation so the returned raster has the same shape as the
    original.

    Parameters
    ----------
    raster : Raster
        The raster to convolve `kernel` with. Can be multibanded.
    kernel : array_like
        2D array of kernel weights
    mode : {'reflect', 'constant', 'nearest', 'wrap'}, optional
        Determines how the data is extended beyond its boundaries. The
        default is 'constant'.

        'reflect' (d c b a | a b c d | d c b a)
            The data pixels are reflected at the boundaries.
        'constant' (k k k k | a b c d | k k k k)
            A constant value determined by `cval` is used to extend the
            data pixels.
        'nearest' (a a a a | a b c d | d d d d)
            The data is extended using the boundary pixels.
        'wrap' (a b c d | a b c d | a b c d)
            The data is extended by wrapping to the opposite side of the
            grid.
    cval : scalar, optional
        Value used to fill when `mode` is 'constant'. Default is 0.0.

    Returns
    -------
    Raster
        The resulting new Raster.
    """
    kernel = np.asarray(kernel)
    check_kernel(kernel)
    kernel = kernel[::-1, ::-1].copy()
    return correlate(raster, kernel, mode=mode, cval=cval)
