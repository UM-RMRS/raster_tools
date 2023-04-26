from collections.abc import Sequence

import dask.array as da
import numpy as np
import xarray as xr

from raster_tools.dtypes import is_int, is_scalar
from raster_tools.masking import get_default_null_value
from raster_tools.raster import Raster, get_raster
from raster_tools.utils import make_raster_ds

__all__ = [
    "constant_raster",
    "empty_like",
    "full_like",
    "ones_like",
    "random_raster",
    "zeros_like",
]

_VALID_RANDOM_DISTRIBUTIONS = frozenset(
    (
        "b",
        "binomial",
        "n",
        "normal",
        "p",
        "poisson",
        "u",
        "uniform",
        "w",
        "weibull",
    )
)


def _copy_mask(template, out_bands):
    if out_bands == template.nbands:
        return template.xmask.copy()
    mask = template.get_bands(1).xmask
    mask = xr.concat([mask] * out_bands, "band")
    mask["band"] = np.arange(out_bands) + 1
    return mask


def _build_result(template, xdata, nbands, copy_mask, copy_nv):
    if template._masked and copy_mask:
        xmask = _copy_mask(template, nbands)
        nv = (
            template.null_value
            if copy_nv
            else get_default_null_value(xdata.dtype)
        )
        xdata = xdata.rio.write_nodata(nv)
    else:
        xmask = xr.zeros_like(xdata, dtype=bool)
    ds = make_raster_ds(xdata, xmask)
    if template.crs is not None:
        ds = ds.rio.write_crs(template.crs)
    return Raster(ds, _fast_path=True).burn_mask()


def random_raster(
    raster_template,
    distribution="normal",
    bands=1,
    params=(1, 0.5),
    copy_mask=False,
):
    """Creates a Raster of random values based on the desired distribution.

    This function uses dask.array.random to generate data. The default is a
    normal distribution with mean of 1 and standard deviation of 0.5.

    Parameters
    ----------
    raster_template : Raster, str
        A template raster used to define rows, columns, crs, resolution, etc.
    distribution : str, optional
        Random distribution type. Default is `'normal'`. See `params` parameter
        below for passing additional distribution  parameters. Valid values
        are:

        'binomial'
            Binomial distribution. Uses two additional parameters: (n, p).
        'normal'
            Normal distribution. Uses two additional parameters: (mean, std).
            This is the default.
        'poisson'
            Poisson distribution. Uses one additional parameter.
        'uniform'
            Uniform distribution in the half-open interval [0, 1). Uses no
            additional.
        'weibull'
            Weibull distribution. Uses one additional parameter.

    bands : int, optional
        Number of bands needed desired. Default is 1.
    params : list of scalars, optional
        Additional parameters for generating the distribution. For example
        `distribution='normal'` and `params=(1, 0.5)` would result in a normal
        distribution with mean of 1 and standard deviation of 0.5. Default is
        ``(1, 0.5)``.
    copy_mask : bool
        If `True`, the template raster's mask is copied to the result raster.
        If `bands` differs from `raster_template`, the first band's mask is
        copied `bands` times.

    Returns
    -------
    Raster
        The resulting raster of random values pulled from the distribution.

    """
    rst = get_raster(raster_template)

    if not is_int(bands):
        try:
            bands = int(bands)
        except Exception:
            raise TypeError(
                f"Could not coerce bands argument to an int: {repr(bands)}"
            )
    if bands < 1:
        raise ValueError("Number of bands must be greater than 0")
    if not isinstance(params, Sequence):
        try:
            params = list(params)
        except Exception:
            raise TypeError(
                f"Could not coerce params argument to a list: {repr(params)}"
            )
    else:
        params = list(params)

    shape = (bands,) + rst.shape[1:]
    chunks = ((1,) * bands,) + rst.data.chunks[1:]

    dist = distribution.lower()
    if dist not in _VALID_RANDOM_DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution type: {repr(distribution)}")
    if dist not in ("u", "uniform") and len(params) == 0:
        raise ValueError(
            "Not enough additional parameters for the distribution"
        )

    if dist in ("n", "normal"):
        if len(params) < 2:
            raise ValueError(
                "Two few additional parameters for normal distribution"
            )
        ndata = da.random.normal(
            params[0], params[1], size=shape, chunks=chunks
        )
    elif dist in ("p", "poisson"):
        ndata = da.random.poisson(params[0], size=shape, chunks=chunks)
    elif dist in ("b", "binomial"):
        if len(params) < 2:
            raise ValueError(
                "Two few additional parameters for binomial distribution"
            )
        ndata = da.random.binomial(
            params[0], params[1], size=shape, chunks=chunks
        )
    elif dist in ("w", "weibull"):
        ndata = da.random.weibull(params[0], size=shape, chunks=chunks)
    elif dist in ("u", "uniform"):
        ndata = da.random.random(size=shape, chunks=chunks)
    # TODO: add more distributions

    cband = np.arange(bands) + 1
    xdata = xr.DataArray(
        ndata, coords=(cband, rst.y, rst.x), dims=("band", "y", "x")
    )
    return _build_result(rst, xdata, bands, copy_mask, False)


def empty_like(raster_template, bands=1, dtype=None, copy_mask=False):
    """Create a Raster filled with uninitialized data like a template raster.

    Parameters
    ----------
    raster_template : Raster, str
        Template raster used to define rows, columns, crs, resolution, etc
    bands : int, optional
        Number of bands desired for output. Default is 1.
    dtype : data-type, optional
        Overrides the result dtype.
    copy_mask : bool
        If `True`, the template raster's mask is copied to the result raster.
        If `bands` differs from `raster_template`, the first band's mask is
        copied `bands` times.

    Returns
    -------
    Raster
        The resulting raster of uninitialized data.

    """
    rst = get_raster(raster_template)

    if not is_int(bands):
        try:
            bands = int(bands)
        except Exception:
            raise TypeError(
                f"Could not coerce bands argument to an int: {repr(bands)}"
            )
    if bands < 1:
        raise ValueError("Number of bands must be greater than 0")
    if dtype is not None:
        try:
            dtype = np.dtype(dtype)
        except TypeError:
            raise ValueError(
                f"Could not understand dtype argument: {repr(dtype)}"
            )

    shape = (bands,) + rst.shape[1:]
    chunks = ((1,) * bands,) + rst.data.chunks[1:]
    ndata = da.empty(shape, chunks=chunks, dtype=dtype)
    xdata = xr.DataArray(
        ndata,
        coords=(np.arange(bands) + 1, rst.y, rst.x),
        dims=("band", "y", "x"),
    )
    copy_null = dtype is None or np.dtype(dtype) == rst.dtype
    return _build_result(rst, xdata, bands, copy_mask, copy_null)


def full_like(raster_template, value, bands=1, dtype=None, copy_mask=False):
    """Create a Raster filled with a constant value like a template raster.

    Parameters
    ----------
    raster_template : Raster, str
        Template raster used to define rows, columns, crs, resolution, etc
    value : scalar
        Value to fill result with.
    bands : int, optional
        Number of bands desired for output. Default is 1.
    dtype : data-type, optional
        Overrides the result dtype.
    copy_mask : bool
        If `True`, the template raster's mask is copied to the result raster.
        If `bands` differs from `raster_template`, the first band's mask is
        copied `bands` times.

    Returns
    -------
    Raster
        The resulting raster of constant values.

    """
    rst = get_raster(raster_template)

    if not is_int(bands):
        try:
            bands = int(bands)
        except Exception:
            raise TypeError(
                f"Could not coerce bands argument to an int: {repr(bands)}"
            )
    if bands < 1:
        raise ValueError("Number of bands must be greater than 0")
    if not is_scalar(value):
        try:
            value = float(value)
        except Exception:
            raise TypeError(
                f"Could not coerce value argument to a scalar: {repr(value)}"
            )
    if dtype is not None:
        try:
            dtype = np.dtype(dtype)
        except TypeError:
            raise ValueError(
                f"Could not understand dtype argument: {repr(dtype)}"
            )

    shape = (bands,) + rst.shape[1:]
    chunks = ((1,) * bands,) + rst.data.chunks[1:]
    ndata = da.full(shape, value, chunks=chunks, dtype=dtype)
    xdata = xr.DataArray(
        ndata,
        coords=(np.arange(bands) + 1, rst.y, rst.x),
        dims=("band", "y", "x"),
    )
    copy_null = dtype is None or np.dtype(dtype) == rst.dtype
    return _build_result(rst, xdata, bands, copy_mask, copy_null)


def constant_raster(raster_template, value=1, bands=1, copy_mask=False):
    """Create a Raster filled with a constant value like a template raster.

    This is a convenience function that wraps :func:`full_like`.

    Parameters
    ----------
    raster_template : Raster, str
        Template raster used to define rows, columns, crs, resolution, etc
    value : scalar, optional
        Value to fill result with. Default is 1.
    bands : int, optional
        Number of bands desired for output. Default is 1.
    copy_mask : bool
        If `True`, the template raster's mask is copied to the result raster.
        If `bands` differs from `raster_template`, the first band's mask is
        copied `bands` times.

    Returns
    -------
    Raster
        The resulting raster of constant values.

    """
    return full_like(raster_template, value, bands=bands, copy_mask=copy_mask)


def zeros_like(raster_template, bands=1, dtype=None, copy_mask=False):
    """Create a Raster filled with zeros like a template raster.

    Parameters
    ----------
    raster_template : Raster, str
        Template raster used to define rows, columns, crs, resolution, etc
    bands : int, optional
        Number of bands desired for output. Default is 1.
    dtype : data-type, optional
        Overrides the result dtype.
    copy_mask : bool
        If `True`, the template raster's mask is copied to the result raster.
        If `bands` differs from `raster_template`, the first band's mask is
        copied `bands` times.

    Returns
    -------
    Raster
        The resulting raster of zreos.

    """
    return full_like(
        raster_template, 0, bands=bands, dtype=dtype, copy_mask=copy_mask
    )


def ones_like(raster_template, bands=1, dtype=None, copy_mask=False):
    """Create a Raster filled with ones like a template raster.

    Parameters
    ----------
    raster_template : Raster, str
        Template raster used to define rows, columns, crs, resolution, etc
    bands : int, optional
        Number of bands desired for output. Default is 1.
    dtype : data-type, optional
        Overrides the result dtype.
    copy_mask : bool
        If `True`, the template raster's mask is copied to the result raster.
        If `bands` differs from `raster_template`, the first band's mask is
        copied `bands` times.

    Returns
    -------
    Raster
        The resulting raster of ones.

    """
    return full_like(
        raster_template, 1, bands=bands, dtype=dtype, copy_mask=copy_mask
    )
