from collections.abc import Sequence

import dask.array as da
import numpy as np
import xarray as xr

from raster_tools.dtypes import is_int, is_scalar
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
        "webull",
    )
)


def random_raster(
    raster_template, distribution="normal", bands=1, params=(1, 0.5)
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
    if rst.crs is not None:
        xdata = xdata.rio.write_crs(rst.crs)
    xmask = xr.zeros_like(xdata, dtype=bool)
    return Raster(make_raster_ds(xdata, xmask), _fast_path=True)


def empty_like(raster_template, bands=1, dtype=None):
    """Create a Raster filled with uninitialized data like a template raster.

    Parameters
    ----------
    raster_template : Raster, str
        Template raster used to define rows, columns, crs, resolution, etc
    bands : int, optional
        Number of bands desired for output. Default is 1.
    dtype : data-type, optional
        Overrides the result dtype.

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
    if rst.crs is not None:
        xdata = xdata.rio.write_crs(rst.crs)
    xmask = xr.zeros_like(xdata, dtype=bool)
    return Raster(make_raster_ds(xdata, xmask), _fast_path=True)


def full_like(raster_template, value, bands=1, dtype=None):
    """Create a Raster filled with a contant value like a template raster.

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
    if rst.crs is not None:
        xdata = xdata.rio.write_crs(rst.crs)
    xmask = xr.zeros_like(xdata, dtype=bool)
    return Raster(make_raster_ds(xdata, xmask), _fast_path=True)


def constant_raster(raster_template, value=1, bands=1):
    """Create a Raster filled with a contant value like a template raster.

    This is a convenience function that wraps :func:`full_like`.

    Parameters
    ----------
    raster_template : Raster, str
        Template raster used to define rows, columns, crs, resolution, etc
    value : scalar, optional
        Value to fill result with. Default is 1.
    bands : int, optional
        Number of bands desired for output. Default is 1.

    Returns
    -------
    Raster
        The resulting raster of constant values.

    """
    return full_like(raster_template, value, bands=bands)


def zeros_like(raster_template, bands=1, dtype=None):
    """Create a Raster filled with zeros like a template raster.

    Parameters
    ----------
    raster_template : Raster, str
        Template raster used to define rows, columns, crs, resolution, etc
    bands : int, optional
        Number of bands desired for output. Default is 1.
    dtype : data-type, optional
        Overrides the result dtype.

    Returns
    -------
    Raster
        The resulting raster of zreos.

    """
    return full_like(raster_template, 0, bands=bands, dtype=dtype)


def ones_like(raster_template, bands=1, dtype=None):
    """Create a Raster filled with ones like a template raster.

    Parameters
    ----------
    raster_template : Raster, str
        Template raster used to define rows, columns, crs, resolution, etc
    bands : int, optional
        Number of bands desired for output. Default is 1.
    dtype : data-type, optional
        Overrides the result dtype.

    Returns
    -------
    Raster
        The resulting raster of ones.

    """
    return full_like(raster_template, 1, bands=bands, dtype=dtype)
