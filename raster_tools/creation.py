from collections.abc import Sequence

import dask.array as da
import numpy as np

from raster_tools.dtypes import is_int, is_scalar
from raster_tools.raster import get_raster

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

    band_list = [1] * bands
    *_, rows, columns = rst.shape
    shape = (bands, rows, columns)
    # Use get_bands to get data with the right chunks
    outrs = rst.get_bands(band_list)
    chunks = outrs._data.chunks

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

    outrs._data = ndata
    outrs._null_value = None
    outrs._mask = da.zeros_like(ndata, dtype=bool)
    return outrs


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

    band_list = [1] * bands
    outrs = rst.get_bands(band_list)
    ndata = da.empty_like(outrs._data, dtype=dtype)
    outrs._data = ndata
    outrs._null_value = None
    outrs._mask = da.zeros_like(ndata, dtype=bool)
    return outrs


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

    band_list = [1] * bands
    outrs = rst.get_bands(band_list)
    ndata = da.full_like(outrs._data, value, dtype=dtype)
    outrs._data = ndata
    outrs._null_value = None
    outrs._mask = da.zeros_like(ndata, dtype=bool)
    return outrs


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
    value : scalar
        Value to fill result with.
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
    value : scalar
        Value to fill result with.
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
