import warnings
from collections.abc import Sequence

import dask.array as da
import numpy as np
from odc.geo.geobox import GeoBox

from raster_tools.dtypes import is_int, is_scalar
from raster_tools.masking import get_default_null_value
from raster_tools.raster import (
    data_to_raster,
    data_to_raster_like,
    get_raster,
)

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

_MISSING = object()


def _coalesce_template(template, raster_template):
    if raster_template is not None:
        if template is not None:
            raise TypeError(
                "Pass `template` only; `raster_template` is deprecated"
            )
        warnings.warn(
            "`raster_template` is deprecated; use `template`",
            DeprecationWarning,
            stacklevel=3,
        )
        return raster_template
    if template is None:
        raise TypeError("A template is required")
    return template


def _resolve_template(template):
    """Normalize a template to grid info.

    Returns
    -------
    rst : Raster or None
        The resolved Raster, or None if `template` is a GeoBox.
    yx_shape : tuple of int
        The (ny, nx) shape from the template.
    yx_chunks : tuple or None
        The (y, x) chunks from the template Raster, or None for a GeoBox.
    affine : affine.Affine or None
        The affine transform (only set for the GeoBox path).
    crs : CRS or None
        The CRS (only set for the GeoBox path).
    """
    if isinstance(template, GeoBox):
        return None, tuple(template.shape), None, template.affine, template.crs
    rst = get_raster(template, strict=False)
    return rst, rst.shape[1:], rst.data.chunks[1:], None, None


def _check_copy_mask(rst, copy_mask):
    if rst is None and copy_mask:
        raise ValueError("copy_mask requires a Raster template, not a GeoBox")


def _get_bands(bands):
    if not is_int(bands):
        try:
            bands = int(bands)
        except Exception:
            raise TypeError(
                f"Could not coerce bands argument to an int: {repr(bands)}"
            ) from None
    if bands < 1:
        raise ValueError("Number of bands must be greater than 0")
    return bands


def _get_dtype(dtype):
    if dtype is not None:
        try:
            dtype = np.dtype(dtype)
        except TypeError:
            raise ValueError(
                f"Could not understand dtype argument: {repr(dtype)}"
            ) from None
    return dtype


def _get_shape_and_chunks(yx_shape, yx_chunks, bands):
    shape = (bands,) + yx_shape
    if yx_chunks is not None:
        chunks = ((1,) * bands,) + yx_chunks
    else:
        # GeoBox path: chunksize 1 in band dim, dask picks spatial chunks
        chunks = (1, "auto", "auto")
    return shape, chunks


def _copy_mask(template, out_bands):
    if out_bands == template.nbands:
        return template.mask.copy()
    mask = template.mask[0]
    mask = da.stack([mask] * out_bands, axis=0)
    return mask


def _build_result(
    rst, data, nbands, copy_mask, copy_nv, affine=None, crs=None
):
    if rst is None:
        return data_to_raster(data, affine=affine, crs=crs)
    if rst._masked and copy_mask:
        mask = _copy_mask(rst, nbands)
        nv = rst.null_value if copy_nv else get_default_null_value(data.dtype)
        burn_mask = True
    else:
        mask = None
        nv = None
        burn_mask = False
    return data_to_raster_like(data, rst, mask=mask, nv=nv, burn=burn_mask)


def random_raster(
    template=None,
    distribution="normal",
    bands=1,
    params=(1, 0.5),
    copy_mask=False,
    *,
    raster_template=None,
):
    """Creates a Raster of random values based on the desired distribution.

    This function uses dask.array.random to generate data. The default is a
    normal distribution with mean of 1 and standard deviation of 0.5.

    Parameters
    ----------
    template : Raster, str, or odc.geo.geobox.GeoBox
        Template defining rows, columns, crs, resolution, etc. A path string
        is loaded as a Raster. A ``GeoBox`` supplies only the grid; in that
        case ``copy_mask`` is invalid and no mask/null value is propagated.
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
        If `bands` differs from the template, the first band's mask is copied
        `bands` times. Only valid when `template` is a Raster.

    Returns
    -------
    Raster
        The resulting raster of random values pulled from the distribution.

    """
    template = _coalesce_template(template, raster_template)
    rst, yx_shape, yx_chunks, affine, crs = _resolve_template(template)
    _check_copy_mask(rst, copy_mask)
    bands = _get_bands(bands)
    if not isinstance(params, Sequence):
        try:
            params = list(params)
        except Exception:
            raise TypeError(
                f"Could not coerce params argument to a list: {repr(params)}"
            ) from None
    else:
        params = list(params)

    shape, chunks = _get_shape_and_chunks(yx_shape, yx_chunks, bands)

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

    return _build_result(
        rst, ndata, bands, copy_mask, False, affine=affine, crs=crs
    )


def empty_like(
    template=None,
    bands=1,
    dtype=None,
    copy_mask=False,
    *,
    raster_template=None,
):
    """Create a Raster filled with uninitialized data like a template.

    Parameters
    ----------
    template : Raster, str, or odc.geo.geobox.GeoBox
        Template defining rows, columns, crs, resolution, etc. A path string
        is loaded as a Raster. A ``GeoBox`` supplies only the grid; in that
        case ``copy_mask`` is invalid and no mask/null value is propagated.
    bands : int, optional
        Number of bands desired for output. Default is 1.
    dtype : data-type, optional
        Overrides the result dtype.
    copy_mask : bool
        If `True`, the template raster's mask is copied to the result raster.
        If `bands` differs from the template, the first band's mask is
        copied `bands` times. Only valid when `template` is a Raster.

    Returns
    -------
    Raster
        The resulting raster of uninitialized data.

    """
    template = _coalesce_template(template, raster_template)
    rst, yx_shape, yx_chunks, affine, crs = _resolve_template(template)
    _check_copy_mask(rst, copy_mask)
    bands = _get_bands(bands)
    dtype = _get_dtype(dtype)

    shape, chunks = _get_shape_and_chunks(yx_shape, yx_chunks, bands)
    ndata = da.empty(shape, chunks=chunks, dtype=dtype)
    copy_null = rst is not None and (
        dtype is None or np.dtype(dtype) == rst.dtype
    )
    return _build_result(
        rst, ndata, bands, copy_mask, copy_null, affine=affine, crs=crs
    )


def full_like(
    template=None,
    value=_MISSING,
    bands=1,
    dtype=None,
    copy_mask=False,
    *,
    raster_template=None,
):
    """Create a Raster filled with a constant value like a template.

    Parameters
    ----------
    template : Raster, str, or odc.geo.geobox.GeoBox
        Template defining rows, columns, crs, resolution, etc. A path string
        is loaded as a Raster. A ``GeoBox`` supplies only the grid; in that
        case ``copy_mask`` is invalid and no mask/null value is propagated.
    value : scalar
        Value to fill result with.
    bands : int, optional
        Number of bands desired for output. Default is 1.
    dtype : data-type, optional
        Overrides the result dtype.
    copy_mask : bool
        If `True`, the template raster's mask is copied to the result raster.
        If `bands` differs from the template, the first band's mask is
        copied `bands` times. Only valid when `template` is a Raster.

    Returns
    -------
    Raster
        The resulting raster of constant values.

    """
    template = _coalesce_template(template, raster_template)
    if value is _MISSING:
        raise TypeError("full_like() missing required argument: 'value'")
    rst, yx_shape, yx_chunks, affine, crs = _resolve_template(template)
    _check_copy_mask(rst, copy_mask)
    bands = _get_bands(bands)
    if not is_scalar(value):
        try:
            value = float(value)
        except Exception:
            raise TypeError(
                f"Could not coerce value argument to a scalar: {repr(value)}"
            ) from None
    dtype = _get_dtype(dtype)

    shape, chunks = _get_shape_and_chunks(yx_shape, yx_chunks, bands)
    ndata = da.full(shape, value, chunks=chunks, dtype=dtype)
    if ndata.npartitions == 1:
        # https://github.com/dask/dask/issues/11531
        # Adding false preserves the dtype
        ndata += False
    copy_null = rst is not None and (
        dtype is None or np.dtype(dtype) == rst.dtype
    )
    return _build_result(
        rst, ndata, bands, copy_mask, copy_null, affine=affine, crs=crs
    )


def constant_raster(
    template=None,
    value=1,
    bands=1,
    dtype=None,
    copy_mask=False,
    *,
    raster_template=None,
):
    """Create a Raster filled with a constant value like a template.

    This is a convenience function that wraps :func:`full_like`.

    Parameters
    ----------
    template : Raster, str, or odc.geo.geobox.GeoBox
        Template defining rows, columns, crs, resolution, etc. A path string
        is loaded as a Raster. A ``GeoBox`` supplies only the grid; in that
        case ``copy_mask`` is invalid and no mask/null value is propagated.
    value : scalar, optional
        Value to fill result with. Default is 1.
    bands : int, optional
        Number of bands desired for output. Default is 1.
    dtype : data-type, optional
        Overrides the result dtype.
    copy_mask : bool
        If `True`, the template raster's mask is copied to the result raster.
        If `bands` differs from the template, the first band's mask is
        copied `bands` times. Only valid when `template` is a Raster.

    Returns
    -------
    Raster
        The resulting raster of constant values.

    """
    return full_like(
        template,
        value,
        bands=bands,
        dtype=dtype,
        copy_mask=copy_mask,
        raster_template=raster_template,
    )


def zeros_like(
    template=None,
    bands=1,
    dtype=None,
    copy_mask=False,
    *,
    raster_template=None,
):
    """Create a Raster filled with zeros like a template.

    Parameters
    ----------
    template : Raster, str, or odc.geo.geobox.GeoBox
        Template defining rows, columns, crs, resolution, etc. A path string
        is loaded as a Raster. A ``GeoBox`` supplies only the grid; in that
        case ``copy_mask`` is invalid and no mask/null value is propagated.
    bands : int, optional
        Number of bands desired for output. Default is 1.
    dtype : data-type, optional
        Overrides the result dtype.
    copy_mask : bool
        If `True`, the template raster's mask is copied to the result raster.
        If `bands` differs from the template, the first band's mask is
        copied `bands` times. Only valid when `template` is a Raster.

    Returns
    -------
    Raster
        The resulting raster of zeros.

    """
    return full_like(
        template,
        0,
        bands=bands,
        dtype=dtype,
        copy_mask=copy_mask,
        raster_template=raster_template,
    )


def ones_like(
    template=None,
    bands=1,
    dtype=None,
    copy_mask=False,
    *,
    raster_template=None,
):
    """Create a Raster filled with ones like a template.

    Parameters
    ----------
    template : Raster, str, or odc.geo.geobox.GeoBox
        Template defining rows, columns, crs, resolution, etc. A path string
        is loaded as a Raster. A ``GeoBox`` supplies only the grid; in that
        case ``copy_mask`` is invalid and no mask/null value is propagated.
    bands : int, optional
        Number of bands desired for output. Default is 1.
    dtype : data-type, optional
        Overrides the result dtype.
    copy_mask : bool
        If `True`, the template raster's mask is copied to the result raster.
        If `bands` differs from the template, the first band's mask is
        copied `bands` times. Only valid when `template` is a Raster.

    Returns
    -------
    Raster
        The resulting raster of ones.

    """
    return full_like(
        template,
        1,
        bands=bands,
        dtype=dtype,
        copy_mask=copy_mask,
        raster_template=raster_template,
    )
