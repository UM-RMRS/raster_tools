import dask
import dask.array as da
import numpy as np
import rasterio as rio
import xarray as xr

import raster_tools as rts
from raster_tools.dtypes import is_bool, is_scalar
from raster_tools.raster import Raster
from raster_tools.utils import (
    is_strictly_decreasing,
    is_strictly_increasing,
    null_values_equal,
)


def _as_3d(data):
    data = np.asarray(data)
    if data.ndim == 1:
        raise ValueError("Single dimension data not allowed")
    if data.ndim < 3:
        data = data[None]
    return data


def _norm_chunks(chunks, shape, dtype):
    if is_scalar(chunks):
        chunks = (1, chunks, chunks)
    elif isinstance(chunks, tuple):
        if len(chunks) == 2:
            chunks = (1, *chunks)
        elif len(chunks) == 3:
            assert chunks[0] == 1
        else:
            raise ValueError("Invalid chunks")
    return da.core.normalize_chunks(chunks, shape=shape, dtype=dtype)


def _apply_null_patterns(data, mask, patterns, nv, depth=0):
    if depth > 1:
        raise ValueError("Too much nesting in null_patterns")
    if isinstance(patterns, str):
        if patterns.startswith("%"):
            modv = int(patterns[1:])
            mask |= data % modv == 0
        elif patterns.startswith(">"):
            x = float(patterns[1:])
            mask |= data > x
        elif patterns.startswith("<"):
            x = float(patterns[1:])
            mask |= data < x
        else:
            raise ValueError(f"Unknown null_pattern string: {patterns!r}")
    elif isinstance(patterns, (slice, tuple)):
        mask[patterns] = True
    elif isinstance(patterns, list):
        if len(patterns) == 0:
            raise ValueError("null_pattern list was empty")
        for p in patterns:
            _apply_null_patterns(data, mask, p, nv, depth=depth + 1)
    else:
        raise ValueError(f"Unknown null_pattern type: {type(patterns)}")


DIRECTIONS = ("E", "W", "N", "S", "NE", "SE", "NW", "SW")


def make_raster(
    content="ones",
    mask=None,
    dtype=None,
    *,
    shape=(1, 6, 6),
    x=None,
    y=None,
    affine=None,
    chunksize=None,
    null=None,
    crs="EPSG:3857",
    xarray=False,
    scale=None,
    offset=None,
    null_pattern=None,
):
    """Build a small lazy ``Raster`` (or ``xarray.DataArray``) for tests.

    Parameters
    ----------
    content : str or list or np.ndarray, optional
        The data to fill the raster with. String presets
        (``"ones"``, ``"zeros"``, ``"arange"``, ``"peak"``, ``"hills"``,
        ``"grad-<DIR>"``) use ``shape`` and ``dtype``; array-like
        content is promoted to 3D (a leading band dim is added if
        needed) and its shape overrides ``shape``. ``"peak"``
        generates ``sin(pi*x)*sin(pi*y)`` over each band, producing
        a smooth surface that peaks at the center of the raster.
        ``"hills"`` does the same, but uses
        ``sin(3*pi*x)*sin(3*pi*y)``, yielding a 3x3 grid of peaks
        and valleys. ``"grad-<DIR>"`` produces a linear gradient
        ramping from 0 to 1, where ``<DIR>`` is one of
        ``E``/``W``/``N``/``S``/``NE``/``SE``/``NW``/``SW``
        (case-insensitive); the gradient increases toward the
        named compass direction (e.g. ``"grad-E"`` is smallest on
        the west edge and largest on the east edge). The default
        is ``"ones"``.
    mask : array-like, optional
        Boolean-like mask to apply to the raster. Non-zero entries are
        treated as masked. Promoted to 3D the same way as ``content``.
        Providing ``mask`` implicitly sets ``null=True`` if ``null`` is
        not given.
    dtype : dtype-like, optional
        Output dtype. For string ``content`` the array is allocated with
        this dtype. For array-like ``content`` the data is cast to this
        dtype if provided, otherwise the input dtype is kept. Defaults
        to ``int64`` internally when unspecified.
    shape : tuple, keyword-only, optional
        Raster shape. Either ``(Y, X)`` (a band dim is prepended) or
        ``(B, Y, X)``. Ignored when ``content`` is array-like. The
        default is ``(1, 6, 6)``.
    x : np.ndarray, keyword-only, optional
        X coordinate values. Must be provided together with ``y``. If
        neither is specified, ``affine`` is used to generate
        coordinates. Passed through to ``data_to_raster``.
    y : np.ndarray, keyword-only, optional
        Y coordinate values. Must be provided together with ``x``. If
        neither is specified, ``affine`` is used to generate
        coordinates. Passed through to ``data_to_raster``.
    affine : affine.Affine, keyword-only, optional
        Affine transform for the raster. If ``None``, the transform is
        derived from ``x`` and ``y`` or defaults to an identity-like
        matrix. Passed through to ``data_to_raster``.
    chunksize : tuple, keyword-only, optional
        Chunk sizes passed to ``dask.array.core.normalize_chunks``. The
        default is ``(1, "auto", "auto")`` so bands are always chunked
        one-at-a-time.
    null : scalar, list, tuple, or True, keyword-only, optional
        Controls the raster's null value.

        - A scalar sets the null value directly with no implicit mask.
        - A list or tuple of values builds a mask from positions where
          ``content`` equals any of them and uses the last value as the
          null value.
        - ``True`` (the default when ``mask`` or ``null_pattern`` is
          given) uses the dtype's default null value from
          ``raster_tools.masking.get_default_null_value``.
        - ``None`` (the default otherwise) leaves the raster with no
          null value.
    crs : CRS-like, keyword-only, optional
        CRS for the raster. Defaults to ``"EPSG:3857"``. Pass ``None``
        for no CRS.
    xarray : bool, keyword-only, optional
        If ``True``, return the underlying ``xarray.DataArray`` instead
        of a ``Raster``. The default is ``False``.
    scale : scalar, keyword-only, optional
        Value used to scale data values. Applies to both string presets and
        array-like ``content``. Defaults to no scaling.
    offset : scalar, keyword-only, optional
        Value added to the data after creation. Applies to both string
        ``content`` presets and array-like ``content``. The value is
        cast to the target ``dtype`` before addition. Defaults to no offset.
    null_pattern : str, tuple, slice, or list, keyword-only, optional
        Pattern-based mask generator used when ``mask`` is not provided.
        Accepted forms:

        - ``"%N"`` -- masks positions where ``data % N == 0``.
        - ``">X"`` -- masks positions where ``data > X``.
        - ``"<X"`` -- masks positions where ``data < X``.
        - ``np.s_[...]`` -- a NumPy index expression (returns a
          ``slice`` or ``tuple``). Sets the selected positions to
          masked.
        - A ``list`` of the above -- each pattern is applied in
          order.

    Returns
    -------
    raster : Raster or xarray.DataArray
        A lazy ``Raster``, or its ``xdata`` if ``xarray=True``.
    """
    if len(shape) == 2:
        shape = (1, *shape)
    elif len(shape) != 3:
        raise ValueError("Shape must be 2D or 3D")

    dtype_is_none = dtype is None
    dtype = np.dtype(dtype) if not dtype_is_none else np.dtype("int64")

    if chunksize is None:
        chunksize = (1, "auto", "auto")

    if scale is not None:
        if not is_scalar(scale):
            raise TypeError("scale must be None or a scalar")
        if is_bool(dtype):
            raise TypeError("scale is not compatible with bool dtype")
    if is_scalar(offset):
        offset = dtype.type(offset)
    elif offset is not None:
        raise TypeError("offset must be None or a scalar")

    content_is_grad = (
        isinstance(content, str)
        and content.startswith("grad-")
        and content[5:].upper() in DIRECTIONS
    )

    if isinstance(content, (list, np.ndarray)):
        data = _as_3d(content)
        if not dtype_is_none:
            data = data.astype(dtype)
        shape = data.shape
    elif content == "ones":
        data = np.ones(shape, dtype=dtype)
    elif content == "zeros":
        data = np.zeros(shape, dtype=dtype)
    elif content == "arange":
        data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    elif content in ("peak", "hills"):
        n = 1 if content == "peak" else 3
        _, ny, nx = shape
        gy = np.sin(n * np.pi * np.linspace(0, 1, ny))
        gx = np.sin(n * np.pi * np.linspace(0, 1, nx))
        band = gy[:, None] * gx[None, :]
        data = np.broadcast_to(band, shape).copy()
    elif content_is_grad:
        dir_ = content[5:].upper()
        _, ny, nx = shape
        xx, yy = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
        if dir_ == "E":
            band = xx
        elif dir_ == "W":
            band = 1 - xx
        elif dir_ == "S":
            band = yy
        elif dir_ == "N":
            band = 1 - yy
        elif dir_ == "SE":
            band = (xx + yy) / 2.0
        elif dir_ == "NW":
            band = 1 - (xx + yy) / 2.0
        elif dir_ == "NE":
            band = (xx + 1 - yy) / 2.0
        else:  # SW
            band = (1 - xx + yy) / 2.0
        data = np.broadcast_to(band, shape).copy()
    else:
        raise ValueError(f"Unknown content: {content}")

    target_dtype = data.dtype if dtype_is_none else dtype
    if scale is not None:
        data = (data * scale).astype(target_dtype)
    # Presets that build data in float space need the final cast.
    content_is_float_preset = content_is_grad or (
        isinstance(content, str) and content in ("peak", "hills")
    )
    if content_is_float_preset:
        data = data.astype(target_dtype)
    if offset is not None:
        data = data + offset

    if mask is not None:
        mask = _as_3d(mask) == 1
        if null is None:
            null = True

    if (mask is not None or null_pattern is not None) and null is None:
        null = True

    nv = None
    if null is not None:
        if is_scalar(null):
            nv = null
        elif isinstance(null, (list, tuple)):
            if mask is None:
                mask = np.zeros_like(data, dtype=bool)
            for v in null:
                mask |= rts.raster.get_mask_from_data(data, v)
            nv = null[-1]
        else:
            nv = rts.masking.get_default_null_value(dtype)

    if mask is None and null_pattern is not None:
        if nv is None:
            nv = rts.masking.get_default_null_value(dtype)
        mask = np.zeros_like(data, dtype=bool)
        _apply_null_patterns(data, mask, null_pattern, nv)
        data[mask] = nv

    chunks = _norm_chunks(chunksize, shape=shape, dtype=dtype)
    raster = rts.data_to_raster(
        data, mask=mask, x=x, y=y, affine=affine, crs=crs, nv=nv, burn=True
    ).chunk(chunks)
    if xarray:
        raster = raster.xdata
    return raster


def assert_coords_equal(ac, bc):
    assert isinstance(
        ac,
        (
            xr.core.coordinates.DataArrayCoordinates,
            xr.core.coordinates.DatasetCoordinates,
        ),
    ) and isinstance(
        bc,
        (
            xr.core.coordinates.DataArrayCoordinates,
            xr.core.coordinates.DatasetCoordinates,
        ),
    )
    assert list(ac.keys()) == list(bc.keys())
    for k in ac:
        assert np.allclose(ac[k], bc[k])


def assert_valid_raster(raster):
    assert isinstance(raster, Raster)
    assert len(raster.shape) == 3
    assert "raster" in raster._ds
    assert "mask" in raster._ds
    assert raster._ds.raster.shape == raster._ds.mask.shape
    # RIO encoding
    assert raster._ds.rio.x_dim == "x"
    assert raster._ds.rio.y_dim == "y"
    # Coords and dims
    # xr.Dataset seems to sort dims alphabetically (sometimes)
    assert set(raster._ds.dims) == {"band", "y", "x"}
    assert raster._ds.raster.dims == ("band", "y", "x")
    assert raster._ds.mask.dims == ("band", "y", "x")
    # Make sure that all dims have coords
    for d in raster._ds.raster.dims:
        assert d in raster._ds.coords
        assert d in raster._ds.raster.coords
        assert d in raster._ds.mask.coords
    # Make sure that coords are consistent
    assert_coords_equal(raster._ds.coords, raster._ds.raster.coords)
    assert_coords_equal(raster._ds.coords, raster._ds.mask.coords)
    if raster.shape[2] > 1:
        assert is_strictly_increasing(raster._ds.x)
    if raster.shape[1] > 1:
        assert is_strictly_decreasing(raster._ds.y)
    assert np.allclose(
        raster._ds.band.to_numpy(), np.arange(raster.shape[0]) + 1
    )
    assert raster._ds.band.to_numpy()[0] == 1
    # is lazy
    assert dask.is_dask_collection(raster._ds)
    assert dask.is_dask_collection(raster._ds.raster)
    assert dask.is_dask_collection(raster._ds.raster.data)
    assert dask.is_dask_collection(raster._ds.mask)
    assert dask.is_dask_collection(raster._ds.mask.data)
    assert raster._ds.raster.chunks == raster._ds.mask.chunks
    # size 1 chunks along band dim
    assert raster._ds.raster.data.chunksize[0] == 1
    assert raster._ds.mask.data.chunksize[0] == 1
    # Null value stuff
    assert is_bool(raster._ds.mask.dtype)
    nv = raster.null_value
    assert nv is None or is_scalar(nv) or is_bool(nv)
    # Materialize data and mask once for the remaining checks
    data, mask = dask.compute(raster._ds.raster.data, raster._ds.mask.data)
    if raster.null_value is not None:
        assert "_FillValue" in raster._ds.raster.attrs
        assert np.allclose(
            raster.null_value, raster._ds.raster.rio.nodata, equal_nan=True
        )
        assert np.allclose(
            raster._ds.raster.attrs["_FillValue"],
            raster._ds.raster.rio.nodata,
            equal_nan=True,
        )
        data_copy = data.copy()
        data_copy[mask] = raster.null_value
        assert np.allclose(data, data_copy, equal_nan=True)
    # Declared chunks should sum to the full shape on every axis
    assert raster.data.chunks == raster._ds.raster.data.chunks
    for axis_chunks, axis_size in zip(raster.data.chunks, raster.shape):
        assert sum(axis_chunks) == axis_size
    assert raster.shape == data.shape
    assert raster.shape == mask.shape
    # CRS
    assert isinstance(raster.crs, rio.crs.CRS) or raster.crs is None
    assert raster.crs == raster._ds.rio.crs
    assert raster.crs == raster._ds.raster.rio.crs
    assert raster.crs == raster._ds.mask.rio.crs


def assert_raster_dataset_data_equal_any_nv(left, right, dtype_match=True):
    if dtype_match:
        assert left.raster.dtype == right.raster.dtype
    assert np.allclose(left.mask.to_numpy(), right.mask.to_numpy())
    assert (right.raster.rio.nodata is not None) == (
        right.raster.rio.nodata is not None
    )
    ldata = left.raster.data.compute()
    rdata = right.raster.data.compute()
    lmask = left.mask.data.compute()
    rmask = right.mask.data.compute()
    assert np.allclose(ldata[~lmask], rdata[~rmask], equal_nan=True)


def assert_dataset_structure_equal(left, right):
    # Create skeletons with same structure but dummy data
    left_skel = left.map(lambda x: xr.zeros_like(x), keep_attrs=True)
    right_skel = right.map(lambda x: xr.zeros_like(x), keep_attrs=True)
    xr.testing.assert_equal(left_skel, right_skel)


def assert_raster_ds_equal(left, right, nv_match=True, dtype_match=True):
    if dtype_match:
        assert left.dtypes == right.dtypes
    if nv_match:
        assert left.equals(right)
        assert null_values_equal(
            left.raster.rio.nodata, right.raster.rio.nodata
        )
    else:
        assert_dataset_structure_equal(left, right)
        assert_raster_dataset_data_equal_any_nv(left, right)


def assert_rasters_equal(
    left, right, check_chunks=True, nv_match=True, dtype_match=True
):
    assert isinstance(left, Raster)
    assert isinstance(right, Raster)
    assert_valid_raster(left)
    assert_valid_raster(right)
    assert left.crs == right.crs
    assert left.affine == right.affine
    assert_raster_ds_equal(
        left._ds, right._ds, nv_match=nv_match, dtype_match=dtype_match
    )
    if check_chunks:
        assert left._ds.chunks == right._ds.chunks


def assert_rasters_similar(left, right, check_nbands=True, check_chunks=True):
    assert isinstance(left, Raster) and isinstance(right, Raster)
    if left is right:
        return

    assert set(left._ds.dims) == set(right._ds.dims)
    assert np.allclose(left.x, right.x)
    assert np.allclose(left.y, right.y)
    assert left.crs == right.crs
    assert left.affine == right.affine
    if check_nbands:
        assert left.nbands == right.nbands
    if check_chunks:
        assert left.data.chunks[1:] == right.data.chunks[1:]


def assert_dataarrays_similar(
    left, right, check_nbands=True, check_chunks=True
):
    assert isinstance(left, xr.DataArray) and isinstance(right, xr.DataArray)
    if left is right:
        return

    assert set(left.dims) == set(right.dims)
    assert np.allclose(left.x, right.x)
    assert np.allclose(left.y, right.y)
    assert left.rio.crs == right.rio.crs
    assert left.rio.transform(True) == right.rio.transform(True)
    if check_nbands:
        assert np.allclose(left.band, right.band)
    if check_chunks:
        assert left.data.chunks[1:] == right.data.chunks[1:]


def assert_datasets_similar(left, right, check_nbands=True, check_chunks=True):
    assert isinstance(left, xr.Dataset) and isinstance(right, xr.Dataset)
    if left is right:
        return

    assert sorted(left.data_vars) == sorted(right.data_Vars)
    for v in list(left.data_vars):
        assert_dataarrays_similar(
            left.data_vars[v],
            right.data_vars[v],
            check_nbands=check_nbands,
            check_chunks=check_chunks,
        )


def arange_nd(shape, dtype=None, mod=np):
    return mod.arange(np.prod(shape), dtype=dtype).reshape(shape)
