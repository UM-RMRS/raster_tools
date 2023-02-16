import numbers
import warnings
from collections import namedtuple

import dask
import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine
from numba import jit
from shapely.geometry import Point

from raster_tools.dask_utils import dask_nanmax, dask_nanmin
from raster_tools.dtypes import (
    DTYPE_INPUT_TO_DTYPE,
    is_bool,
    is_float,
    is_int,
    is_scalar,
    is_str,
    promote_dtype_to_float,
    should_promote_to_fit,
)
from raster_tools.masking import (
    create_null_mask,
    get_default_null_value,
    reconcile_nullvalue_with_dtype,
)
from raster_tools.utils import can_broadcast, make_raster_ds, merge_masks

from .io import (
    IO_UNDERSTOOD_TYPES,
    RasterDataError,
    RasterIOError,
    chunk,
    is_batch_file,
    normalize_xarray_data,
    open_raster_from_path,
    write_raster,
)


class RasterDeviceMismatchError(BaseException):
    pass


class RasterDeviceError(BaseException):
    pass


class RasterNoDataError(BaseException):
    pass


_REDUCTION_FUNCS = (
    "all",
    "any",
)
_NAN_REDUCTION_FUNCS = (
    "max",
    "mean",
    "min",
    "prod",
    "std",
    "sum",
    "var",
)


_REDUCTION_DOCSTRING = """\
    Reduce the raster to a single dask value by applying `{}` across all bands.

    All arguments are ignored.
"""


def _inject_reductions(cls):
    pass
    funcs = [(name, getattr(np, name)) for name in _REDUCTION_FUNCS]
    funcs += [
        (name, getattr(np, "nan" + name)) for name in _NAN_REDUCTION_FUNCS
    ]
    for name, f in funcs:
        func = cls._build_reduce_method(f)
        func.__name__ = name
        func.__doc__ = _REDUCTION_DOCSTRING.format(name)
        setattr(cls, name, func)


_MIN_MAX_FUNC_MAP = {
    np.max: dask_nanmax,
    np.nanmax: dask_nanmax,
    np.min: dask_nanmin,
    np.nanmin: dask_nanmin,
}


class _ReductionsMixin:
    """
    This mixin class adds reduction methods like `all`, `sum`, etc to the
    Raster class. Having these methods also allows numpy reduction functions to
    use them for dynamic dispatch. So `np.sum(raster)` will call the class'
    `sum` method. All reductions return dask results.
    """

    __slots__ = ()

    def __init_subclass__(cls, **kwargs):
        _inject_reductions(cls)
        super().__init_subclass__(**kwargs)

    @classmethod
    def _build_reduce_method(cls, func):
        if func in _MIN_MAX_FUNC_MAP:
            func = _MIN_MAX_FUNC_MAP[func]

        def method(self, *args, **kwargs):
            # Take args and kwargs to stay compatible with numpy
            data = self.data
            values = data[~self._ds.mask.data]
            return func(values)

        return method


def _normalize_ufunc_other(other, this):
    if other is None or isinstance(other, numbers.Number):
        return other

    if isinstance(other, xr.Dataset):
        return Raster(other)
    elif isinstance(other, (list, tuple, range, np.ndarray, da.Array)):
        # allow only if broadcastable
        if isinstance(other, da.Array) and np.isnan(other.size):
            raise ValueError("Dask arrays must have known chunks")
        other = np.atleast_1d(other)
        if not can_broadcast(this.shape, other.shape):
            msg = (
                "Raster could not broadcast together with array of shape"
                f" {other.shape}."
            )
            if other.squeeze().ndim == 1 and other.size == this.shape[0]:
                msg += " Did you mean to use Raster.bandwise?"
            raise ValueError(msg)
        return other
    elif isinstance(other, xr.DataArray):
        return Raster(other)
    else:
        assert isinstance(other, Raster)
    return other


def _apply_ufunc(ufunc, this, left, right=None, kwargs=None, out=None):
    args = [left]
    if right is not None:
        args.append(right)
    other = left if left is not this else right
    types = [getattr(a, "dtype", type(a)) for a in args]
    masked = any(getattr(r, "_masked", False) for r in args)
    ufunc_args = [getattr(a, "xdata", a) for a in args]
    kwargs = kwargs or {}
    out_crs = None
    if this.crs is not None:
        out_crs = this.crs
    elif isinstance(other, Raster) and other.crs is not None:
        out_crs = other.crs

    ufname = ufunc.__name__
    if ufname.startswith("bitwise") and any(is_float(t) for t in types):
        raise TypeError(
            "Bitwise operations are not compatible with float dtypes. You may"
            " want to convert one or more inputs to a boolean or integer dtype"
            " (e.g. 'raster > 0' or 'raster.astype(int)')."
        )
    if ufname.endswith("shift") and any(is_float(t) for t in types):
        raise TypeError(
            "right_shift and left_shift operations are not compatible with"
            " float dtypes. You may want to convert one or more inputs to an"
            " integer dtype (e.g. 'raster.astype(int)')."
        )

    xr_out = ufunc(*ufunc_args, **kwargs)
    multi_out = ufunc.nout > 1
    if not masked:
        mask = create_null_mask(xr_out, None)
        if not multi_out:
            xmask = xr.DataArray(mask, dims=xr_out.dims, coords=xr_out.coords)
            xr_out = xr_out.rio.write_nodata(None)
            ds_out = make_raster_ds(xr_out, xmask)
        else:
            xmask = xr.DataArray(
                mask, dims=xr_out[0].dims, coords=xr_out[0].coords
            )
            xr_out = [x.rio.write_nodata(None) for x in xr_out]
            ds_out = [make_raster_ds(x, xmask) for x in xr_out]
    else:
        xmask = merge_masks([r.xmask for r in args if isinstance(r, Raster)])
        if not multi_out:
            nv = get_default_null_value(xr_out.dtype)
            xr_out = xr.where(xmask, nv, xr_out).rio.write_nodata(nv)
            ds_out = make_raster_ds(xr_out, xmask)
        else:
            nvs = [get_default_null_value(x.dtype) for x in xr_out]
            xr_out = [
                xr.where(xmask, nv, x).rio.write_nodata(nv)
                for x, nv in zip(xr_out, nvs)
            ]
            ds_out = [make_raster_ds(x, xmask) for x in xr_out]
    if out_crs is not None:
        if multi_out:
            ds_out = [x.rio.write_crs(out_crs) for x in ds_out]
        else:
            ds_out = ds_out.rio.write_crs(out_crs)

    if out is not None:
        # "Inplace"
        out._ds = ds_out
        return out

    if not multi_out:
        return Raster(ds_out, _fast_path=True)

    rs_outs = tuple(Raster(x, _fast_path=True) for x in ds_out)
    return rs_outs


_UNARY_UFUNCS = frozenset(
    (np.absolute, np.invert, np.logical_not, np.negative, np.positive)
)
_UNSUPPORED_UFUNCS = frozenset((np.isnat, np.matmul))


class _RasterBase(np.lib.mixins.NDArrayOperatorsMixin, _ReductionsMixin):
    """This class implements methods for handling numpy ufuncs."""

    __slots__ = ()

    _HANDLED_TYPES = (
        numbers.Number,
        xr.DataArray,
        np.ndarray,
        da.Array,
        list,
        tuple,
        range,
    )
    # Higher than xarray objects
    __array_priority__ = 70

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.pop("out", ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (_RasterBase,)):
                return NotImplemented

        if ufunc in _UNSUPPORED_UFUNCS:
            raise TypeError(
                f"Raster objects are not supported for ufunc: '{ufunc}'."
            )

        if ufunc.signature is not None:
            raise NotImplementedError("Raster does not support gufuncs")

        if method != "__call__":
            raise NotImplementedError(
                f"{method} for ufunc {ufunc} is not implemented on Raster "
                "objects."
            )

        if len(inputs) > ufunc.nin:
            raise TypeError(
                "Too many inputs for ufunc:"
                f" inputs={len(inputs)}, ufunc={ufunc!r}"
            )

        if len(out):
            if len(out) > 1:
                raise NotImplementedError(
                    "The 'out' keyword is only supported for inplace "
                    "operations, not operations with multiple outputs."
                )
            (out,) = out
            if out is not self:
                raise NotImplementedError(
                    "'out' must be the raster being operated on."
                )
        else:
            out = None

        left = inputs[0]
        right = inputs[1] if ufunc.nin == 2 else None
        other = left if left is not self else right
        other = _normalize_ufunc_other(other, self)
        if self is left:
            right = other
        else:
            left = other
        return _apply_ufunc(ufunc, self, left, right=right, out=out)

    def __array__(self, dtype=None):
        return self._ds.raster.__array__(dtype)


def _normalize_bandwise_other(other, target_shape):
    other = np.atleast_1d(other)
    if isinstance(other, da.Array) and np.isnan(other.size):
        raise ValueError("Dask arrays must have known chunks")
    if other.size == 1:
        return other.ravel()
    if can_broadcast(other.shape, target_shape):
        raise ValueError(
            "Operands can be broadcast together. A bandwise operation is not"
            " needed here."
        )
    if other.ndim == 1 and other.size == target_shape[0]:
        # Map a list/array of values that has same length as # of bands to
        # each band
        return other.reshape((-1, 1, 1))
    raise ValueError(
        "Received array with incompatible shape in bandwise operation: "
        f"{other.shape}"
    )


class BandwiseOperationAdapter(np.lib.mixins.NDArrayOperatorsMixin):
    __slots__ = ("_raster",)

    _HANDLED_TYPES = (
        numbers.Number,
        np.ndarray,
        da.Array,
        list,
        tuple,
        range,
    )
    # Higher than xarray objects
    __array_priority__ = 70

    def __init__(self, raster):
        self._raster = raster

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.pop("out", None)
        for x in inputs:
            if not isinstance(
                x,
                self._HANDLED_TYPES
                + (
                    Raster,
                    BandwiseOperationAdapter,
                ),
            ):
                return NotImplemented

        if len(inputs) < 2:
            raise ValueError(
                "Bandwise operations are binary and require 2 inputs. Only"
                " one given."
            )
        if len(inputs) > 2:
            raise ValueError(
                "Bandwise operations are binary and require 2 inputs."
                f" {len(inputs)} given."
            )

        if ufunc in _UNSUPPORED_UFUNCS:
            raise TypeError(
                "The given ufunc is not supported for bandwise operations:"
                f" {ufunc!r}."
            )

        if ufunc.signature is not None:
            raise NotImplementedError("Raster does not support gufuncs")

        if method != "__call__":
            raise NotImplementedError(
                f"{method} for ufunc {ufunc} is not implemented for bandwise "
                " operations."
            )

        if len(inputs) > ufunc.nin:
            raise TypeError(
                "Too many inputs for ufunc:"
                f" inputs={len(inputs)}, ufunc={ufunc!r}"
            )

        if out is not None:
            raise NotImplementedError(
                "The 'out' keyword is not supported for bandwise operations"
            )

        left, right = inputs
        other = left if left is not self else right
        other = _normalize_bandwise_other(other, self._raster.shape)
        if self is left:
            left = self._raster
            right = other
        else:
            left = other
            right = self._raster

        return ufunc(left, right, **kwargs)


def _array_to_xarray(ar):
    if len(ar.shape) > 3 or len(ar.shape) < 2:
        raise ValueError(f"Invalid raster shape for numpy array: {ar.shape}")
    if len(ar.shape) == 2:
        # Add band dim
        ar = ar[None]
    if not isinstance(ar, da.Array):
        has_nan = is_float(ar.dtype) and np.isnan(ar).any()
    else:
        has_nan = is_float(ar.dtype)
    # The band dimension needs to start at 1 to match raster conventions
    coords = [np.arange(1, ar.shape[0] + 1)]
    # Add 0.5 offset to move coords to center of cell
    # y
    coords.append(np.arange(ar.shape[1])[::-1] + 0.5)
    coords.append(np.arange(ar.shape[2]) + 0.5)
    xrs = xr.DataArray(ar, dims=["band", "y", "x"], coords=coords)
    if has_nan:
        xrs.attrs["_FillValue"] = np.nan
    else:
        # No way to know null value
        xrs.attrs["_FillValue"] = None
    return xrs


def _try_to_get_null_value_xarray(xrs):
    if (
        "_FillValue" in xrs.attrs
        and xrs.attrs["_FillValue"] is None
        and xrs.rio.nodata is None
        and xrs.rio.encoded_nodata is None
        and not dask.is_dask_collection(xrs)
    ):
        return None
    null = xrs.attrs.get("_FillValue", None)
    if null is not None:
        return null
    nv1 = xrs.rio.nodata
    nv2 = xrs.rio.encoded_nodata
    # All are None, return nan for float and None otherwise
    if all(nv is None for nv in [nv1, nv2]):
        if is_float(xrs.dtype):
            if dask.is_dask_collection(xrs) or np.isnan(xrs.data).any():
                return np.nan
            else:
                return None
        else:
            return None
    # All are not None, return nv1
    if all(nv is not None for nv in [nv1, nv2]):
        return nv1
    # One is None and the other is not, return the valid null value
    return nv1 if nv1 is not None else nv2


def _dataarry_to_raster_ds(xin):
    nv = _try_to_get_null_value_xarray(xin)
    xin = normalize_xarray_data(xin)
    if not dask.is_dask_collection(xin) or xin.data.chunksize[0] != 1:
        xin = chunk(xin)

    if nv is None:
        mask = create_null_mask(xin, nv)
        mask = xr.DataArray(
            create_null_mask(xin, nv), dims=xin.dims, coords=xin.coords
        )
        if xin.rio.crs is not None:
            mask = mask.rio.write_crs(xin.rio.crs)
    elif np.isnan(nv):
        mask = np.isnan(xin)
    else:
        mask = xin == nv

    orig_nv = nv
    nv = reconcile_nullvalue_with_dtype(nv, xin.dtype)
    crs = xin.rio.crs
    if nv is not None and (np.isnan(orig_nv) or orig_nv != nv):
        xin = xr.where(mask, nv, xin)
        if crs is not None:
            xin = xin.rio.write_crs(crs)
    xin = xin.rio.write_nodata(nv)
    return make_raster_ds(xin, mask)


def _dataset_to_raster_ds(xin):
    if (
        dask.is_dask_collection(xin)
        and set(xin.data_vars) == set(["raster", "mask"])
        and dask.is_dask_collection(xin.raster)
        and dask.is_dask_collection(xin.mask)
        and is_bool(xin.mask.dtype)
        and xin.raster.dims == ("band", "y", "x")
        and xin.mask.dims == ("band", "y", "x")
        and xin.raster.chunks == xin.mask.chunks
        and xin.raster.rio.crs == xin.mask.rio.crs
    ):
        return xin

    nvars = len(xin.data_vars)
    if nvars < 2 or nvars > 2 or set(xin.data_vars) != set(["raster", "mask"]):
        raise ValueError(
            "Dataset input must have only 'raster' and 'mask' variables"
        )

    raster = xin.raster
    mask = xin.mask
    if not is_bool(mask.dtype):
        raise TypeError("mask must have boolean dtype")

    raster = normalize_xarray_data(raster)
    if not dask.is_dask_collection(raster):
        raster = chunk(raster)
    mask = normalize_xarray_data(mask)
    if mask.shape != raster.shape:
        raise ValueError("raster and mask dimensions do not match")
    if not dask.is_dask_collection(mask):
        mask = mask.chunk(chunks=raster.chunks)

    nv = _try_to_get_null_value_xarray(raster)
    if nv is None:
        nv = get_default_null_value(raster.dtype)
    else:
        nv = reconcile_nullvalue_with_dtype(nv, raster.dtype)
    if nv is not None:
        raster.data = xr.where(mask, nv, raster.data)
    raster.rio.write_nodata(nv, inplace=True)
    crs = raster.rio.crs
    if crs != mask.rio.crs:
        mask.rio.write_crs(crs, inplace=True)
    return make_raster_ds(raster, mask)


def _xarray_to_raster_ds(xin):
    if isinstance(xin, xr.DataArray):
        return _dataarry_to_raster_ds(xin)
    return _dataset_to_raster_ds(xin)


def get_raster_ds(raster):
    # Copy to take ownership
    if isinstance(raster, Raster):
        ds = raster._ds.copy(deep=True)
    elif isinstance(raster, (xr.DataArray, xr.Dataset)):
        ds = _xarray_to_raster_ds(raster.copy())
    elif isinstance(raster, (np.ndarray, da.Array)):
        ds = _xarray_to_raster_ds(_array_to_xarray(raster.copy()))
    elif type(raster) in IO_UNDERSTOOD_TYPES:
        if is_batch_file(raster):
            # Import here to avoid circular import errors
            from raster_tools.batch import parse_batch_script

            ds = parse_batch_script(raster).final_raster._ds
        else:
            rs, mask, nv = open_raster_from_path(raster)
            xmask = xr.DataArray(mask, dims=rs.dims, coords=rs.coords)
            ds = make_raster_ds(rs.rio.write_nodata(nv), xmask)
            ds = _xarray_to_raster_ds(ds)
    else:
        raise TypeError(f"Could not resolve input to a raster: {raster!r}")
    return ds


RasterQuadrantsResult = namedtuple(
    "RasterQuadrantsResult", ("nw", "ne", "sw", "se")
)


class Raster(_RasterBase):
    """Abstraction of georeferenced raster data with lazy function evaluation.

    Raster is a wrapper around xarray Datasets and DataArrays. It takes
    advantage of xarray's dask integration to allow lazy loading and
    evaluation. It allows a pipeline of operations on underlying raster
    sources to be built in a lazy fashion and then evaluated effiently.
    Most mathematical operations have been overloaded so operations such as
    ``z = x - y`` and ``r = x**2`` are possible.

    All operations on a Raster return a new Raster.

    Parameters
    ----------
    raster : str, Raster, xarray.DataArray, xarray.Dataset, numpy.ndarray
        The raster source to use for this Raster. If `raster` is a string,
        it is treated like a path. If `raster` is a Raster, a copy is made
        and its raster source is used. If `raster` is an xarray DataArray,
        Dataset, or numpy array, it is used as the source. Dataset objects must
        contain 'raster' and 'mask' variables. The dimensions of these
        variables are assumed to be "band", "y", "x", in that order. The 'mask'
        variable must have boolean dtype.

    """

    __slots__ = ("_ds",)

    def __init__(self, raster, _fast_path=False):
        if _fast_path:
            self._ds = raster
        else:
            self._ds = get_raster_ds(raster)

    def __repr__(self):
        # TODO: implement
        return repr(self._ds.raster)

    @property
    def null_value(self):
        """The raster's null value used to fill missing or invalid entries."""
        return self._ds.raster.rio.nodata

    @property
    def _masked(self):
        return self.null_value is not None

    @property
    def xdata(self):
        """The underlying :class:`xarray.DataArray` (read only)"""
        return self._ds.raster

    @property
    def data(self):
        """The underlying dask array of data"""
        return self._ds.raster.data

    @property
    def values(self):
        """
        The raw internal raster as a numpy array.

        .. note::
           This triggers computation and loads the raster data into memory.
        """
        return self._ds.raster.values

    @property
    def mask(self):
        """The null value mask as a dask array."""
        return self._ds.mask.data

    @property
    def xmask(self):
        """The null value mask as an xarray DataArray."""
        return self._ds.mask

    @property
    def band(self):
        """The band coordinate values."""
        return self._ds.band.data

    @property
    def x(self):
        """The x (horizontal) coordinate values."""
        return self._ds.x.data

    @property
    def y(self):
        """The y (vertical) coordinate values."""
        return self._ds.y.data

    @property
    def dtype(self):
        """The dtype of the data."""
        return self._ds.raster.dtype

    @property
    def shape(self):
        """The shape of the underlying raster. Dim 0 is the band dimension.

        The shape will always be of the form ``(B, Y, X)`` where ``B`` is the
        band dimension, ``Y`` is the y dimension, and ``X`` is the x dimension.

        """
        return self._ds.raster.shape

    @property
    def nbands(self):
        """The number of bands."""
        return self._ds.band.size

    @property
    def crs(self):
        """The raster's CRS.

        This is a :obj:`rasterio.crs.CRS` object."""
        return self._ds.rio.crs

    @property
    def affine(self):
        """The affine transformation for the raster data.

        This is an :obj:`affine.Affine` object.

        """
        return self._ds.rio.transform(True)

    @property
    def resolution(self):
        """The x and y cell sizes as a tuple. Values are always positive."""
        return self._ds.rio.resolution(True)

    @property
    def bounds(self):
        """Bounds tuple (minx, miny, maxx, maxy)"""
        r, c = self.shape[1:]
        minx, maxy = self.xy(0, 0, offset="ul")
        maxx, miny = self.xy(r - 1, c - 1, "lr")
        return (minx, miny, maxx, maxy)

    @property
    def bandwise(self):
        """Returns an adapter for band-wise operations on this raster

        The returned adapter allows lists/arrays of values to be mapped to the
        raster's bands for binary operations.

        """
        return BandwiseOperationAdapter(self)

    @property
    def _rs(self):
        warnings.warn(
            "'Raster._rs' is deprecated. It will soon be removed. Use xdata.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._ds.raster

    @property
    def _data(self):
        warnings.warn(
            "'Raster._data' is deprecated. It will soon be removed. Use data.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._ds.raster.data

    @property
    def _values(self):
        warnings.warn(
            "'Raster._values' is deprecated. It will soon be removed. Use "
            "values.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._ds.raster.values

    @property
    def _mask(self):
        warnings.warn(
            "'Raster._mask' is deprecated. It will soon be removed. Use mask.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._ds.mask.data

    @property
    def _null_value(self):
        warnings.warn(
            "'Raster._null_value' is deprecated. It will soon be removed. Use "
            "null_value.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.null_value

    @property
    def xrs(self):
        warnings.warn(
            "'Raster.xrs' is deprecated. It will soon be removed. Use xdata.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._ds.raster

    @property
    def pxrs(self):
        warnings.warn(
            "'Raster.pxrs' is deprecated. It will soon be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        return get_raster(self, null_to_nan=True)._ds.raster

    def chunk(self, chunks):
        """Rechunk the underlying raster

        Parameters
        ----------
        chunks : tuple of int
            The chunk sizes for each dimension ("band", "y", "x", in that
            order). Must be a 3-tuple.

        Returns
        -------
        Raster
            The rechunked raster

        """
        assert len(chunks) == 3
        return Raster(
            self._ds.chunk(
                {d: cs for d, cs in zip(["band", "y", "x"], chunks)}
            ),
            _fast_path=True,
        )

    def to_dataset(self):
        """Returns the underlying `xarray.Dataset`."""
        return self._ds

    def plot(self, *args, **kwargs):
        """
        Plot the raster. Args and kwargs are passed to xarray's plot function.
        """
        return get_raster(self, null_to_nan=True).xdata.plot(*args, **kwargs)

    def get_chunked_coords(self):
        """Get lazy coordinate arrays, in x-y order, chunked to match data."""
        xc = da.from_array(self.x, chunks=self.data.chunks[2]).reshape(
            (1, 1, -1)
        )
        yc = da.from_array(self.y, chunks=self.data.chunks[1]).reshape(
            (1, -1, 1)
        )
        return xc, yc

    def close(self):
        """Close the underlying source"""
        self._ds.close()

    def save(self, path, no_data_value=None, **gdal_kwargs):
        """Compute the final raster and save it to the provided location.

        Parameters
        ----------
        path : str
            The target location to save the raster to.
        no_data_value : scalar, optional
            A new null value to use when saving.
        **gdal_kwargs : kwargs, optional
            Additional keyword arguments to to pass to rasterio and GDAL when
            writing the raster data.

        Returns
        -------
        Raster
            A raster pointing to the saved location.

        """
        # TODO: warn of overwrite
        rs = self
        if no_data_value is not None:
            rs = self.set_null_value(no_data_value)
        if rs._masked and is_bool(rs.dtype):
            # Cast to uint and burn in mask
            rs = rs.astype("uint8").set_null_value(
                get_default_null_value("uint8")
            )
        xrs = rs.xdata
        write_raster(
            xrs,
            path,
            rs.null_value if no_data_value is None else no_data_value,
            **gdal_kwargs,
        )
        return Raster(path)

    def eval(self):
        """Compute any applied operations and return the result as new Raster.

        Note that the unerlying sources will be loaded into memory for the
        computations and the result will be fixed in memory. The original
        Raster will be unaltered.

        """
        ds = chunk(self._ds.compute())
        # A new raster is returned to mirror the xarray and dask APIs
        return Raster(ds)

    def copy(self):
        """Returns a copy of this Raster."""
        return Raster(self)

    def astype(self, dtype, warn_about_null_change=True):
        """Return a copy of the Raster cast to the specified type.

        Parameters
        ----------
        dtype : str, type, numpy.dtype
            The new type to cast the raster to. The null value will also be
            cast to this type. If the null value cannot be safely cast to the
            new type, a default null value for the given `dtype` is used. This
            will produce a warning unless `warn_about_null_change` is set to
            ``False``.
        warn_about_null_change : bool, optional
            Can be used to silence warnings. The default is to always warn
            about null value changes.

        Returns
        -------
        Raster
            The new `dtype` raster.

        """
        if dtype not in DTYPE_INPUT_TO_DTYPE:
            raise ValueError(f"Unsupported type: '{dtype}'")
        dtype = DTYPE_INPUT_TO_DTYPE[dtype]

        if dtype == self.dtype:
            return self.copy()

        xrs = self._ds.raster
        nv = self.null_value
        mask = self._ds.mask

        xrs = xrs.astype(dtype)
        if self._masked:
            nv = reconcile_nullvalue_with_dtype(
                nv, dtype, warn_about_null_change
            )
            if nv != self.null_value:
                xrs = xr.where(mask, nv, xrs).rio.write_nodata(nv)
        ds = make_raster_ds(xrs, mask)
        if self.crs is not None:
            ds = ds.rio.write_crs(self.crs)
        return Raster(ds, _fast_path=True)

    def get_bands(self, bands):
        """Retrieve the specified bands as a new Raster. Indexing starts at 1.

        Parameters
        ----------
        bands : int or sequence of ints
            The band or bands to be retrieved. A single band Raster is returned
            if `bands` is an int and a multiband Raster is returned if it is a
            sequence. The band numbers may be out of order and contain repeated
            elements. if `bands` is a sequence, the multiband raster's bands
            will be in the order provided.

        Returns
        -------
        Raster
            The resulting raster composed of the specified bands in the order
            given.

        """
        if is_int(bands):
            bands = [bands]
        if not all(is_int(b) for b in bands):
            raise TypeError("All band numbers must be integers")
        bands = list(bands)
        if len(bands) == 0:
            raise ValueError("No bands provided")
        if any(b < 1 or b > self.nbands for b in bands):
            raise IndexError(
                f"One or more band numbers were out of bounds: {bands}"
            )

        if len(bands) == 1 and self.nbands == 1:
            return self.copy()
        # We have to do sel() followed by concat() in order to get size 1
        # chunking along the band dim
        dss = [self._ds.sel(band=b) for b in bands]
        ds = xr.concat(dss, dim="band")
        # Reset band numbering
        ds["band"] = np.arange(len(bands)) + 1
        return Raster(ds, _fast_path=True)

    def set_crs(self, crs):
        """Set the CRS for the underlying data.

        Parameters
        ----------
        crs : object
            The desired CRS. This can be anything accepted by
            :obj:`rasterio.crs.CRS.from_user_input` (i.e. `4326`,
            `"epsg:4326"`, etc).

        Returns
        -------
        Raster
            A new raster with the specified CRS.

        """
        return Raster(self._ds.rio.write_crs(crs), _fast_path=True)

    def set_null_value(self, value):
        """Sets or replaces the null value for the raster.

        If there was previously no null value for the raster, one is set. If
        there was already a null value, then it is replaced. If the raster has
        an integer dtype and `value` is a float, the dtype will be promoted to
        a float dtype. If `value` is None, the null value is cleared. The
        raster data is not changed in this case.

        Parameters
        ----------
        value : scalar or None
            The new null value for the raster. If None, the resulting raster
            will have no null value, i.e. the null value is cleared.

        Returns
        -------
        Raster
            The new resulting Raster.

        """
        if value is not None and not (is_scalar(value) or is_bool(value)):
            raise TypeError(f"Value must be a scalar or None: {value}")

        xrs = self._ds.raster.copy()
        # Cast up to float if needed
        if should_promote_to_fit(self.dtype, value):
            xrs = xrs.astype(promote_dtype_to_float(self.dtype))

        if value is None:
            # Burn in current mask values and then clear null value
            # TODO: burning should not be needed as the values should already
            # be set
            xrs = self.burn_mask().xdata.rio.write_nodata(None)
            mask = xr.zeros_like(xrs, dtype=bool)
            return Raster(make_raster_ds(xrs, mask), _fast_path=True)

        # Update mask
        mask = self._ds.mask
        temp_mask = np.isnan(xrs) if np.isnan(value) else xrs == value
        if self._masked:
            mask = mask | temp_mask
        else:
            mask = temp_mask
        xrs = xrs.rio.write_nodata(value)
        return Raster(make_raster_ds(xrs, mask), _fast_path=True).burn_mask()

    def burn_mask(self):
        """Fill null-masked cells with null value.

        Use this as a way to return the underlying data to a known state. If
        dtype is boolean, the null cells are set to ``True`` instead of
        promoting to fit the null value.
        """
        if not self._masked:
            return self
        nv = self.null_value
        if is_bool(self.dtype):
            nv = get_default_null_value(self.dtype)
        # call write_nodata because xr.where drops the nodata info
        xrs = xr.where(self._ds.mask, nv, self._ds.raster).rio.write_nodata(nv)
        if self.crs is not None:
            xrs = xrs.rio.write_crs(self.crs)
        return Raster(make_raster_ds(xrs, self._ds.mask), _fast_path=True)

    def to_null_mask(self):
        """
        Returns a boolean Raster with True at null values and False otherwise.

        Returns
        -------
        Raster
            The resulting mask Raster. It is True where this raster contains
            null values and False everywhere else.

        """
        return Raster(
            make_raster_ds(
                self._ds.mask.rio.write_nodata(None),
                xr.zeros_like(self._ds.mask),
            ),
            _fast_path=True,
        )

    def replace_null(self, value):
        """Replaces null values with `value`.

        Parameters
        ----------
        value : scalar
            The new value to replace null values with.

        Returns
        -------
        Raster
            The new resulting Raster. If `value` is a float and the raster
            dtype is int, the raster type will be promoted to float.

        """
        if not is_scalar(value):
            raise TypeError("value must be a scalar")

        return self.set_null_value(value)

    def where(self, condition, other):
        """Filter elements from this raster according to `condition`.

        Parameters
        ----------
        condition : str or Raster
            A boolean or int raster that indicates where elements in this
            raster should be preserved and where `other` should be used. If
            the condition is an int raster, it is coerced to bool using
            `condition > 0`.  ``True`` cells pull values from this raster and
            ``False`` cells pull from `other`. *str* is treated as a path to a
            raster.
        other : scalar, str or Raster
            A raster or value to use in locations where `condition` is
            ``False``. *str* is treated as a path to a raster.

        Returns
        -------
        Raster
            The resulting filtered Raster.

        """
        from raster_tools.general import where

        return where(condition, self, other)

    def remap_range(self, mapping, inclusivity="left"):
        """Remaps values based on a mapping or list of mappings.

        Mappings are applied all at once with earlier mappings taking
        precedence.

        Parameters
        ----------
        mapping : 3-tuple of scalars or list of 3-tuples of scalars
            A tuple or list of tuples containing ``(min, max, new_value)``
            scalars. The mappiing(s) map values between the min and max to the
            ``new_value``. If `mapping` is a list and there are mappings that
            conflict or overlap, earlier mappings take precedence.
            `inclusivity` determines which sides of the range are inclusive and
            exclusive.
        inclusivity : str, optional
            Determines whether to be inclusive or exclusive on either end of
            the range. Default is `'left'`.

            'left' [min, max)
                Left (min) side is inclusive and right (max) side is exclusive.
            'right' (min, max]
                Left (min) side is exclusive and right (max) side is inclusive.
            'both' [min, max]
                Both sides are inclusive.
            'none' (min, max)
                Both sides are exclusive.

        Returns
        -------
        Raster
            The resulting Raster.

        See Also
        --------
        raster_tools.general.remap_range

        """
        # local import to avoid circular import
        from raster_tools.general import remap_range

        return remap_range(self, mapping, inclusivity=inclusivity)

    def reclassify(self, remapping, unmapped_to_null=False):
        """Reclassify raster values based on a mapping.

        The raster must have an integer type.

        Parameters
        ----------
        remapping : str, dict
            Can be either a ``dict`` or a path string. If a ``dict`` is
            provided, the keys will be reclassified to the corresponding
            values. If a path string, it is treated as an ASCII remap file
            where each line looks like ``a:b`` and ``a`` and ``b`` are
            integers. All remap values (both from and to) must be integers.
        unmapped_to_null : bool, optional
            If ``True``, values not included in the mapping are instead mapped
            to the null value. Default is ``False``.

        Returns
        -------
        Raster
            The remapped raster.

        """
        from raster_tools.general import reclassify

        return reclassify(self, remapping, unmapped_to_null=unmapped_to_null)

    def round(self, decimals=0):
        """Evenly round to the given number of decimals

        Parameters
        ----------
        decimals : int, optional
            The number of decimal places to round to. If negative, value
            specifies the number of positions to the left of the decimal point.
            Default is 0.

        Returns
        -------
        Raster
            Rounded raster.

        """
        rast = self._ds.raster
        if self._masked:
            rast = xr.where(
                self._ds.mask, self.null_value, rast.round(decimals=decimals)
            ).rio.write_nodata(self.null_value)
        else:
            rast = rast.round(decimals=decimals)
        ds = make_raster_ds(rast, self._ds.mask)
        if self.crs is not None:
            ds = ds.rio.write_crs(self.crs)
        return Raster(ds, _fast_path=True)

    def xy(self, row, col, offset="center"):
        """Return the `(x, y)` coordinates of the pixel at `(row, col)`.

        Parameters
        ----------
        row : int, float, array-like
            row value(s) in the raster's CRS
        col : int, float, array-like
            row value(s) in the raster's CRS
        offset : str, optional
            Determines if the returned coordinates are for the center or a
            corner. Default is `'center'`

            'center'
                The pixel center.
            'ul'
                The upper left corner.
            'ur'
                The upper right corner.
            'll'
                The lower left corner.
            'lr'
                The lower right corner.

        Returns
        -------
        tuple
            (x value(s), y value(s)). If the inputs where array-like, the
            output values will be arrays.

        """
        return rowcol_to_xy(row, col, self.affine, offset)

    def index(self, x, y):
        """Return the `(row, col)` index of the pixel at `(x, y)`.

        Parameters
        ----------
        x : float, array-like
            x value(s) in the raster's CRS
        y : float, array-like
            y value(s) in the raster's CRS

        Returns
        -------
        tuple
            (row value(s), col value(s)). If the inputs where array-like, the
            output values will be arrays.

        """
        return xy_to_rowcol(x, y, self.affine)

    def to_vector(self):
        """
        Convert the raster into a vector where each non-null cell is a point.

        The resulting points are located at the center of each grid cell.

        Returns
        -------
        result
            The result is a dask_geopandas GeoDataFrame with the following
            columns: value, row, col, geometry. The value is the cell value
            at the row and column in the raster.

        """
        xrs = self._ds.raster
        mask = self._ds.mask.data
        data_delayed = xrs.data.to_delayed()
        chunks_shape = data_delayed.shape
        data_delayed = data_delayed.ravel()
        mask_delayed = mask.to_delayed().ravel()

        xc = da.from_array(xrs.x.data, chunks=xrs.chunks[2])
        xc_delayed = xc.to_delayed()
        yc = da.from_array(xrs.y.data, chunks=xrs.chunks[1])
        yc_delayed = yc.to_delayed()
        # See issue for inspiration: https://github.com/dask/dask/issues/7589
        # Group chunk data with corresponding coordinate data
        chunks = []
        for k, (d, m) in enumerate(zip(data_delayed, mask_delayed)):
            # band index, chunk y index, chunk x index
            b, i, j = np.unravel_index(k, chunks_shape)
            xck = xc_delayed[j]
            yck = yc_delayed[i]
            # Pass affine as tuple because of how dask handles namedtuple
            # subclasses.
            # ref: https://github.com/rasterio/affine/issues/79
            chunks.append(
                (d, m, xck, yck, b + 1, self.crs, tuple(self.affine))
            )

        meta = gpd.GeoDataFrame(
            {
                "value": [self.dtype.type(1)],
                "band": [1],
                "row": [0],
                "col": [0],
                "geometry": [Point(xrs.x.data[0], xrs.y.data[0])],
            },
            crs=self.crs,
        )
        results = [
            dd.from_delayed(_vectorize(*chunk), meta=meta) for chunk in chunks
        ]
        return dd.concat(results)

    def to_quadrants(self):
        """Split the raster into quadrants

        This returns the quadrants of the raster in the order northwest,
        northeast, southwest, southeast.

        Returns
        -------
        result : RasterQuadrantsResult
            The returned result is a `namedtuple` with attributes: `nw`, `ne`,
            `sw`, and `se`. Unpacking or indexing the object provides the
            quadrants in the stated order.

        """
        _, ny, nx = self.shape
        slice_nw = np.s_[:, : ny // 2, : nx // 2]
        slice_ne = np.s_[:, : ny // 2, nx // 2 :]
        slice_sw = np.s_[:, ny // 2 :, : nx // 2]
        slice_se = np.s_[:, ny // 2 :, nx // 2 :]
        slices = [slice_nw, slice_ne, slice_sw, slice_se]
        data = self.data.copy()
        mask = self.mask.copy()
        results = []
        for s in slices:
            data_quad = data[s]
            mask_quad = mask[s]
            x_quad = self.x[s[2]]
            y_quad = self.y[s[1]]
            xdata_quad = xr.DataArray(
                data_quad,
                coords=(self.band, y_quad, x_quad),
                dims=("band", "y", "x"),
            ).rio.write_nodata(self.null_value)
            xmask_quad = xr.DataArray(
                mask_quad,
                coords=(self.band, y_quad, x_quad),
                dims=("band", "y", "x"),
            )
            ds = make_raster_ds(xdata_quad, xmask_quad)
            if self.crs is not None:
                ds = ds.rio.write_crs(self.crs)
            results.append(Raster(ds, _fast_path=True))
        return RasterQuadrantsResult(*results)


_offset_name_to_rc_offset = {
    "center": (0.5, 0.5),
    "ul": (0, 0),
    "ur": (0, 1),
    "ll": (1, 0),
    "lr": (1, 1),
}


def rowcol_to_xy(row, col, affine, offset):
    """
    Convert (row, col) index values to (x, y) coords using the transformation.
    """
    roffset, coffset = _offset_name_to_rc_offset[offset]
    T = Affine.identity().translation(coffset, roffset)
    return affine * T * (col, row)


def xy_to_rowcol(x, y, affine):
    """
    Convert (x, y) coords to (row, col) index values using the transformation.
    """
    col, row = (~affine) * (x, y)
    row = np.floor(row).astype(int)
    col = np.floor(col).astype(int)
    return row, col


@jit(nopython=True, nogil=True)
def _extract_points(mask, xc, yc):
    n = mask.size - np.sum(mask)
    rx = np.empty(n, dtype=xc.dtype)
    ry = np.empty(n, dtype=yc.dtype)
    if n == 0:
        return rx, ry

    k = 0
    for i in range(mask.shape[1]):
        for j in range(mask.shape[2]):
            if mask[0, i, j]:
                continue
            rx[k] = xc[j]
            ry[k] = yc[i]
            k += 1
    return (rx, ry)


@jit(nopython=True, nogil=True)
def _extract_values(data, mask):
    n = mask.size - np.sum(mask)
    results = np.empty(n, dtype=data.dtype)
    if n == 0:
        return results

    k = 0
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if mask[0, i, j]:
                continue
            results[k] = data[0, i, j]
            k += 1
    return results


@dask.delayed
def _vectorize(data, mask, xc, yc, band, crs, affine_tuple):
    affine = Affine(*affine_tuple[:6])
    xpoints, ypoints = _extract_points(mask, xc, yc)
    if len(xpoints):
        values = _extract_values(data, mask)
        points = gpd.GeoSeries.from_xy(xpoints, ypoints, crs=crs)
        rows, cols = xy_to_rowcol(xpoints, ypoints, affine)
        bands = [band] * len(values)
    else:
        values = []
        points = gpd.GeoSeries([], crs=crs)
        bands = []
        rows = []
        cols = []
    df = gpd.GeoDataFrame(
        {
            "value": values,
            "band": bands,
            "row": rows,
            "col": cols,
            "geometry": points,
        },
        crs=crs,
    )
    if len(xpoints):
        return df
    # Add astype() as a workaround for
    # https://github.com/geopandas/dask-geopandas/issues/190
    return df.astype(
        {"value": data.dtype, "band": int, "row": int, "col": int}
    )


def get_raster(src, strict=True, null_to_nan=False):
    rs = None
    if isinstance(src, Raster):
        rs = src
    elif is_str(src):
        rs = Raster(src)
    elif strict:
        raise TypeError(
            f"Input must be a Raster or path string. Got {repr(type(src))}"
        )
    else:
        try:
            rs = Raster(src)
        except (ValueError, TypeError, RasterDataError, RasterIOError):
            raise ValueError(f"Could not convert input to Raster: {repr(src)}")

    if null_to_nan and rs._masked:
        rs = rs.copy()
        data = rs.data
        new_dtype = promote_dtype_to_float(data.dtype)
        if new_dtype != data.dtype:
            data = data.astype(new_dtype)
        rs.xdata.data = da.where(rs.mask, np.nan, data)
    return rs
