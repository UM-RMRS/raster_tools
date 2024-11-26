import numbers
import warnings
from collections import namedtuple

import dask
import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import odc.geo.xr as odcxr
import rasterio as rio
import rioxarray as xrio
import shapely
import xarray as xr
from affine import Affine
from numba import jit
from odc.geo.geobox import GeoBox
from shapely.geometry import box

from raster_tools.dask_utils import (
    chunks_to_array_locations,
    dask_nanmax,
    dask_nanmin,
)
from raster_tools.dtypes import (
    BOOL,
    DTYPE_INPUT_TO_DTYPE,
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    INT_DTYPE_TO_FLOAT_DTYPE,
    U8,
    U16,
    U32,
    U64,
    get_common_dtype,
    is_bool,
    is_float,
    is_int,
    is_scalar,
    is_str,
    promote_dtype_to_float,
)
from raster_tools.exceptions import RasterDataError, RasterIOError
from raster_tools.masking import (
    get_default_null_value,
    reconcile_nullvalue_with_dtype,
)
from raster_tools.utils import (
    can_broadcast,
    is_strictly_decreasing,
    is_strictly_increasing,
    merge_masks,
    to_chunk_dict,
)

from .io import (
    IO_UNDERSTOOD_TYPES,
    chunk,
    is_batch_file,
    open_raster_from_path_or_url,
    write_raster,
)

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
        mask = get_mask_from_data(xr_out, None)
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


def xr_where_with_meta(cond, left, right, crs=None, nv=None):
    result = xr.where(cond, left, right)
    if crs is not None:
        result = result.rio.write_crs(crs)
    if nv is not None:
        result = result.rio.write_nodata(nv)
    return result


def get_mask_from_data(data, nv=None):
    if nv is None:
        if isinstance(data, (list, np.ndarray)):
            mod = np
        elif isinstance(data, da.Array):
            mod = da
        elif isinstance(data, xr.DataArray):
            mod = xr
        else:
            raise TypeError()
        mask = mod.zeros_like(data, dtype=bool)
        if (mod == da and mask.npartitions == 1) or (
            mod == xr and getattr(mask.data, "npartitions", 2) == 1
        ):
            # https://github.com/dask/dask/issues/11531
            mask |= False
    else:
        mask = np.isnan(data) if np.isnan(nv) else data == nv
    if isinstance(data, xr.DataArray):
        mask = mask.rio.write_nodata(None)
    return mask


def normalize_data(data, yx_chunks=None):
    """2d/3d np.ndarray or da.Array --> 3D da.Array.

    This transforms the data array into the standard data format, which has the
    following properties:

    * 3D with band dimension first.
    * Is a dask array
    * Chunksize of 1 in the band dim.

    Parameters
    ----------
    data : np.ndarray, dask.array.Array
        The data array.
    yx_chunks : tuple, optional
        Tuple of chunks for the y and x dimensions, in that order. The default
        is to leave the chunks as is in this dimension or to chunk with
        ``"auto"``, if `data` is a numpy array.

    Returns
    -------
    data : dask.array.Array
        The resulting dask array.

    """
    if not isinstance(data, (np.ndarray, da.Array)):
        raise TypeError("data must be a numpy or dask array")
    if data.ndim not in (2, 3):
        raise ValueError("data must be 2d or 3d")

    if data.ndim == 2:
        data = data[None]
    chunks = (1, "auto", "auto")
    if yx_chunks is not None:
        chunks = ((1,) * data.shape[0], *yx_chunks)
    if not dask.is_dask_collection(data):
        data = da.from_array(data, chunks=chunks)
    elif data.chunksize[0] != 1 or yx_chunks is not None:
        data = data.rechunk(chunks)
    return data


def has_spatial_dims(xobj):
    try:
        _ = xobj.rio.x_dim
        _ = xobj.rio.y_dim
        return True
    except xrio.exceptions.MissingSpatialDimensionError:
        return False


def find_spatial_dims(obj):
    if has_spatial_dims(obj):
        return (obj.rio.y_dim, obj.rio.x_dim)
    dims = obj.dims
    if len(dims) < 2:
        return None
    sdims = odcxr.spatial_dims(obj, relaxed=True)
    if sdims is not None:
        return sdims
    dims = [
        dim
        for dim in dims
        if dim not in ("band", "bands", "time", "wavelength", "wavelengths")
    ]
    dim1 = dims[-2]
    dim2 = dims[-1]
    if (obj.coords[dim1].dtype.kind != "f") or (
        obj.coords[dim2].dtype.kind != "f"
    ):
        # odc.geo.xr.spatial_dims will throw out dims that are not floats. We
        # relax that here
        return (dim1, dim2)
    return None


def normalize_xarray_data(xdata):
    """Transform the DataArray to the standard raster DataArray format.

    A standard raster DataArray has the following:

    * 3 dimensions in this order: ``("band", "y", "x")``
    * The band dim coordinates starts at one. ex:  ``[1, 2, ...]``
    * The x dim coordinates are always increasing.
    * The y dim coordinates are always decreasing.
    * The CRS is encoded at `obj.rio.crs`
    * The nodata/null value is encoded at `obj.rio.nodata`
    * The data within the DataArray is a dask array.
    * The dask chunks have size 1 in the band dim.

    """
    if not dask.is_dask_collection(xdata):
        xdata = xdata.chunk()
    if len(xdata.shape) > 3 or len(xdata.shape) < 2:
        raise ValueError(
            "Invalid shape. xarray.DataArray objects must have 2D or 3D "
            "shapes."
        )
    if len(xdata.shape) == 2:
        # Add band dim
        xdata = xdata.expand_dims({"band": [1]})

    dims = xdata.dims
    rename_map = {}
    if "time" in dims:
        rename_map["time"] = "band"
    if "lon" in dims:
        rename_map["lon"] = "x"
    if "lat" in dims:
        rename_map["lat"] = "y"
    xdata = xdata.rename(rename_map)
    sdims = find_spatial_dims(xdata)
    if sdims is None:
        # No easy way to figure out how best to transpose based on dim names so
        # just assume the order is valid and rename.
        dims = xdata.dims
        if "band" in dims:
            dims.remove("band")
            name_mapping = {dims[0]: "y", dims[1]: "x"}
        else:
            name_mapping = {
                d: nd for d, nd in zip(dims, ("band", "y", "x")) if d != nd
            }
        xdata = xdata.rename(name_mapping)
    else:
        y_dim, x_dim = sdims
        dims = xdata.dims
        band_dim = (set(dims) - set(sdims)).pop()
        name_mapping = {}
        if y_dim != "y":
            name_mapping[y_dim] = "y"
        if x_dim != "x":
            name_mapping[x_dim] = "x"
        if band_dim != "band":
            name_mapping[band_dim] = "band"
        xdata = xdata.rename(name_mapping)
    xdata = xdata.transpose("band", "y", "x", transpose_coords=True)

    band_coords = np.arange(1, len(xdata.band) + 1)
    if not np.allclose(xdata.band.to_numpy(), band_coords):
        xdata["band"] = band_coords
    if any(dim not in xdata.coords for dim in xdata.dims):
        raise ValueError(
            "Invalid coordinates on xarray.DataArray object:\n{xdata!r}"
        )
    if (xdata.rio.x_dim, xdata.rio.y_dim) != ("x", "y"):
        xdata = xdata.rio.set_spatial_dims(x_dim="x", y_dim="y")
    if xdata.data.chunksize[0] != 1:
        xdata = xdata.chunk({"band": 1, "y": "auto", "x": "auto"})
    # Make sure that x is always increasing and y is always decreasing. xarray
    # will auto align rasters but when a raster is converted to a numpy or dask
    # array, the data may not be aligned. This ensures that rasters converted
    # to non-georeferenecd formats will be oriented the same.
    if is_strictly_decreasing(xdata.x):
        xdata = xdata.isel(x=slice(None, None, -1))
    if is_strictly_increasing(xdata.y):
        xdata = xdata.isel(y=slice(None, None, -1))
    # This MUST BE DONE. Need to recompute and write the transform so that
    # rechunking to (N,1,1) doesn't cause the transform to be dropped. This
    # only happens for hard to reproduce and test edge-cases far down the
    # pipeline.
    # TODO: Find test to check for just this. For right now,
    # tests/test_raster.py::test_to_points catches it.
    xdata = xdata.rio.write_transform(xdata.rio.transform(True))
    return xdata


def is_normalized(xdata):
    if not isinstance(xdata, xr.DataArray):
        raise TypeError("Expected a xarray.DataArray object")
    if any(dim not in xdata.coords for dim in xdata.dims):
        raise ValueError(
            "Invalid coordinates on xarray.DataArray object:\n{xdata!r}"
        )

    return (
        xdata.ndim == 3
        and xdata.dims == ("band", "y", "x")
        and has_spatial_dims(xdata)
        and xdata.rio.y_dim == "y"
        and xdata.rio.x_dim == "x"
        and all(dim in xdata.coords for dim in xdata.dims)
        and dask.is_dask_collection(xdata)
        and xdata.data.chunksize[0] == 1
        and np.allclose(xdata.band, np.arange(1, len(xdata.band) + 1))
        and is_strictly_increasing(xdata.x)
        and is_strictly_decreasing(xdata.y)
    )


def data_to_xr_raster(data, x=None, y=None, affine=None, crs=None, nv=None):
    """Create a standard raster DataArray from a data array.

    Parameters
    ----------
    data : np.ndarray, dask.array.Array
        The data array.
    x : list, np.ndarra, optional
        The x coordinate value. If `x` and `y` are not specified, `affine` is
        used to generate x and y coordinates.
    y : list, np.ndarra, optional
        The y coordinate value. If `x` and `y` are not specified, `affine` is
        used to generate x and y coordinates.
    affine : affine.Affine, optional
        If ``None``, the affine matrix is created using `x` and `y`. If
        `affine` is ``None`` and `x` and `y` are not specified, default affine
        matrix of:
        ::

            | 1.0 0.0 0.0 |
            | 0.0 1.0   N |

        where N is the size of the y dim.
    crs : int, str, rasterio.CRS, optional
        The CRS to use for the result. The default is no CRS.
    nv : scalar, optional
        The nodata/null value to use. The default is no null value.

    Returns
    -------
    raster : xarray.DataArray
        The resulting raster DataArray.

    See Also
    --------
    data_to_raster, data_to_raster_like, data_to_xr_raster_ds,
    data_to_xr_raster_ds_like, data_to_xr_raster_like, dataarray_to_raster,
    dataarray_to_xr_raster, dataarray_to_xr_raster_ds

    """
    data = normalize_data(data)
    if affine is None:
        if x is None and y is None:
            # Add 0.5 offset to move coords to center of cell
            x = np.arange(data.shape[2]) + 0.5
            y = np.arange(data.shape[1])[::-1] + 0.5
        elif any(xi is None for xi in (x, y)):
            raise ValueError("Must specify both x and y or neither.")
        if not isinstance(x, np.ndarray) or not isinstance(x, np.ndarray):
            raise TypeError("x and y must be numpy arrays")
        x = x.ravel()
        y = y.ravel()
        if (y.size, x.size) != data.shape[1:]:
            raise ValueError("x and y do not match data shape")
    else:
        x = _build_x_coord(affine, data.shape)
        y = _build_y_coord(affine, data.shape)
    # The band dimension needs to start at 1 to match raster conventions
    band = np.arange(data.shape[0]) + 1
    xdata = xr.DataArray(
        data, dims=("band", "y", "x"), coords=(band, y, x)
    ).rio.set_spatial_dims(y_dim="y", x_dim="x")
    if crs is not None:
        xdata = xdata.rio.write_crs(crs)
    xdata = xdata.rio.write_nodata(nv)
    return normalize_xarray_data(xdata)


def data_to_xr_raster_like(
    data, xlike, nv=None, match_band_dim=False, match_chunks=True
):
    """Create a standard raster DataArray based on a template DataArray.

    The CRS and x/y information are pulled from `xlike`.

    Parameters
    ----------
    data : np.ndarray, dask.array.Array
        The data array.
    xlike : xarray.DataArray
        The template to pull geospatial information from.
    nv : scalar, optional
        The nodata/null value to use. The default is no null value.
    match_band_dim : bool, optional
        If `data` has only 1 band, this will stack `data` to match the number
        of bands in `xlike`. The default is to not match the number of bands.
    match_chunks : bool, optional
        If ``True``, the chunks of the output will match the chunks in `xlike`.
        The default is ``True``.

    Returns
    -------
    raster : xarray.DataArray
        The resulting raster DataArray matcing `xlike`.

    See Also
    --------
    data_to_raster, data_to_raster_like, data_to_xr_raster,
    data_to_xr_raster_ds, data_to_xr_raster_ds_like, dataarray_to_raster,
    dataarray_to_xr_raster, dataarray_to_xr_raster_ds

    """
    yx_chunks = xlike.data.chunks[1:] if match_chunks else None
    data = normalize_data(data, yx_chunks=yx_chunks)
    if data.shape[-2:] != xlike.shape[1:]:
        raise ValueError("data x/y dims did not match xlike")

    if data.shape[0] == 1 and match_band_dim:
        data = da.stack([data[0] for i in range(xlike.shape[0])], axis=0)
    return data_to_xr_raster(
        data,
        x=xlike.x.to_numpy(),
        y=xlike.y.to_numpy(),
        crs=xlike.rio.crs,
        nv=nv,
    )


def make_raster_ds(raster_dataarray, mask_dataarray):
    """
    Takes data and mask DataArrays and produces a raster Dataset with the data
    in the data variable "raster" and the mask in the "mask" data variable.
    """
    return xr.Dataset({"raster": raster_dataarray, "mask": mask_dataarray})


def data_to_xr_raster_ds(
    data, mask=None, x=None, y=None, affine=None, crs=None, nv=None, burn=False
):
    """
    Create a standard raster Dataset from a data array.

    Parameters
    ----------
    data : np.ndarray, dask.array.Array
        The data array.
    mask : np.ndarray, dask.array.Array
        A boolean mask array. The default is to generate a mask from the data
        using `nv`.
    x : list, np.ndarra, optional
        The x coordinate value. If `x` and `y` are not specified, `affine` is
        used to generate x and y coordinates.
    y : list, np.ndarra, optional
        The y coordinate value. If `x` and `y` are not specified, `affine` is
        used to generate x and y coordinates.
    affine : affine.Affine, optional
        If ``None``, the affine matrix is created using `x` and `y`. If
        `affine` is ``None`` and `x` and `y` are not specified, default affine
        matrix of:
        ::

            | 1.0 0.0 0.0 |
            | 0.0 1.0   N |

        where N is the size of the y dim.
    crs : int, str, rasterio.CRS, optional
        The CRS to use for the result. The default is no CRS.
    nv : scalar, optional
        The nodata/null value to use. The default is no null value.
    burn : bool, optional
        If ``True``, `mask` is used to 'burn' the null value into the data
        array (e.g. `np.where(mask, nv, data)`). The default is ``False``.

    Returns
    -------
    raster : xarray.Dataset
        Theresulting raster Dataset object.

    See Also
    --------
    data_to_raster, data_to_raster_like, data_to_xr_raster,
    data_to_xr_raster_ds_like, data_to_xr_raster_like, dataarray_to_raster,
    dataarray_to_xr_raster, dataarray_to_xr_raster_ds

    """
    data = normalize_data(data)
    if mask is None:
        mask = get_mask_from_data(data, nv)
    else:
        mask = normalize_data(mask)
        if data.shape != mask.shape:
            raise ValueError("data and mask dimensions do not match")
        nv = get_default_null_value(data.dtype) if nv is None else nv
        if burn:
            data = da.where(mask, nv, data)
    xdata = data_to_xr_raster(data, x=x, y=y, affine=affine, crs=crs, nv=nv)
    xmask = data_to_xr_raster(mask, x=x, y=y, affine=affine, crs=crs, nv=None)
    return make_raster_ds(xdata, xmask)


def data_to_xr_raster_ds_like(
    data, xlike, mask=None, nv=None, burn=False, match_chunks=True
):
    """
    Create a standard raster Dataset using a template raster DataArray.

    The CRS and x/y information are pulled from `xlike`.

    Parameters
    ----------
    data : np.ndarray, dask.array.Array
        The data array.
    xlike : xarray.DataArray
        The template to pull geospatial information from.
    mask : np.ndarray, dask.array.Array
        A boolean mask array. The default is to generate a mask from the data
        using `nv`.
    nv : scalar, optional
        The nodata/null value to use. The default is no null value.
    burn : bool, optional
        If ``True``, `mask` is used to 'burn' the null value into the data
        array (e.g. `np.where(mask, nv, data)`). The default is ``False``.
    match_chunks : bool, optional
        If ``True``, the chunks of the output will match the chunks in `xlike`.
        The default is ``True``

    Returns
    -------
    raster : xarray.Dataset
        The resulting raster Dataset object matching `xlike`.

    See Also
    --------
    data_to_raster, data_to_raster_like, data_to_xr_raster,
    data_to_xr_raster_ds, data_to_xr_raster_like, dataarray_to_raster,
    dataarray_to_xr_raster, dataarray_to_xr_raster_ds

    """
    yx_chunks = xlike.data.chunks[1:] if match_chunks else None
    data = normalize_data(data, yx_chunks=yx_chunks)
    if data.shape[-2:] != xlike.shape[1:]:
        raise ValueError("data x/y dims did not match xlike")
    if mask is not None:
        if nv is None:
            nv = get_default_null_value(data.dtype)
        mask = normalize_data(mask, yx_chunks=yx_chunks)
        if mask.shape != data.shape:
            raise ValueError("data and mask dimensions do not match")
    return data_to_xr_raster_ds(
        data,
        mask=mask,
        x=xlike.x.to_numpy(),
        y=xlike.y.to_numpy(),
        crs=xlike.rio.crs,
        nv=nv,
        burn=burn,
    )


def data_to_raster(
    data, mask=None, x=None, y=None, affine=None, crs=None, nv=None, burn=False
):
    """Create a Raster from a data array.

    Parameters
    ----------
    data : np.ndarray, dask.array.Array
        The data array.
    mask : np.ndarray, dask.array.Array
        A boolean mask array. The default is to generate a mask from the data
        using `nv`.
    x : list, np.ndarra, optional
        The x coordinate value. If `x` and `y` are not specified, `affine` is
        used to generate x and y coordinates.
    y : list, np.ndarra, optional
        The y coordinate value. If `x` and `y` are not specified, `affine` is
        used to generate x and y coordinates.
    affine : affine.Affine, optional
        If ``None``, the affine matrix is created using `x` and `y`. If
        `affine` is ``None`` and `x` and `y` are not specified, default affine
        matrix of:
        ::

            | 1.0 0.0 0.0 |
            | 0.0 1.0   N |

        where N is the size of the y dim.
    crs : int, str, rasterio.CRS, optional
        The CRS to use for the result. The default is no CRS.
    nv : scalar, optional
        The nodata/null value to use. The default is no null value.
    burn : bool, optional
        If ``True``, `mask` is used to 'burn' the null value into the data
        array (e.g. `np.where(mask, nv, data)`). The default is ``False``.

    Returns
    -------
    raster : Raster
        The resulting Raster object.

    See Also
    --------
    data_to_raster_like, data_to_xr_raster, data_to_xr_raster_ds,
    data_to_xr_raster_ds_like, data_to_xr_raster_like, dataarray_to_raster,
    dataarray_to_xr_raster, dataarray_to_xr_raster_ds

    """
    ds = data_to_xr_raster_ds(
        data,
        mask=mask,
        x=x,
        y=y,
        affine=affine,
        crs=crs,
        nv=nv,
        burn=burn,
    )
    return Raster(ds, _fast_path=True)


def data_to_raster_like(
    data, like, mask=None, nv=None, burn=False, match_chunks=True
):
    """Create a Raster, based on a template Raster, from a data array.

    The CRS and x/y information are pulled from `xlike`.

    Parameters
    ----------
    data : np.ndarray, dask.array.Array
        The data array.
    like : xarray.DataArray, Raster
        The template to pull geospatial information from. Can be a DataArray or
        Raster.
    mask : np.ndarray, dask.array.Array
        A boolean mask array. The default is to generate a mask from the data
        using `nv`.
    nv : scalar, optional
        The nodata/null value to use. The default is no null value.
    burn : bool, optional
        If ``True``, `mask` is used to 'burn' the null value into the data
        array (e.g. `np.where(mask, nv, data)`). The default is ``False``.
    match_chunks : bool, optional
        If ``True``, the chunks of the output will match the chunks in `xlike`.
        The default is ``True``.

    Returns
    -------
    raster : Raster
        The resulting raster matching `like`.

    See Also
    --------
    data_to_raster, data_to_xr_raster, data_to_xr_raster_ds,
    data_to_xr_raster_ds_like, data_to_xr_raster_like, dataarray_to_raster,
    dataarray_to_xr_raster, dataarray_to_xr_raster_ds

    """
    if isinstance(like, Raster):
        like = like.xdata
    ds = data_to_xr_raster_ds_like(
        data, like, mask=mask, nv=nv, burn=burn, match_chunks=match_chunks
    )
    return Raster(ds, _fast_path=True)


def dataarray_to_xr_raster(xdata):
    """Transform a DataArray object into standard raster DataArray.

    The CRS and null value are pulled from `xdata` using `rioxarray`.

    Parameters
    ----------
    xdata : xarray.DataArray,
        The object to convert to a raster form.

    Returns
    -------
    raster : xarray.DataArray
        The resulting DataArray in standard raster format with dims: ``("band",
        "y", "x")`` and encoded CRS and null value.

    See Also
    --------
    data_to_raster, data_to_raster_like, data_to_xr_raster,
    data_to_xr_raster_ds, data_to_xr_raster_ds_like, data_to_xr_raster_like,
    dataarray_to_raster, dataarray_to_xr_raster_ds

    """
    return normalize_xarray_data(xdata)


def dataarray_to_xr_raster_ds(xdata, xmask=None, crs=None):
    """Create a raster Dataset object from a DataArray object.

    The null value is pulled from `xdata` using `rioxarray`.

    Parameters
    ----------
    xdata : xarray.DataArray,
        The object to convert to a raster Dataset
    xmask : xarray.DataArray[bool], optional
        The matching mask to use with the `xdata` object when creating the
        raster Dataset. Must have boolean dtype. The default is to generate a
        mask from `xdata` based on the value returned by `xdata.rio.nodata`.
    crs : int, str, rasterio.CRS, optional
        The CRS to use when creating the result. The default is to take the CRS
        from `xdata`, if present.

    Returns
    -------
    raster : xarray.Dataset
        The resulting raster Dataset with two data variables: "raster" and
        "mask".

    See Also
    --------
    data_to_raster, data_to_raster_like, data_to_xr_raster,
    data_to_xr_raster_ds, data_to_xr_raster_ds_like, data_to_xr_raster_like,
    dataarray_to_raster, dataarray_to_xr_raster

    """
    xdata = dataarray_to_xr_raster(xdata)
    if xmask is None:
        xmask = get_mask_from_data(xdata, xdata.rio.nodata)
    else:
        xmask = dataarray_to_xr_raster(xmask)
    ds = make_raster_ds(xdata, xmask)
    if crs is not None:
        ds = ds.rio.write_crs(crs)
    return ds


def dataarray_to_raster(xdata, xmask=None, crs=None):
    """Create a Raster from a DataArray object.

    The null value is pulled from `xdata` using `rioxarray`.

    Parameters
    ----------
    xdata : xarray.DataArray
        The object to make a Raster from.
    xmask : xarray.DataArray, optional
        The matching mask to use with the `xdata` object when creating the
        raster Dataset. Must have boolean dtype. The default is to generate a
        mask from `xdata` based on the value returned by `xdata.rio.nodata`.
    crs : int, str, rasterio.CRS, optional
        The CRS to use when creating the result. The default is to take the CRS
        from `xdata`, if present.

    Returns
    -------
    raster : Raster
        The resulting Raster.

    See Also
    --------
    data_to_raster, data_to_raster_like, data_to_xr_raster,
    data_to_xr_raster_ds, data_to_xr_raster_ds_like, data_to_xr_raster_like,
    dataarray_to_xr_raster, dataarray_to_xr_raster_ds

    """
    return Raster(
        dataarray_to_xr_raster_ds(xdata, xmask=xmask, crs=crs), _fast_path=True
    )


def _array_input_to_raster_ds(ar):
    if not isinstance(ar, da.Array):
        has_nan = is_float(ar.dtype) and np.isnan(ar).any()
    else:
        has_nan = is_float(ar.dtype)
    data = normalize_data(ar)
    nv = np.nan if has_nan else None
    ds = data_to_xr_raster_ds(data, nv=nv)

    # Maintain compatibility
    orig_nv = nv
    nv = reconcile_nullvalue_with_dtype(nv, data.dtype)
    if nv is not None and (np.isnan(orig_nv) or orig_nv != nv):
        ds["raster"] = xr_where_with_meta(
            ds.mask, nv, ds.raster
        ).rio.write_nodata(nv)
    return ds


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


def _dataarray_input_to_raster_ds(xdata):
    nv = _try_to_get_null_value_xarray(xdata)
    xdata = normalize_xarray_data(xdata)
    if not dask.is_dask_collection(xdata) or xdata.data.chunksize[0] != 1:
        xdata = chunk(xdata)

    xmask = get_mask_from_data(xdata, nv)
    orig_nv = nv
    nv = reconcile_nullvalue_with_dtype(nv, xdata.dtype)
    crs = xdata.rio.crs
    if nv is not None and (np.isnan(orig_nv) or orig_nv != nv):
        xdata = xr_where_with_meta(xmask, nv, xdata, crs=crs)
    xdata = xdata.rio.write_nodata(nv)
    return dataarray_to_xr_raster_ds(xdata, xmask=xmask, crs=crs)


def _dataset_to_raster_ds(ds_in):
    if (
        dask.is_dask_collection(ds_in)
        and set(ds_in.data_vars) == {"raster", "mask"}
        and dask.is_dask_collection(ds_in.raster)
        and dask.is_dask_collection(ds_in.mask)
        and is_bool(ds_in.mask.dtype)
        and ds_in.raster.dims == ("band", "y", "x")
        and ds_in.mask.dims == ("band", "y", "x")
        and ds_in.raster.chunks == ds_in.mask.chunks
        and ds_in.raster.rio.crs == ds_in.mask.rio.crs
    ):
        return ds_in

    nvars = len(ds_in.data_vars)
    if nvars < 2 or nvars > 2 or set(ds_in.data_vars) != {"raster", "mask"}:
        raise ValueError(
            "Dataset input must have only 'raster' and 'mask' variables"
        )

    xdata = ds_in.raster
    xmask = ds_in.mask
    crs = ds_in.rio.crs
    if not is_bool(xmask.dtype):
        raise TypeError("mask must have boolean dtype")

    xdata = normalize_xarray_data(xdata)
    if not dask.is_dask_collection(xdata):
        xdata = chunk(xdata)
    xmask = normalize_xarray_data(xmask)
    if xmask.shape != xdata.shape:
        raise ValueError("raster and mask dimensions do not match")
    if not dask.is_dask_collection(xmask):
        xmask = xmask.chunk(chunks=to_chunk_dict(xdata.chunks))

    nv = _try_to_get_null_value_xarray(xdata)
    if nv is None:
        nv = get_default_null_value(xdata.dtype)
    else:
        nv = reconcile_nullvalue_with_dtype(nv, xdata.dtype)
    if nv is not None:
        xdata = xr_where_with_meta(xmask, nv, xdata, nv=nv)
    return dataarray_to_xr_raster_ds(xdata, xmask, crs=crs)


def _xarray_input_to_raster_ds(xin):
    if isinstance(xin, xr.DataArray):
        return _dataarray_input_to_raster_ds(xin)
    return _dataset_to_raster_ds(xin)


def get_raster_ds(raster):
    # Copy to take ownership
    if isinstance(raster, Raster):
        ds = raster._ds.copy(deep=True)
    elif isinstance(raster, (xr.DataArray, xr.Dataset)):
        ds = _xarray_input_to_raster_ds(raster.copy())
    elif isinstance(raster, (np.ndarray, da.Array)):
        ds = _array_input_to_raster_ds(raster.copy())
    elif type(raster) in IO_UNDERSTOOD_TYPES:
        if is_batch_file(raster):
            # Import here to avoid circular import errors
            from raster_tools.batch import parse_batch_script

            ds = parse_batch_script(raster).final_raster._ds
        else:
            xdata = open_raster_from_path_or_url(raster)
            ds = dataarray_to_xr_raster_ds(xdata)
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
        crs = self.crs
        masked = self._masked
        return (
            f"<raster_tools.Raster (crs='{crs}', masked={masked})>\n"
            f"{repr(self._ds.raster)}"
        )

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
        return self.to_numpy()

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
    def size(self):
        """
        The number of grid cells across all bands.
        """
        return self.data.size

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
    def geobox(self):
        """GeoBox object describing the raster's grid."""
        return self._ds.odc.geobox

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
        return self._ds.raster.to_numpy()

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

    @property
    def geochunks(self):
        """
        Produce an array of GeoChunks that correspond to the underlying chunks
        in the data.
        """
        affine = self.affine
        data = self.data
        chunks = data.chunks
        chunks_shape = data.numblocks
        geochunks = np.empty(chunks_shape, dtype=object)
        band_locations = chunks_to_array_locations(chunks[0])
        y_locations = chunks_to_array_locations(chunks[1])
        x_locations = chunks_to_array_locations(chunks[2])
        chunk_rasters = self.get_chunk_rasters()
        for bi, yi, xi in np.ndindex(chunks_shape):
            chunk_raster = chunk_rasters[bi, yi, xi]
            geochunks[bi, yi, xi] = GeoChunk(
                chunk_raster.shape,
                chunk_raster.geobox,
                affine,
                self.shape,
                [
                    band_locations[bi],
                    y_locations[yi],
                    x_locations[xi],
                ],
                (bi, yi, xi),
            )
        return GeoChunkArray(geochunks)

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
            self._ds.chunk(dict(zip(["band", "y", "x"], chunks))),
            _fast_path=True,
        )

    def to_dataset(self):
        """Returns the underlying `xarray.Dataset`."""
        return self._ds

    def to_numpy(self):
        """
        Return the raw internal raster as a numpy array.

        .. note::
           This triggers computation and loads the raster data into memory.

        """
        return self._ds.raster.to_numpy()

    def plot(self, *args, **kwargs):
        """
        Plot the raster. Args and kwargs are passed to xarray's plot function.
        """
        return get_raster(self, null_to_nan=True).xdata.plot(*args, **kwargs)

    def explore(self, band, *args, **kwargs):
        """Plot the raster band on an interactive :py:mod:`folium` map.

        This allows for rapid data exploration. Any extra arguments or keyword
        arguments are passed on to `odc-geo`'s `explore` function.

        .. note::
            This function is very experimental.

        Parameters
        ----------
        band : int
            The band to plot. Bands use 1-based indexing so this value must be
            greater than 0.

        Returns
        -------
        map : folium.folium.Map
            The resulting py:mod:`folium` map.

        """
        if not is_int(band):
            raise TypeError("Band value must be an integer")
        if band < 1:
            raise ValueError("Specified band must be greater than 0")
        return self.xdata.sel(band=band).odc.explore(*args, **kwargs)

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

    def load(self):
        """Compute delayed operations and load the result into memory.

        This computes all delayed operations built up so far and then stores
        the resulting raster in memory. This can have significant performance
        implications if the raster data being computed is large relative to the
        available computer memory. The original raster is unaltered.

        """
        ds = chunk(self._ds.compute())
        # A new raster is returned to mirror the xarray and dask APIs
        return Raster(ds)

    def eval(self):  # noqa: A003
        """Compute any applied operations and return the result as new Raster.

        Note that the unerlying sources will be loaded into memory for the
        computations and the result will be fixed in memory. The original
        Raster will be unaltered.

        .. note::
            This method has been replaced by :meth:`load` and will eventually
            be deprecated and then removed.

        """
        # TODO: deprecate in favor of load
        return self.load()

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
        there was already a null value, then it is replaced. The raster dtype
        will be promoted, if needed, to accommodate `value`. If `value` is
        None, the null value is cleared. The raster data is not changed in this
        case.

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
        if value is None:
            # Burn in current mask values and then clear null value
            # TODO: burning should not be needed as the values should already
            # be set
            xrs = self.burn_mask().xdata.rio.write_nodata(None)
            mask = xr.zeros_like(xrs, dtype=bool)
            return Raster(make_raster_ds(xrs, mask), _fast_path=True)

        if not (is_scalar(value) or is_bool(value)):
            raise TypeError(f"Value must be a scalar or None: {value}")

        dtype = self.dtype
        new_dtype = None
        if np.isnan(value):
            if is_int(dtype):
                new_dtype = INT_DTYPE_TO_FLOAT_DTYPE[dtype]
            # else do nothing
        elif is_int(value) and is_float(dtype):
            # INFO: Numpy refuses to let small int values be put in small float
            # dtypes. This catches and fixes the issue.
            fvalue = float(value)
            new_dtype = get_common_dtype([fvalue, dtype])
        else:
            new_dtype = get_common_dtype([value, dtype])

        xrs = self._ds.raster.copy()
        if new_dtype is not None:
            xrs = xrs.astype(new_dtype)
        else:
            value = dtype.type(value)

        # Update mask
        mask = self.xmask
        temp_mask = get_mask_from_data(xrs, value)
        mask = mask | temp_mask if self._masked else temp_mask
        xrs = xrs.rio.write_nodata(value)
        return Raster(make_raster_ds(xrs, mask), _fast_path=True).burn_mask()

    def set_null(self, mask_raster):
        """Update the raster's null pixels using the provided mask

        Parameters
        ----------
        mask_raster : str, Raster
            Raster or path to a raster that is used to update the masked out
            pixels. This raster updates the mask. It does not replace the mask.
            Pixels that were already marked as null stay null and pixels that
            are marked as null in `mask_raster` become marked as null in the
            resulting raster. This is a logical "OR" operation. `mask_raster`
            must have data type of boolean, int8, or uint8. `mask_raster` must
            have either 1 band or the same number of bands as the raster it is
            being applied to. A single band `mask_raster` is broadcast across
            all bands of the raster being modified.

        Returns
        -------
        Raster
            The resulting raster with updated mask.

        """
        mask_raster = get_raster(mask_raster)
        if mask_raster.nbands > 1 and mask_raster.nbands != self.nbands:
            raise ValueError(
                "The number of bands in mask_raster must be 1 or match"
                f" this raster. Got {mask_raster.nbands}"
            )
        if mask_raster.shape[1:] != self.shape[1:]:
            raise ValueError(
                "x and y dims for mask_raster do not match this raster."
                f" {mask_raster.shape[1:]} vs {self.shape[1:]}"
            )
        dtype = mask_raster.dtype
        if dtype not in {BOOL, I8, U8}:
            raise TypeError("mask_raster must be boolean, int8, or uint8")
        elif not is_bool(dtype):
            mask_raster = mask_raster.astype(bool)

        out_raster = self.copy()
        new_mask_data = out_raster._ds.mask.data
        # Rely on numpy broadcasting when applying the new mask data
        if mask_raster._masked:
            new_mask_data |= mask_raster.data & (~mask_raster.mask)
        else:
            new_mask_data |= mask_raster.data
        out_raster._ds.mask.data = new_mask_data
        if not self._masked:
            out_raster._ds["raster"] = out_raster._ds.raster.rio.write_nodata(
                get_default_null_value(self.dtype)
            )
        # Burn mask to set null values in newly masked regions
        return out_raster.burn_mask()

    def burn_mask(self):
        """Fill null-masked cells with null value.

        Use this as a way to return the underlying data to a known state. If
        dtype is boolean, the null cells are set to ``True`` instead of
        promoting to fit the null value.
        """
        if not self._masked:
            return self.copy()
        nv = self.null_value
        if is_bool(self.dtype):
            # Sanity check to make sure that boolean rasters get a boolean null
            # value
            nv = get_default_null_value(self.dtype)
        out_raster = self.copy()
        # Work with .data to avoid dropping attributes caused by using xarray
        # and rioxarrays' APIs
        out_raster._ds.raster.data = da.where(self.mask, nv, self.data)
        return out_raster

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

        The mask for the result is a combination of all three rasters. Any
        cells that are masked in the condition raster will be masked in the
        output. The rest of the cells are masked based on which raster they
        were taken from, as determined by the condition raster, and if the cell
        in that raster was null. Effectively, the resulting mask is determined
        as follows: `where(condition.mask, True, where(condition,
        true_rast.mask, false_rast.mask)`.

        Parameters
        ----------
        condition : str or Raster
            A boolean or int raster that indicates where elements in this
            raster should be preserved and where `other` should be used. If
            the condition is an int raster, it is coerced to bool using
            `condition > 0`.  ``True`` cells pull values from this raster and
            ``False`` cells pull from `other`. *str* is treated as a path to a
            raster.
        other : scalar, str or Raster, None
            A raster or value to use in locations where `condition` is
            ``False``. *str* is treated as a path to a raster. If None, this is
            treated as a null value.

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
            ``new_value``. If ``new_value`` is ``None``, the matching pixels
            will be marked as null. If `mapping` is a list and there are
            mappings that conflict or overlap, earlier mappings take
            precedence. `inclusivity` determines which sides of the range are
            inclusive and exclusive.
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

    def model_predict(self, model, n_outputs=1):
        """
        Generate a new raster using the provided model to predict new values.

        The raster's values are used as the predictor inputs for `model`.
        Each band in the input raster is used as a separate input variable.
        Outputs are raster surfaces where each band corresponds to a variable
        output by `model`.

        The `model` argument must provide a `predict` method. If the desired
        model does not provide a `predict` function,
        :class:`ModelPredictAdaptor` can be used to wrap it and make it
        compatible with this function.

        Parameters
        ----------
        model : object
            The model used to estimate new values. It must have a `predict`
            method that takes an array-like object of shape `(N, M)`, where `N`
            is the number of samples and `M` is the number of
            features/predictor variables. The `predict` method should return an
            `(N, [n_outputs])` shape result. If only one variable is resurned,
            then the `n_outputs` dimension is optional.
        n_outputs : int, optional
            The number of output variables from the model. Each output variable
            produced by the model is converted to a band in output raster. The
            default is ``1``.

        Returns
        -------
        Raster
            The resulting raster of estimated values. Each band corresponds to
            an output variable produced by the `model`.
        """
        from raster_tools.general import model_predict_raster

        return model_predict_raster(self, model, n_outputs)

    def reclassify(self, remapping, unmapped_to_null=False):
        """Reclassify raster values based on a mapping.

        The raster must have an integer type.

        Parameters
        ----------
        remapping : str, dict
            Can be either a ``dict`` or a path string. If a ``dict`` is
            provided, the keys will be reclassified to the corresponding
            values. It is possible to map values to the null value by providing
            ``None`` in the mapping. If a path string, it is treated as an
            ASCII remap file where each line looks like ``a:b`` and ``a`` and
            ``b`` are scalars. ``b`` can also be "NoData". This indicates that
            ``a`` will be mapped to the null value. The output values of the
            mapping can cause type promotion. If the input raster has integer
            data and one of the outputs in the mapping is a float, the result
            will be a float raster.
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

    def round(self, decimals=0):  # noqa: A003
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

    def reproject(
        self, crs_or_geobox=None, resample_method="nearest", resolution=None
    ):
        """Reproject to a new projection or resolution.

        This is a lazy operation.

        Parameters
        ----------
        crs_or_geobox : int, str, CRS, GeoBox, optional
            The target grid to reproject the raster to. This can be a
            projection string, EPSG code string or integer, a CRS object, or a
            GeoBox object. `resolution` can also be specified to change the
            output raster's resolution in the new CRS. If `crs_or_geobox` is
            not provided, `resolution` must be specified.
        resample_method : str, optional
            The data resampling method to use. Null pixels are ignored for all
            methods. Some methods require specific versions of GDAL. These are
            noted below. Valid methods are:

            'nearest'
                Nearest neighbor resampling. This is the default.
            'bilinear'
                Bilinear resmapling.
            'cubic'
                Cubic resmapling.
            'cubic_spline'
                Cubic spline resmapling.
            'lanczos'
                Lanczos windowed sinc resmapling.
            'average'
                Average resampling, computes the weighted average of all
                contributing pixels.
            'mode'
                Mode resampling, selects the value which appears most often.
            'max'
                Maximum resampling. (GDAL 2.0+)
            'min'
                Minimum resampling. (GDAL 2.0+)
            'med'
                Median resampling. (GDAL 2.0+)
            'q1'
                Q1, first quartile resampling. (GDAL 2.0+)
            'q3'
                Q3, third quartile resampling. (GDAL 2.0+)
            'sum'
                Sum, compute the weighted sum. (GDAL 3.1+)
            'rms'
                RMS, root mean square/quadratic mean. (GDAL 3.3+)
        resolution : int, float, tuple of int or float, optional
            The desired resolution of the reprojected raster. If
            `crs_or_geobox` is unspecified, this is used to reproject to the
            new resolution while maintaining the same CRS. One of
            `crs_or_geobox` or `resolution` must be provided. Both can also be
            provided.

        Returns
        -------
        Raster
            The reprojected raster on the new grid.

        """
        from raster_tools.warp import reproject

        return reproject(
            self,
            crs_or_geobox=crs_or_geobox,
            resample_method=resample_method,
            resolution=resolution,
        )

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

    def to_polygons(self, neighbors=4):
        """Convert the raster to a vector of polygons.

        Null cells are ignored. The resulting vector is randomly ordered.

        Parameters
        ----------
        neighbors : {4, 8} int, optional
            This determines how many neighboring cells to consider when joining
            cells together to form polygons. Valid values are ``4`` and ``8``.
            ``8`` causes diagonal neighbors to be used. Default is ``4``.

        Returns
        -------
        dask_geopandas.GeoDataFrame
            A `GeoDataFrame` with columns: value, band, geometry. The geometry
            column contains polygons/multi-polygons.

        Notes
        -----
        The algorithm used by this method is best used with simple thematic
        data. This is because the algorithm consumes memory proportional to the
        number and complexity of the produced polygons. Data that produces a
        large number of polygons, i.e. data with high pixel-to-pixel
        variability, will cause a large amount of memory to be consumed.

        See Also
        --------
        Raster.to_vector

        """
        if neighbors not in (4, 8):
            raise ValueError("neighbors can only be 4 or 8")
        meta = gpd.GeoDataFrame(
            {
                "value": np.array((), dtype=self.dtype),
                "band": np.array((), dtype=I64),
                "geometry": gpd.GeoSeries(()),
            },
            crs=self.crs,
        )
        # Iterate over bands because dissolve will join polygons from different
        # bands, otherwise.
        band_results = []
        for iband in range(self.nbands):
            chunks = list(self.data[iband].to_delayed().ravel())
            mask_chunks = list(self.mask[iband].to_delayed().ravel())
            transforms = [
                r.affine
                for r in self.get_bands(iband + 1).get_chunk_rasters().ravel()
            ]
            partitions = []
            for chnk, mask, transform in zip(chunks, mask_chunks, transforms):
                part = dd.from_delayed(
                    _shapes_delayed(
                        chnk, mask, neighbors, transform, iband + 1, self.crs
                    ),
                    meta=meta.copy(),
                )
                partitions.append(part)
            # Dissolve uses the 'by' column to create the new index. Reset the
            # index to turn it back into a regular column.
            band_results.append(
                dd.concat(partitions).dissolve(by="value").reset_index()
            )
        return (
            dd.concat(band_results)
            .repartition(npartitions=1)
            .reset_index(drop=True)
        )

    def to_vector(self, as_polygons=False, neighbors=4):
        """Convert the raster to a vector.

        Parameters
        ----------
        as_polygons : bool, optional
            If ``True``, the raster is converted to polygons/multi-polygon,
            each polyon/mult-polygon is made up of like-valued cells. Null
            cells are ignored. :meth:`to_polygons` for more information. If
            ``False``, the raster is converted to points. Each non-null cell is
            a point in the output. See :meth:`to_points` for more information.
            Default is ``False``.
        neighbors : {4, 8} int, optional
            Used when `as_polygons` is ``True``. This determines how many
            neighboring cells to consider when joining cells together to form
            polygons. Valid values are ``4`` and ``8``. ``8`` causes diagonal
            neighbors to be used. Default is ``4``.

        Returns
        -------
        dask_geopandas.GeoDataFrame
            A `GeoDataFrame`. If `as_polygons` is ``True``, the result is a
            dataframe with columns: value, band, geometry. The geometry column
            will contain polygons. If ``False``, the result has columns: value,
            band, row, col, geometry. The geometry column will contain points.

        See Also
        --------
        Raster.to_points, Raster.to_polygons

        """
        if as_polygons:
            return self.to_polygons(neighbors=neighbors)
        return self.to_points()

    def to_points(self):
        """
        Convert the raster into a vector where each non-null cell is a point.

        The resulting points are located at the center of each grid cell.

        Returns
        -------
        dask_geopandas.GeoDataFrame
            The result is a dask_geopandas GeoDataFrame with the following
            columns: value, band, row, col, geometry. The value is the cell
            value at the row and column in the raster. Each row has a unique
            integer in the index. This is true across data frame partitions.

        See Also
        --------
        Raster.to_vector

        """
        xrs = self._ds.raster
        mask = self._ds.mask.data
        data_delayed = xrs.data.to_delayed()
        data_delayed = data_delayed.ravel()
        mask_delayed = mask.to_delayed().ravel()
        geochunks = self.geochunks.ravel()

        # See issue for inspiration: https://github.com/dask/dask/issues/7589
        # Group chunk data with corresponding coordinate data
        chunks = list(zip(data_delayed, mask_delayed, geochunks))

        meta = gpd.GeoDataFrame(
            {
                "value": np.array((), dtype=self.dtype),
                "band": np.array((), dtype="int64"),
                "row": np.array((), dtype="int64"),
                "col": np.array((), dtype="int64"),
                "geometry": gpd.GeoSeries(),
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

    def get_chunk_bounding_boxes(self, include_band=False):
        """Return GeoDataFrame with the chunk bounding boxes.

        This method generates a GeoDataFrame of bounding boxes for the
        underlying data chunks. By default it does this for a single band since
        the bounding boxes are the same across bands. The result also contains
        columns for the chunk position in the data.

        By default, the result has two position columns: `'chunk_row'` and
        `'chunk_col'`. Setting `include_band` adds `'chunk_band'`.

        Parameters
        ----------
        include_band : bool, optional
            Duplicates the result across bands and adds a third position column
            named `'chunk_band'`. Default is ``False``.

        Examples
        --------

        >>> dem = Raster("data/dem.tif")
        >>> dem.get_chunk_bounding_boxes()
            chunk_row  chunk_col                                           geometry
        0           0          0  POLYGON ((-32863.383 53183.938, -32863.383 681...
        1           0          1  POLYGON ((-17863.383 53183.938, -17863.383 681...
        2           0          2  POLYGON ((-2863.383 53183.938, -2863.383 68183...
        3           0          3  POLYGON ((12136.617 53183.938, 12136.617 68183...
        4           0          4  POLYGON ((27136.617 53183.938, 27136.617 68183...
        ...
        >>> dem.get_chunk_bounding_boxes(True)
            chunk_band  chunk_row  chunk_col                                           geometry
        0            0          0          0  POLYGON ((-32863.383 53183.938, -32863.383 681...
        1            0          0          1  POLYGON ((-17863.383 53183.938, -17863.383 681...
        2            0          0          2  POLYGON ((-2863.383 53183.938, -2863.383 68183...
        3            0          0          3  POLYGON ((12136.617 53183.938, 12136.617 68183...
        4            0          0          4  POLYGON ((27136.617 53183.938, 27136.617 68183...
        ...

        """  # noqa: E501
        _, ychunks, xchunks = self.data.chunks
        i = 0
        j = 0
        chunk_boxes = []
        rows = []
        cols = []
        for row, yc in enumerate(ychunks):
            for col, xc in enumerate(xchunks):
                y = self.y[i : i + yc]
                x = self.x[j : j + xc]
                j += xc
                bounds = xr.DataArray(
                    da.empty((yc, xc)), dims=("y", "x"), coords=(y, x)
                ).rio.bounds()
                chunk_boxes.append(box(*bounds))
                rows.append(row)
                cols.append(col)
            i += yc
            j = 0
        if not include_band:
            data = {
                "chunk_row": rows,
                "chunk_col": cols,
                "geometry": chunk_boxes,
            }
        else:
            chunks_per_band = np.prod(self.data.blocks.shape[1:])
            bands = []
            for i in range(self.nbands):
                bands += [i] * chunks_per_band
            rows *= self.nbands
            cols *= self.nbands
            chunk_boxes *= self.nbands
            data = {
                "chunk_band": bands,
                "chunk_row": rows,
                "chunk_col": cols,
                "geometry": chunk_boxes,
            }
        return gpd.GeoDataFrame(data, crs=self.crs)

    def get_chunk_rasters(self):
        """Return the underlying data chunks as an array of Rasters.

        This returns a 3D array where each element corresponds to a chunk of
        data. Each element is a Raster wrapping the corresponding chunk. The
        dimensions of the array are (n-bands, y-chunks, x-chunks). For
        example, if a Raster has 2 bands, 3 chunks in the y direction, and 4
        chunks in the x direction, this method would produce an array of
        Rasters with shape ``(2, 3, 4)``.

        Returns
        -------
        numpy.ndarray
            The 3D numpy array of chunk Rasters.

        """
        bchunks, ychunks, xchunks = self.data.chunks
        b = 0
        y = 0
        x = 0
        out = np.empty(self.data.numblocks, dtype=object)
        for band, bc in enumerate(bchunks):
            for row, yc in enumerate(ychunks):
                for col, xc in enumerate(xchunks):
                    ds = (
                        self._ds.isel(
                            band=slice(b, b + bc),
                            y=slice(y, y + yc),
                            x=slice(x, x + xc),
                        )
                        .copy()
                        .assign_coords(band=[1])
                    )
                    rs = Raster(ds, _fast_path=True)
                    out[band, row, col] = rs
                    x += xc
                y += yc
                x = 0
            b += bc
            y = 0
        return out


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
    transform = Affine.identity().translation(coffset, roffset)
    return affine * transform * (col, row)


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


@jit(nopython=True, nogil=True)
def _extract(data, mask, band, x, y, inv_affine):
    n = mask.size - np.sum(mask)

    bands = np.empty(n, dtype="int64")
    # x, y, 1 for each column
    xy1 = np.ones((3, n), dtype="float64")
    values = np.empty(n, dtype=data.dtype)
    if n == 0:
        return (
            values,
            bands,
            np.array((), dtype="int64"),
            np.array((), dtype="int64"),
            np.array((), dtype="float64"),
            np.array((), dtype="float64"),
        )

    k = 0
    for idx in np.ndindex(data.shape):
        if mask[idx]:
            continue
        values[k] = data[idx]
        bands[k] = band[idx[0]]
        xy1[1, k] = y[idx[1]]
        xy1[0, k] = x[idx[2]]
        k += 1
    # Use the inverse affine matrix to generate column and row values
    cr1 = inv_affine @ xy1
    return (
        values,
        bands,
        # rows
        np.floor(cr1[1]).astype("int64"),
        # cols
        np.floor(cr1[0]).astype("int64"),
        # x
        xy1[0],
        # y
        xy1[1],
    )


@dask.delayed
def _vectorize(data, mask, geochunk):
    values, bands, rows, cols, xs, ys = _extract(
        data,
        mask,
        geochunk.band,
        geochunk.x,
        geochunk.y,
        # Inverse affine transform matrix
        np.array(list(~geochunk.parent_affine), dtype=F64).reshape(3, 3),
    )
    # Use the index of each point in the flattened parent array as the pandas
    # index.
    index = np.ravel_multi_index((bands, rows, cols), geochunk.parent_shape)
    bands += 1
    points = gpd.points_from_xy(xs, ys, crs=geochunk.crs)
    vec = gpd.GeoDataFrame(
        {
            "value": values,
            "band": bands,
            "row": rows,
            "col": cols,
            "geometry": points,
        },
        crs=geochunk.crs,
        index=index,
    )
    return vec


def _get_rio_shapes_dtype(dtype):
    if dtype in (I16, I32, U8, U16, F32):
        return dtype

    new_dtype = dtype
    if is_bool(dtype):
        new_dtype = U8
    elif dtype == I8:
        new_dtype = I16
    elif dtype in (I64, U32, U64):
        new_dtype = I32
    elif dtype in (F16, F64):
        new_dtype = F32
    return new_dtype


@dask.delayed
def _shapes_delayed(chunk, mask, neighbors, transform, band, crs):
    # Rasterio's shapes function only supports a small set of dtypes. This may
    # cause the dtype to be downcast.
    # ref: rasterio._features._shapes
    orig_dtype = chunk.dtype
    working_dtype = _get_rio_shapes_dtype(orig_dtype)
    if working_dtype != orig_dtype:
        # Replace null values with a value that will work for all dtypes. Null
        # values are the most likely to produce warnings when down casting. The
        # null cells will be ignored in rio.features.shapes so the value
        # does not matter for computation.
        chunk[mask] = 0
        chunk = chunk.astype(working_dtype)

    # Invert because rasterio's shapes func uses False for masked cells.
    mask = ~mask
    shapes = []
    values = []
    for geojson, value in rio.features.shapes(
        chunk, mask, neighbors, transform
    ):
        shapes.append(shapely.geometry.shape(geojson))
        values.append(value)
    values = np.array(values)

    # Cast back to the original dtype, if needed
    if working_dtype != orig_dtype:
        values = values.astype(orig_dtype)
    return gpd.GeoDataFrame(
        {
            "value": values,
            "band": [band] * len(values),
            "geometry": gpd.GeoSeries(shapes),
        },
        crs=crs,
    )


def _build_x_coord(affine, shape_3d):
    nx = shape_3d[2]
    # Cell size for x dim
    a = affine.a
    tmatrix = np.array(affine).reshape((3, 3))
    xc = (tmatrix @ np.array([np.arange(nx), np.zeros(nx), np.ones(nx)]))[0]
    xc += a / 2
    # Copy to trim off excess base array
    return xc.copy()


def _build_y_coord(affine, shape_3d):
    ny = shape_3d[1]
    # Cell size for y dim (should be < 0)
    e = affine.e
    tmatrix = np.array(affine).reshape((3, 3))
    yc = (tmatrix @ np.array([np.zeros(ny), np.arange(ny), np.ones(ny)]))[1]
    yc += e / 2
    # Copy to trim off excess base array
    return yc.copy()


def _build_coords(affine, shape):
    return _build_x_coord(affine, shape), _build_y_coord(affine, shape)


class GeoChunk:
    """Object representing a geo-located chunk.

    A GeoChunk contains information needed to geo-locate a chunk and locate it
    within the parent array. It also has helper methods for manipulating that
    information. It is meant to be used to provide information to raster
    functions inside dask map operations.

    """

    def __init__(
        self,
        shape,
        geobox,
        parent_affine,
        parent_shape,
        array_location,
        chunk_location,
    ):
        self.shape = shape
        self.geobox = geobox
        self.parent_affine = parent_affine
        self.affine = geobox.affine
        self.crs = geobox.crs
        self.bbox = geobox.extent.geom
        self.parent_shape = parent_shape
        self.array_location = array_location
        self.chunk_location = chunk_location

    def __repr__(self):
        return (
            f"<{GeoChunk.__module__}.{GeoChunk.__name__}"
            f" shape: {self.shape}, chunk_location: {self.chunk_location})>"
        )

    @property
    def x(self):
        return _build_x_coord(self.affine, self.shape)

    @property
    def y(self):
        return _build_y_coord(self.affine, self.shape)

    @property
    def band(self):
        return np.arange(*self.array_location[0])

    def resize_dim(self, left, right, dim):
        """Resize the given dimension in the left and right directions.

        Parameters
        ----------
        left : int
            if negative, the dimension is trimmed by the given value on its
            left-hand or top side. If positive the dimension is expanded on its
            left-hand or top side.
        right : int
            if negative, the dimension is trimmed by the given value on its
            right-hand or bottom side. If positive the dimension is expanded on
            its left-hand or bottom side.
        dim : int {0, 1}
            Can be 0 or 1. 0 indicates the y or row dimension. 1 indicates the
            x or colomn dimension.

        Returns
        -------
        GeoChunk
            The resized GeoChunk.

        """
        if not is_int(left) or not is_int(right):
            raise TypeError(
                f"Cannot resize by a non-integer value: {left}, {right}"
            )
        dim += 1
        assert dim > 0
        new_shape = tuple(
            self.shape[d] if d != dim else self.shape[d] + left + right
            for d in range(len(self.shape))
        )
        new_affine = self.affine
        if left != 0:
            # translation(col, row) aka translation(xoffset, yoffset)
            if dim == 1:
                translation = Affine.translation(0, -left)
            else:
                translation = Affine.translation(-left, 0)
            new_affine *= translation
        new_geobox = GeoBox(new_shape[1:], new_affine, self.crs)
        new_location = []
        for i, loc in enumerate(self.array_location):
            if i == dim:
                nloc = (loc[0] - left, loc[1] + right)
                new_location.append(nloc)
            else:
                new_location.append(loc)
        return GeoChunk(
            new_shape,
            new_geobox,
            self.parent_affine,
            self.shape,
            new_location,
            self.chunk_location,
        )

    def pad_rows(self, nrows):
        """Pad both sides of the chunk in the row dimension."""
        return self.resize_dim(nrows, nrows, 0)

    def pad_cols(self, ncols):
        """Pad both sides of the chunk in the column dimension."""
        return self.resize_dim(ncols, ncols, 1)

    def pad(self, nrows, ncols=None):
        """Pad both sides of the chunk in the x and y directions.

        The size of the chunk along a given axis will grow by twice the
        specified number because both sides of the chunk are padded.

        """
        if ncols is None:
            ncols = nrows
        return self.pad_rows(nrows).pad_cols(ncols)

    def trim_left(self, ncols):
        """Trim the chunk's left most columns by `ncols`."""
        return self.resize_dim(-ncols, 0, 0)

    def trim_right(self, ncols):
        """Trim the chunk's right most columns by `ncols`."""
        return self.resize_dim(0, -ncols, 0)

    def trim_top(self, nrows):
        """Trim the chunk's top rows by `nrows`."""
        return self.resize_dim(-nrows, 0, 1)

    def trim_bottom(self, nrows):
        """Trim the chunk's bottom rows by `nrows`."""
        return self.resize_dim(0, -nrows, 1)

    def trim_rows(self, nrows):
        """Trim the chunk's rows on both sides by `nrows`."""
        return self.resize_dim(-nrows, -nrows, 0)

    def trim_cols(self, ncols):
        """Trim the chunk's columns on both sides by `ncols`."""
        return self.resize_dim(-ncols, -ncols, 1)

    def trim(self, nrows, ncols=None):
        """
        Trim the chunk's columns and rows on both sides by `nrows` and `ncols`.
        """
        if ncols is None:
            ncols = nrows
        return self.trim_rows(nrows).trim_cols(ncols)

    def shift_rows(self, nrows):
        """Shift the chunk up or down.

        Negative values shift the chunk up.

        """
        return self.resize_dim(-nrows, nrows, 0)

    def shift_cols(self, ncols):
        """Shift the chunk left or right.

        Negative values shift the chunk left.

        """
        return self.resize_dim(-ncols, ncols, 1)

    def shift(self, nrows, ncols=None):
        """Shift the chunk in both row and column dimensions."""
        if ncols is None:
            ncols = nrows
        return self.shift_rows(nrows).shift_cols(ncols)


class GeoChunkArray:
    def __init__(self, geochunks):
        self._array = geochunks

    def __getitem__(self, idx):
        result = self._array[idx]
        if isinstance(result, GeoChunk):
            return result
        else:
            return GeoChunkArray(result)

    def __eq__(self, other):
        if isinstance(other, GeoChunkArray):
            return self._array is other._array
        return NotImplemented

    def __array__(self):
        return self._array

    def __repr__(self):
        return (
            f"<{GeoChunkArray.__module__}.{GeoChunkArray.__name__}"
            f" (shape: {self.shape})>"
        )

    @property
    def size(self):
        return self._array.size

    @property
    def shape(self):
        return self._array.shape

    def ravel(self):
        """Return the array as a flattened list of geochunks."""
        return list(self._array.ravel())

    def map(self, func, *args, **kwargs):
        new = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            new[idx] = func(self._array[idx], *args, **kwargs)
        return GeoChunkArray(new)

    def to_numpy(self):
        return self._array

    def to_dask(self):
        """Return as a dask array with chunksize of 1."""
        return da.from_array(self.to_numpy(), chunks=1)


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
        except (ValueError, TypeError, RasterDataError, RasterIOError) as err:
            raise ValueError(
                f"Could not convert input to Raster: {repr(src)}"
            ) from err

    if null_to_nan and rs._masked:
        rs = rs.copy()
        data = rs.data
        new_dtype = promote_dtype_to_float(data.dtype)
        if new_dtype != data.dtype:
            data = data.astype(new_dtype)
        rs.xdata.data = da.where(rs.mask, np.nan, data)
    return rs
