import numbers
import warnings
from collections import abc

import dask
import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import rasterio as rio
import xarray as xr
from numba import jit
from shapely.geometry import Point

from raster_tools.dask_utils import dask_nanmax, dask_nanmin
from raster_tools.dtypes import (
    DTYPE_INPUT_TO_DTYPE,
    get_default_null_value,
    is_bool,
    is_float,
    is_int,
    is_scalar,
    is_str,
    promote_dtype_to_float,
    should_promote_to_fit,
)

from ._utils import (
    create_null_mask,
    is_dask,
    is_numpy,
    is_numpy_masked,
    is_xarray,
)
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
            data = self._rs.data
            values = data[~self._mask]
            return func(values)

        return method


def _normalize_ndarray_for_ufunc(other, target_shape):
    if isinstance(other, da.Array) and np.isnan(other.size):
        raise ValueError(
            "Arithmetic with dask arrays only works for arrays with "
            "known chunks."
        )
    if other.size == 1:
        return other.ravel()
    try:
        if np.broadcast_shapes(target_shape, other.shape) == target_shape:
            return other
    except ValueError:
        if len(other.shape) == 1 and other.size == target_shape[0]:
            # Map a list/array of values that has same length as # of bands to
            # each band
            return other.reshape((other.size, 1, 1))
    raise ValueError(
        "Received array with incompatible shape in arithmetic: "
        f"{other.shape}"
    )


def _normalize_input_for_ufunc(other, target_shape):
    from raster_tools.raster import Raster

    if isinstance(other, (np.ndarray, da.Array)):
        other = _normalize_ndarray_for_ufunc(other, target_shape)
    elif isinstance(other, (list, tuple, range)):
        other = _normalize_ndarray_for_ufunc(
            np.atleast_1d(other), target_shape
        )
    elif isinstance(other, xr.DataArray):
        other = Raster(other)
    return other


def _merge_null_values(values):
    if all(v is None for v in values):
        return None
    values = [v for v in values if v is not None]
    # Take left most value
    return values[0]


def _apply_ufunc(ufunc, *args, kwargs=None, out=None):
    raster_args = [a for a in args if isinstance(a, Raster)]
    xr_args = [getattr(a, "_rs", a) for a in args]
    kwargs = kwargs or {}

    with xr.set_options(keep_attrs=True):
        ufname = ufunc.__name__
        if ufname.startswith("bitwise") or ufname.startswith("logical"):
            # Extend bitwise operations to non-boolean dtypes by coercing the
            # inputs to boolean.
            tmp = []
            for arg in xr_args:
                if not is_bool(getattr(arg, "dtype", arg)):
                    # TODO: Come to consensus on best coercion operation
                    arg = arg > 0
                tmp.append(arg)
            xr_args = tmp
        xr_out = ufunc(*xr_args, **kwargs)
    xmask = None
    for r in raster_args:
        # Use xarray to align grids
        if xmask is None:
            xmask = xr.DataArray(r._mask, coords=r.xrs.coords, dims=r.xrs.dims)
        elif r._masked:
            xmask |= xr.DataArray(
                r._mask, coords=r.xrs.coords, dims=r.xrs.dims
            )
    mask = xmask.data
    nv = _merge_null_values([r.null_value for r in raster_args])

    if out is not None:
        # Inplace
        return out._replace_inplace(
            xr_out, mask=mask, null_value=nv, null_value_none=nv is None
        )

    if isinstance(xr_out, xr.DataArray):
        return Raster(xr_out)._replace(
            mask=mask, null_value=nv, null_value_none=nv is None
        )

    rs_outs = tuple(
        Raster(o)._replace(
            mask=mask, null_value=nv, null_value_none=nv is None
        )
        for o in xr_out
    )
    return rs_outs


_UNARY_UFUNCS = frozenset((np.absolute, np.invert, np.negative, np.positive))
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
        out = kwargs.get("out", ())
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

        if ufunc in _UNARY_UFUNCS:
            if ufunc == np.invert and is_float(self.dtype):
                raise TypeError("ufunc 'invert' not supported for float types")
            return self._replace(ufunc(self._rs, **kwargs))

        inputs = [_normalize_input_for_ufunc(i, self.shape) for i in inputs]
        return _apply_ufunc(ufunc, *inputs, out=out)

    def __array__(self, dtype=None):
        return self._rs.__array__(dtype)


def _array_to_xarray(ar):
    if len(ar.shape) > 3 or len(ar.shape) < 2:
        raise ValueError(f"Invalid raster shape for numpy array: {ar.shape}")
    if len(ar.shape) == 2:
        # Add band dim
        ar = ar[None]
    if not is_dask(ar):
        ar = da.from_array(ar)
    # The band dimension needs to start at 1 to match raster conventions
    coords = [np.arange(1, ar.shape[0] + 1)]
    # Add 0.5 offset to move coords to center of cell
    coords.extend([np.arange(d) + 0.5 for d in ar.shape[1:]])
    xrs = xr.DataArray(ar, dims=["band", "y", "x"], coords=coords)
    if is_numpy_masked(ar._meta):
        xrs.attrs["_FillValue"] = ar._meta.fill_value
    else:
        if is_float(ar.dtype):
            xrs.attrs["_FillValue"] = np.nan
        else:
            # No way to know null value
            xrs.attrs["_FillValue"] = None
    return xrs


def _try_to_get_null_value_xarray(xrs):
    null = xrs.attrs.get("_FillValue", None)
    if null is not None:
        return null
    nv1 = xrs.rio.nodata
    nv2 = xrs.rio.encoded_nodata
    # All are None, return nan for float and None otherwise
    if all(nv is None for nv in [nv1, nv2]):
        if is_float(xrs.dtype):
            return np.nan
        else:
            return None
    # All are not None, return nv1
    if all(nv is not None for nv in [nv1, nv2]):
        return nv1
    # One is None and the other is not, return the valid null value
    return nv1 if nv1 is not None else nv2


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
    raster : str, Raster, xarray.DataArray, numpy.ndarray
        The raster source to use for this Raster. If `raster` is a string,
        it is treated like a path. If `raster` is a Raster, a copy is made
        and its raster source is used. If `raster` is an xarray DataArray or
        numpy array, it is used as the source.

    """

    __slots__ = ("_rs", "_mask")

    def __init__(self, raster):
        self._mask = None
        self._rs = None

        if isinstance(raster, Raster):
            self._rs = raster._rs.copy()
            self._mask = raster._mask.copy()
            self._null_value = raster._null_value
        elif is_xarray(raster):
            if isinstance(raster, xr.Dataset):
                raise TypeError("Unable to handle xarray.Dataset objects")
            raster = normalize_xarray_data(raster)
            if dask.is_dask_collection(raster):
                raster = chunk(raster)
            self._rs = raster
            null = _try_to_get_null_value_xarray(raster)
            self._mask = create_null_mask(self._rs, null)
            self._null_value = null
        elif is_numpy(raster) or is_dask(raster):
            # Copy to take ownership
            raster = chunk(_array_to_xarray(raster.copy()))
            self._rs = normalize_xarray_data(raster)
            nv = raster.attrs.get("_FillValue", None)
            self._mask = create_null_mask(self._rs, nv)
            self._null_value = nv
        elif type(raster) in IO_UNDERSTOOD_TYPES:
            if is_batch_file(raster):
                # Import here to avoid circular import errors
                from raster_tools.batch import parse_batch_script

                rs = parse_batch_script(raster).final_raster
                self._rs = rs._rs
                self._mask = rs._mask
                self._null_value = rs._null_value
            else:
                rs, mask, nv = open_raster_from_path(raster)
                self._rs = rs
                self._mask = mask
                self._null_value = nv
        else:
            raise TypeError(f"Could not resolve input to a raster: {raster!r}")

    def __repr__(self):
        # TODO: implement
        return repr(self._rs)

    def _replace(
        self,
        new_xrs=None,
        attrs=None,
        mask=None,
        null_value=None,
        null_value_none=False,
    ):
        if new_xrs is not None:
            new_rs = Raster(new_xrs)
        else:
            new_rs = self.copy()
        new_rs._attrs = attrs or self._attrs
        new_rs._mask = mask if mask is not None else self._mask
        if new_rs._data.chunks != new_rs._mask.chunks:
            new_rs._mask = da.rechunk(new_rs._mask, new_rs._data.chunks)
        if not null_value_none:
            new_rs._null_value = (
                null_value if null_value is not None else self.null_value
            )
        else:
            new_rs._null_value = None
        return new_rs

    def _replace_inplace(
        self,
        new_xrs,
        attrs=None,
        mask=None,
        null_value=None,
        null_value_none=False,
    ):
        old_attrs = self._attrs
        old_nv = self.null_value
        self._rs = new_xrs
        self._attrs = attrs or old_attrs
        self._mask = mask if mask is not None else self._mask
        if not null_value_none:
            self._null_value = null_value if null_value is not None else old_nv
        else:
            self._null_value = None
        return self

    def _to_presentable_xarray(self):
        """Returns a DataArray with null locations filled by the null value."""
        xrs = self._rs
        if self._masked:
            xrs = xrs.where(~self._mask, self.null_value)
        return xrs

    @property
    def _attrs(self):
        # Dict containing raster metadata like projection, etc.
        return self._rs.attrs.copy()

    @_attrs.setter
    def _attrs(self, attrs):
        if attrs is not None and isinstance(attrs, abc.Mapping):
            self._rs.attrs = attrs.copy()
        else:
            raise TypeError("attrs cannot be None and must be mapping type")

    @property
    def _masked(self):
        return self.null_value is not None

    @property
    def _values(self):
        """The raw internal values. Note: this triggers computation."""
        return self._rs.values

    @property
    def xrs(self):
        """The underlying :class:`xarray.DataArray` (read only)"""
        return self._rs

    @property
    def pxrs(self):
        """Plottable xrs. Same as :attr:`~Raster.xrs` but null cells are filled
        with NaN.

        This makes for nicer plots when calling :meth:`xarray.DataArray.plot`.
        """
        if not self._masked:
            return self.xrs
        return xr.where(self._mask, np.nan, self._rs)

    @property
    def _data(self):
        return self.xrs.data

    @_data.setter
    def _data(self, data):
        self.xrs.data = data

    @property
    def _null_value(self):
        return self._rs.attrs.get("_FillValue", None)

    @_null_value.setter
    def _null_value(self, value):
        if value is not None and not is_scalar(value):
            raise TypeError("Null value must be a scalar or None")
        self._rs.attrs["_FillValue"] = value

    @property
    def null_value(self):
        """The raster's null value used to fill missing or invalid entries."""
        return self._null_value

    @property
    def dtype(self):
        """The dtype of the data."""
        return self._rs.dtype

    @property
    def shape(self):
        """The shape of the underlying raster. Dim 0 is the band dimension.

        The shape will always be of the form ``(B, Y, X)`` where ``B`` is the
        band dimension, ``Y`` is the y dimension, and ``X`` is the x dimension.

        """
        return self._rs.shape

    @property
    def crs(self):
        """The raster's CRS.

        This is a :obj:`rasterio.crs.CRS` object."""
        return self._rs.rio.crs

    @property
    def affine(self):
        """The affine transformation for the raster data.

        This is an :obj:`affine.Affine` object.

        """
        return self._rs.rio.transform(True)

    @property
    def resolution(self):
        """The x and y cell sizes as a tuple. Values are always positive."""
        return self._rs.rio.resolution(True)

    @property
    def bounds(self):
        """Bounds tuple (minx, miny, maxx, maxy)"""
        minx, miny = self.xy(0, 0, offset="ll")
        r, c = self.shape[1:]
        maxx, maxy = self.xy(r - 1, c - 1, "ur")
        return (minx, miny, maxx, maxy)

    def to_dask(self):
        """Returns the underlying data as a dask array."""
        rs = self
        if not dask.is_dask_collection(self._rs):
            rs = self._replace(chunk(self._rs))
        return rs._data

    def close(self):
        """Close the underlying source"""
        self._rs.close()

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
        xrs = rs._rs
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
        rs = chunk(self.xrs.compute())
        mask = da.from_array(self._mask.compute())
        # A new raster is returned to mirror the xarray and dask APIs
        return self._replace(rs, mask=mask)

    def copy(self):
        """Returns a copy of this Raster."""
        return Raster(self)

    def astype(self, dtype):
        """Return a copy of the Raster cast to the specified type.

        Parameters
        ----------
        dtype : str, type, numpy.dtype
            The new type to cast the raster to. The null value will also be
            cast to this type.

        Returns
        -------
        Raster
            The new `dtype` raster.

        """
        if isinstance(dtype, str):
            dtype = dtype.lower()
        if dtype not in DTYPE_INPUT_TO_DTYPE:
            raise ValueError(f"Unsupported type: '{dtype}'")
        dtype = DTYPE_INPUT_TO_DTYPE[dtype]

        if dtype == self.dtype:
            return self.copy()

        xrs = self.xrs
        nv = self.null_value
        mask = self._mask
        if self._masked:
            if is_float(xrs.dtype) and is_int(dtype):
                if np.isnan(nv):
                    nv = get_default_null_value(dtype)
                    warnings.warn(
                        f"Null value is NaN but new dtype is {dtype},"
                        f" using default null value for that dtype: {nv}",
                        RuntimeWarning,
                    )
                    # Reset mask just to be safe
                    mask = np.isnan(xrs)
                    xrs = xrs.fillna(nv)
        xrs = xrs.astype(dtype)
        return self._replace(xrs, mask=mask, null_value=nv)

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
        n_bands, *_ = self.shape
        if is_int(bands):
            bands = [bands]
        if not all(is_int(b) for b in bands):
            raise TypeError("All band numbers must be integers")
        bands = list(bands)
        if len(bands) == 0:
            raise ValueError("No bands provided")
        if any(b < 1 or b > n_bands for b in bands):
            raise IndexError(
                f"One or more band numbers were out of bounds: {bands}"
            )
        bands = [b - 1 for b in bands]
        if len(bands) == 1 and n_bands == 1:
            return self.copy()
        rs = self.xrs[bands]
        rs["band"] = list(range(1, len(rs) + 1))
        mask = self._mask[bands]
        # TODO: look into making attrs consistant with bands
        return self._replace(rs, mask=mask)

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
        xrs = self.xrs.rio.write_crs(crs)
        return self._replace(xrs, attrs=xrs.attrs)

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
        if value is not None and not is_scalar(value):
            raise TypeError(f"Value must be a scalar or None: {value}")

        xrs = self.xrs.copy()
        # Cast up to float if needed
        if should_promote_to_fit(self.dtype, value):
            xrs = xrs.astype(promote_dtype_to_float(self.dtype))

        if value is None:
            mask = create_null_mask(xrs, value)
            return self._replace(xrs, mask=mask, null_value_none=True)

        # Update mask
        mask = self._mask
        temp_mask = (
            np.isnan(xrs.data) if np.isnan(value) else xrs.data == value
        )
        if self._masked:
            mask = mask | temp_mask
        else:
            mask = temp_mask
        return self._replace(xrs, mask=mask, null_value=value)

    def burn_mask(self):
        """Fill null-masked cells with null value.

        Use this as a way to return the underlying data to a known state. If
        dtype is boolean, the null cells are set to ``False`` instead of
        promoting to fit the null value.
        """
        if not self._masked:
            return self
        nv = self.null_value
        if is_bool(self.dtype):
            nv = False
        xrs = xr.where(self._mask, nv, self._rs)
        return self._replace(xrs)

    def to_null_mask(self):
        """
        Returns a boolean Raster with True at null values and False otherwise.

        Returns
        -------
        Raster
            The resulting mask Raster. It is True where this raster contains
            null values and False everywhere else.

        """
        xrs = self.xrs.copy()
        xrs.data = self._mask.copy()
        return self._replace(xrs, null_value=True)

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

        xrs = self.xrs
        if should_promote_to_fit(xrs.dtype, value):
            xrs = xrs.astype(promote_dtype_to_float(xrs.dtype))
        if self._masked:
            xrs = xrs.where(~self._mask, value)
        mask = da.zeros_like(xrs.data, dtype=bool)
        return self._replace(xrs, mask=mask)

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
        condition = get_raster(condition)
        if not is_scalar(other):
            try:
                other = get_raster(other)
            except TypeError:
                raise TypeError(
                    f"Could not understand other argument. Got: {other!r}"
                )

        if not is_bool(condition.dtype) and not is_int(condition.dtype):
            raise TypeError(
                "Condition argument must be a boolean or integer raster"
            )

        xrs = self.xrs
        other_arg = other.xrs if isinstance(other, Raster) else other
        xcondition = condition.xrs.copy()
        mask = condition._mask
        if is_int(condition.dtype):
            # if condition.dtype is not bool then must be an int raster so
            # assume that condition is raster of 0 and 1 values.
            # condition > 0 will grab all 1/True values.
            xcondition = xcondition > 0
        xrs = xrs.where(xcondition, other_arg)
        # Drop null cells from both the condition raster and this
        if condition._masked or self._masked:
            mask = mask | self._mask
            nv = self.null_value if self._masked else condition.null_value
            # Fill null areas
            xmask = xr.DataArray(mask, dims=xrs.dims, coords=xrs.coords)
            xrs = xrs.where(~xmask, nv)
        else:
            mask = create_null_mask(xrs, None)
        return self._replace(xrs, mask=mask)

    def remap_range(self, mapping):
        """Remaps values in a range [`min`, `max`) to a `new_value`.

        Mappings are applied all at once with earlier mappings taking
        precedence.

        Parameters
        ----------
        mapping : 3-tuple of scalars or list of 3-tuples of scalars
            A tuple or list of tuples containing ``(min, max, new_value)``
            scalars. The mappiing(s) map values between the min (inclusive) and
            max (exclusive) to the ``new_value``. If `mapping` is a list and
            there are mappings that conflict or overlap, earlier mappings take
            precedence.

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

        return remap_range(self, mapping)

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
        return self._replace(self._rs.round(decimals=decimals))

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
        xrs = self.xrs
        mask = self._mask
        data_delayed = xrs.data.to_delayed()
        chunks_shape = data_delayed.shape
        data_delayed = data_delayed.ravel()
        mask_delayed = mask.to_delayed().ravel()

        # Mapping from dim name to list of slices for coordinate arrays
        slices = {}
        for d, r in xrs.chunksizes.items():
            r = np.add.accumulate([0, *r])
            s = []
            for i in range(len(r) - 1):
                s.append(slice(*r[i : i + 2]))
            slices[d] = s
        x = xrs.x.data
        y = xrs.y.data
        # See issue for inspiration: https://github.com/dask/dask/issues/7589
        # Group chunk data with corresponding coordinate data
        chunks = []
        for k, (d, m) in enumerate(zip(data_delayed, mask_delayed)):
            # band index, chunk y index, chunk x index
            b, i, j = np.unravel_index(k, chunks_shape)
            yslice = y[slices["y"][i]]
            xslice = x[slices["x"][j]]
            chunks.append((d, m, xslice, yslice, b + 1, self.crs, self.affine))

        meta = gpd.GeoDataFrame(
            {
                "value": [self.dtype.type(1)],
                "band": [1],
                "row": [0],
                "col": [0],
                "geometry": [Point(x[0], y[0])],
            },
            crs=self.crs,
        )
        results = [
            dd.from_delayed(_vectorize(*chunk), meta=meta) for chunk in chunks
        ]
        return dd.concat(results)


_XY_OFFSET_REMAP = {"ul": "ll", "ll": "ul", "ur": "lr", "lr": "ur"}


def rowcol_to_xy(row, col, affine, offset):
    """
    Convert (row, col) index values to (x, y) coords using the transformation.
    """
    # Invert the north/south dim so that upper always gives north and lower
    # gives south
    if offset in _XY_OFFSET_REMAP:
        offset = _XY_OFFSET_REMAP[offset]
    result = rio.transform.xy(affine, row, col, offset=offset)
    if is_scalar(row):
        return result
    return tuple(np.array(v) for v in result)


def xy_to_rowcol(x, y, affine):
    """
    Convert (x, y) coords to (row, col) index values using the transformation.
    """
    result = rio.transform.rowcol(affine, x, y)
    if is_scalar(x):
        return result
    return tuple(np.array(v) for v in result)


@jit(nopython=True, nogil=True, cache=True)
def _extract_points(mask, cx, cy):
    shape = mask.shape
    rx = []
    ry = []
    for i in range(shape[1]):
        for j in range(shape[2]):
            if mask[0, i, j]:
                continue
            rx.append(cx[j])
            ry.append(cy[i])
    return (rx, ry)


@jit(nopython=True, nogil=True, cache=True)
def _extract_values(data, mask):
    shape = data.shape
    results = []
    for i in range(shape[1]):
        for j in range(shape[2]):
            if mask[0, i, j]:
                continue
            results.append(data[0, i, j])
    return results


@dask.delayed
def _vectorize(data, mask, cx, cy, band, crs, affine):
    xpoints, ypoints = _extract_points(mask, cx, cy)
    if len(xpoints):
        values = _extract_values(data, mask)
        points = [Point(x, y) for x, y in zip(xpoints, ypoints)]
        rows, cols = xy_to_rowcol(xpoints, ypoints, affine)
        bands = [band] * len(values)
    else:
        values = []
        points = []
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
        data = rs._data
        new_dtype = promote_dtype_to_float(data.dtype)
        if new_dtype != data.dtype:
            data = data.astype(new_dtype)
        rs._data = da.where(rs._mask, np.nan, data)
    return rs
