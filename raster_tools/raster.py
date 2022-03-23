import operator
import warnings
from collections import abc

import dask
import dask.array as da
import numpy as np
import xarray as xr

from raster_tools.dtypes import (
    BOOL,
    DTYPE_INPUT_TO_DTYPE,
    F16,
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


def _is_using_dask(raster):
    rs = raster.xrs if isinstance(raster, Raster) else raster
    return dask.is_dask_collection(rs)


class RasterDeviceMismatchError(BaseException):
    pass


class RasterDeviceError(BaseException):
    pass


class RasterNoDataError(BaseException):
    pass


_BINARY_ARITHMETIC_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
    "%": operator.mod,
}
_BINARY_LOGICAL_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
}
_BINARY_BITWISE_OPS = {
    "&": operator.and_,
    "|": operator.or_,
}


def _map_chunk_function(raster, func, args, **kwargs):
    """Map a function to the dask chunks of a raster."""
    if _is_using_dask(raster):
        raster._data = raster._data.map_blocks(func, args, **kwargs)
    else:
        raster._data = func(raster._data, args)
    return raster


def _gt0(x):
    return x > 0


def _cast_to_bool(x):
    if is_scalar(x):
        return bool(x)
    return x.astype(BOOL)


def _coerce_to_bool_for_and_or(operands, to_bool_op):
    if is_str(to_bool_op):
        if to_bool_op == "gt0":
            to_bool_op = _gt0
        elif to_bool_op == "cast":
            to_bool_op = _cast_to_bool
        else:
            raise ValueError("Unknown conversion to bool")
    elif not callable(to_bool_op):
        raise TypeError("Invalid conversion to bool")

    boperands = []
    for opr in operands:
        if not is_scalar(opr) and (is_bool(opr) or is_bool(opr.dtype)):
            boperands.append(opr)
        else:
            boperands.append(to_bool_op(opr))
    return boperands


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
    coords.extend([np.arange(d) for d in ar.shape[1:]])
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


class Raster:
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
            if _is_using_dask(raster):
                raster = chunk(raster)
            self._rs = normalize_xarray_data(raster)
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

    def __array__(self, dtype=None):
        return self._rs.__array__(dtype)

    def _replace(
        self,
        new_xrs,
        attrs=None,
        mask=None,
        null_value=None,
        null_value_none=False,
    ):
        new_rs = Raster(new_xrs)
        new_rs._attrs = attrs or self._attrs
        new_rs._mask = mask if mask is not None else self._mask
        if not null_value_none:
            new_rs._null_value = (
                null_value if null_value is not None else self.null_value
            )
        else:
            new_rs._null_value = None
        return new_rs

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
        """The underlying xarray DataArray"""
        return self._rs

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

    def to_dask(self):
        """Returns the underlying data as a dask array."""
        rs = self
        if not _is_using_dask(self):
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

    def round(self, round_null_value=True):
        """Round the data to the nearest integer value. Return a new Raster.

        Parameters
        ----------
        round_null_value : bool, optional
            If ``True``, the resulting raster will have its null value rounded
            as well. Default is ``True``,

        Returns
        -------
        Raster
            The resulting rounded raster.

        """
        xrs = self.xrs
        if self._masked:
            xrs = xr.where(~self._mask, xrs.round(), self.null_value)
        else:
            xrs = xrs.round()
        nv = self.null_value
        if round_null_value and self._masked:
            nv = np.round(nv)
        return self._replace(xrs, null_value=nv)

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

    def _input_to_raster(self, raster_input):
        if isinstance(raster_input, Raster):
            raster = raster_input
        else:
            raster = Raster(raster_input)
        if raster.xrs.size == 0:
            raise ValueError(
                f"Input raster is empty with shape {raster.xrs.shape}"
            )
        return raster

    def _handle_binary_op_input(self, raster_or_scalar, xarray=True):
        if is_scalar(raster_or_scalar):
            operand = raster_or_scalar
        else:
            operand = self._input_to_raster(raster_or_scalar)
            if xarray:
                operand = operand.xrs
        return operand

    def _binary_arithmetic(self, raster_or_scalar, op, swap=False):
        # TODO: handle mapping of list of values to bands
        # TODO: handle case where shapes match but geo references don't
        if op not in _BINARY_ARITHMETIC_OPS:
            raise ValueError(f"Unknown arithmetic operation: '{op}'")
        operand = self._handle_binary_op_input(raster_or_scalar, False)
        parsed_operand = raster_or_scalar
        if isinstance(operand, Raster):
            parsed_operand = operand
            operand = operand.xrs
        left, right = self.xrs, operand
        if swap:
            left, right = right, left
        xrs = _BINARY_ARITHMETIC_OPS[op](left, right)

        mask = self._mask
        if isinstance(parsed_operand, Raster):
            mask = mask | parsed_operand._mask

        return self._replace(xrs, mask=mask)

    def _binary_logical(self, raster_or_scalar, op):
        if op not in _BINARY_LOGICAL_OPS:
            raise ValueError(f"Unknown arithmetic operation: '{op}'")
        operand = self._handle_binary_op_input(raster_or_scalar, False)
        parsed_operand = raster_or_scalar
        if isinstance(operand, Raster):
            parsed_operand = operand
            operand = operand.xrs
        xrs = _BINARY_LOGICAL_OPS[op](self.xrs, operand)

        mask = self._mask
        if isinstance(parsed_operand, Raster):
            mask = mask | parsed_operand._mask
        return self._replace(xrs, mask=mask)

    def add(self, raster_or_scalar):
        """Add this Raster with another Raster or scalar.

        Parameters
        ----------
        raster_or_scalar : scalar, str, or Raster
            The scalar or Raster to add to this raster.

        Returns
        -------
        Raster
            Returns the resulting Raster.

        """
        return self._binary_arithmetic(raster_or_scalar, "+")

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def subtract(self, raster_or_scalar):
        """Subtract another Raster or scalar from This Raster.

        Parameters
        ----------
        raster_or_scalar : scalar, str, or Raster
            The scalar or Raster to subtract from this raster.

        Returns
        -------
        Raster
            Returns the resulting Raster.

        """
        return self._binary_arithmetic(raster_or_scalar, "-")

    def __sub__(self, other):
        return self.subtract(other)

    def __rsub__(self, other):
        return self.negate().add(other)

    def multiply(self, raster_or_scalar):
        """Multiply this Raster with another Raster or scalar.

        Parameters
        ----------
        raster_or_scalar : scalar, str, or Raster
            The scalar or Raster to multiply with this raster.

        Returns
        -------
        Raster
            Returns the resulting Raster.

        """
        return self._binary_arithmetic(raster_or_scalar, "*")

    def __mul__(self, other):
        return self.multiply(other)

    def __rmul__(self, other):
        return self.multiply(other)

    def divide(self, raster_or_scalar):
        """Divide this Raster by another Raster or scalar.

        Parameters
        ----------
        raster_or_scalar : scalar, str, or Raster
            The scalar or Raster to divide this raster by.

        Returns
        -------
        Raster
            Returns the resulting Raster.

        """
        return self._binary_arithmetic(raster_or_scalar, "/")

    def __truediv__(self, other):
        return self.divide(other)

    def __rtruediv__(self, other):
        return self._binary_arithmetic(other, "/", swap=True)

    def mod(self, raster_or_scalar):
        """Perform the modulo operation on this Raster with another Raster or
        scalar.

        Parameters
        ----------
        raster_or_scalar : scalar, str, or Raster
            The scalar or Raster to mod this raster with.

        Returns
        -------
        Raster
            Returns the resulting Raster.

        """
        return self._binary_arithmetic(raster_or_scalar, "%")

    def __mod__(self, other):
        return self.mod(other)

    def __rmod__(self, other):
        return self._binary_arithmetic(other, "%", swap=True)

    def pow(self, value):
        """
        Raise this raster by another Raster or scalar.

        Parameters
        ----------
        raster_or_scalar : scalar, str, or Raster
            The scalar or Raster to raise this raster to.

        Returns
        -------
        Raster
            Returns the resulting Raster.

        """
        return self._binary_arithmetic(value, "**")

    def __pow__(self, value):
        return self.pow(value)

    def __rpow__(self, value):
        return self._binary_arithmetic(value, "**", swap=True)

    def sqrt(self):
        """Take the square root of the raster.

        Returns a new Raster.

        """
        xrs = np.sqrt(self.xrs)
        return self._replace(xrs)

    def __pos__(self):
        return self

    def negate(self):
        """Negate this Raster.

        Returns a new Raster.

        """
        return self._replace(-self.xrs)

    def __neg__(self):
        return self.negate()

    def log(self):
        """Take the natural logarithm of this Raster.

        Returns a new Raster.

        """
        xrs = np.log(self.xrs)
        return self._replace(xrs)

    def log10(self):
        """Take the base-10 logarithm of this Raster.

        Returns a new Raster.

        """
        xrs = np.log10(self.xrs)
        return self._replace(xrs)

    def eq(self, other):
        """
        Perform element-wise equality test against `other`.

        Parameters
        ----------
        other : scalar, str, or Raster
            The value or raster to compare with.

        Returns
        -------
        Raster
            The resulting boolean raster.

        """
        return self._binary_logical(other, "==")

    def __eq__(self, other):
        return self.eq(other)

    def ne(self, other):
        """Perform element-wise not-equal test against `other`.

        Parameters
        ----------
        other : scalar, str, or Raster
            The value or raster to compare with.

        Returns
        -------
        Raster
            The resulting boolean raster.

        """
        return self._binary_logical(other, "!=")

    def __ne__(self, other):
        return self.ne(other)

    def le(self, other):
        """Perform element-wise less-than-or-equal test against `other`.

        Parameters
        ----------
        other : scalar, str, or Raster
            The value or raster to compare with.

        Returns
        -------
        Raster
            The resulting boolean raster.

        """
        return self._binary_logical(other, "<=")

    def __le__(self, other):
        return self.le(other)

    def ge(self, other):
        """Perform element-wise greater-than-or-equal test against `other`.

        Parameters
        ----------
        other : scalar, str, or Raster
            The value or raster to compare with.

        Returns
        -------
        Raster
            The resulting boolean raster.

        """
        return self._binary_logical(other, ">=")

    def __ge__(self, other):
        return self.ge(other)

    def lt(self, other):
        """Perform element-wise less-than test against `other`.

        Parameters
        ----------
        other : scalar, str, or Raster
            The value or raster to compare with.

        Returns
        -------
        Raster
            The resulting boolean raster.

        """
        return self._binary_logical(other, "<")

    def __lt__(self, other):
        return self.lt(other)

    def gt(self, other):
        """Perform element-wise greater-than test against `other`.

        Parameters
        ----------
        other : scalar, str, or Raster
            The value or raster to compare with.

        Returns
        -------
        Raster
            The resulting boolean raster.

        """
        return self._binary_logical(other, ">")

    def __gt__(self, other):
        return self.gt(other)

    def _and_or(self, other, and_or, to_bool_op):
        if isinstance(other, (bool, np.bool_)):
            operand = other
        else:
            operand = self._handle_binary_op_input(other, False)
        parsed_operand = operand
        if isinstance(operand, Raster):
            operand = operand.xrs
        left, right = _coerce_to_bool_for_and_or(
            [self.xrs, operand], to_bool_op
        )
        xrs = _BINARY_BITWISE_OPS[and_or](left, right).astype(F16)

        mask = self._mask
        if isinstance(parsed_operand, Raster):
            mask = mask | parsed_operand._mask
        return self._replace(xrs, mask=mask)

    def and_(self, other, to_bool_op="gt0"):
        """Returns this Raster and'd with another. Both are coerced to bools
        according to `to_bool_op`.

        Parameters
        ----------
        other : Raster or path or bool or scalar
            The raster to and this raster with
        to_bool_op : {'gt0', 'cast'} or callable, optional
            Controls how the two rasters are coerced to dtype bool. If a
            callable, to_bool_op is called on this raster and `other`
            separately to convert them to bool types. For a str:

            'gt0'
                The two operands are compared against 0 using greater-than.
                Default.
            'cast'
                The two operands are cast to bool.

        Returns
        -------
        Raster
            The resulting Raster of zeros and ones.

        """
        return self._and_or(other, "&", to_bool_op)

    def __and__(self, other):
        return self._and_or(other, "&", "gt0")

    def __rand__(self, other):
        return self._and_or(other, "&", "gt0")

    def or_(self, other, to_bool_op="gt0"):
        """
        Returns this Raster or'd with another. Both are coerced to bools
        according to `to_bool_op`.

        Parameters
        ----------
        other : Raster or path or bool or scalar
            The raster to and this raster with
        to_bool_op : {'gt0', 'cast'} or callable, optional
            Controls how the two rasters are coerced to dtype bool. If a
            callable, to_bool_op is called on this raster and `other`
            separately to convert them to bool types. For a str:

            'gt0'
                The two operands are compared against 0 using greater-than.
                Default.
            'cast'
                The two operands are cast to bool.

        Returns
        -------
        Raster
            The resulting Raster of zeros and ones.
        """
        return self._and_or(other, "|", to_bool_op)

    def __or__(self, other):
        return self._and_or(other, "|", "gt0")

    def __ror__(self, other):
        return self._and_or(other, "|", "gt0")

    def __invert__(self):
        if is_bool(self.dtype) or is_int(self.dtype):
            xrs = self.xrs.copy()
            if self._masked:
                xrs.data = da.where(self._mask, self.null_value, ~xrs.data)
            else:
                xrs = ~xrs
            return self._replace(xrs)
        if is_float(self.dtype):
            raise TypeError(
                "Bitwise complement operation not supported for floating point"
                " rasters"
            )
        raise TypeError(
            "Bitwise complement operation not supported for this raster dtype"
        )
