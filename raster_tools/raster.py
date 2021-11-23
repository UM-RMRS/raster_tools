import collections
import operator
import os
import re
import warnings

import dask
import dask.array as da
import numpy as np
import xarray as xr

try:
    import cupy as cp

    GPU_ENABLED = True
except (ImportError, ModuleNotFoundError):
    GPU_ENABLED = False

from ._types import (
    BOOL,
    DTYPE_INPUT_TO_DTYPE,
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    promote_dtype_to_float,
    should_promote_to_fit,
)
from ._utils import (
    create_null_mask,
    is_bool,
    is_dask,
    is_float,
    is_int,
    is_numpy,
    is_numpy_masked,
    is_scalar,
    is_str,
    is_xarray,
    validate_file,
)
from .io import (
    IO_UNDERSTOOD_TYPES,
    chunk,
    is_batch_file,
    normalize_xarray_data,
    open_raster_from_path,
    write_raster,
)


def is_raster_class(value):
    return isinstance(value, Raster)


def _is_using_dask(raster):
    rs = raster._rs if is_raster_class(raster) else raster
    return dask.is_dask_collection(rs)


class RasterDeviceMismatchError(BaseException):
    pass


class RasterDeviceError(BaseException):
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
        raster._rs.data = raster._rs.data.map_blocks(func, args, **kwargs)
    else:
        raster._rs.data = func(raster._rs.data, args)
    return raster


def _chunk_to_cpu(chunk, *args):
    try:
        return chunk.get()
    except AttributeError:
        # assume chunk is already on the cpu
        return chunk


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


GPU = "gpu"
CPU = "cpu"


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


def _dask_from_array_with_device(arr, device):
    arr = da.from_array(arr)
    if GPU_ENABLED and device == GPU:
        arr = arr.map_blocks(cp.asarray)
    return arr


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


DTYPE_TO_DEFAULT_NULL = {
    BOOL: True,
    U8: 99,
    U16: 9999,
    U32: 999999,
    U64: 999999,
    I8: -99,
    I16: -9999,
    I32: -999999,
    I64: -999999,
    F16: -999999.0,
    F32: -999999.0,
    F64: -999999.0,
}


def get_default_null_value(dtype):
    """Get the default null value for a given dtype."""
    return DTYPE_TO_DEFAULT_NULL[dtype]


def _raster_like(
    orig_rs,
    new_xrs,
    attrs=None,
    device=None,
    mask=None,
    null_value=None,
    null_value_none=False,
):
    new_rs = Raster(new_xrs)
    new_rs._attrs = attrs or orig_rs._attrs
    new_rs._set_device(device or orig_rs.device)
    new_rs._mask = mask if mask is not None else orig_rs._mask
    if not null_value_none:
        new_rs._null_value = (
            null_value if null_value is not None else orig_rs.null_value
        )
    else:
        new_rs._null_value = None
    return new_rs


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
        self._device = CPU
        self._mask = None
        self._rs = None

        if is_raster_class(raster):
            self._rs = raster._rs.copy()
            self._set_device(raster.device)
            self._mask = raster._mask.copy()
            self._null_value = raster._null_value
        elif is_xarray(raster):
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
                rs = BatchScript(raster).parse().final_raster
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
        if attrs is not None and isinstance(attrs, collections.Mapping):
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
    def device(self):
        """The hardware device where the raster is processed, `CPU` or `GPU`"""
        return self._device

    def _set_device(self, dev):
        if dev not in (CPU, GPU):
            raise RasterDeviceError(f"Unknown device: '{dev}'")
        if not GPU_ENABLED and dev == GPU:
            raise RasterDeviceError("GPU support is not enabled")
        self._device = dev

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

    def to_xarray(self):
        """Returns the underlying data as an xarray.DataArray.

        This may be a reference to the underlying datastructure so changes made
        to the resulting DataArray may also affect this Raster.

        """
        return self._rs

    def to_dask(self):
        """Returns the underlying data as a dask array."""
        rs = self
        if not _is_using_dask(self):
            rs = _raster_like(self, chunk(self._rs))
        return rs._rs.data

    def close(self):
        """Close the underlying source"""
        self._rs.close()

    def save(
        self,
        path,
        no_data_value=None,
        **gdal_kwargs,
    ):
        """Compute the final raster and save it to the provided location.

        Parameters
        ----------
        path : str
            The target location to save the raster to.
        no_data_value : scalar, optional
            A new null value to use when saving.
        **rio_gdal_kwargs : kwargs, optional
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
        rs = self._rs.compute()
        mask = self._mask.compute()
        # A new raster is returned to mirror the xarray and dask APIs
        return _raster_like(self, rs, mask=mask)

    def to_lazy(self):
        """Convert a non-lazy Raster to a lazy one.

        If this Raster is already lazy, a copy is returned.

        Returns
        -------
        Raster
            The new lazy Raster or a copy of the already lazy Raster

        """
        if _is_using_dask(self._rs):
            return self.copy()
        return _raster_like(self, chunk(self._rs))

    def copy(self):
        """Returns a copy of this Raster."""
        return Raster(self)

    def gpu(self):
        if not GPU_ENABLED:
            raise RasterDeviceError("GPU support is not enabled")
        if self.device == GPU:
            return self
        rs = _map_chunk_function(self.copy(), cp.asarray, args=None)
        rs._set_device(GPU)
        return rs

    def _check_device_mismatch(self, other):
        if is_scalar(other):
            return
        if is_raster_class(other) and (self.device == other.device):
            return
        raise RasterDeviceMismatchError(
            f"Raster devices must match: {self.device} != {other.device}"
        )

    def cpu(self):
        if self.device == CPU:
            return self
        rs = _map_chunk_function(self.copy(), _chunk_to_cpu, args=None)
        rs._set_device(CPU)
        return rs

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

        xrs = self._rs
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
        return _raster_like(self, xrs, mask=mask, null_value=nv)

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
        xrs = self._rs
        if self._masked:
            xrs = xr.where(~self._mask, xrs.round(), self.null_value)
        else:
            xrs = xrs.round()
        nv = self.null_value
        if round_null_value and self._masked:
            nv = np.round(nv)
        return _raster_like(self, xrs, null_value=nv)

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
            return self
        rs = self._rs[bands]
        mask = self._mask[bands]
        # TODO: look into making attrs consistant with bands
        return _raster_like(self, rs, mask=mask)

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
        xrs = self._rs.rio.write_crs(crs)
        return _raster_like(self, xrs, attrs=xrs.attrs)

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

        xrs = self._rs.copy()
        # Cast up to float if needed
        if should_promote_to_fit(self.dtype, value):
            xrs = xrs.astype(promote_dtype_to_float(self.dtype))

        if value is None:
            mask = create_null_mask(xrs, value)
            return _raster_like(self, xrs, mask=mask, null_value_none=True)

        # Update mask
        mask = self._mask
        temp_mask = (
            np.isnan(xrs.data) if np.isnan(value) else xrs.data == value
        )
        if self._masked:
            mask = mask | temp_mask
        else:
            mask = temp_mask
        return _raster_like(self, xrs, mask=mask, null_value=value)

    def to_null_mask(self):
        """
        Returns a boolean Raster with True at null values and False otherwise.

        Returns
        -------
        Raster
            The resulting mask Raster. It is True where this raster contains
            null values and False everywhere else.

        """
        xrs = self._rs.copy()
        xrs.data = self._mask.copy()
        return _raster_like(self, xrs, null_value=True)

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

        xrs = self._rs
        if should_promote_to_fit(xrs.dtype, value):
            xrs = xrs.astype(promote_dtype_to_float(xrs.dtype))
        if self._masked:
            xrs = xrs.where(~self._mask, value)
        mask = da.zeros(xrs.shape, dtype=bool)
        return _raster_like(self, xrs, mask=mask)

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
        if not is_raster_class(condition) and not is_str(condition):
            raise TypeError(
                f"Invalid type for condition argument: {type(condition)}"
            )
        if (
            not is_scalar(other)
            and not is_str(other)
            and not is_raster_class(other)
        ):
            raise TypeError(f"Invalid type for `other`: {type(other)}")
        if is_str(condition):
            condition = Raster(condition)
        if not is_bool(condition.dtype) and not is_int(condition.dtype):
            raise TypeError(
                "Condition argument must be a boolean or integer raster"
            )
        if is_str(other):
            try:
                other = Raster(other)
            except ValueError:
                raise ValueError("Could not resolve other to a raster")

        xrs = self._rs
        other_arg = other._rs if is_raster_class(other) else other
        xcondition = condition._rs.copy()
        mask = condition._mask
        if is_int(condition.dtype):
            # if condition.dtype is not bool then must be an int raster so
            # assume that condition is raster of 0 and 1 values.
            # condition > 0 will grab all 1/True values.
            xcondition = xcondition > 0
        # Drop null cells from both the condition raster and this
        mask = mask | self._mask
        xrs = xr.where(xcondition, xrs, other_arg)
        # Fill null areas
        xrs = xr.where(mask, self.null_value, xrs)
        return _raster_like(self, xrs, mask=mask)

    def remap_range(self, min, max, new_value, *args):
        """Remaps values in the range [`min`, `max`) to `new_value`.

        Parameters
        ----------
        min : scalar
            The minimum value of the mapping range (inclusive).
        max : scalar
            The maximum value of the mapping range (exclusive).
        new_value : scalar
            The new value to map the range to.
        *args : tuple
            Additional remap groups allowing for multiple ranges to be
            remapped. This allows calls like
            `remap_range(0, 12, 0, 12, 20, 1, 20, 30, 2)`. An error is raised
            if the number of additional remap args is not a multiple of 3. The
            remap groups are applied sequentially.

        Returns
        -------
        Raster
            The resulting Raster.

        """
        remaps = [(min, max, new_value)]
        if len(args):
            if len(args) % 3 != 0:
                raise RuntimeError(
                    "Too few additional args to form a remap operation."
                    " Additional args must be in groups of 3."
                )
            remaps += [args[i : i + 3] for i in range(0, len(args), 3)]
        rs = self
        for (min, max, new_value) in remaps:
            if not all([is_scalar(v) for v in (min, max, new_value)]):
                raise TypeError("min, max, and new_value must all be scalars")
            if np.isnan((min, max)).any():
                raise ValueError("min and max cannot be NaN")
            if min >= max:
                raise ValueError(f"min must be less than max: ({min}, {max})")
            condition = rs.copy()
            crs = rs._rs >= min
            crs &= rs._rs < max
            # Invert for where operation
            condition._rs = ~crs
            rs = rs.where(condition, new_value)
        return rs

    def _input_to_raster(self, raster_input):
        if is_raster_class(raster_input):
            self._check_device_mismatch(raster_input)
            raster = raster_input
        else:
            raster = Raster(raster_input)
            if self.device == GPU:
                raster = raster.gpu()
        if raster._rs.size == 0:
            raise ValueError(
                f"Input raster is empty with shape {raster._rs.shape}"
            )
        return raster

    def _handle_binary_op_input(self, raster_or_scalar, xarray=True):
        if is_scalar(raster_or_scalar):
            operand = raster_or_scalar
        else:
            operand = self._input_to_raster(raster_or_scalar)
            if xarray:
                operand = operand._rs
        return operand

    def _binary_arithmetic(self, raster_or_scalar, op, swap=False):
        # TODO: handle mapping of list of values to bands
        # TODO: handle case where shapes match but geo references don't
        if op not in _BINARY_ARITHMETIC_OPS:
            raise ValueError(f"Unknown arithmetic operation: '{op}'")
        operand = self._handle_binary_op_input(raster_or_scalar, False)
        parsed_operand = raster_or_scalar
        if is_raster_class(operand):
            parsed_operand = operand
            operand = operand._rs
        left, right = self._rs, operand
        if swap:
            left, right = right, left
        xrs = _BINARY_ARITHMETIC_OPS[op](left, right)

        mask = self._mask
        if is_raster_class(parsed_operand):
            mask = mask | parsed_operand._mask

        return _raster_like(self, xrs, mask=mask)

    def _binary_logical(self, raster_or_scalar, op):
        if op not in _BINARY_LOGICAL_OPS:
            raise ValueError(f"Unknown arithmetic operation: '{op}'")
        operand = self._handle_binary_op_input(raster_or_scalar, False)
        parsed_operand = raster_or_scalar
        if is_raster_class(operand):
            parsed_operand = operand
            operand = operand._rs
        xrs = _BINARY_LOGICAL_OPS[op](self._rs, operand)

        mask = self._mask
        if is_raster_class(parsed_operand):
            mask = mask | parsed_operand._mask
        return _raster_like(self, xrs, mask=mask)

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
        xrs = np.sqrt(self._rs)
        return _raster_like(self, xrs)

    def __pos__(self):
        return self

    def negate(self):
        """Negate this Raster.

        Returns a new Raster.

        """
        return _raster_like(self, -self._rs)

    def __neg__(self):
        return self.negate()

    def log(self):
        """Take the natural logarithm of this Raster.

        Returns a new Raster.

        """
        xrs = np.log(self._rs)
        return _raster_like(self, xrs)

    def log10(self):
        """Take the base-10 logarithm of this Raster.

        Returns a new Raster.

        """
        xrs = np.log10(self._rs)
        return _raster_like(self, xrs)

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
        if is_raster_class(operand):
            operand = operand._rs
        left, right = _coerce_to_bool_for_and_or(
            [self._rs, operand], to_bool_op
        )
        xrs = _BINARY_BITWISE_OPS[and_or](left, right).astype(F16)

        mask = self._mask
        if is_raster_class(parsed_operand):
            mask = mask | parsed_operand._mask
        return _raster_like(self, xrs, mask=mask)

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
            xrs = self._rs.copy()
            if self._masked:
                xrs.data = da.where(self._mask, self.null_value, ~xrs.data)
            else:
                xrs = ~xrs
            return _raster_like(self, xrs)
        if is_float(self.dtype):
            raise TypeError(
                "Bitwise complement operation not supported for floating point"
                " rasters"
            )
        raise TypeError(
            "Bitwise complement operation not supported for this raster dtype"
        )

    def clip_box(self, minx, miny, maxx, maxy):
        """Clip the raster to the specified box.

        Parameters
        ----------
        minx : scalar
            The minimum x coordinate bound.
        miny : scalar
            The minimum y coordinate bound.
        maxx : scalar
            The maximum x coordinate bound.
        maxy : scalar
            The maximum y coordinate bound.

        Returns
        -------
        Raster
            The clipped raster.

        """
        xrs = self._rs.rio.clip_box(minx, miny, maxx, maxy)
        if self._masked:
            xmask = xr.DataArray(
                self._mask, dims=self._rs.dims, coords=self._rs.coords
            )
            mask = xmask.rio.clip_box(minx, miny, maxx, maxy).data
        else:
            mask = da.zeros_like(xrs.data, dtype=bool)
        # TODO: This will throw a rioxarray.exceptions.MissingCRS exception if
        # no crs is set. Add code to fall back on
        return _raster_like(self, xrs, mask=mask)


class BatchScriptParseError(BaseException):
    pass


def _split_strip(s, delimeter):
    return [si.strip() for si in s.split(delimeter)]


FTYPE_TO_EXT = {
    "TIFF": "tif",
}


_ESRI_OP_TO_OP = {
    "esriRasterPlus": "+",
    "+": "+",
    "esriRasterMinus": "-",
    "-": "-",
    "esriRasterMultiply": "*",
    "*": "*",
    "esriRasterDivide": "/",
    "/": "/",
    "esriRasterMode": "%",
    "%": "%",
    "esriRasterPower": "**",
    "**": "**",
}
_ARITHMETIC_OPS_MAP = {}
_FUNC_PATTERN = re.compile(r"^(?P<func>[A-Za-z]+)\((?P<args>[^\(\)]+)\)$")


class BatchScript:
    def __init__(self, path):
        validate_file(path)
        self.path = os.path.abspath(path)
        self.location = os.path.dirname(self.path)
        self.rasters = {}
        self.final_raster = None

    def parse(self):
        with open(self.path) as fd:
            lines = fd.readlines()
        last_raster = None
        for i, line in enumerate(lines):
            # Ignore comments
            line, *_ = _split_strip(line, "#")
            if not line:
                continue
            lh, rh = _split_strip(line, "=")
            self.rasters[lh] = self._parse_raster(lh, rh, i + 1)
            last_raster = lh
        self.final_raster = self.rasters[last_raster]
        return self

    def _parse_raster(self, dst, expr, line_no):
        mat = _FUNC_PATTERN.match(expr)
        if mat is None:
            raise BatchScriptParseError(
                f"Could not parse function on line {line_no}"
            )
        func = mat["func"].upper()
        args = mat["args"]
        raster = None
        if func == "ARITHMETIC":
            raster = self._arithmetic_args_to_raster(args, line_no)
        elif func == "NULLTOVALUE":
            return self._null_to_value_args_to_raster(args, line_no)
        elif func == "REMAP":
            raster = self._remap_args_to_raster(args, line_no)
        elif func == "COMPOSITE":
            raster = self._composite_args_to_raster(args, line_no)
        elif func == "OPENRASTER":
            raster = Raster(args)
        elif func == "SAVEFUNCTIONRASTER":
            raster = self._save_args_to_raster(args, line_no)
        else:
            raise BatchScriptParseError(
                f"Unknown function on line {line_no}: '{func}'"
            )
        if raster is None:
            raise BatchScriptParseError(f"Could not parse line: {line_no}")
        return raster

    def _arithmetic_args_to_raster(self, args_str, line_no):
        left_arg, right_arg, op = _split_strip(args_str, ";")
        op = _ESRI_OP_TO_OP[op]
        if op not in _ESRI_OP_TO_OP:
            raise BatchScriptParseError(
                f"Unknown arithmetic operation on line {line_no}: '{op}'"
            )
        op = _ESRI_OP_TO_OP[op]
        try:
            left = float(left_arg)
        except ValueError:
            left = self._get_raster(left_arg)
        try:
            right = float(right_arg)
        except ValueError:
            right = self._get_raster(right_arg)
        return left._binary_arithmetic(right, op)

    def _null_to_value_args_to_raster(self, args_str, line_no):
        on_line = f" on line {line_no}"
        left, *right = _split_strip(args_str, ";")
        if len(right) > 1:
            raise BatchScriptParseError(
                "NULLTOVALUE Error: Too many arguments" + on_line
            )
        value = float(right[0])
        return self._get_raster(left).replace_null(value)

    def _remap_args_to_raster(self, args_str, line_no):
        on_line = f" on line {line_no}"
        raster, *args = _split_strip(args_str, ";")
        if len(args) > 1:
            raise BatchScriptParseError(
                "REMAP Error: Too many arguments dividers" + on_line
            )
        args = args[0]
        remaps = []
        for group in _split_strip(args, ","):
            try:
                values = [float(v) for v in _split_strip(group, ":")]
            except ValueError:
                raise BatchScriptParseError(
                    "REMAP Error: values must be numbers" + on_line
                )
            if len(values) != 3:
                raise BatchScriptParseError(
                    "REMAP Error: requires 3 values separated by ':'" + on_line
                )
            left, right, new = values
            if right <= left:
                raise BatchScriptParseError(
                    "REMAP Error: the min value must be less than the max"
                    " value" + on_line
                )
            remaps.append((left, right, new))
        if len(remaps) == 0:
            raise BatchScriptParseError(
                "REMAP Error: No remap values found" + on_line
            )
        args = []
        for group in remaps:
            args.extend(group)
        return self._get_raster(raster).remap_range(*args)

    def _composite_args_to_raster(self, args_str, line_no):
        on_line = f" on line {line_no}"
        rasters = [
            self._get_raster(path) for path in _split_strip(args_str, ";")
        ]
        if len(rasters) < 2:
            raise BatchScriptParseError(
                "COMPOSITE Error: at least 2 rasters are required" + on_line
            )
        return rasters[0].band_concat(rasters[1:])

    def _save_args_to_raster(self, args_str, line_no):
        on_line = f" on line {line_no}"
        # From c# files:
        #  (inRaster;outName;outWorkspace;rasterType;nodata;blockwidth;blockheight)
        # nodata;blockwidth;blockheight are optional
        try:
            in_rs, out_name, out_dir, type_, *extra = _split_strip(
                args_str, ";"
            )
        except ValueError:
            raise BatchScriptParseError(
                "SAVEFUNCTIONRASTER Error: Incorrect number of arguments"
                + on_line
            )
        n = len(extra)
        bwidth = None
        bheight = None
        nodata = 0
        if n >= 1:
            nodata = float(extra[0])
        if n >= 2:
            bwidth = int(extra[1])
        if n == 3:
            bheight = int(extra[2])
        if n > 3:
            raise BatchScriptParseError(
                "SAVEFUNCTIONRASTER Error: Too many arguments" + on_line
            )
        if type_ not in FTYPE_TO_EXT:
            raise BatchScriptParseError(
                "SAVEFUNCTIONRASTER Error: Unknown file type" + on_line
            )
        raster = self._get_raster(in_rs)
        out_name = os.path.join(out_dir, out_name)
        ext = FTYPE_TO_EXT[type_]
        out_name += f".{ext}"
        return raster.save(out_name, nodata, bwidth, bheight)

    def _get_raster(self, name_or_path):
        if name_or_path in self.rasters:
            return self.rasters[name_or_path]
        else:
            # Handle relative paths. Assume they are relative to the batch file
            if not os.path.isabs(name_or_path):
                name_or_path = os.path.join(self.location, name_or_path)
            validate_file(name_or_path)
            return Raster(name_or_path)
