import collections
import dask
import dask.array as da
import numpy as np
import operator
import os
import re
import xarray as xr

try:
    import cupy as cp

    GPU_ENABLED = True
except (ImportError, ModuleNotFoundError):
    GPU_ENABLED = False

from .focal import (
    FOCAL_PROMOTING_OPS,
    FOCAL_STATS,
    check_kernel,
    correlate,
    focal,
    get_focal_window,
)
from .io import (
    Encoding,
    chunk,
    is_batch_file,
    open_raster_from_path,
    write_raster,
)
from ._types import (
    DTYPE_INPUT_TO_DTYPE,
    I32,
    F16,
    F64,
    BOOL,
    U8,
    maybe_promote,
    promote_data_dtype,
    should_promote_to_fit,
)
from ._utils import (
    is_bool,
    is_float,
    is_int,
    is_numpy,
    is_scalar,
    is_str,
    is_xarray,
    validate_file,
)


def _is_raster_class(value):
    return isinstance(value, Raster)


def _is_using_dask(raster):
    rs = raster._rs if _is_raster_class(raster) else raster
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


def _chunk_replace_null(chunk, args):
    """Replace null values in a chunk."""
    null_values, new_value = args
    null_values = set(null_values)
    if np.nan in null_values:
        null_values.remove(np.nan)
    if GPU_ENABLED:
        xp = cp.get_array_module(chunk)
    else:
        xp = np
    match = xp.isnan(chunk)
    chunk[match] = new_value
    for nv in null_values:
        xp.equal(chunk, nv, out=match)
        chunk[match] = new_value
    return chunk


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


def _np_to_xarray(nprs):
    if len(nprs.shape) > 3 or len(nprs.shape) < 2:
        raise ValueError(f"Invalid raster shape for numpy array: {nprs.shape}")
    if len(nprs.shape) == 2:
        nprs = np.expand_dims(nprs, axis=0)
    nprs = da.from_array(nprs)
    coords = [list(range(d)) for d in nprs.shape]
    return xr.DataArray(nprs, dims=["band", "y", "x"], coords=coords)


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
    if all(nv is None for nv in [nv1, nv2]):
        return np.nan
    if nv1 is None:
        nv1 = np.nan
    if nv2 is None:
        nv2 = np.nan
    if not np.isnan(nv1):
        return nv1
    if not np.isnan(nv2):
        return nv2
    return np.nan


def _reconcile_encodings_after_op(left, right=None, should_promote=False):
    encoding = Encoding()
    if left is None and right is not None:
        raise ValueError("Use left instead of only right")
    if not _is_raster_class(left):
        raise TypeError("left must be a Raster")

    left = left.encoding
    if is_scalar(right) or is_bool(right):
        right = Encoding(False, np.min_scalar_type(right), np.nan)
    elif _is_raster_class(right):
        right = right.encoding
    elif right is not None:
        raise TypeError("Could not understand right")

    if right is not None:
        # result was created from two parents (one may have been a scalar)
        masked = left.masked or right.masked
        dtype = np.result_type(left.dtype, right.dtype)
        null = np.nan
        if masked:
            dtype = maybe_promote(dtype)
            nvs = [enc.null_value for enc in [left, right] if enc.masked]
            if len(nvs) == 1:
                null = nvs[0]
            elif all(np.isnan(v) for v in nvs):
                null = np.nan
            elif nvs[0] == nvs[1]:
                null = nvs[0]
            else:
                # Have to make a decision so give precedence to left
                null = nvs[0]
        encoding = Encoding(masked, dtype, null)
    else:
        # result was created from a single raster. Simply copy the encoding
        encoding = left.copy()
    if should_promote:
        encoding.dtype = maybe_promote(encoding.dtype)
    return encoding


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
        self._encoding = Encoding()
        self._rs = None

        if _is_raster_class(raster):
            self._rs = raster._rs.copy()
            self._set_device(raster.device)
            self._encoding = raster.encoding.copy()
        elif is_xarray(raster):
            self._rs = raster
            null = _try_to_get_null_value_xarray(raster)
            masked = null is not None and not np.isnan(null)
            self._encoding = Encoding(masked, raster.dtype, null)
        elif is_numpy(raster):
            raster = _np_to_xarray(raster)
            self._rs = chunk(raster)
            self._encoding = Encoding(False, self._rs.dtype, np.nan)
        elif is_batch_file(raster):
            rs = BatchScript(raster).parse().final_raster
            self._rs = rs._rs
            self._encoding = rs.encoding
        else:
            self._rs, self._encoding = open_raster_from_path(raster)

    def _new_like_self(self, rs, attrs=None, device=None, encoding=None):
        new_rs = Raster(rs)
        new_rs._attrs = attrs or self._attrs
        new_rs._set_device(device or self.device)
        new_rs._encoding = encoding or self.encoding.copy()
        return new_rs

    def _to_presentable_xarray(self):
        """Returns a DataArray with nans replaced with the null value"""
        rs = self._rs
        null_value = self.encoding.null_value
        if self._masked and not np.isnan(null_value):
            rs = rs.fillna(null_value).astype(self.encoding.dtype)
        return rs

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
        return self.encoding.masked

    @property
    def _values(self):
        """The raw internal values. Note: this triggers computation."""
        return self._rs.values

    @property
    def _values_encoded(self):
        """
        The data as it would be written to disk. Note: this triggers
        computation.
        """
        return self._to_presentable_xarray().values

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
        """The dtype of the data when loaded.

        This is different from the encoded dtype. The dtype that will be used
        when writing to disk can be found in `Raster.encoding`.

        """
        return self._rs.dtype

    @property
    def encoding(self):
        """Stores encoding information used when saving the raster to disk.

        The encoding information includes the masked status, the final dtype to
        use when saving and the null or fill value for the raster. The masked
        status indicates whether or not there are missing values in the raster.

        """
        return self._encoding

    @property
    def shape(self):
        """The shape of the underlying raster. Dim 0 is the band dimension.

        The shape will always be of the form ``(B, Y, X)`` where ``B`` is the
        band dimension, ``Y`` is the y dimension, and ``X`` is the x dimension.

        """
        return self._rs.shape

    @property
    def resolution(self):
        """The x and y cell sizes as a tuple."""
        return self._attrs.get("res")

    def to_xarray(self):
        """Returns the underlying data as an xarray.DataArray.

        Changes made to the resulting DataArray may affect this Raster. This
        is the internal representation of the data and may differ from the
        result that will be written to disk. See `Raster.encoding`.

        """
        return self._rs

    def to_dask(self):
        """Returns the underlying data as a dask array."""
        rs = self
        if not _is_using_dask(self):
            rs = self._new_like_self(chunk(self._rs))
        return rs._rs.data

    def as_encoded(self):
        """Returns a raster encoded according to the encoding information.

        The underlying data of the result looks like what would be written to
        disk with :func: `~Raster.save`. This can be convenient when using
        :func: `~Raster.to_xarray` and :func: `~Raster.to_dask`.

        """
        rs = self.copy()
        enc = rs.encoding
        if not np.isnan(enc.null_value):
            rs._rs = rs._rs.fillna(enc.null_value)
        rs._rs = rs._rs.astype(enc.dtype)
        # Turn masked flag off to prevent errors if used with masked aware
        # funtions later
        rs.encoding.masked = False
        return rs

    def close(self):
        """Close the underlying source"""
        self._rs.close()

    def save(
        self, path, no_data_value=None, blockwidth=None, blockheight=None
    ):
        """Compute the final raster and save it to the provided location."""
        # TODO: add tiling flag
        # TODO: warn of overwrite
        rs = self
        if no_data_value is not None:
            rs = self.set_null_value(no_data_value)
        xrs = rs._rs
        write_raster(
            xrs,
            rs.encoding.copy(),
            path,
            blockwidth=blockwidth,
            blockheight=blockheight,
        )
        return self

    def eval(self):
        """Compute any applied operations and return the result as new Raster.

        Note that the unerlying sources will be loaded into memory for the
        computations and the result will be fixed in memory. The original
        Raster will be unaltered.

        """
        rs = self._rs.compute()
        # A new raster is returned to mirror the xarray and dask APIs
        return self._new_like_self(rs)

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
        return self._new_like_self(chunk(self._rs))

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
        if _is_raster_class(other) and (self.device == other.device):
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
        """Return a copy of the Raster cast to the specified type."""
        if isinstance(dtype, str):
            dtype = dtype.lower()
        if dtype not in DTYPE_INPUT_TO_DTYPE:
            raise ValueError(f"Unsupported type: '{dtype}'")
        dtype = DTYPE_INPUT_TO_DTYPE[dtype]

        if dtype == self.dtype and dtype == self.encoding.dtype:
            return self.copy()

        xrs = self._rs
        encoding = self.encoding.copy()
        if self._masked:
            encoding.dtype = dtype
            xrs = promote_data_dtype(xrs)
        else:
            xrs = xrs.astype(dtype)
            encoding.dtype = dtype
        return self._new_like_self(xrs, encoding=encoding)

    def round(self):
        """Round the data to the nearest integer value. Return a new Raster."""
        return self._new_like_self(self._rs.round())

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
        # TODO: look into making attrs consistant with bands
        return self._new_like_self(rs)

    def band_concat(self, rasters):
        """Join this and a sequence of rasters along the band dimension.

        Parameters
        ----------
        rasters : sequence of Rasters and/or paths
            The rasters to concatenate to with this raster. These can be a mix
            of Rasters and paths. All rasters must have the same shape in the
            last two dimensions.

        Returns
        -------
        Raster
            The resulting concatenated Raster.

        """
        rasters = [self._input_to_raster(other) for other in rasters]
        if not rasters:
            raise ValueError("No rasters provided")
        shapes = [r.shape for r in rasters]
        if any(len(s) > 3 for s in shapes):
            raise ValueError("Unexpected dimension on input raster")
        if any(len(s) < 2 for s in shapes):
            raise ValueError("Too few dimensions")
        # NOTE: xarray.concat allows for arrays to be missing the first
        # dimension, e.g. concat([(2, 3, 3), (3, 3)]) works. This
        # differs from numpy.
        shapes = set([s[-2:] for s in shapes])
        if len(shapes) != 1:
            raise ValueError("Final dimensions must match for input rasters")
        shapes.add(self.shape[-2:])
        if len(shapes) != 1:
            raise ValueError(
                "Final dimensions of input rasters must match this raster"
            )
        # TODO: make sure band dim is "band"
        rasters = [self._rs] + [r._rs for r in rasters]
        rs = xr.concat(rasters, "band")
        # Make sure that band is now an increaseing list starting at 1 and
        # incrementing by 1. For xrs1 (1, N, M) and xrs2 (1, N, M),
        # concat([xrs1, xrs2]) sets the band dim to [1, 1], which causes errors
        # in other operations, so this fixes that. It also keeps the band dim
        # values in line with what open_rasterio() returns for multiband
        # rasters.
        rs["band"] = list(range(1, rs.shape[0] + 1))
        return self._new_like_self(rs)

    def set_null_value(self, value):
        """Sets or replaces the null value for the raster.

        If there was previously no null value for the raster, one is set. If
        there was already a null value, then it is replaced. If the raster has
        an integer encoding dtype and `value` is a float, the encoding dtype
        will be promoted to a float dtype.

        Parameters
        ----------
        value : scalar
            The new null value for the raster.

        Returns
        -------
        Raster
            The new resulting Raster.

        """
        if not is_scalar(value):
            raise TypeError(f"Value must be a scalar: {value}")

        encoding = self.encoding.copy()
        rs = self._rs
        if encoding.masked:
            encoding.null_value = value
        else:
            encoding.masked = True
            encoding.null_value = value
            enc_dtype = encoding.dtype
            # Promote the encoding dtype if needed to make the new null value
            # valid
            if should_promote_to_fit(enc_dtype, value):
                enc_dtype = maybe_promote(enc_dtype)
            encoding.dtype = enc_dtype
            # Promote actual dtype if needed to allow nans
            dtype = maybe_promote(rs.dtype)
            if rs.dtype != dtype:
                rs = rs.astype(dtype)
        if not np.isnan(value):
            rs = rs.where(rs != value, np.nan)
        rs.attrs["_FillValue"] = value
        return self._new_like_self(rs, encoding=encoding, attrs=rs.attrs)

    def replace_null(self, value):
        """Replaces null and nan values with `value`.

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
        if self._masked:
            rs = self._rs.fillna(value)
        else:
            if is_float(value) and is_int(self.dtype):
                dtype = maybe_promote(np.result_type(value))
                rs = self._rs.astype(dtype)
        return self._new_like_self(rs)

    def where(self, condition, other):
        """Filter elements from this raster according to `condition`.

        Parameters
        ----------
        condition : str or Raster
            A boolean raster that indicates where elements in this raster
            should be preserved and where `other` should be used. ``True``
            indicates this raster and ``False`` indicates `other`. The dtype
            must be *bool*. *str* is treated as a path to a raster.
        other : scalar, str or Raster
            A raster or value to use in locations where `condition` is
            ``False``. *str* is treated as a path to a raster.

        Returns
        -------
        Raster
            The resulting filtered Raster.

        """
        if not _is_raster_class(condition) and not is_str(condition):
            raise TypeError(
                f"Invalid type for condition argument: {type(condition)}"
            )
        if (
            not is_scalar(other)
            and not is_str(other)
            and not _is_raster_class(other)
        ):
            raise TypeError(f"Invalid type for `other`: {type(other)}")
        if is_str(condition):
            condition = Raster(condition)
        if not is_bool(condition.dtype):
            raise TypeError("Condition argument must be a boolean raster")
        if is_str(other):
            other = Raster(other)

        xrs = self._rs
        xrs = xrs.where(condition._rs, other)
        encoding = _reconcile_encodings_after_op(
            self, other if _is_raster_class(other) else None
        )
        return self._new_like_self(xrs, encoding=encoding)

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
            if the additional remap args are not a multiple of 3. The remap
            groups are applied sequentially.

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
            if is_int(rs.dtype) and is_float(new_value):
                rs = rs.astype(maybe_promote(rs.dtype))
            rs = rs.where(condition, new_value)
        return rs

    def _input_to_raster(self, raster_input):
        if _is_raster_class(raster_input):
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
        if _is_raster_class(operand):
            parsed_operand = operand
            operand = operand._rs
        left, right = self._rs, operand
        if swap:
            left, right = right, left
        xrs = _BINARY_ARITHMETIC_OPS[op](left, right)

        should_promote = op == "/" or (op == "**" and is_float(operand))
        new_encoding = _reconcile_encodings_after_op(
            self, parsed_operand, should_promote=should_promote
        )
        return self._new_like_self(xrs, encoding=new_encoding)

    def _binary_logical(self, raster_or_scalar, op):
        if op not in _BINARY_LOGICAL_OPS:
            raise ValueError(f"Unknown arithmetic operation: '{op}'")
        operand = self._handle_binary_op_input(raster_or_scalar, False)
        parsed_operand = raster_or_scalar
        if _is_raster_class(operand):
            parsed_operand = operand
            operand = operand._rs
        xrs = _BINARY_LOGICAL_OPS[op](self._rs, operand)

        new_encoding = _reconcile_encodings_after_op(self, parsed_operand)
        xrs = xrs.astype(F16)
        if new_encoding.masked:
            # Mask out null values
            mask = np.isnan(self._rs)
            if not is_scalar(operand):
                mask |= np.isnan(operand)
            xrs = xrs.where(~mask, np.nan)
            # Determine the smallest type that can hold the null value since
            # the other values are just 1s and 0s
            new_encoding.dtype = np.min_scalar_type(new_encoding.null_value)
        else:
            new_encoding.dtype = U8
        return self._new_like_self(xrs, encoding=new_encoding)

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
        encoding = _reconcile_encodings_after_op(self, should_promote=True)
        return self._new_like_self(xrs, encoding=encoding)

    def __pos__(self):
        return self

    def negate(self):
        """Negate this Raster.

        Returns a new Raster.

        """
        return self._new_like_self(-self._rs)

    def __neg__(self):
        return self.negate()

    def log(self):
        """Take the natural logarithm of this Raster.

        Returns a new Raster.

        """
        xrs = np.log(self._rs)
        encoding = _reconcile_encodings_after_op(self, should_promote=True)
        return self._new_like_self(xrs, encoding=encoding)

    def log10(self):
        """Take the base-10 logarithm of this Raster.

        Returns a new Raster.

        """
        xrs = np.log10(self._rs)
        encoding = _reconcile_encodings_after_op(self, should_promote=True)
        return self._new_like_self(xrs, encoding=encoding)

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
        if _is_raster_class(operand):
            operand = operand._rs
        left, right = _coerce_to_bool_for_and_or(
            [self._rs, operand], to_bool_op
        )
        xrs = _BINARY_BITWISE_OPS[and_or](left, right).astype(F16)

        new_encoding = _reconcile_encodings_after_op(self, parsed_operand)
        if new_encoding.masked:
            # Mask out null values
            mask = np.isnan(self._rs)
            if is_xarray(operand):
                mask |= np.isnan(operand)
            xrs = xrs.where(~mask, np.nan)
            # Determine the smallest type that can hold the null value since
            # the other values are just 1s and 0s
            new_encoding.dtype = np.min_scalar_type(new_encoding.null_value)
        else:
            new_encoding.dtype = U8
        return self._new_like_self(xrs, encoding=new_encoding)

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

    def correlate(self, kernel, mode="constant", cval=0.0):
        """Cross-correlate `kernel` with each band individually. Returns a new
        Raster.

        The kernel is applied to each band in isolation so returned raster has
        the same shape as the original.

        Parameters
        ----------
        kernel : array_like
            2D array of kernel weights
        mode : {'reflect', 'constant', 'nearest', 'wrap'}, optional
            Determines how the data is extended beyond its boundaries. The
            default is 'constant'.

            'reflect' (d c b a | a b c d | d c b a)
                The data pixels are reflected at the boundaries.
            'constant' (k k k k | a b c d | k k k k)
                A constant value determined by `cval` is used to extend the
                data pixels.
            'nearest' (a a a a | a b c d | d d d d)
                The data is extended using the boundary pixels.
            'wrap' (a b c d | a b c d | a b c d)
                The data is extended by wrapping to the opposite side of the
                grid.
        cval : scalar, optional
            Value used to fill when `mode` is 'constant'. Default is 0.0.

        Returns
        -------
        Raster
            The resulting new Raster.
        """
        kernel = np.asarray(kernel)
        check_kernel(kernel)
        rs = self.copy()
        if is_float(kernel.dtype) and is_int(rs.dtype):
            rs._rs = promote_data_dtype(rs._rs)
            rs.encoding.dtype = rs.dtype
        data = rs._rs.data

        for bnd in range(data.shape[0]):
            data[bnd] = correlate(
                data[bnd], kernel, mode=mode, cval=cval, nan_aware=rs._masked
            )
        return rs

    def convolve(self, kernel, mode="constant", cval=0.0):
        """Convolve `kernel` with each band individually. Returns a new Raster.

        This is the same as correlation but the kernel is rotated 180 degrees,
        e.g. ``kernel = kernel[::-1, ::-1]``.  The kernel is applied to each
        band in isolation so the returned raster has the same shape as the
        original.

        Parameters
        ----------
        kernel : array_like
            2D array of kernel weights
        mode : {'reflect', 'constant', 'nearest', 'wrap'}, optional
            Determines how the data is extended beyond its boundaries. The
            default is 'constant'.

            'reflect' (d c b a | a b c d | d c b a)
                The data pixels are reflected at the boundaries.
            'constant' (k k k k | a b c d | k k k k)
                A constant value determined by `cval` is used to extend the
                data pixels.
            'nearest' (a a a a | a b c d | d d d d)
                The data is extended using the boundary pixels.
            'wrap' (a b c d | a b c d | a b c d)
                The data is extended by wrapping to the opposite side of the
                grid.
        cval : scalar, optional
            Value used to fill when `mode` is 'constant'. Default is 0.0.

        Returns
        -------
        Raster
            The resulting new Raster.
        """
        kernel = np.asarray(kernel)
        check_kernel(kernel)
        kernel = kernel[::-1, ::-1].copy()
        return self.correlate(kernel, mode=mode, cval=cval)

    def focal(
        self,
        focal_type,
        width_or_radius,
        height=None,
    ):
        """
        Applies a focal filter to raster bands individually.

        The filter uses a window/footprint that is created using the
        `width_or_radius` and `height` parameters. The window can be a
        rectangle, circle or annulus.

        Parameters
        ----------
        focal_type : str
            Specifies the aggregation function to apply to the focal
            neighborhood at each pixel. Can be one of the following string
            values:

            'min'
                Finds the minimum value in the neighborhood.
            'max'
                Finds the maximum value in the neighborhood.
            'mean'
                Finds the mean of the neighborhood.
            'median'
                Finds the median of the neighborhood.
            'mode'
                Finds the mode of the neighborhood.
            'sum'
                Finds the sum of the neighborhood.
            'std'
                Finds the standard deviation of the neighborhood.
            'var'
                Finds the variance of the neighborhood.
            'asm'
                Angular second moment. Applies -sum(P(g)**2) where P(g) gives
                the probability of g within the neighborhood.
            'entropy'
                Calculates the entropy. Applies -sum(P(g) * log(P(g))). See
                'asm' above.
            'unique'
                Calculates the number of unique values in the neighborhood.
        width_or_radius : int or 2-tuple of ints
            If an int and `height` is `None`, specifies the radius of a circle
            window. If an int and `height` is also an int, specifies the width
            of a rectangle window. If a 2-tuple of ints, the values specify the
            inner and outer radii of an annulus window.
        height : int or None
            If `None` (default), `width_or_radius` will be used to construct a
            circle or annulus window. If an int, specifies the height of a
            rectangle window.

        Returns
        -------
        Raster
            The resulting raster with focal filter applied to each band. The
            bands will have the same shape as the original Raster.
        """
        if focal_type not in FOCAL_STATS:
            raise ValueError(f"Unknown focal operation: '{focal_type}'")

        window = get_focal_window(width_or_radius, height)
        rs = self.copy()
        if focal_type in FOCAL_PROMOTING_OPS:
            rs._rs = promote_data_dtype(rs._rs)
            rs.encoding.dtype = rs._rs.dtype
        nan_aware = self._masked
        data = rs._rs.data

        for bnd in range(data.shape[0]):
            data[bnd] = focal(data[bnd], window, focal_type, nan_aware)
        return rs

    def __repr__(self):
        # TODO: implement
        return repr(self._rs)


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
