import collections
import dask
import dask.array as da
import numpy as np
import operator
import os
import re
import xarray as xr
from dask_image import ndfilters
from numbers import Integral, Number

try:
    import cupy as cp

    GPU_ENABLED = True
except (ImportError, ModuleNotFoundError):
    GPU_ENABLED = False

from .io import chunk, is_batch_file, open_raster_from_path, write_raster
from ._utils import validate_file


def _is_str(value):
    return isinstance(value, str)


def _is_scalar(value):
    return isinstance(value, Number)


def _is_int(value):
    return isinstance(value, Integral)


def _is_raster_class(value):
    return isinstance(value, Raster)


def _is_xarray(rs):
    return isinstance(rs, (xr.DataArray, xr.Dataset))


def _is_using_dask(raster):
    rs = raster._rs if _is_raster_class(raster) else raster
    return dask.is_dask_collection(rs)


class RasterDeviceMismatchError(BaseException):
    pass


class RasterDeviceError(BaseException):
    pass


U8 = np.dtype(np.uint8)
U16 = np.dtype(np.uint16)
U32 = np.dtype(np.uint32)
U64 = np.dtype(np.uint64)
I8 = np.dtype(np.int8)
I16 = np.dtype(np.int16)
I32 = np.dtype(np.int32)
I64 = np.dtype(np.int64)
F16 = np.dtype(np.float16)
F32 = np.dtype(np.float32)
F64 = np.dtype(np.float64)
# Some machines don't support np.float128 so use longdouble instead. This
# aliases f128 on machines that support it and f64 on machines that don't
F128 = np.dtype(np.longdouble)
BOOL = np.dtype(bool)
_DTYPE_INPUT_TO_DTYPE = {
    # Unsigned int
    U8: U8,
    "uint8": U8,
    np.dtype("uint8"): U8,
    U16: U16,
    "uint16": U16,
    np.dtype("uint16"): U16,
    U32: U32,
    "uint32": U32,
    np.dtype("uint32"): U32,
    U64: U64,
    "uint64": U64,
    np.dtype("uint64"): U64,
    # Signed int
    I8: I8,
    "int8": I8,
    np.dtype("int8"): I8,
    I16: I16,
    "int16": I16,
    np.dtype("int16"): I16,
    I32: I32,
    "int32": I32,
    np.dtype("int32"): I32,
    I64: I64,
    "int64": I64,
    np.dtype("int64"): I64,
    int: I64,
    "int": I64,
    # Float
    F16: F16,
    "float16": F16,
    np.dtype("float16"): F16,
    F32: F32,
    "float32": F32,
    np.dtype("float32"): F32,
    F64: F64,
    "float64": F64,
    np.dtype("float64"): F64,
    F128: F128,
    "float128": F128,
    np.dtype("longdouble"): F128,
    float: F64,
    "float": F64,
    # Boolean
    BOOL: BOOL,
    bool: BOOL,
    "bool": BOOL,
    np.dtype("bool"): BOOL,
}


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


def _chunk_remap_range(chunk, args):
    """Remap a range of value to a new value within a chunk."""
    min_, max_, new_value = args
    match = chunk >= min_
    match &= chunk < max_
    chunk[match] = new_value
    return chunk


def _chunk_to_cpu(chunk, *args):
    try:
        return chunk.get()
    except AttributeError:
        # assume chunk is already on the cpu
        return chunk


def _coerce_to_bool_for_and_or(operands, to_bool_op):
    if _is_str(to_bool_op):
        if to_bool_op == "gt0":

            def to_bool(x):
                return x > 0

            to_bool_op = to_bool
        elif to_bool_op == "cast":

            def to_bool(x):
                if _is_scalar(x):
                    return bool(x)
                return x.astype(BOOL)

            to_bool_op = to_bool
        else:
            raise ValueError("Unknown conversion to bool")
    elif callable(to_bool_op):
        pass
    else:
        raise TypeError("Invalid conversion to bool")

    boperands = []
    for opr in operands:
        if not _is_scalar(opr) and (
            isinstance(opr, (bool, np.bool_)) or opr.dtype == np.dtype(bool)
        ):
            boperands.append(opr)
            continue
        boperands.append(to_bool_op(opr))
    return boperands


def _get_output_type_for_and_or(dt1, dt2):
    dt1 = np.dtype(dt1)
    dt2 = np.dtype(dt2)
    dts = [dt1, dt2]
    kinds = [d.kind for d in dts]
    if len(set(kinds)) == 1:
        # Return the widest type
        return dts[np.argmax([d.itemsize for d in dts])]
    # Prioritize float over int over uint over bool
    if "f" in kinds:
        return dt1 if dt1.kind == "f" else dt2
    if "i" in kinds:
        return dt1 if dt1.kind == "i" else dt2
    if "u" in kinds:
        return dt1 if dt1.kind == "u" else dt2
    if "b" in kinds:
        return dt1 if dt1.kind == "b" else dt2
    return dt1


def _get_focal_window(width_or_radius, height=None):
    width = width_or_radius
    window = None
    if height is None:
        width = ((width - 1) * 2) + 1
        height = width
        r = (width - 1) // 2
        window = np.zeros((width, height), dtype=I32)
        for x in range(width):
            for y in range(height):
                rxy = np.sqrt((x - r) ** 2 + (y - r) ** 2)
                if rxy <= r:
                    window[x, y] = 1
    else:
        window = np.ones((width, height), dtype=I32)
    return window


GPU = "gpu"
CPU = "cpu"


def _dask_from_array_with_device(arr, device):
    arr = da.from_array(arr)
    if GPU_ENABLED and device == GPU:
        arr = arr.map_blocks(cp.asarray)
    return arr


class Raster:
    """
    An abstraction of georeferenced raster data with lazy function evaluation.

    Raster is a wrapper around xarray Datasets and DataArrays. It takes
    advantage of xarray's dask integration to allow lazy loading and
    evaluation. It allows a pipeline of operations on underlying raster
    sources to be built in a lazy fashion and then evaluated effiently.
    Most mathematical operations have been overloaded so operations such as
    `z = x - y` and `r = x**2` are possible.

    All operations on a Raster return a new Raster.

    Parameters
    ----------
    raster : str, Raster, xarray.Dataset, xarray.DataArray
        The raster source to use for this Raster. If `raster` is a string,
        it is treated like a path. If `raster` is a Raster, a copy is made
        and its raster source is used. If `raster` is and xarray data
        structure, it is used as the source.
    """

    def __init__(self, raster):
        self.device = CPU
        if _is_raster_class(raster):
            self._rs = raster._rs.copy()
            self.device = raster.device
        elif _is_xarray(raster):
            self._rs = raster
        elif is_batch_file(raster):
            self._rs = BatchScript(raster).parse().final_raster._rs
        else:
            self._rs = open_raster_from_path(raster)

    def _new_like_self(self, rs):
        new_rs = Raster(rs)
        new_rs._attrs = self._attrs
        new_rs.device = self.device
        return new_rs

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
    def device(self):
        return self._device

    @device.setter
    def device(self, dev):
        if dev not in (CPU, GPU):
            raise RasterDeviceError(f"Unknown device: '{dev}'")
        if not GPU_ENABLED and dev == GPU:
            raise RasterDeviceError("GPU support is not enabled")
        self._device = dev

    @property
    def dtype(self):
        return self._rs.dtype

    @property
    def shape(self):
        return self._rs.shape

    def to_xarray(self):
        """Returns the underlying data as an xarray.DataArray.

        Changes made to the resulting DataArray may affect this Raster.
        """
        return self._rs

    def to_dask(self):
        """Returns the underlying data as a dask array."""
        if _is_using_dask(self):
            return self._rs.data
        else:
            return chunk(self._rs).data

    def close(self):
        """Close the underlying source"""
        self._rs.close()

    def save(
        self, path, no_data_value=None, blockwidth=None, blockheight=None
    ):
        """Compute the final raster and save it to the provided location."""
        # TODO: add tiling flag
        # TODO: warn of overwrite
        write_raster(
            self._rs,
            path,
            no_data_value=no_data_value,
            blockwidth=blockwidth,
            blockheight=blockheight,
        )
        return self

    def eval(self):
        """
        Compute any applied operations and return the result as a new Raster.

        Note that the unerlying sources will be loaded into memory for the
        computations and the result will be fixed in memory. The original
        Raster will be unaltered.
        """
        rs = self._rs.compute()
        # A new raster is returned to mirror the xarray and dask APIs
        return self._new_like_self(rs)

    def to_lazy(self):
        """
        Convert a non-lazy Raster to a lazy one.

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
        rs.device = GPU
        return rs

    def _check_device_mismatch(self, other):
        if _is_scalar(other):
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
        rs.device = CPU
        return rs

    def astype(self, dtype):
        """Return a copy of the Raster, cast to the specified type."""
        if isinstance(dtype, str):
            dtype = dtype.lower()
        if dtype not in _DTYPE_INPUT_TO_DTYPE:
            raise ValueError(f"Unsupported type: '{dtype}'")
        dtype = _DTYPE_INPUT_TO_DTYPE[dtype]
        if dtype != self.dtype:
            return self._new_like_self(self._rs.astype(dtype))
        return self.copy()

    def get_bands(self, bands):
        """
        Retrieve the specified bands as a new Raster. Indexing starts at 1.

        Parameters
        ----------
        bands : int or sequence of ints
            The band or bands to be retrieved. A single band Raster is returned
            if `bands` is an int and a multiband Raster is returned if it is a
            sequence. The band numbers may be out of order and contain repeated
            elements. if `bands` is a sequence, the multiband raster's bands
            will be in the order provided.
        """
        n_bands, *_ = self.shape
        if _is_int(bands):
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

    def replace_null(self, value):
        """
        Replaces null values with a new value. Returns a new Raster.

        Null values are NaN and the values specified by the underlying source.

        Parameters
        ----------
        value : scalar
            The new value to replace null values with.

        Returns
        -------
        Raster
            The new resulting Raster.
        """
        if not _is_scalar(value):
            raise TypeError("value must be a scalar")
        null_values = self._attrs["nodatavals"]
        rs = _map_chunk_function(
            self.copy(), _chunk_replace_null, (null_values, value)
        )
        return rs

    def remap_range(self, min, max, new_value, *args):
        """
        Remaps values in the range [`min`, `max`) to `new_value`. Returns a
        new Raster.

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
            if not all([_is_scalar(v) for v in (min, max, new_value)]):
                raise TypeError("min, max, and new_value must all be scalars")
            if np.isnan((min, max)).any():
                raise ValueError("min and max cannot be NaN")
            if min >= max:
                raise ValueError(f"min must be less than max: ({min}, {max})")
            rs = _map_chunk_function(
                rs.copy(), _chunk_remap_range, (min, max, new_value)
            )
        return rs

    def _input_to_raster(self, raster_input):
        if _is_raster_class(raster_input):
            self._check_device_mismatch(raster_input)
            raster = raster_input
        else:
            raster = Raster(open_raster_from_path(raster_input))
            if self.device == GPU:
                raster = raster.gpu()
        if raster._rs.size == 0:
            raise ValueError(
                f"Input raster is empty with shape {raster._rs.shape}"
            )
        return raster

    def _handle_binary_op_input(self, raster_or_scalar, xarray=True):
        if _is_scalar(raster_or_scalar):
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
        operand = self._handle_binary_op_input(raster_or_scalar)
        # Attributes are not propagated through math ops
        if not swap:
            rs = self._new_like_self(
                _BINARY_ARITHMETIC_OPS[op](self._rs, operand)
            )
            return rs
        else:
            rs = self._new_like_self(
                _BINARY_ARITHMETIC_OPS[op](operand, self._rs)
            )
            return rs

    def _binary_logical(self, raster_or_scalar, op):
        if op not in _BINARY_LOGICAL_OPS:
            raise ValueError(f"Unknown arithmetic operation: '{op}'")
        operand = self._handle_binary_op_input(raster_or_scalar)
        rs = self._new_like_self(_BINARY_LOGICAL_OPS[op](self._rs, operand))
        return rs.astype(F32)

    def add(self, raster_or_scalar):
        """
        Add this Raster with another Raster or scalar. Returns a new Raster.
        """
        return self._binary_arithmetic(raster_or_scalar, "+")

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def subtract(self, raster_or_scalar):
        """
        Subtract another Raster or scalar from This Raster. Returns a new
        Raster.
        """
        return self._binary_arithmetic(raster_or_scalar, "-")

    def __sub__(self, other):
        return self.subtract(other)

    def __rsub__(self, other):
        return self.negate().add(other)

    def multiply(self, raster_or_scalar):
        """
        Multiply this Raster with another Raster or scalar. Returns a new
        Raster.
        """
        return self._binary_arithmetic(raster_or_scalar, "*")

    def __mul__(self, other):
        return self.multiply(other)

    def __rmul__(self, other):
        return self.multiply(other)

    def divide(self, raster_or_scalar):
        """
        Divide this Raster by another Raster or scalar. Returns a new Raster.
        """
        return self._binary_arithmetic(raster_or_scalar, "/")

    def __truediv__(self, other):
        return self.divide(other)

    def __rtruediv__(self, other):
        return self._binary_arithmetic(other, "/", swap=True)

    def mod(self, raster_or_scalar):
        """
        Perform the modulo operation on this Raster with another Raster or
        scalar. Returns a new Raster.
        """
        return self._binary_arithmetic(raster_or_scalar, "%")

    def __mod__(self, other):
        return self.mod(other)

    def __rmod__(self, other):
        return self._binary_arithmetic(other, "%", swap=True)

    def pow(self, value):
        """
        Raise this raster by another Raster or scalar. Returns a new Raster.
        """
        return self._binary_arithmetic(value, "**")

    def __pow__(self, value):
        return self.pow(value)

    def __rpow__(self, value):
        return self._binary_arithmetic(value, "**", swap=True)

    def sqrt(self):
        """Take the square root of the raster. Returns a new Raster."""
        return self._new_like_self(np.sqrt(self._rs))

    def __pos__(self):
        return self

    def negate(self):
        """Negate this Raster. Returns a new Raster."""
        # Don't need to copy attrs here
        return self._new_like_self(-self._rs)

    def __neg__(self):
        return self.negate()

    def log(self):
        """Take the natural logarithm of this Raster. Returns a new Raster."""
        return self._new_like_self(np.log(self._rs))

    def log10(self):
        """Take the base-10 logarithm of this Raster. Returns a new Raster."""
        return self._new_like_self(np.log10(self._rs))

    def eq(self, other):
        """
        Perform element-wise equality test against `other`. Returns a new
        Raster.
        """
        return self._binary_logical(other, "==")

    def __eq__(self, other):
        return self.eq(other)

    def ne(self, other):
        """
        Perform element-wise not-equal test against `other`. Returns a new
        Raster.
        """
        return self._binary_logical(other, "!=")

    def __ne__(self, other):
        return self.ne(other)

    def le(self, other):
        """
        Perform element-wise less-than-or-equal test against `other`. Returns a
        new Raster.
        """
        return self._binary_logical(other, "<=")

    def __le__(self, other):
        return self.le(other)

    def ge(self, other):
        """
        Perform element-wise greater-than-or-equal test against `other`.
        Returns a new Raster.
        """
        return self._binary_logical(other, ">=")

    def __ge__(self, other):
        return self.ge(other)

    def lt(self, other):
        """
        Perform element-wise less-than test against `other`.  Returns a new
        Raster.
        """
        return self._binary_logical(other, "<")

    def __lt__(self, other):
        return self.lt(other)

    def gt(self, other):
        """
        Perform element-wise greater-than test against `other`.  Returns a new
        Raster.
        """
        return self._binary_logical(other, ">")

    def __gt__(self, other):
        return self.gt(other)

    def _and_or(self, other, and_or, to_bool_op):
        if isinstance(other, (bool, np.bool_)):
            operand = other
        else:
            operand = self._handle_binary_op_input(other, False)
        other_type = (
            type(other) if not _is_raster_class(other) else other.dtype
        )
        out_type = _get_output_type_for_and_or(self.dtype, other_type)
        if _is_raster_class(operand):
            operand = operand._rs
        left, right = _coerce_to_bool_for_and_or(
            [self._rs, operand], to_bool_op
        )
        rs = self._new_like_self(_BINARY_BITWISE_OPS[and_or](left, right))
        return rs.astype(out_type)

    def and_(self, other, to_bool_op="gt0"):
        """
        Returns this Raster and'd with another. Both are coerced to bools
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
            The resulting Raster of zeros and ones or bools. The output dtype
            is determined by the input types. If both are from the same family
            (i.e. float, int, uint, bool), then the widest of the two is used
            for the output dtype. Otherwise, a dtype is chosen from the two in
            this order of priority: float > int > uint > bool.
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
            The resulting Raster of zeros and ones or bools. The output dtype
            is determined by the input types. If both are from the same family
            (i.e. float, int, uint, bool), then the widest of the two is used
            for the output dtype. Otherwise, a dtype is chosen from the two in
            this order of priority: float > int > uint > bool.
        """
        return self._and_or(other, "|", to_bool_op)

    def __or__(self, other):
        return self._and_or(other, "|", "gt0")

    def __ror__(self, other):
        return self._and_or(other, "|", "gt0")

    def convolve(self, kernel, mode="constant", cval=0.0):
        """Convolve `kernel` with each band individually. Returns a new Raster.

        The kernel is applied to each band in isolation so returned raster has
        the same shape as the original.

        Parameters
        ----------
        kernel : array_like
            2D array of kernel weights
        mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
            Determines how the data is extended beyond its boundaries. The
            default is 'constant'.
            'reflect' (d c b a | a b c d | d c b a)
                The data pixels are reflected at the boundaries.
            'constant' (k k k k | a b c d | k k k k)
                A constant value determined by `cval` is used to extend the
                data pixels.
            'nearest' (a a a a | a b c d | d d d d)
                The data is extended using the boundary pixels.
            'mirror' (d c b | a b c d | c b a)
                Like 'reflect' but the reflection is centered on the boundary
                pixel.
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
        kernel = np.asarray(kernel, dtype=F64)
        if len(kernel.shape) != 2:
            raise ValueError(f"Kernel must be 2D. Got {kernel.shape}")
        # TODO: astype F32 to match GPU words?
        kernel = _dask_from_array_with_device(kernel, self.device)
        rs = self.copy()
        data = rs._rs.data
        for bnd in range(data.shape[0]):
            data[bnd] = ndfilters.convolve(
                data[bnd], kernel, mode=mode, cval=cval
            )
        return rs

    def focal(
        self,
        focal_type,
        width_or_radius,
        height=None,
        mode="constant",
        cval=0.0,
    ):
        if focal_type not in (
            # "asm",
            # "entropy",
            "max",
            "mean",
            "median",
            "min",
            # "mode",
            "std",
            "sum",
            # "unique",
            "variance",
        ):
            raise ValueError(f"Unknown focal operation: '{focal_type}'")
        if not _is_int(width_or_radius):
            raise TypeError(
                f"width_or_radius must be an integer: {width_or_radius}"
            )
        elif width_or_radius <= 0:
            raise ValueError(
                "Window width or radius must be greater than 0."
                f" Got {width_or_radius}"
            )
        if height is not None:
            if not _is_int(height):
                raise TypeError(f"height must be an integer or None: {height}")
            elif height <= 0:
                raise ValueError(
                    f"Window height must be greater than 0. Got {height}"
                )

        window = _get_focal_window(width_or_radius, height)
        window = _dask_from_array_with_device(window, self.device)
        rs = self.copy()
        data = rs._rs.data

        if focal_type == "asm":
            raise NotImplementedError()
        elif focal_type == "entropy":
            raise NotImplementedError()
        elif focal_type == "max":
            for bnd in range(data.shape[0]):
                data[bnd] = ndfilters.maximum_filter(
                    data[bnd], footprint=window, mode=mode, cval=cval
                )
        elif focal_type in ["mean", "sum"]:
            for bnd in range(data.shape[0]):
                n = window.sum()
                data[bnd] = ndfilters.convolve(
                    data[bnd], window, mode=mode, cval=cval
                )
                if focal_type == "mean":
                    data[bnd] /= n
        elif focal_type == "median":
            for bnd in range(data.shape[0]):
                data[bnd] = ndfilters.median_filter(
                    data[bnd], footprint=window, mode=mode, cval=cval
                )
        elif focal_type == "min":
            for bnd in range(data.shape[0]):
                data[bnd] = ndfilters.minimum_filter(
                    data[bnd], footprint=window, mode=mode, cval=cval
                )
        elif focal_type == "mode":
            raise NotImplementedError()
        elif focal_type in ["std", "variance"]:
            data_sq = data ** 2
            n = window.sum()
            for bnd in range(data.shape[0]):
                data[bnd] = ndfilters.convolve(
                    data[bnd], window, mode=mode, cval=cval
                )
                data_sq[bnd] = ndfilters.convolve(
                    data_sq[bnd], window, mode=mode, cval=cval
                )
                data = (data_sq - ((data ** 2) / n)) / n
                if focal_type == "std":
                    data = np.sqrt(data)
        elif focal_type == "unique":
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown focal operation: '{focal_type}'")
        rs._rs.data = data
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
