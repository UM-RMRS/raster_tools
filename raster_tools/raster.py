import collections
import cupy as cp
import dask
import numpy as np
import operator
import os
import re
import xarray as xr
from numbers import Number

from .io import chunk, is_batch_file, open_raster_from_path, write_raster
from ._utils import validate_file


def _is_str(value):
    return isinstance(value, str)


def _is_scalar(value):
    return isinstance(value, Number)


def _is_raster_class(value):
    return isinstance(value, Raster)


def _is_xarray(rs):
    return isinstance(rs, (xr.DataArray, xr.Dataset))


def _is_using_dask(raster):
    rs = raster._rs if _is_raster_class(raster) else raster
    return dask.is_dask_collection(rs)


class RasterDeviceMismatchError(BaseException):
    pass


U8 = np.uint8
U16 = np.uint16
U32 = np.uint32
U64 = np.uint64
I8 = np.int8
I16 = np.int16
I32 = np.int32
I64 = np.int64
F16 = np.float16
F32 = np.float32
F64 = np.float64
F128 = np.float128
BOOL = np.bool_
_DTYPE_INPUT_TO_DTYPE = {
    # Unsigned int
    U8: U8,
    "uint8": U8,
    U16: U16,
    "uint16": U16,
    U32: U32,
    "uint32": U32,
    U64: U64,
    "uint64": U64,
    # Signed int
    I8: I8,
    "int8": I8,
    I16: I16,
    "int16": I16,
    I32: I32,
    "int32": I32,
    I64: I64,
    "int64": I64,
    int: I64,
    "int": I64,
    # Float
    F16: F16,
    "float16": F16,
    F32: F32,
    "float32": F32,
    F64: F64,
    "float64": F64,
    F128: F128,
    "float128": F128,
    float: F64,
    "float": F64,
    # Boolean
    BOOL: BOOL,
    bool: BOOL,
    "bool": BOOL,
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
    xp = cp.get_array_module(chunk)
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


GPU = "gpu"
CPU = "cpu"


def _new_raster_set_attrs(rs, attrs):
    new_rs = Raster(rs)
    new_rs._attrs = attrs
    return new_rs


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
        self.shape = self._rs.shape

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
    def dtype(self):
        return self._rs.dtype

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
        return Raster(rs)

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
        return Raster(chunk(self._rs))

    def copy(self):
        """Returns a copy of this Raster."""
        return Raster(self)

    def gpu(self):
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
        return Raster(self._rs.astype(dtype))

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

    def remap_range(self, min, max, new_value):
        """
        Remaps values in a the range [`min`, `max`) to `new_value`. Returns a
        new Raster.

        Parameters
        ----------
        min : scalar
            The minimum value of the mapping range (inclusive).
        max : scalar
            The maximum value of the mapping range (exclusive).
        new_value : scalar
            The new value to map the range to.

        Returns
        -------
        Raster
            The resulting Raster.
        """
        if not all([_is_scalar(v) for v in (min, max, new_value)]):
            raise TypeError("min, max, and new_value must all be scalars")
        if np.isnan((min, max)).any():
            raise ValueError("min and max cannot be NaN")
        if min >= max:
            raise ValueError(f"min must be less than max: ({min}, {max})")
        rs = _map_chunk_function(
            self.copy(), _chunk_remap_range, (min, max, new_value)
        )
        return rs

    def _handle_binary_op_input(self, raster_or_scalar):
        if _is_scalar(raster_or_scalar):
            operand = raster_or_scalar
        elif _is_raster_class(raster_or_scalar):
            self._check_device_mismatch(raster_or_scalar)
            operand = raster_or_scalar._rs
        else:
            operand = open_raster_from_path(raster_or_scalar)
            if self.device == GPU:
                operand = operand.gpu()._rs
        return operand

    def _binary_arithmetic(self, raster_or_scalar, op, swap=False):
        # TODO: handle mapping of list of values to bands
        # TODO: handle case where shapes match but geo references don't
        if op not in _BINARY_ARITHMETIC_OPS:
            raise ValueError(f"Unknown arithmetic operation: '{op}'")
        operand = self._handle_binary_op_input(raster_or_scalar)
        # Attributes are not propagated through math ops
        if not swap:
            rs = _new_raster_set_attrs(
                _BINARY_ARITHMETIC_OPS[op](self._rs, operand), self._attrs
            )
            rs.device = self.device
            return rs
        else:
            rs = _new_raster_set_attrs(
                _BINARY_ARITHMETIC_OPS[op](operand, self._rs), self._attrs
            )
            rs.device = self.device
            return rs

    def _binary_logical(self, raster_or_scalar, op):
        if op not in _BINARY_LOGICAL_OPS:
            raise ValueError(f"Unknown arithmetic operation: '{op}'")
        operand = self._handle_binary_op_input(raster_or_scalar)
        rs = _new_raster_set_attrs(
            _BINARY_LOGICAL_OPS[op](self._rs, operand), self._attrs
        )
        rs.device = self.device
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

    def __pos__(self):
        return self

    def negate(self):
        """Negate this Raster. Returns a new Raster."""
        # Don't need to copy attrs here
        return Raster(-self._rs)

    def __neg__(self):
        return self.negate()

    def log(self):
        """Take the natural logarithm of this Raster. Returns a new Raster."""
        # Don't need to copy attrs here
        return Raster(np.log(self._rs))

    def log10(self):
        """Take the base-10 logarithm of this Raster. Returns a new Raster."""
        # Don't need to copy attrs here
        return Raster(np.log10(self._rs))

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

    def convolve2d(self, kernel, fill_value=0):
        """Convolve this Raster with a kernel. Warning: experimental"""
        # TODO: validate kernel
        nr, nc = kernel.shape
        kernel = xr.DataArray(kernel, dims=("kx", "ky"))
        min_periods = (nr // 2 + 1) * (nc // 2 + 1)
        rs_out = (
            self._rs.rolling(x=nr, y=nc, min_periods=min_periods, center=True)
            .construct(x="kx", y="ky", fill_value=fill_value)
            .dot(kernel)
        )
        # There seems to be a bug where the attributes aren't propagated
        # through construct().
        return _new_raster_set_attrs(rs_out, self._attrs)

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
            raise NotImplementedError()
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
                "REMAP Error: Too many arguments" + on_line
            )
        args = args[0]
        try:
            values = [float(v) for v in _split_strip(args, ":")]
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
                "REMAP Error: the min value must be less than the max value"
                + on_line
            )
        return self._get_raster(raster).remap_range(left, right, new)

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
