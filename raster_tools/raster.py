import collections
import cupy as cp
import dask
import numpy as np
import operator
import xarray as xr
from numbers import Number

from .io import chunk, open_raster_from_path, write_raster


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


_BINARY_ARITHMETIC_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
    "%": operator.mod,
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

    def close(self):
        """Close the underlying source"""
        self._rs.close()

    def save(self, path):
        """Compute the final raster and save it to the provided location."""
        write_raster(self._rs, path)
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

    def _binary_arithmetic(self, raster_or_scalar, op, swap=False):
        # TODO: handle mapping of list of values to bands
        # TODO: handle case where shapes match but geo references don't
        if op not in _BINARY_ARITHMETIC_OPS:
            raise ValueError(f"Unknown arithmetic operation: '{op}'")
        if _is_scalar(raster_or_scalar):
            operand = raster_or_scalar
        elif _is_raster_class(raster_or_scalar):
            self._check_device_mismatch(raster_or_scalar)
            operand = raster_or_scalar._rs
        else:
            operand = open_raster_from_path(raster_or_scalar)
            if self.device == GPU:
                operand = operand.gpu()._rs
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
