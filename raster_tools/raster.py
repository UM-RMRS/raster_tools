import collections
import operator
import os
import rioxarray  # noqa: F401; adds ability to save tiffs to xarray
import xarray as xr
from numbers import Number
from pathlib import Path


def _validate_file(path):
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"Could not find file: '{path}'")


def _is_str(value):
    return isinstance(value, str)


def _is_scalar(value):
    return isinstance(value, Number)


def _is_raster_class(value):
    return isinstance(value, Raster)


def _is_xarray(rs):
    return isinstance(rs, (xr.DataArray, xr.Dataset))


def _get_extension(path):
    return os.path.splitext(path)[-1].lower()


TIFF_EXTS = frozenset((".tif", ".tiff"))
BATCH_EXTS = frozenset((".bch",))
NC_EXTS = frozenset((".nc",))


class RasterInputError(BaseException):
    pass


def _open_raster_from_path(path, open_lazy=True):
    if isinstance(path, Path) or _is_str(path):
        path = str(path)
        path = os.path.abspath(path)
    else:
        raise RasterInputError(
            f"Could not resolve input to a raster: '{path}'"
        )
    _validate_file(path)
    ext = _get_extension(path)
    if not ext:
        raise RasterInputError("Could not determine file type")
    if ext in TIFF_EXTS:
        # TODO: smarter chunking logic
        rs = xr.open_rasterio(path)
        if open_lazy:
            rs = rs.chunk({"band": 1, "x": 4000, "y": 4000})
        return rs
    elif ext in BATCH_EXTS:
        raise NotImplementedError()
    elif ext in NC_EXTS:
        # TODO: chunking logic
        return xr.open_dataset(path)
    else:
        raise RasterInputError("Unknown file type")


_BINARY_ARITHMETIC_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
    "%": operator.mod,
}


class Raster:
    def __init__(self, raster, attrs=None, open_lazy=True):
        if _is_raster_class(raster):
            self._rs = raster._rs
        elif _is_xarray(raster):
            self._rs = raster
        else:
            self._rs = _open_raster_from_path(raster, open_lazy)
        self.shape = self._rs.shape
        # Dict containing raster metadata like projection, etc.
        if attrs is not None and isinstance(attrs, collections.Mapping):
            self._rs.attrs = attrs.copy()

    @property
    def _attrs(self):
        return self._rs.attrs.copy()

    @_attrs.setter
    def _attrs(self, attrs):
        self._rs.attrs = attrs.copy()

    def close(self):
        self._rs.close()

    def save(self, path):
        ext = _get_extension(path)
        if ext in TIFF_EXTS:
            # XXX: only works for single band
            # TODO: figure out method for multi-band tiffs
            self._rs.rio.to_raster(path, compute=True)
        elif ext in NC_EXTS:
            self._rs.to_netcdf(path, compute=True)
        else:
            # TODO: populate
            raise NotImplementedError()

    def eval(self):
        self._rs.compute()
        return self

    def _binary_arithmetic(self, raster_or_scalar, op):
        # TODO: handle mapping of list of values to bands
        if op not in _BINARY_ARITHMETIC_OPS:
            raise ValueError(f"Unknown arithmetic operation: '{op}'")
        # TODO: consider disallowing xarray objects
        if _is_scalar(raster_or_scalar) or _is_xarray(raster_or_scalar):
            operand = raster_or_scalar
        elif _is_raster_class(raster_or_scalar):
            operand = raster_or_scalar._rs
        else:
            operand = _open_raster_from_path(raster_or_scalar)
        # Attributes are not propagated through math ops
        return Raster(
            _BINARY_ARITHMETIC_OPS[op](self._rs, operand), self._attrs
        )

    def add(self, raster_or_scalar):
        return self._binary_arithmetic(raster_or_scalar, "+")

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def subtract(self, raster_or_scalar):
        return self._binary_arithmetic(raster_or_scalar, "-")

    def __sub__(self, other):
        return self.subtract(other)

    def __rsub__(self, other):
        return self.negate().add(other)

    def multiply(self, raster_or_scalar):
        return self._binary_arithmetic(raster_or_scalar, "*")

    def __mul__(self, other):
        return self.multiply(other)

    def __rmul__(self, other):
        return self.multiply(other)

    def divide(self, raster_or_scalar):
        return self._binary_arithmetic(raster_or_scalar, "/")

    def __truediv__(self, other):
        return self.divide(other)

    def __rtruediv__(self, other):
        return self.pow(-1).multiply(other)

    def mod(self, raster_or_scalar):
        return self._binary_arithmetic(raster_or_scalar, "%")

    def __mod__(self, other):
        return self.mod(other)

    def __rmod__(self, other):
        return Raster(other % self._rs, self._attrs)

    def pow(self, value):
        return self._binary_arithmetic(value, "**")

    def __pow__(self, value):
        return self.pow(value)

    def __rpow__(self, value):
        return Raster(value ** self._rs, self._attrs)

    def __pos__(self):
        return self

    def negate(self):
        # Don't need to copy attrs here
        return Raster(-self._rs)

    def __neg__(self):
        return self.negate()

    def convolve2d(self, kernel, fill_value=0):
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
        return Raster(rs_out, self._attrs)
