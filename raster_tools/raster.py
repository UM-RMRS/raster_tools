import collections
import operator
import os
import rioxarray  # noqa: F401; adds ability to save tiffs to xarray
import xarray as xr
from numbers import Number


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


TIFF_EXTS = frozenset((".tif", ".tiff"))
BATCH_EXTS = frozenset((".bch",))
NC_EXTS = frozenset((".nc",))


def _parse_path(path):
    _validate_file(path)
    ext = os.path.splitext(path)[-1].lower()
    if not ext:
        raise ValueError("Could not determine file type")
    if ext in TIFF_EXTS:
        # TODO: chunking logic
        return xr.open_rasterio(path)
    elif ext in BATCH_EXTS:
        raise NotImplementedError()
    else:
        raise ValueError("Unknown file type")


def _parse_input(rs_in):
    if _is_str(rs_in):
        return _parse_path(rs_in)
    elif isinstance(rs_in, Raster):
        return rs_in
    elif isinstance(rs_in, (xr.DataArray, xr.Dataset)):
        return rs_in


_ARITHMETIC_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
}


class Raster:
    def __init__(self, raster, attrs=None):
        rs = _parse_input(raster)
        if _is_raster_class(rs):
            self._rs = rs._rs
        else:
            self._rs = rs
        self.shape = self._rs.shape
        if attrs is not None and isinstance(attrs, collections.Mapping):
            self._rs.attrs = attrs
        # Dict containing raster metadata like projection, etc.
        self._attrs = self._rs.attrs

    def close(self):
        self._rs.close()

    def save(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext in TIFF_EXTS:
            # XXX: only works for single band
            # TODO: figure out method for multi-band tiffs
            self._rs.rio.to_raster(path)
        elif ext in NC_EXTS:
            self._rs.to_netcdf(path)
        else:
            # TODO: populate
            raise NotImplementedError()

    def eval(self):
        self._rs.compute()
        return self

    def arithmetic(self, raster_or_scalar, op):
        # TODO: handle mapping of list of values to bands
        if op not in _ARITHMETIC_OPS:
            raise ValueError(f"Unknown arithmetic operation: '{op}'")
        # TODO:Fix this ugly block
        if _is_scalar(raster_or_scalar):
            operand = raster_or_scalar
        elif _is_raster_class(raster_or_scalar):
            operand = raster_or_scalar._rs
        else:
            operand = _parse_input(raster_or_scalar)
        # Attributes are not propagated through math ops
        return Raster(_ARITHMETIC_OPS[op](self._rs, operand), self._attrs)

    def add(self, raster_or_scalar):
        return self.arithmetic(raster_or_scalar, "+")

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def subtract(self, raster_or_scalar):
        return self.arithmetic(raster_or_scalar, "-")

    def __sub__(self, other):
        return self.subtract(other)

    def __rsub__(self, other):
        return self.negate().add(other)

    def multiply(self, raster_or_scalar):
        return self.arithmetic(raster_or_scalar, "*")

    def __mul__(self, other):
        return self.multiply(other)

    def __rmul__(self, other):
        return self.multiply(other)

    def divide(self, raster_or_scalar):
        return self.arithmetic(raster_or_scalar, "/")

    def __truediv__(self, other):
        return self.divide(other)

    def __rtruediv__(self, other):
        return self.pow(-1).multiply(other)

    def pow(self, value):
        return self.arithmetic(value, "**")

    def __pow__(self, value):
        return self.pow(value)

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
