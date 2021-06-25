import collections
import cupy
import dask
import numpy as np
import operator
import os
import rasterio as rio
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


def _is_using_dask(raster):
    rs = raster._rs if _is_raster_class(raster) else raster
    return dask.is_dask_collection(rs)


def _get_extension(path):
    return os.path.splitext(path)[-1].lower()


TIFF_EXTS = frozenset((".tif", ".tiff"))
BATCH_EXTS = frozenset((".bch",))
NC_EXTS = frozenset((".nc",))


class RasterInputError(BaseException):
    pass


def _write_tif_with_rasterio(rs, path, tile=False, compress=False, **kwargs):
    if len(rs.shape) == 3:
        bands, rows, cols = rs.shape
    else:
        rows, cols = rs.shape
        bands = 1
    compress = None if not compress else "lzw"
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=bands,
        dtype=rs.dtype,
        nodata=rs.nodatavals[0],
        crs=rs.crs,
        transform=rs.transform,
        tile=tile,
        compress=compress,
    ) as dst:
        for band in range(bands):
            if len(rs.shape) == 3:
                values = rs[band].values
            else:
                values = rs.values
            dst.write(values, band + 1)


def _chunk(xrs):
    # TODO: smarter chunking logic
    return xrs.chunk({"band": 1, "x": 4000, "y": 4000})


def _open_raster_from_path(path):
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
        rs = xr.open_rasterio(path)
        # XXX: comments on a few xarray issues mention better performance when
        # using the chunks keyword in open_*(). Consider combining opening and
        # chunking.
        rs = _chunk(rs)
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


def _map_chunk_function(raster, func, args, **kwargs):
    if _is_using_dask(raster):
        raster._rs.data = raster._rs.data.map_blocks(func, args, **kwargs)
    else:
        raster._rs.data = func(raster._rs.data, args)
    return raster


def _chunk_replace_null(chunk, args):
    null_values, new_value = args
    null_values = set(null_values)
    if np.nan in null_values:
        null_values.remove(np.nan)
    mod = np if isinstance(chunk, np.ndarray) else cupy
    match = mod.isnan(chunk)
    chunk[match] = new_value
    for nv in null_values:
        mod.equal(chunk, nv, out=match)
        chunk[match] = new_value
    return chunk


def _chunk_remap_range(chunk, args):
    min_, max_, new_value = args
    match = chunk >= min_
    match &= chunk < max_
    chunk[match] = new_value
    return chunk


class Raster:
    def __init__(self, raster, attrs=None):
        if _is_raster_class(raster):
            self._rs = raster._rs
        elif _is_xarray(raster):
            self._rs = raster
        else:
            self._rs = _open_raster_from_path(raster)
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
            # TODO: figure out method for multi-band tiffs that respects dask
            # lazy eval/loading
            nbands = 1
            if len(self.shape) == 3:
                nbands = self.shape[0]
            if nbands == 1:
                self._rs.rio.to_raster(path, compute=True)
            else:
                _write_tif_with_rasterio(self._rs, path)
        elif ext in NC_EXTS:
            self._rs.to_netcdf(path, compute=True)
        else:
            # TODO: populate
            raise NotImplementedError()

    def eval(self):
        """
        Compute any applied operations and return the resulting raster.

        Note that the unerlying sources will be loaded into memory for the
        computations and the result will be fixed in memory.
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
        return Raster(_chunk(self._rs))

    def copy(self):
        return Raster(self._rs.copy())

    def replace_null(self, value):
        if not _is_scalar(value):
            raise TypeError("value must be a scalar")
        null_values = self._attrs["nodatavals"]
        rs = _map_chunk_function(
            self.copy(), _chunk_replace_null, (null_values, value)
        )
        return rs

    def remap_range(self, min, max, new_value):
        if np.isnan((min, max)).any():
            raise ValueError("min and max cannot be NaN")
        if min >= max:
            raise ValueError(f"min must be less than max: ({min}, {max})")
        rs = _map_chunk_function(
            self.copy(), _chunk_remap_range, (min, max, new_value)
        )
        return rs

    def _binary_arithmetic(self, raster_or_scalar, op):
        # TODO: handle mapping of list of values to bands
        # TODO: handle case where shapes match but geo references don't
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
        # Fall back to xarray implementation
        return Raster(value ** self._rs, self._attrs)

    def __pos__(self):
        return self

    def negate(self):
        # Don't need to copy attrs here
        return Raster(-self._rs)

    def __neg__(self):
        return self.negate()

    def log(self):
        # Don't need to copy attrs here
        return Raster(np.log(self._rs))

    def log10(self):
        # Don't need to copy attrs here
        return Raster(np.log10(self._rs))

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

    def __repr__(self):
        return repr(self._rs)
