"""Geo-aware block descriptors for use with dask map_blocks/map_overlap.

This module provides :class:`GeoBlockInfo`, the geo-aware analog of dask's
``block_info`` dict: a per-block metadata object carrying the block's
:class:`~odc.geo.geobox.GeoBox`, its band/row/col slices into the parent
array, and helpers for padding/shifting the block window and reconstructing
an :class:`xarray.DataArray` for the block's data.

The future ``geo_map_blocks`` / ``geo_map_overlap`` wrappers are designed to
build on this primitive: each user callback receives a coordinated
``DataArray`` (with band/y/x coords, CRS, nodata) instead of a raw NumPy
block.
"""

import dask.array as da
import numpy as np
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

from raster_tools.dask_utils import chunks_to_array_locations
from raster_tools.dtypes import is_int

__all__ = ["GeoBlockInfo", "geo_block_infos_as_dask"]


class GeoBlockInfo:
    """Geo-aware metadata for a single dask block.

    Mirrors dask's per-block ``block_info`` dict, but carries the block's
    geographic grid (a 2D :class:`~odc.geo.geobox.GeoBox`) plus enough
    parent-array context for chunk-local code to locate itself within the
    full raster. Holds no pixel data.

    Parameters
    ----------
    geobox : odc.geo.geobox.GeoBox
        The 2D grid (``(ny, nx)``) for this block. Source of truth for
        ``affine``, ``crs``, and ``bbox``.
    band_slice, row_slice, col_slice : slice
        Half-open ``[start, stop)`` slices identifying where this block
        sits inside the parent array's band, row, and column axes.
    chunk_location : tuple of int
        The ``(bi, yi, xi)`` block index in the parent array's ``numblocks``
        grid. Preserved across mutators (still names the originating block
        even after padding/shifting).
    parent_affine : affine.Affine
        Affine of the full raster this block came from. Needed by callers
        like ``Raster.to_points`` to map block-local cells back to parent
        row/col.
    parent_shape : tuple of int
        ``(nbands, ny, nx)`` of the full raster. Used for parent-array
        ravel-index computations.
    """

    def __init__(
        self,
        geobox,
        band_slice,
        row_slice,
        col_slice,
        chunk_location,
        parent_affine,
        parent_shape,
    ):
        self.geobox = geobox
        self.band_slice = band_slice
        self.row_slice = row_slice
        self.col_slice = col_slice
        self.chunk_location = chunk_location
        self.parent_affine = parent_affine
        self.parent_shape = parent_shape

    def __repr__(self):
        return (
            f"<{type(self).__module__}.{type(self).__name__}"
            f" shape={self.shape}, chunk_location={self.chunk_location}>"
        )

    @property
    def affine(self):
        return self.geobox.affine

    @property
    def crs(self):
        return self.geobox.crs

    @property
    def bbox(self):
        return self.geobox.extent.geom

    @property
    def shape(self):
        """3D ``(nbands, ny, nx)`` block shape."""
        return (
            self.band_slice.stop - self.band_slice.start,
            self.geobox.shape.y,
            self.geobox.shape.x,
        )

    @property
    def x(self):
        return self.geobox.coordinates["x"].values

    @property
    def y(self):
        return self.geobox.coordinates["y"].values

    @property
    def band(self):
        return np.arange(self.band_slice.start, self.band_slice.stop)

    def to_dataarray(self, data, *, mask=None, nodata=None):
        """
        Wrap a block's pixel data as a coordinated :class:`xarray.DataArray`.

        Builds an ``xr.DataArray`` with ``band``, ``y``, ``x`` coords drawn
        from this block's geobox and band slice, attaches the CRS via
        ``rio.write_crs``, and (optionally) writes ``nodata``.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel data with shape matching :attr:`shape`.
        mask : numpy.ndarray, optional
            Boolean mask with the same shape as ``data``. If provided,
            this method returns a 2-tuple of ``(xdata, xmask)`` with
            matching coords.
        nodata : scalar, optional
            Nodata value to write into the data DataArray's metadata.

        Returns
        -------
        xarray.DataArray or tuple of xarray.DataArray
            ``xdata`` if ``mask`` is None; otherwise ``(xdata, xmask)``.
        """
        if data.shape != self.shape:
            raise ValueError(
                f"data shape {data.shape} does not match block shape "
                f"{self.shape}"
            )
        coords = {"band": self.band, "y": self.y, "x": self.x}
        xdata = xr.DataArray(
            data, dims=("band", "y", "x"), coords=coords
        ).rio.set_spatial_dims(y_dim="y", x_dim="x")
        if self.crs is not None:
            xdata = xdata.rio.write_crs(self.crs)
        if nodata is not None:
            xdata = xdata.rio.write_nodata(nodata)
        if mask is None:
            return xdata
        if mask.shape != self.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match block shape "
                f"{self.shape}"
            )
        xmask = xr.DataArray(
            mask, dims=("band", "y", "x"), coords=coords
        ).rio.set_spatial_dims(y_dim="y", x_dim="x")
        if self.crs is not None:
            xmask = xmask.rio.write_crs(self.crs)
        return xdata, xmask

    # -- mutators ----------------------------------------------------------
    #
    # All mutators return a NEW GeoBlockInfo (immutable semantics). They
    # adjust the geobox (translating the affine when growing on the top/left)
    # and update row_slice / col_slice to track the window within the parent
    # array. parent_affine, parent_shape, band_slice, and chunk_location are
    # preserved.
    #
    # Negative pad values trim that side. Positive shift values move the
    # window down (pad_y) or right (pad_x).

    def _resize(self, dy_top, dy_bottom, dx_left, dx_right):
        if not all(is_int(v) for v in (dy_top, dy_bottom, dx_left, dx_right)):
            raise TypeError("resize amounts must be integers")
        new_ny = self.geobox.shape.y + dy_top + dy_bottom
        new_nx = self.geobox.shape.x + dx_left + dx_right
        if new_ny < 0 or new_nx < 0:
            raise ValueError(
                f"resize would produce negative shape: ({new_ny}, {new_nx})"
            )
        new_affine = self.geobox.affine
        if dy_top != 0 or dx_left != 0:
            # Affine.translation(xoff, yoff): xoff in cols, yoff in rows.
            # Growing on top/left moves the upper-left pixel center
            # outward, which means a negative col/row translation in the
            # affine's input space.
            new_affine = new_affine * Affine.translation(-dx_left, -dy_top)
        new_geobox = GeoBox((new_ny, new_nx), new_affine, self.crs)
        return GeoBlockInfo(
            new_geobox,
            self.band_slice,
            slice(
                self.row_slice.start - dy_top,
                self.row_slice.stop + dy_bottom,
            ),
            slice(
                self.col_slice.start - dx_left,
                self.col_slice.stop + dx_right,
            ),
            self.chunk_location,
            self.parent_affine,
            self.parent_shape,
        )

    def pad_y(self, top, bottom):
        """Grow (or trim, with negative values) the block on top/bottom."""
        return self._resize(top, bottom, 0, 0)

    def pad_x(self, left, right):
        """Grow (or trim, with negative values) the block on left/right."""
        return self._resize(0, 0, left, right)

    def pad(self, y, x=None):
        """Symmetrically pad the block in both axes by ``y`` and ``x``.

        ``x`` defaults to ``y``. Each axis is grown by twice the given
        value because both sides are padded.
        """
        if x is None:
            x = y
        return self._resize(y, y, x, x)

    def shift_y(self, n):
        """Slide the block window vertically. Positive ``n`` moves down."""
        return self._resize(-n, n, 0, 0)

    def shift_x(self, n):
        """Slide the block window horizontally. Positive ``n`` moves right."""
        return self._resize(0, 0, -n, n)

    def shift(self, y, x=None):
        """Slide the window in both axes."""
        if x is None:
            x = y
        return self._resize(-y, y, -x, x)


def _slices_from_locations(locations):
    return [slice(start, stop) for start, stop in locations]


def geo_block_infos(raster):
    """Build a NumPy object-array of :class:`GeoBlockInfo` for a Raster.

    The returned array has shape ``raster.data.numblocks`` and each entry
    describes the corresponding dask block.
    """
    parent_geobox = raster.geobox
    parent_affine = raster.affine
    parent_shape = raster.shape
    chunks = raster.data.chunks
    numblocks = raster.data.numblocks
    band_slices = _slices_from_locations(chunks_to_array_locations(chunks[0]))
    row_slices = _slices_from_locations(chunks_to_array_locations(chunks[1]))
    col_slices = _slices_from_locations(chunks_to_array_locations(chunks[2]))
    out = np.empty(numblocks, dtype=object)
    for bi, yi, xi in np.ndindex(numblocks):
        rs = row_slices[yi]
        cs = col_slices[xi]
        block_geobox = parent_geobox[rs, cs]
        out[bi, yi, xi] = GeoBlockInfo(
            block_geobox,
            band_slices[bi],
            rs,
            cs,
            (bi, yi, xi),
            parent_affine,
            parent_shape,
        )
    return out


def geo_block_infos_as_dask(raster):
    """Return a chunksize-1 dask object-array of :class:`GeoBlockInfo`.

    Suitable as a side-input to :func:`dask.array.map_blocks` aligned 1:1
    with the data blocks.
    """
    return da.from_array(geo_block_infos(raster), chunks=1)
