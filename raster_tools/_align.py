from typing import NamedTuple

import dask.array as da
import numpy as np
import rasterio as rio
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

from raster_tools import _grids
from raster_tools._grids import EMPTY_INTERSECTION_MSG, build_coords
from raster_tools.masking import get_default_null_value
from raster_tools.raster import (
    Raster,
    data_to_raster,
    get_mask_from_data,
    get_raster,
    grid_transform,
)
from raster_tools.warp import SUPPORTED_RESAMPLE_METHODS, reproject

__all__ = ["align"]

_DIMS = ("x", "y")
_MISSING_CRS_MSG = (
    "When a raster or the destination grid has no CRS, the raster can only"
    " be aligned to the grid if they share a cell lattice (the same cell size"
    " and a whole-cell offset). Any other alignment is a reprojection, which"
    " requires a CRS. Set a missing CRS with Raster.set_crs first."
)


class Target(NamedTuple):
    """A destination grid and the exact coordinates given to every output.

    `crs_given` is False when the grid had no CRS and took the CRS of the
    rasters instead.
    """

    geobox: GeoBox
    x: np.ndarray
    y: np.ndarray
    crs_given: bool = True

    @property
    def crs(self):
        return self.geobox.crs

    def coords(self, dim):
        return self.x if dim == "x" else self.y

    def step(self, dim):
        affine = self.geobox.affine
        return affine.a if dim == "x" else affine.e


def _coords(raster, dim):
    return raster._ds[dim].data


def _stored_transform(raster):
    """The affine transform stored with the raster's data."""
    return raster._ds.rio.transform()


def _grid_affine(raster):
    """The affine transform of the raster's grid.

    The origin comes from the coordinates. The cell size comes from the
    coordinates along an axis with more than one cell, and from the stored
    transform along a length-1 axis.
    """
    return grid_transform(raster._ds)


def _raster_geobox(raster):
    """The raster's grid as a GeoBox.

    Built from the raster's own transform rather than Raster.geobox, which
    is None for a raster with no CRS that is one cell wide along an axis.
    """
    ny, nx = raster.shape[1:]
    return GeoBox((ny, nx), _grid_affine(raster), raster.crs)


def _step(affine, dim):
    return affine.a if dim == "x" else affine.e


def _origin(affine, dim):
    return affine.c if dim == "x" else affine.f


def target_from_geobox(geobox, crs_given=True):
    """Build a Target from a GeoBox, flipped so x increases and y decreases."""
    affine = geobox.affine
    ny, nx = geobox.shape
    a, c, e, f = affine.a, affine.c, affine.e, affine.f
    if a < 0:
        a, c = -a, c + a * nx
    if e > 0:
        e, f = -e, f + e * ny
    if (a, c, e, f) != (affine.a, affine.c, affine.e, affine.f):
        geobox = GeoBox(geobox.shape, Affine(a, 0, c, 0, e, f), geobox.crs)
    x, y = build_coords(geobox.affine, (1, ny, nx))
    return Target(geobox, x, y, crs_given)


def parse_dst_grid(dst_grid, dst_crs, resolution):
    """Validate the destination grid arguments and return a GeoBox or None."""
    if dst_grid is None:
        return None
    if isinstance(dst_grid, (str, Raster)):
        dst_grid = _raster_geobox(get_raster(dst_grid))
    elif not isinstance(dst_grid, GeoBox):
        raise TypeError(
            f"Expected dst_grid to have type GeoBox. Got {type(dst_grid)}"
        )
    affine = dst_grid.affine
    if affine.b or affine.d:
        raise ValueError(
            "dst_grid must not be rotated or sheared: its affine transform"
            f" has nonzero rotation/shear terms (b={affine.b}, d={affine.d})."
        )
    if dst_crs is not None:
        parsed = rio.CRS.from_user_input(dst_crs)
        if dst_grid.crs != parsed:
            raise ValueError(
                "dst_crs does not match dst_grid.crs: "
                f"{parsed} vs {dst_grid.crs}"
            )
    if resolution is not None:
        raise ValueError(
            "resolution cannot be specified together with dst_grid"
        )
    return dst_grid


def _common_crs(rasters):
    """The CRS shared by the rasters that have one.

    Raises GridMismatchError if two rasters have different CRSs.
    """
    crs = None
    for r in rasters:
        rcrs = r.crs
        if rcrs is None:
            continue
        if crs is None:
            crs = rcrs
        elif rcrs != crs:
            raise _grids.GridMismatchError("CRS differs")
    return crs


def _crs_compatible(a, b):
    return a is None or b is None or a == b


def _lattice_target(rasters, join, dst_crs, resolution):
    """Build the target grid on the cell lattice that the rasters share.

    The rasters share a lattice if they have the same CRS (a missing CRS
    matches any CRS) and cell size, and are offset from each other by whole
    numbers of cells. Returns None if they do not share one, or if `dst_crs`
    or `resolution` call for a different lattice.
    """
    affines = [_grid_affine(r) for r in rasters]
    try:
        crs = _common_crs(rasters)
        steps = {}
        offsets = {}
        for dim in _DIMS:
            coords = [_coords(r, dim) for r in rasters]
            ncells = max(len(c) for c in coords)
            step = _grids.lattice_step(
                [_step(a, dim) for a in affines], ncells, dim
            )
            offsets[dim] = [
                _grids.cell_offset(coords[0][0], c[0], step, dim)
                for c in coords
            ]
            steps[dim] = step
    except _grids.GridMismatchError:
        return None
    if dst_crs is not None:
        if crs is not None and crs != dst_crs:
            return None
        crs = dst_crs
    if resolution is not None and not all(
        np.isclose(abs(step), resolution) for step in steps.values()
    ):
        return None

    windows = {}
    for dim in _DIMS:
        spans = [
            (off, off + len(_coords(r, dim)))
            for r, off in zip(rasters, offsets[dim], strict=True)
        ]
        if join == "inner":
            window = (max(s[0] for s in spans), min(s[1] for s in spans))
        else:
            window = (min(s[0] for s in spans), max(s[1] for s in spans))
        windows[dim] = window
    if any(start >= stop for start, stop in windows.values()):
        raise ValueError(EMPTY_INTERSECTION_MSG)

    axes = {}
    for dim in _DIMS:
        start, stop = windows[dim]
        # Reuse the coordinates and transform of a raster that spans the
        # window exactly, so that it can pass through unchanged.
        source = next(
            (
                (r, a)
                for r, a, off in zip(
                    rasters, affines, offsets[dim], strict=True
                )
                if (off, off + len(_coords(r, dim))) == (start, stop)
            ),
            None,
        )
        if source is None:
            step = steps[dim]
            coords = (
                _coords(rasters[0], dim)[0] + np.arange(start, stop) * step
            )
            axes[dim] = (step, coords[0] - step / 2, coords)
        else:
            r, a = source
            axes[dim] = (_step(a, dim), _origin(a, dim), _coords(r, dim))
    (xstep, xorigin, x), (ystep, yorigin, y) = axes["x"], axes["y"]
    affine = Affine(xstep, 0, xorigin, 0, ystep, yorigin)
    geobox = GeoBox((len(y), len(x)), affine, crs)
    return Target(geobox, x, y)


def build_target(rasters, join, dst_grid=None, dst_crs=None, resolution=None):
    """Build the Target that `rasters` are aligned to.

    `dst_grid` must already be validated by `parse_dst_grid`. If it is None,
    the target is built from the rasters according to `join`, `dst_crs` and
    `resolution`.
    """
    if dst_grid is not None:
        if dst_grid.crs is not None:
            return target_from_geobox(dst_grid)
        # A grid without a CRS takes the CRS of the rasters
        try:
            crs = _common_crs(rasters)
        except _grids.GridMismatchError:
            raise ValueError(
                "dst_grid has no CRS and the rasters have different CRSs, so"
                " the CRS of the result is ambiguous. Give dst_grid a CRS."
            ) from None
        if crs is None:
            return target_from_geobox(dst_grid)
        return target_from_geobox(
            GeoBox(dst_grid.shape, dst_grid.affine, crs), crs_given=False
        )
    if dst_crs is not None:
        dst_crs = rio.CRS.from_user_input(dst_crs)
    target = _lattice_target(rasters, join, dst_crs, resolution)
    if target is not None:
        return target
    if any(r.crs is None for r in rasters):
        raise ValueError(_MISSING_CRS_MSG)
    grid = _grids.combine_grids(
        [_raster_geobox(r) for r in rasters],
        how="union" if join == "outer" else "intersection",
        dst_crs=dst_crs,
        resolution=resolution,
    )
    return target_from_geobox(grid)


def _target_offsets(raster, target):
    """Whole-cell offsets of the raster's first cell in the target grid.

    Returns a dict mapping dim to offset, or None if the raster does not
    share the target's cell lattice.
    """
    affine = _grid_affine(raster)
    offsets = {}
    for dim in _DIMS:
        tcoords = target.coords(dim)
        coords = _coords(raster, dim)
        ncells = max(len(tcoords), len(coords))
        try:
            step = _grids.lattice_step(
                [target.step(dim), _step(affine, dim)], ncells, dim
            )
            offsets[dim] = _grids.cell_offset(tcoords[0], coords[0], step, dim)
        except _grids.GridMismatchError:
            return None
    return offsets


def _is_on_target(raster, target):
    return (
        target.crs == raster.crs
        and all(
            np.array_equal(_coords(raster, dim), target.coords(dim))
            for dim in _DIMS
        )
        # The stored transform also gives the cell size along a length-1
        # axis, so it must match exactly too.
        and _stored_transform(raster) == target.geobox.affine
    )


def _put_on_target(ds, target):
    """Give a raster Dataset the target's coordinates, CRS and transform."""
    crs = target.crs if target.crs is not None else ds.rio.crs
    ds = ds.assign_coords(x=target.x, y=target.y)
    if crs is not None:
        ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_transform(target.geobox.affine)
    return Raster(ds, _fast_path=True)


def _slice_and_pad(raster, target, offsets):
    """Cut the raster to the target grid and fill uncovered cells with null.

    Filled cells are masked. An unmasked raster that needs filling gets the
    default null value for its dtype, and its cells that hold that value are
    masked too, as reprojecting would do.
    """
    ds = raster._ds
    shape = {"y": len(target.y), "x": len(target.x)}
    slices = {}
    pads = {}
    for dim in ("y", "x"):
        offset = offsets[dim]
        size = shape[dim]
        lo = min(max(offset, 0), size)
        hi = min(max(offset + ds.sizes[dim], 0), size)
        slices[dim] = slice(lo - offset, max(hi, lo) - offset)
        pads[dim] = (lo, size - max(hi, lo))
    nv = raster.null_value
    padded = any(p != (0, 0) for p in pads.values())
    new_nv = padded and nv is None
    if new_nv:
        nv = get_default_null_value(raster.dtype)

    if any(s.start == s.stop for s in slices.values()):
        # The raster lies entirely outside the target grid
        out_shape = (raster.nbands, shape["y"], shape["x"])
        chunks = (1, *raster.data.chunksize[1:])
        data = da.full(out_shape, nv, dtype=raster.dtype, chunks=chunks)
        mask = da.ones(out_shape, dtype=bool, chunks=chunks)
    else:
        index = (slice(None), slices["y"], slices["x"])
        data = raster.data[index]
        mask = raster.mask[index]
        if new_nv:
            mask = mask | get_mask_from_data(data, nv)
        if padded:
            pad_width = ((0, 0), pads["y"], pads["x"])
            data = da.pad(data, pad_width, constant_values=nv)
            mask = da.pad(mask, pad_width, constant_values=True)

    dims = ("band", "y", "x")
    xdata = xr.DataArray(data, dims=dims, attrs=ds.raster.attrs)
    xdata.encoding = dict(ds.raster.encoding)
    if nv is not None:
        xdata = xdata.rio.write_nodata(nv)
    xmask = xr.DataArray(mask, dims=dims, attrs=ds.mask.attrs)
    out = xr.Dataset(
        {"raster": xdata, "mask": xmask},
        coords={"band": ds.band.data},
    )
    return _put_on_target(out, target)


def conform(raster, target, resampling_method):
    """Put a raster on the target grid.

    Rasters already on the target grid are returned as new wrappers of the
    same data. Rasters that share the target's cell lattice are cut and
    padded. All other rasters are reprojected.
    """
    crs = raster.crs
    if _crs_compatible(crs, target.crs):
        offsets = _target_offsets(raster, target)
        if offsets is not None:
            if _is_on_target(raster, target):
                return Raster(raster._ds, _fast_path=True)
            return _slice_and_pad(raster, target, offsets)
        if crs is None or target.crs is None or not target.crs_given:
            raise ValueError(_MISSING_CRS_MSG)
    reprojected = reproject(
        raster, target.geobox, resample_method=resampling_method
    )
    return _put_on_target(reprojected._ds, target)


def raster_on_target(data, target, nv, mask=None):
    """Create a Raster on the target grid from a (band, y, x) data array."""
    raster = data_to_raster(
        data, mask=mask, x=target.x, y=target.y, crs=target.crs, nv=nv
    )
    return _put_on_target(raster._ds, target)


def align(
    rasters,
    *,
    join="inner",
    dst_grid=None,
    dst_crs=None,
    resolution=None,
    resampling_method="nearest",
):
    """Put rasters on a common grid.

    Every output has the same CRS, affine transform, and x/y coordinates, so
    the outputs can be combined cell by cell. Bands, dtypes, and chunking are
    left as they are. This is a lazy operation.

    Each raster is handled in the cheapest way that puts it on the
    destination grid:

    * A raster that is already on the destination grid is returned as is,
      wrapped in a new Raster object.
    * A raster that has the destination grid's CRS and cell size and is
      offset from it by a whole number of cells is cut to the destination
      extent and padded with null cells where it does not cover it. Its
      values are not resampled.
    * Any other raster is reprojected onto the destination grid using
      `resampling_method`.

    A raster without a CRS is treated as having the destination CRS, as long
    as it does not need to be reprojected.

    Parameters
    ----------
    rasters : list of Raster or str
        The rasters to align. Path strings are opened with
        :func:`raster_tools.get_raster`.
    join : str, optional
        How to combine the input grids when building the destination grid.
        Only consulted when `dst_grid` is not provided. Valid options are:

        'inner'
            Use the intersection of the input grid bounds. This is the
            default. Inputs that do not overlap raise a ``ValueError``.
        'outer'
            Use the union of the input grid bounds.
    dst_grid : odc.geo.GeoBox, raster_tools.Raster, str, optional
        The destination grid. This can be a :py:class:`odc.geo.GeoBox`,
        :py:class:`raster_tools.Raster` object, or a path str. If the input
        is a raster object or path, only its grid is used. The output grid is
        `dst_grid` in north-up orientation: x increases and y decreases, even
        if `dst_grid` is flipped. A rotated or sheared `dst_grid` raises a
        ``ValueError``. A `dst_grid` without a CRS takes the CRS shared by the
        inputs, and raises a ``ValueError`` if the inputs have different CRSs.
        The default is to build a grid from the inputs according to `join`.
        Passing `dst_grid` together with a `dst_crs` that differs from its
        CRS, or together with `resolution`, is an error.
    dst_crs : CRS-like, str, int, optional
        The CRS of the destination grid. This can be anything that can be
        parsed by :py:meth:`rasterio.CRS.from_user_input`. Only consulted
        when `dst_grid` is not provided. The default is the CRS of the
        inputs.
    resolution : scalar, optional
        Cell size of the destination grid, in units of the destination CRS.
        Only consulted when `dst_grid` is not provided. The default is the
        cell size of the inputs.
    resampling_method : str, optional
        Resampling method used when reprojecting rasters onto the destination
        grid. The default is ``'nearest'``. See
        :func:`raster_tools.reproject` for the valid methods.

    Returns
    -------
    tuple of Raster
        The aligned rasters, in input order.

    Notes
    -----
    When `dst_grid` is not provided and the inputs share a CRS, cell size,
    and cell lattice (whole-cell offsets), the destination grid lies on that
    lattice and no input is resampled. Otherwise, the destination CRS and
    cell size default to those of the first raster in `rasters`, and
    reordering the inputs can change the destination grid.

    Each raster's cell size is taken from its grid (:attr:`Raster.affine`).
    Along an axis where a raster is one cell wide, that is the cell size
    recorded in its transform, or the cell size of its other axis if the
    raster was built from coordinates alone.

    A raster with no null value that is padded or reprojected is given the
    default null value for its dtype, so that the new cells can be marked
    as null. Any of its existing cells that hold that value are then null
    too. For a bool raster the default null value is ``True``, so its padded
    cells hold ``True`` but are masked, as are its existing ``True`` cells.

    A raster whose cell size and offset are within 0.001 of a cell of the
    destination lattice is cut and padded rather than resampled, so
    sub-pixel noise in its transform is dropped.

    Chunking is preserved, except that padding adds chunks at the edges of
    the grid. Use :meth:`Raster.chunk` to rechunk the outputs if needed.

    Examples
    --------
    Crop two overlapping tiles to their shared extent, then add them:

    >>> a, b = rts.align([tile_a, tile_b])  # doctest: +SKIP
    >>> total = a + b  # doctest: +SKIP

    Put a raster on the grid of a reference raster:

    >>> (aligned,) = rts.align([src], dst_grid=reference)  # doctest: +SKIP

    """
    if isinstance(rasters, (str, Raster)):
        raise TypeError("rasters must be a list of rasters or paths")
    if join not in ("inner", "outer"):
        raise ValueError("join must be one of 'inner' or 'outer'")
    resampling_method = resampling_method or "nearest"
    if resampling_method not in SUPPORTED_RESAMPLE_METHODS:
        raise ValueError("Invalid resampling method")
    dst_grid = parse_dst_grid(dst_grid, dst_crs, resolution)
    rasters = [get_raster(r) for r in rasters]
    if not rasters:
        raise ValueError("No rasters provided")

    target = build_target(rasters, join, dst_grid, dst_crs, resolution)
    return tuple(conform(r, target, resampling_method) for r in rasters)
