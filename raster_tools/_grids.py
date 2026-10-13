import math

import geopandas as gpd
import numpy as np
import shapely
from affine import Affine
from odc.geo import resxy_
from odc.geo.crs import norm_crs
from odc.geo.geobox import GeoBox
from rasterio.warp import calculate_default_transform, transform_bounds


def build_coords(affine, shape_3d):
    """Build (x, y) 1D coordinate arrays from an affine and 3D shape.

    The affine must be axis-aligned (no shear/rotation).
    """
    ny, nx = shape_3d[1], shape_3d[2]
    y, x = (
        c.values for c in GeoBox((ny, nx), affine, None).coordinates.values()
    )
    return x, y


def build_x_coord(affine, shape_3d):
    return build_coords(affine, shape_3d)[0]


def build_y_coord(affine, shape_3d):
    return build_coords(affine, shape_3d)[1]


# Tolerance for geobox comparisons, as a fraction of pixel size. Some
# published products carry sub-pixel FP noise (observed up to ~1e-4 in CRS
# units) in what are otherwise shared grids; strict odc-geo equality would
# reject those, forcing unnecessary reprojections.
GRID_PIXEL_TOLERANCE = 1e-3


def grids_close(a, b, pixel_tolerance=GRID_PIXEL_TOLERANCE):
    """True if grids `a` and `b` share a CRS and shape and their cells line up.

    The origins must agree to within `pixel_tolerance` of a cell, and any
    difference in the cell size or rotation terms must not move a cell by
    more than that again across the whole grid. A per-cell difference that
    looks negligible can still shift the far cells of a large grid by whole
    cells, so it is judged by its effect across the grid.
    """
    if a.crs != b.crs or a.shape != b.shape:
        return False
    aa, bb = a.affine, b.affine
    cell = max(math.hypot(aa.a, aa.d), math.hypot(aa.b, aa.e))
    atol = pixel_tolerance * cell
    if abs(aa.c - bb.c) > atol or abs(aa.f - bb.f) > atol:
        return False
    ny, nx = a.shape
    drift_x = nx * abs(aa.a - bb.a) + ny * abs(aa.b - bb.b)
    drift_y = nx * abs(aa.d - bb.d) + ny * abs(aa.e - bb.e)
    return drift_x <= atol and drift_y <= atol


class GridMismatchError(ValueError):
    """Raised when grids do not share a cell lattice."""


def axis_step(coords):
    """Signed cell step along an axis, derived from its coordinates alone.

    Returns None for an axis of length 1.
    """
    if len(coords) < 2:
        # A length-1 axis has no cell size derivable from its coordinates,
        # so only lattice alignment can be checked along it.
        return None
    return (coords[-1] - coords[0]) / (len(coords) - 1)


def lattice_step(steps, ncells, dim):
    """The cell step shared by the given axis steps, or None if unknown.

    `steps` holds the signed step of each grid along `dim`, with None for an
    unknown step. `ncells` is the length of the longest axis. The first known
    step is returned. Raises GridMismatchError if two known steps differ in
    orientation, or in size by enough that the grids drift apart by more than
    GRID_PIXEL_TOLERANCE of a cell across `ncells` cells.
    """
    lattice = None
    for step in steps:
        if step is None:
            continue
        if lattice is None:
            lattice = step
            continue
        if np.sign(step) != np.sign(lattice):
            raise GridMismatchError(
                f"Raster grids do not match: the {dim} axis orientation"
                " differs (the coordinates increase in one raster and"
                " decrease in the other)."
            )
        drift = ncells * abs(step - lattice) / abs(lattice)
        if drift > GRID_PIXEL_TOLERANCE:
            raise GridMismatchError(
                "Raster grids do not match: resolution differs along"
                f" {dim} ({abs(lattice)} vs {abs(step)}; across {ncells}"
                f" cells the grids drift apart by {drift:g} cells)."
            )
    return lattice


def cell_offset(ref_start, start, lattice, dim):
    """Whole-cell offset of the cell center `start` from `ref_start`.

    The offset is in units of `lattice`, the shared signed cell step along
    `dim`. If `lattice` is None, the cell size is unknown and the coordinates
    must be equal, up to float noise. Raises GridMismatchError if the offset
    is not a whole number of cells, within GRID_PIXEL_TOLERANCE of a cell.
    """
    if lattice is None:
        # With no cell size to scale a tolerance, allow only float noise
        if math.isclose(start, ref_start, rel_tol=1e-9):
            return 0
        raise GridMismatchError(
            f"Raster grids do not match: the rasters are one cell wide along"
            f" {dim}, so the cell size is unknown, and their cell centers"
            f" differ ({ref_start} vs {start})."
        )
    shift = (start - ref_start) / lattice
    ishift = round(shift)
    if abs(shift - ishift) > GRID_PIXEL_TOLERANCE:
        raise GridMismatchError(
            "Raster grids do not match: grids are offset by a non-integer"
            f" number of cells ({shift:g} along {dim})."
        )
    return ishift


def are_all_grids_same(grids):
    if not grids:
        return True

    grids = [getattr(g, "geobox", g) for g in grids]
    gtest = grids[0]
    return all(grids_close(gtest, g) for g in grids[1:])


def _fit_between_poles(grid):
    """Keep a north-up geographic `grid` between latitudes -90 and 90.

    Rounding a grid up to whole cells can carry it past a pole, which
    reprojection cannot handle. Such a grid is moved north to end at -90, or
    south to end at 90. If its rows span more than 180 degrees, they are
    shortened so that the same number of rows spans -90 to 90 exactly, which
    keeps everything the grid covers at the cost of slightly non-square
    cells.
    """
    a = grid.affine
    if grid.crs is None or not grid.crs.geographic or a.e >= 0:
        return grid
    ny, nx = grid.shape
    cell = -a.e
    # Allow float noise, but no more: odc-geo cannot reproject onto a grid
    # even 1e-10 degrees past a pole
    eps = 1e-12
    top = a.f
    if top - ny * cell >= -90 - eps and top <= 90 + eps:
        return grid
    if ny * cell >= 180 - eps:
        top, cell = 90, 180 / ny
    elif top > 90:
        top = 90
    else:
        top = -90 + ny * cell
    return GeoBox((ny, nx), Affine(a.a, a.b, a.c, a.d, -cell, top), grid.crs)


def reproject_grid(grid, crs, resolution=None):
    """GDAL's default grid for reprojecting `grid` to `crs`.

    The grid is the one GDAL's ``GDALSuggestedWarpOutput2`` suggests, via
    ``rasterio.warp.calculate_default_transform``, which handles poles inside
    `grid`, crossing the antimeridian, and edge points that are off the
    globe. Where GDAL suggests no rows, or one row of zero height, as it does
    for a small grid around a pole, one row spanning the grid's latitudes
    from ``rasterio.warp.transform_bounds`` is used instead. If `resolution`
    is given, it replaces GDAL's cell size, keeping the top-left corner, and
    the cell counts are rounded up as rasterio does, so the grid still covers
    GDAL's extent. A geographic grid is then kept
    between the poles (see :func:`_fit_between_poles`). If `crs` is `grid`'s
    CRS, `grid` is returned, with its cell size changed to `resolution`
    (keeping the origin) if given.
    """
    if resolution is not None and resolution <= 0:
        raise ValueError("Resolution must be a positive value")
    dst_crs = norm_crs(crs)
    if dst_crs == grid.crs:
        if resolution is None:
            return grid
        return grid.zoom_to(resolution=resolution)
    ny, nx = grid.shape
    a, width, height = calculate_default_transform(
        grid.crs, dst_crs, nx, ny, *grid.boundingbox
    )
    if height == 0 or a.e == 0:
        # For a small grid around a pole, GDAL's cells are a few degrees
        # tall and its row count is rounded to the nearest whole number,
        # which gives no rows, or one row of zero height at the south pole.
        # Use one row spanning the grid's latitudes instead.
        _, south, _, north = transform_bounds(
            grid.crs, dst_crs, *grid.boundingbox, densify_pts=21
        )
        height = 1
        a = Affine(a.a, a.b, a.c, a.d, south - north, north)
    width = max(width, 1)
    if resolution is not None:
        # The same sizing as calculate_default_transform's resolution option
        width = math.ceil(width * (a.a / resolution))
        height = math.ceil(height * (a.e / -resolution))
        a = Affine(resolution, a.b, a.c, a.d, -resolution, a.f)
    return _fit_between_poles(GeoBox((height, width), a, dst_crs))


def get_grid_bbox(grid):
    return grid.extent.geom


def get_grid_bounds(grid):
    return get_grid_bbox(grid).bounds


EMPTY_INTERSECTION_MSG = (
    "The intersection of the given grids is empty: the grids do not overlap"
)


def combine_grids(grids, how=None, dst_crs=None, resolution=None):
    """Produce a grid that combines the input grids

    Parameters
    ----------
    grids : list of GeoBox
        The input GeoBox grids to combine.
    how : str, optional
        How to combine the grids. Either ``"union"`` or ``"intersection"``.
        Union takes the bounding box of the convex hull of the individual grid
        bounding boxes. Intersection takes the bounding box of the intersection
        of the bounding boxes of the grids, and raises a ``ValueError`` if the
        grids do not overlap, including grids that only touch. Default is
        ``"union"``.
    dst_crs : CRS-like, optional
        The CRS of the resulting grid. If ``None``, the CRS of the first input
        grid is used. When the input grids do not share a CRS, they are
        reprojected to ``dst_crs`` (or the first grid's CRS) before being
        combined, each taking the extent of GDAL's default grid for it in
        ``dst_crs``. A geographic result built from reprojected grids is kept
        between latitudes -90 and 90 (see :func:`_fit_between_poles`).
        Default is ``None``.
    resolution : scalar, optional
        Pixel resolution of the resulting grid, in units of ``dst_crs``. If
        ``None``, the x and y resolutions of the first input grid are used
        when `dst_crs` is its CRS. Otherwise, the x resolution of GDAL's
        default grid for reprojecting the first input grid to `dst_crs` is
        used for both axes, except that a geographic grid's rows can be
        shortened to fit between the poles. Default is ``None``.

    Returns
    -------
    GeoBox
        The resulting GeoBox object.

    """
    if how is None:
        how = "union"
    elif how not in ("intersection", "union"):
        raise ValueError("how must be one of intersection, union, or None")

    if are_all_grids_same(grids):
        if dst_crs is None and resolution is None:
            return grids[0]
        return reproject_grid(
            grids[0],
            dst_crs if dst_crs is not None else grids[0].crs,
            resolution=resolution,
        )

    dst_crs = norm_crs(dst_crs if dst_crs is not None else grids[0].crs)
    dst_resolution = resolution
    if resolution is None:
        res = reproject_grid(grids[0], dst_crs).resolution
        resolution = abs(res.x)
        dst_resolution = resolution
        if dst_crs == grids[0].crs:
            # Keep the first grid's cell size along both axes
            dst_resolution = resxy_(abs(res.x), -abs(res.y))
    # A grid in dst_crs is sized to the resolution; any other takes the
    # extent of GDAL's default grid for it in dst_crs.
    grids_dst = [
        reproject_grid(
            g, dst_crs, resolution=resolution if g.crs == dst_crs else None
        )
        for g in grids
    ]
    bboxes_dst = [get_grid_bbox(g) for g in grids_dst]
    if how == "union":
        total_bounds_dst = gpd.GeoSeries(bboxes_dst, crs=dst_crs).total_bounds
        dst_grid = GeoBox.from_bbox(
            total_bounds_dst,
            crs=dst_crs,
            resolution=dst_resolution,
            tight=True,
        )
    else:
        bbox = shapely.intersection_all(bboxes_dst).normalize()
        # Grids that only touch intersect in a line or point, which has no
        # cells to keep.
        if bbox.is_empty or bbox.area == 0:
            raise ValueError(EMPTY_INTERSECTION_MSG)
        dst_grid = GeoBox.from_bbox(
            bbox.bounds, crs=dst_crs, resolution=dst_resolution, tight=True
        )
    if any(g.crs != dst_crs for g in grids):
        dst_grid = _fit_between_poles(dst_grid)
    return dst_grid
