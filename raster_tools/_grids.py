import dask.array as da
import geopandas as gpd
import numpy as np
import shapely
from odc.geo.geobox import GeoBox

# Tolerance for geobox comparisons, as a fraction of pixel size. Some
# published products carry sub-pixel FP noise (observed up to ~1e-4 in CRS
# units) in what are otherwise shared grids; strict odc-geo equality would
# reject those, forcing unnecessary reprojections.
GRID_PIXEL_TOLERANCE = 1e-3


def grids_close(a, b, pixel_tolerance=GRID_PIXEL_TOLERANCE):
    if a.crs != b.crs or a.shape != b.shape:
        return False
    aa, bb = a.affine, b.affine
    atol = pixel_tolerance * max(abs(aa.a), abs(aa.e))
    return all(
        abs(x - y) <= atol
        for x, y in zip(
            (aa.a, aa.b, aa.c, aa.d, aa.e, aa.f),
            (bb.a, bb.b, bb.c, bb.d, bb.e, bb.f),
            strict=True,
        )
    )


def are_all_grids_same(grids):
    if not grids:
        return True

    grids = [getattr(g, "geobox", g) for g in grids]
    gtest = grids[0]
    return all(grids_close(gtest, g) for g in grids[1:])


def _build_empty_raster_from_grid(grid, dtype, nodata):
    import raster_tools as rts

    data = da.full((grid.shape.y, grid.shape.x), nodata, dtype=dtype)
    # coordinates is an ordered (y-axis, x-axis) mapping; key names vary by
    # CRS (e.g. "x"/"y" for projected, "longitude"/"latitude" for 4326).
    y_coord, x_coord = grid.coordinates.values()
    return rts.data_to_raster(
        data, x=x_coord.values, y=y_coord.values, crs=grid.crs, nv=nodata
    )


def reproject_grid(grid, crs, resolution=None):
    dummy_raster = _build_empty_raster_from_grid(grid, int, 0)
    return dummy_raster.reproject(crs, resolution=resolution).geobox


def get_grid_bbox(grid):
    return grid.extent.geom


def get_grid_bounds(grid):
    return get_grid_bbox(grid).bounds


def combine_grids(grids, how=None, dst_crs=None):
    """Produce a grid that combines the input grids

    Parameters
    ----------
    grids : list of GeoBox
        The input GeoBox grids to combine.
    how : str, optional
        How to combine the grids. Either ``"union"`` or ``"intersection"``.
        Union takes the bounding box of the convex hull of the individual grid
        bounding boxes. Intersection takes the bounding box of the intersection
        of the bounding boxes of the grids. Default is ``"union"``.
    dst_crs : CRS-like, optional
        The CRS of the resulting grid. If ``None``, the CRS of the first input
        grid is used. When the input grids do not share a CRS, they are
        reprojected to ``dst_crs`` (or the first grid's CRS) before being
        combined. Default is ``None``.

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
        if dst_crs is None:
            return grids[0]
        else:
            return reproject_grid(grids[0], dst_crs)

    if dst_crs is None:
        dst_crs = grids[0].crs
    resolution = np.abs(grids[0].resolution.x)
    grids_dst = [
        reproject_grid(g, dst_crs, resolution=resolution) for g in grids
    ]
    bboxes_dst = [get_grid_bbox(g) for g in grids_dst]
    if how == "union":
        total_bounds_dst = gpd.GeoSeries(bboxes_dst, crs=dst_crs).total_bounds
        dst_grid = GeoBox.from_bbox(
            total_bounds_dst, crs=dst_crs, resolution=resolution, tight=True
        )
    else:
        bbox = shapely.intersection_all(bboxes_dst).normalize()
        if bbox.is_empty:
            raise ValueError("The intersection of the given grids is empty")
        dst_grid = GeoBox.from_bbox(
            bbox.bounds, crs=dst_crs, resolution=resolution, tight=True
        )
    return dst_grid
