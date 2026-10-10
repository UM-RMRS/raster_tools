"""Public operations keep the cell size of rasters one cell wide on an axis.

Along an axis that is one cell wide, the cell size cannot be derived from the
coordinates. It comes only from the transform stored on the grid-mapping
coordinate (see raster_tools.raster.grid_transform). xarray operations such as
xr.where drop that coordinate's attributes, and an operation that loses it
silently turns non-square cells into square ones. Each case in OPERATIONS runs
on non-square one-row, one-column and single-cell rasters, with and without
nulls, and checks the cell size of every Raster it returns.

To cover a new operation, add a Case to OPERATIONS. Operations left out:

* reprojection to another CRS (Raster.reproject, warp.reproject, align,
  mosaic and stack_bands with dst_crs): the output grid is legitimately
  different and its cell size is chosen by GDAL/odc-geo.
* Raster.to_quadrants: splits the grid and needs at least two cells per axis.
* Reductions (Raster.sum, Raster.mean, ...), Raster.xy and Raster.index:
  return scalars or arrays, not Rasters.
* zonal.zonal_stats, zonal.extract_points_eager, Raster.to_polygons,
  Raster.to_vector, Raster.to_points and general.model_predict_vector: return
  tables or vectors.
* Raster.save, Raster.save_chunks, open_dataset and read_color_table: file
  I/O, where GDAL stores the transform in the file.
* distance.pa_proximity, pa_allocation, pa_direction, cda_cost_distance,
  cda_traceback and cda_allocation: covered by the proximity_analysis and
  cost_distance_analysis cases, which return all of their outputs.
"""

from typing import Callable, NamedTuple

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from odc.geo.geobox import GeoBox
from shapely.geometry import LineString, box

import raster_tools as rts
from raster_tools import clipping, focal, general, line_stats, surface
from raster_tools.raster import (
    Raster,
    data_to_raster,
    data_to_raster_like,
    data_to_xr_raster,
    data_to_xr_raster_ds,
    data_to_xr_raster_ds_like,
    data_to_xr_raster_like,
    dataarray_to_raster,
    dataarray_to_xr_raster,
    dataarray_to_xr_raster_ds,
    get_raster,
)
from tests.utils import assert_valid_raster

NON_SQUARE_AFFINE = Affine(3, 0, 0, 0, -2, 30)
# A single cell twice the size of the input cells in each direction, with
# the same origin
COARSE_AFFINE = Affine(6, 0, 0, 0, -4, 30)
SHAPES = {"row": (1, 4), "column": (4, 1), "cell": (1, 1)}
ALL_SHAPES = tuple(SHAPES)
LONG_AXIS_SHAPES = ("row", "column")


def _one_cell_wide_raster(shape, masked):
    # Non-square cells, so the cell size along the length-1 axis can only
    # come from the stored transform
    data = np.arange(1.0, np.prod(shape) + 1).reshape((1, *shape))
    if masked:
        data[0, -1, -1] = np.nan
    return data_to_raster(
        data,
        affine=NON_SQUARE_AFFINE,
        crs=5070,
        nv=np.nan if masked else None,
    )


def _coarse_geobox():
    return GeoBox((1, 1), COARSE_AFFINE, 5070)


def _same_grid(shape):
    return NON_SQUARE_AFFINE


def _coarse_grid(shape):
    return COARSE_AFFINE


def _grid_padded_by_one_cell(shape):
    return Affine(3, 0, -3, 0, -2, 32)


def _padded_bounds(raster):
    xmin, ymin, xmax, ymax = raster.bounds
    return (xmin - 3, ymin - 2, xmax + 3, ymax + 2)


def _pairwise_window(shape):
    # Window of 2 cells along the long axis and 1 along the one-cell axis
    return (2 if shape[0] > 1 else 1, 2 if shape[1] > 1 else 1)


def _aggregate_pairs(raster):
    return general.aggregate(raster, _pairwise_window(raster.shape[1:]), "sum")


def _aggregated_pairs_grid(shape):
    fy, fx = _pairwise_window(shape)
    return Affine(3 * fx, 0, 0, 0, -2 * fy, 30)


def _aggregate_whole_axis(raster):
    return general.aggregate(raster, raster.shape[1:], "sum")


def _aggregated_whole_axis_grid(shape):
    ny, nx = shape
    return Affine(3 * nx, 0, 0, 0, -2 * ny, 30)


def _two_bands(raster):
    return general.band_concat([raster, raster + 1])


def _features(raster):
    # A polygon covering the raster with a value field
    return rts.Vector(
        gpd.GeoDataFrame(
            {"value": [5]}, geometry=[box(*raster.bounds)], crs=5070
        )
    )


def _line_through(raster):
    xmin, ymin, xmax, ymax = raster.bounds
    return rts.Vector(
        gpd.GeoDataFrame(
            geometry=[LineString([(xmin, ymin), (xmax, ymax)])], crs=5070
        )
    )


def _cost_distance(raster):
    costs = rts.ones_like(raster).set_null_value(None)
    # Integer sources with a null value; the first cell is the only source
    sources = rts.zeros_like(raster, dtype="int32").set_null_value(-1)
    not_first_cell = data_to_raster_like(
        np.arange(raster.size).reshape(raster.shape) > 0, raster
    )
    return rts.distance.cost_distance_analysis(
        costs, sources.set_null(not_first_cell)
    )


def _in_place_add(raster):
    raster = raster.copy()
    raster += 1
    return raster


def _identity_block(block, **kwargs):
    return block


class _SumModel:
    def predict(self, x):
        return x.sum(axis=-1, keepdims=True)


class Case(NamedTuple):
    """One operation in the sweep.

    `op` takes the input raster and returns a Raster or a sequence of
    Rasters. `grid` maps the input (rows, columns) shape to the affine
    expected on every output. `xfail`, if given, is applied to the cases
    whose masked flag is in `xfail_masked`.
    """

    id: str  # noqa: A003
    op: Callable
    grid: Callable = _same_grid
    shapes: tuple = ALL_SHAPES
    xfail: object = None
    xfail_masked: tuple = (False, True)


OPERATIONS = [
    # Arithmetic, comparison, logical and unary operators
    Case("add", lambda r: r + 1),
    Case("radd", lambda r: 1 + r),
    Case("subtract_raster", lambda r: r - r),
    Case("multiply_raster", lambda r: r * r),
    Case("divide", lambda r: r / 2),
    Case("floor_divide", lambda r: r // 2),
    Case("mod", lambda r: r % 2),
    Case("power", lambda r: r**2),
    Case("add_dataarray", lambda r: r + r.xdata),
    Case("in_place_add", _in_place_add),
    Case("negate", lambda r: -r),
    Case("positive", lambda r: +r),
    Case("abs", abs),
    Case("greater", lambda r: r > 1),
    Case("equal_raster", lambda r: r == r),
    Case("and", lambda r: (r > 1) & (r < 4)),
    Case("or", lambda r: (r > 1) | (r < 0)),
    Case("invert", lambda r: ~(r > 1)),
    Case("bandwise", lambda r: r.bandwise + [1]),
    # NumPy ufuncs
    Case("np_sqrt", np.sqrt),
    Case("np_add", lambda r: np.add(r, r)),
    Case("np_maximum", lambda r: np.maximum(r, 2)),
    Case("np_isnan", np.isnan),
    # Raster methods
    Case("astype_float32", lambda r: r.astype("float32")),
    Case(
        "astype_int16",
        lambda r: r.astype("int16", warn_about_null_change=False),
    ),
    Case("astype_new_null", lambda r: r.astype("float32", new_null_value=-1)),
    Case("round", lambda r: r.round(1)),
    Case("set_null_value", lambda r: r.set_null_value(-5.0)),
    Case("set_null_value_none", lambda r: r.set_null_value(None)),
    Case("set_null", lambda r: r.set_null(r > 2)),
    Case("replace_null", lambda r: r.replace_null(0)),
    Case("burn_mask", lambda r: r.burn_mask()),
    Case("to_null_mask", lambda r: r.to_null_mask()),
    Case("set_crs", lambda r: r.set_crs(5070)),
    Case("chunk", lambda r: r.chunk((1, 1, 1))),
    Case("copy", lambda r: r.copy()),
    Case("load", lambda r: r.load()),
    Case("eval", lambda r: r.eval()),
    Case("get_bands", lambda r: _two_bands(r).get_bands(2)),
    Case("split_bands_method", lambda r: _two_bands(r).split_bands()),
    Case(
        "get_chunk_rasters",
        lambda r: list(r.chunk((1, 1, 1)).get_chunk_rasters().ravel()[:1]),
    ),
    Case("remap_range_method", lambda r: r.remap_range([(0, 2, 9)])),
    Case("reclassify_method", lambda r: r.reclassify({1.0: 5})),
    Case("where_method", lambda r: r.where(r > 1, 0)),
    Case("model_predict_method", lambda r: r.model_predict(_SumModel())),
    Case("map_blocks_method", lambda r: r.map_blocks(np.negative)),
    Case(
        "map_overlap_method",
        lambda r: r.map_overlap(np.negative, depth=0),
    ),
    Case("geo_map_blocks_method", lambda r: r.geo_map_blocks(_identity_block)),
    Case(
        "geo_map_overlap_method",
        lambda r: r.geo_map_overlap(_identity_block, depth=0),
    ),
    Case("pad_method_same_bounds", lambda r: r.pad(r.bounds, fill_values=0)),
    Case(
        "pad_method",
        lambda r: r.pad(_padded_bounds(r), fill_values=0),
        _grid_padded_by_one_cell,
    ),
    Case("reproject_method_own_grid", lambda r: r.reproject(r.geobox)),
    Case(
        "reproject_method_coarse_grid",
        lambda r: r.reproject(_coarse_geobox()),
        _coarse_grid,
    ),
    # Construction from a Raster or its data
    Case("raster_from_raster", Raster),
    Case("raster_from_dataarray", lambda r: Raster(r.xdata)),
    Case("raster_from_dataset", lambda r: Raster(r.to_dataset())),
    Case("get_raster", get_raster),
    Case("dataarray_to_raster", lambda r: dataarray_to_raster(r.xdata)),
    Case(
        "dataarray_to_xr_raster",
        lambda r: Raster(dataarray_to_xr_raster(r.xdata)),
    ),
    Case(
        "dataarray_to_xr_raster_ds",
        lambda r: Raster(dataarray_to_xr_raster_ds(r.xdata, r.xmask)),
    ),
    Case(
        "data_to_xr_raster",
        lambda r: Raster(
            data_to_xr_raster(
                r.data, affine=r.affine, crs=r.crs, nv=r.null_value
            )
        ),
    ),
    Case(
        "data_to_xr_raster_ds",
        lambda r: Raster(
            data_to_xr_raster_ds(
                r.data,
                mask=r.mask,
                affine=r.affine,
                crs=r.crs,
                nv=r.null_value,
            )
        ),
    ),
    Case("data_to_raster_like", lambda r: data_to_raster_like(r.data, r)),
    Case(
        "data_to_xr_raster_like",
        lambda r: Raster(data_to_xr_raster_like(r.data, r.xdata)),
    ),
    Case(
        "data_to_xr_raster_ds_like",
        lambda r: Raster(
            data_to_xr_raster_ds_like(
                r.data, r.xdata, mask=r.mask, nv=r.null_value
            )
        ),
    ),
    # raster_tools.creation
    Case("empty_like", rts.empty_like),
    Case("empty_like_copy_mask", lambda r: rts.empty_like(r, copy_mask=True)),
    Case("full_like", lambda r: rts.full_like(r, 1)),
    Case("full_like_two_bands", lambda r: rts.full_like(r, 1, bands=2)),
    Case("zeros_like", rts.zeros_like),
    Case("ones_like", rts.ones_like),
    Case("constant_raster", lambda r: rts.constant_raster(r, 1)),
    Case("random_raster", rts.random_raster),
    Case("random_raster_two_bands", lambda r: rts.random_raster(r, bands=2)),
    # raster_tools.general
    Case("band_concat", lambda r: rts.band_concat([r, r])),
    Case("remap_range", lambda r: rts.remap_range(r, [(0, 2, 9)])),
    Case("reclassify", lambda r: rts.reclassify(r, {1.0: 5})),
    Case("where", lambda r: general.where(r > 1, r, 0)),
    Case("where_two_rasters", lambda r: general.where(r > 1, r, -r)),
    Case("local_stats", lambda r: general.local_stats(r, "sum")),
    Case(
        "local_stats_two_bands",
        lambda r: general.local_stats(_two_bands(r), "mean"),
    ),
    Case("regions", lambda r: general.regions(r)),
    Case("dilate", lambda r: general.dilate(r, 3)),
    Case("erode", lambda r: general.erode(r, 3)),
    Case(
        "model_predict_raster",
        lambda r: general.model_predict_raster(r, _SumModel()),
    ),
    Case(
        "aggregate_keeping_one_cell_axis",
        _aggregate_pairs,
        _aggregated_pairs_grid,
        shapes=LONG_AXIS_SHAPES,
    ),
    Case(
        "aggregate_to_one_cell",
        _aggregate_whole_axis,
        _aggregated_whole_axis_grid,
        shapes=LONG_AXIS_SHAPES,
    ),
    # raster_tools.focal
    Case("focal_mean", lambda r: focal.focal(r, "mean", 3, 3)),
    Case("focal_median", lambda r: focal.focal(r, "median", 3, 3)),
    Case("focal_max", lambda r: focal.focal(r, "max", 3, 3)),
    Case(
        "focal_ignore_null",
        lambda r: focal.focal(r, "mean", 3, 3, ignore_null=True),
    ),
    Case(
        "focal_window_wider_than_raster",
        lambda r: focal.focal(r, "mean", 5, 5),
        xfail=pytest.mark.xfail(
            raises=ValueError,
            strict=True,
            reason="focal fails when its window reaches further than the "
            "raster extends along an axis "
            "(https://github.com/UM-RMRS/raster_tools/issues/88)",
        ),
    ),
    Case("correlate", lambda r: focal.correlate(r, np.ones((3, 3)))),
    Case("convolve", lambda r: focal.convolve(r, np.ones((3, 3)))),
    # raster_tools.surface
    Case("slope", surface.slope),
    Case("aspect", surface.aspect),
    Case("curvature", surface.curvature),
    Case("northing", surface.northing),
    Case("easting", surface.easting),
    Case("hillshade", surface.hillshade),
    Case("tpi", lambda r: surface.tpi(r, 1, 2)),
    Case("surface_area_3d", surface.surface_area_3d),
    # raster_tools.blocks
    Case("map_blocks", lambda r: rts.map_blocks(np.negative, r)),
    Case("map_overlap", lambda r: rts.map_overlap(np.negative, r, depth=0)),
    Case("geo_map_blocks", lambda r: rts.geo_map_blocks(_identity_block, r)),
    Case(
        "geo_map_overlap",
        lambda r: rts.geo_map_overlap(_identity_block, r, depth=0),
    ),
    # Padding, stacking, aligning and mosaicking
    Case("pad_same_bounds", lambda r: rts.pad(r, r.bounds, fill_values=0)),
    Case(
        "pad",
        lambda r: rts.pad(r, _padded_bounds(r), fill_values=0),
        _grid_padded_by_one_cell,
    ),
    Case("split_bands", lambda r: rts.split_bands(_two_bands(r))),
    Case("stack_bands", lambda r: rts.stack_bands([r, r])),
    Case(
        "stack_bands_coarse_grid",
        lambda r: rts.stack_bands([r, r], dst_grid=_coarse_geobox()),
        _coarse_grid,
    ),
    Case("align", lambda r: rts.align([r, r + 1])),
    Case(
        "align_coarse_grid",
        lambda r: rts.align([r, r + 1], dst_grid=_coarse_geobox()),
        _coarse_grid,
    ),
    Case("mosaic", lambda r: rts.mosaic([r, r + 1])),
    Case(
        "mosaic_coarse_grid",
        lambda r: rts.mosaic([r, r + 1], dst_grid=_coarse_geobox()),
        _coarse_grid,
    ),
    Case("reproject_own_grid", lambda r: rts.reproject(r, r.geobox)),
    Case(
        "reproject_coarse_grid",
        lambda r: rts.reproject(r, _coarse_geobox()),
        _coarse_grid,
    ),
    # raster_tools.clipping
    Case("mask", lambda r: clipping.mask(_features(r), r)),
    Case("mask_invert", lambda r: clipping.mask(_features(r), r, invert=True)),
    Case("clip", lambda r: clipping.clip(_features(r), r)),
    Case("erase", lambda r: clipping.erase(_features(r), r)),
    Case("envelope", lambda r: clipping.envelope(_features(r), r)),
    Case("clip_box", lambda r: clipping.clip_box(r, r.bounds)),
    # Rasterizing vectors onto the raster's grid
    Case("to_raster", lambda r: _features(r).to_raster(r, "value")),
    Case("to_raster_mask", lambda r: _features(r).to_raster(r, mask=True)),
    Case(
        "to_raster_spatial_aware",
        lambda r: _features(r).to_raster(r, "value", use_spatial_aware=True),
    ),
    Case("line_length", lambda r: line_stats.length(_line_through(r), r, 1)),
    # raster_tools.distance
    Case(
        "proximity_analysis",
        lambda r: rts.distance.proximity_analysis(r > 1),
    ),
    Case("cost_distance_analysis", _cost_distance),
]


def _sweep_params():
    for case in OPERATIONS:
        for shape_id in case.shapes:
            for masked in (False, True):
                marks = []
                if case.xfail is not None and masked in case.xfail_masked:
                    marks.append(case.xfail)
                yield pytest.param(
                    case,
                    shape_id,
                    masked,
                    id="-".join(
                        (
                            case.id,
                            shape_id,
                            "masked" if masked else "unmasked",
                        )
                    ),
                    marks=marks,
                )


def test_operation_ids_are_unique():
    ids = [case.id for case in OPERATIONS]
    assert len(ids) == len(set(ids))


@pytest.mark.filterwarnings("ignore:The null value ")
@pytest.mark.parametrize("case,shape_id,masked", _sweep_params())
def test_operation_keeps_cell_size_of_one_cell_wide_raster(
    case, shape_id, masked
):
    shape = SHAPES[shape_id]
    raster = _one_cell_wide_raster(shape, masked)
    expected = case.grid(shape)
    results = case.op(raster)
    if isinstance(results, Raster):
        results = [results]
    assert len(results) > 0
    for result in results:
        assert_valid_raster(result)
        assert result.affine == expected
        assert tuple(abs(v) for v in result.resolution) == (
            abs(expected.a),
            abs(expected.e),
        )
        assert result._ds.raster.rio.transform() == expected
        assert result._ds.mask.rio.transform() == expected
