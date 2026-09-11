# isort: off
# TODO(pygeos): remove this once shapely is the default backend for geopandas.
# Force raster_tools._compat to be loaded before geopandas when running tests
import raster_tools as rts

# isort: on

import dask.array as da
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from affine import Affine
from rasterio.features import rasterize as rio_rasterize

from raster_tools import rasterize
from raster_tools.masking import get_default_null_value
from tests import testdata
from tests.utils import (
    assert_rasters_similar,
    assert_valid_raster,
    make_raster,
)


def test_rasterize_partition_chunk_matches():
    like = testdata.raster.dem.chunk((1, 500, 500))
    features = dgpd.read_file(
        "tests/data/vector/pods.shp", chunksize=40
    ).to_crs(like.crs)
    features.calculate_spatial_partitions()

    matches = rasterize._compute_partition_chunk_matches(features, like)

    chunk_boxes = [
        shapely.geometry.box(*rast.bounds)
        for rast in like.get_chunk_rasters().ravel()
    ]
    expected_pairs = {
        (ipart, ichunk)
        for ipart, g in enumerate(
            features.spatial_partitions.geometry.to_numpy()
        )
        for ichunk, bbox in enumerate(chunk_boxes)
        if g.intersects(bbox)
    }
    actual_pairs = set(
        zip(
            matches.part_idx.to_numpy(),
            matches.flat_idx.to_numpy(),
            strict=True,
        )
    )
    assert actual_pairs == expected_pairs
    # Spatial awareness must actually prune pairs on this data
    assert len(actual_pairs) < features.npartitions * len(chunk_boxes)


def test_rasterize_spatial_matches_builds_one_task_per_match():
    like = testdata.raster.dem.chunk((1, 500, 500)).get_bands(1)
    features = dgpd.read_file(
        "tests/data/vector/pods.shp", chunksize=40
    ).to_crs(like.crs)
    features.calculate_spatial_partitions()

    matches = rasterize._compute_partition_chunk_matches(features, like)
    chunk_rasters = list(like.get_chunk_rasters().ravel())
    out_chunks = rasterize._rasterize_spatial_matches(
        matches,
        features,
        chunk_rasters,
        all_touched=True,
        fill=0,
        target_dtype=np.dtype("uint8"),
        overlap_resolve_method="last",
    )
    # One rasterization array per match, each in its matched chunk's slot
    n_arrays = sum(len(oc) for oc in out_chunks if oc is not None)
    assert n_arrays == len(matches)
    touched = {i for i, oc in enumerate(out_chunks) if oc is not None}
    assert touched == set(matches.flat_idx)


def test_rasterize_spatial_aware_routing(monkeypatch):
    like = testdata.raster.dem.chunk((1, 500, 500))
    features = dgpd.read_file(
        "tests/data/vector/pods.shp", chunksize=40
    ).to_crs(like.crs)

    calls = {"naive": 0, "aware": 0}
    real_naive = rasterize._rasterize_spatial_naive
    real_aware = rasterize._rasterize_spatial_aware

    def naive_wrapper(*args, **kwargs):
        calls["naive"] += 1
        return real_naive(*args, **kwargs)

    def aware_wrapper(*args, **kwargs):
        calls["aware"] += 1
        return real_aware(*args, **kwargs)

    # The naive path handles a frame with no spatial partitions by fabricating
    # full-extent partitions and delegating to the spatial-aware path, so the
    # aware function fires on both branches. The naive function fires only when
    # spatial partitions are absent, making it the branch marker.
    monkeypatch.setattr(rasterize, "_rasterize_spatial_naive", naive_wrapper)
    monkeypatch.setattr(rasterize, "_rasterize_spatial_aware", aware_wrapper)

    # No spatial partitions: dispatched through the naive path.
    rasterize.rasterize(features, like)
    assert calls == {"naive": 1, "aware": 1}

    # Spatial partitions present: dispatched straight to the aware path,
    # bypassing the naive path entirely.
    features.calculate_spatial_partitions()
    rasterize.rasterize(features, like)
    assert calls == {"naive": 1, "aware": 2}


def _direct_rasterize_inputs():
    # Geometries, values, and grid info for calling the low level wrappers
    # directly with no dask involved.
    like = testdata.raster.dem_small
    gdf = testdata.vector.test_circles_small.data.compute()
    geometry = gdf.geometry.to_numpy()
    values = gdf["values"].to_numpy()
    shape = like.shape[1:]
    return like.affine, shape, geometry, values


@pytest.mark.parametrize("budget", [1, 150, 10**9])
@pytest.mark.parametrize("all_touched", [False, True])
def test_rio_rasterize_wrapper_batching_matches_single_call(
    monkeypatch, budget, all_touched
):
    transform, shape, geometry, values = _direct_rasterize_inputs()
    # Unbounded budget is a single call and is the reference result.
    monkeypatch.setattr(rasterize, "RASTERIZE_COORD_BUDGET", 10**9)
    expected = rasterize._rio_rasterize_wrapper(
        shape, transform, geometry, values, values.dtype, -1, all_touched
    )
    monkeypatch.setattr(rasterize, "RASTERIZE_COORD_BUDGET", budget)
    result = rasterize._rio_rasterize_wrapper(
        shape, transform, geometry, values, values.dtype, -1, all_touched
    )
    assert result.dtype == values.dtype
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("budget", [1, 150, 10**9])
@pytest.mark.parametrize("all_touched", [False, True])
@pytest.mark.parametrize("invert", [False, True])
def test_rio_mask_batching_matches_single_call(
    monkeypatch, budget, all_touched, invert
):
    transform, shape, geometry, _ = _direct_rasterize_inputs()
    monkeypatch.setattr(rasterize, "RASTERIZE_COORD_BUDGET", 10**9)
    expected = rasterize._rio_mask(
        geometry, shape, transform, all_touched, invert
    )
    monkeypatch.setattr(rasterize, "RASTERIZE_COORD_BUDGET", budget)
    result = rasterize._rio_mask(
        geometry, shape, transform, all_touched, invert
    )
    assert result.dtype == np.uint8
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("budget", [1, 150, 10**9])
def test_rio_mask_accepts_geoseries(monkeypatch, budget):
    transform, shape, geometry, _ = _direct_rasterize_inputs()
    monkeypatch.setattr(rasterize, "RASTERIZE_COORD_BUDGET", 10**9)
    expected = rasterize._rio_mask(geometry, shape, transform, True, False)
    # line_stats passes a GeoSeries with a non-default index.
    n = len(geometry)
    series = gpd.GeoSeries(geometry, index=np.arange(n) + 10)
    monkeypatch.setattr(rasterize, "RASTERIZE_COORD_BUDGET", budget)
    result = rasterize._rio_mask(series, shape, transform, True, False)
    assert result.dtype == np.uint8
    assert np.array_equal(result, expected)


def test_rio_rasterize_wrapper_int8_roundtrip(monkeypatch):
    transform, shape, geometry, values = _direct_rasterize_inputs()
    values = values.astype("int8")
    out_dtype = np.dtype("int8")
    monkeypatch.setattr(rasterize, "RASTERIZE_COORD_BUDGET", 10**9)
    expected = rasterize._rio_rasterize_wrapper(
        shape, transform, geometry, values, out_dtype, -1, True
    )
    monkeypatch.setattr(rasterize, "RASTERIZE_COORD_BUDGET", 1)
    result = rasterize._rio_rasterize_wrapper(
        shape, transform, geometry, values, out_dtype, -1, True
    )
    assert result.dtype == out_dtype
    assert np.array_equal(result, expected)


def test_iter_geom_batches_boundaries():
    ov = rasterize.GEOM_COORD_OVERHEAD
    small = shapely.Point(0, 0)  # weight ov + 1
    big = shapely.LineString([(i, 0) for i in range(5 * (ov + 1))])

    def batches(geoms, budget):
        return list(
            rasterize._iter_geom_batches(np.array(geoms, dtype=object), budget)
        )

    assert batches([], 10) == []
    assert batches([small] * 3, 3 * (ov + 1)) == [(0, 3)]
    assert batches([small] * 5, 2 * (ov + 1)) == [(0, 2), (2, 4), (4, 5)]
    assert batches([small, big, small], ov + 1) == [(0, 1), (1, 2), (2, 3)]
    assert batches([big, small, small], 2 * (ov + 1)) == [(0, 1), (1, 3)]
    assert batches([small, small, big], 2 * (ov + 1)) == [(0, 2), (2, 3)]
    assert batches([small] * 3, 1) == [(0, 1), (1, 2), (2, 3)]


def test_iter_geom_batches_tiles_pods_small():
    gdf = testdata.vector.pods_small.data.compute()
    geometry = gdf.geometry.to_numpy()
    n = len(geometry)
    weights = shapely.get_num_coordinates(geometry).astype(np.int64)
    weights += rasterize.GEOM_COORD_OVERHEAD
    budget = 5000
    bounds = list(rasterize._iter_geom_batches(geometry, budget))
    assert bounds[0][0] == 0
    assert bounds[-1][1] == n
    for (_, end), (nxt_start, _) in zip(bounds, bounds[1:], strict=False):
        assert end == nxt_start
    for start, end in bounds:
        assert end > start
        assert int(weights[start:end].sum()) < budget + int(weights[start])


@pytest.mark.parametrize("budget_in_geoms", [1, 2, 4])
def test_rio_rasterize_wrapper_call_count(monkeypatch, budget_in_geoms):
    transform, shape, geometry, values = _direct_rasterize_inputs()
    n = len(geometry)
    per_geom = int(shapely.get_num_coordinates(geometry[0]))
    per_geom += rasterize.GEOM_COORD_OVERHEAD
    # All circles have the same coordinate count so the batch count is exact.
    monkeypatch.setattr(
        rasterize, "RASTERIZE_COORD_BUDGET", per_geom * budget_in_geoms
    )
    calls = []
    real = rasterize.rio_rasterize

    def spy(shapes, **kwargs):
        shapes = list(shapes)
        calls.append(len(shapes))
        return real(shapes, **kwargs)

    monkeypatch.setattr(rasterize, "rio_rasterize", spy)
    rasterize._rio_rasterize_wrapper(
        shape, transform, geometry, values, values.dtype, 0, True
    )
    assert len(calls) == -(-n // budget_in_geoms)
    assert sum(calls) == n


def test_rio_mask_call_count_and_shared_out(monkeypatch):
    transform, shape, geometry, _ = _direct_rasterize_inputs()
    n = len(geometry)
    per_geom = int(shapely.get_num_coordinates(geometry[0]))
    per_geom += rasterize.GEOM_COORD_OVERHEAD
    monkeypatch.setattr(rasterize, "RASTERIZE_COORD_BUDGET", per_geom * 2)
    outs = []
    real = rasterize.rio_rasterize

    def spy(shapes, **kwargs):
        outs.append(kwargs["out"])
        return real(shapes, **kwargs)

    monkeypatch.setattr(rasterize, "rio_rasterize", spy)
    rasterize._rio_mask(geometry, shape, transform, True, False)
    assert len(outs) == -(-n // 2)
    assert all(o is outs[0] for o in outs)


def calc_spatial_parts(x):
    x.calculate_spatial_partitions()
    return x


def _edge_like(chunks=(1, 10, 10)):
    # 20x20 unit-cell grid, origin (0, 20), chunk edges at x=10 and y=10
    return make_raster(
        "zeros",
        shape=(1, 20, 20),
        affine=Affine(1, 0, 0, 0, -1, 20),
        chunksize=chunks,
    )


def _edge_features():
    # Values 1..n. Features chosen to exercise chunk edges: a polygon
    # crossing both edges diagonally, a polygon containing a whole chunk, a
    # polygon whose hole straddles both edges, a MultiPolygon spanning three
    # chunks, a polygon abutting x=10 from the left, a line crossing both
    # edges, a line along x=10, a line ending on x=10 from the left, and
    # points on and off the edges.
    geoms = [
        shapely.Polygon([(10, 2), (18, 10), (10, 18), (2, 10)]),
        shapely.box(-2, -2, 12, 12),
        shapely.Polygon(
            [(6, 6), (19, 6), (19, 19), (6, 19)],
            holes=[[(8, 8), (13, 8), (13, 13), (8, 13)]],
        ),
        shapely.MultiPolygon(
            [
                shapely.box(1, 1, 3, 3),
                shapely.box(13, 13, 15, 15),
                shapely.box(1, 13, 3, 15),
            ]
        ),
        shapely.box(6, 2, 10, 6),
        shapely.LineString([(4, 4), (16, 16)]),
        shapely.LineString([(10, 2), (10, 18)]),
        shapely.LineString([(4, 15), (10, 15)]),
        shapely.Point(10, 5),
        shapely.Point(5, 10),
        shapely.Point(10, 10),
        shapely.Point(0, 15),
        shapely.Point(20, 5),
    ]
    return gpd.GeoDataFrame(
        {"values": np.arange(1, len(geoms) + 1)},
        geometry=geoms,
        crs="EPSG:3857",
    )


def _reference(gdf, like, all_touched, fill=0):
    return rio_rasterize(
        zip(gdf.geometry, gdf["values"], strict=True),
        out_shape=like.shape[1:],
        transform=like.affine,
        fill=fill,
        all_touched=all_touched,
        dtype="int64",
    )


def _sort_like_chunk(gdf, overlap_resolve_method):
    # Mirror the per-chunk overlap-resolution sort in _rasterize_onto_chunk.
    if overlap_resolve_method == "first":
        return gdf.iloc[::-1]
    if overlap_resolve_method == "last":
        return gdf
    if overlap_resolve_method == "min":
        return gdf.sort_values(
            by=["values"], ascending=False, na_position="first"
        )
    return gdf.sort_values(by=["values"], na_position="first")


@pytest.mark.parametrize("all_touched", [False, True])
@pytest.mark.parametrize("overlap_resolve_method", ["first", "last"])
def test_rasterize_chunk_edges_match_whole_raster(
    all_touched, overlap_resolve_method
):
    gdf = _edge_features()
    like = _edge_like()
    ref_gdf = gdf.iloc[::-1] if overlap_resolve_method == "first" else gdf
    expected = _reference(ref_gdf, like, all_touched)

    result = rasterize.rasterize(
        gdf,
        like,
        field="values",
        all_touched=all_touched,
        overlap_resolve_method=overlap_resolve_method,
        null_value=0,
    )
    single = rasterize.rasterize(
        gdf,
        _edge_like(chunks=(1, 20, 20)),
        field="values",
        all_touched=all_touched,
        overlap_resolve_method=overlap_resolve_method,
        null_value=0,
    )

    np.testing.assert_array_equal(result.to_numpy()[0], expected)
    np.testing.assert_array_equal(single.to_numpy()[0], expected)


@pytest.mark.parametrize("all_touched", [False, True])
@pytest.mark.parametrize("mask_invert", [False, True])
def test_rasterize_mask_chunk_edges_match_whole_raster(
    all_touched, mask_invert
):
    gdf = _edge_features()
    like = _edge_like()
    base = rio_rasterize(
        gdf.geometry,
        out_shape=like.shape[1:],
        transform=like.affine,
        fill=0,
        default_value=1,
        all_touched=all_touched,
        dtype="uint8",
    )
    expected = (1 - base) if mask_invert else base

    result = rasterize.rasterize(
        gdf,
        like,
        mask=True,
        mask_invert=mask_invert,
        all_touched=all_touched,
    )
    single = rasterize.rasterize(
        gdf,
        _edge_like(chunks=(1, 20, 20)),
        mask=True,
        mask_invert=mask_invert,
        all_touched=all_touched,
    )

    np.testing.assert_array_equal(result.to_numpy()[0], expected)
    np.testing.assert_array_equal(single.to_numpy()[0], expected)


@pytest.mark.parametrize(
    "overlap_resolve_method", ["first", "last", "min", "max"]
)
def test_rasterize_overlap_order_without_resort(overlap_resolve_method):
    rng = np.random.default_rng(0)
    n = 30
    xs = rng.uniform(0, 14, n)
    ys = rng.uniform(0, 14, n)
    boxes = [
        shapely.box(x, y, x + 6, y + 6) for x, y in zip(xs, ys, strict=True)
    ]
    values = rng.permutation(np.arange(1, n + 1))
    gdf = gpd.GeoDataFrame({"values": values}, geometry=boxes, crs="EPSG:3857")
    # Make the spatial order deliberately non-monotone so a dropped re-sort
    # would show up as misordered overlaps.
    gdf = gdf.iloc[rng.permutation(n)].reset_index(drop=True)

    ref_gdf = _sort_like_chunk(gdf, overlap_resolve_method)
    expected = _reference(ref_gdf, _edge_like(), False)

    result = rasterize.rasterize(
        gdf,
        _edge_like(),
        field="values",
        overlap_resolve_method=overlap_resolve_method,
        all_touched=False,
        null_value=0,
    )

    np.testing.assert_array_equal(result.to_numpy()[0], expected)


def test_clip_polygons_to_chunk_unit():
    bounds = (0.0, 0.0, 10.0, 10.0)

    # Lines and points are never clipped; the same array object comes back.
    line_pts = np.array(
        [shapely.LineString([(-5, 5), (15, 5)]), shapely.Point(5, 5)],
        dtype=object,
    )
    assert rasterize._clip_polygons_to_chunk(line_pts, bounds) is line_pts

    # A straddling polygon is clipped down to the chunk bounds.
    straddle = shapely.box(-5, -5, 5, 5)
    arr = np.array([straddle], dtype=object)
    out = rasterize._clip_polygons_to_chunk(arr, bounds)
    b = shapely.bounds(out[0])
    assert b[0] >= 0 and b[1] >= 0 and b[2] <= 10 and b[3] <= 10
    # The input array is not mutated.
    assert arr[0] is straddle

    # A polygon touching only along the edge clips to a lower dimension, so
    # the original polygon is passed through unchanged.
    edge = shapely.box(-5, 2, 0, 6)
    arr2 = np.array([edge], dtype=object)
    out2 = rasterize._clip_polygons_to_chunk(arr2, bounds)
    assert out2[0] is edge


def test_chunk_intersects_mask_drops_missing_and_empty():
    bounds = (0.0, 0.0, 10.0, 10.0)
    geoms = np.array(
        [
            shapely.box(2, 2, 8, 8),
            None,
            shapely.Polygon(),
            shapely.box(20, 20, 30, 30),
        ],
        dtype=object,
    )
    mask = rasterize._chunk_intersects_mask(geoms, bounds)
    np.testing.assert_array_equal(mask, [True, False, False, False])


def test_rasterize_onto_chunk_index_plus_one_with_missing_rows():
    like = _edge_like(chunks=(1, 20, 20))
    geoms = [
        shapely.box(2, 12, 8, 18),
        None,
        shapely.Polygon(),
        shapely.box(2, 2, 8, 8),
    ]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:3857")

    out = rasterize._rasterize_onto_chunk(
        gdf,
        like.affine,
        np.dtype("int64"),
        0,
        False,
        "last",
        block_info={None: {"chunk-shape": (20, 20)}},
    )

    # Index-plus-one values survive the boolean filter of missing rows:
    # index 0 burns 1 and index 3 burns 4; the dropped rows never appear.
    assert 1 in out
    assert 4 in out
    assert 2 not in out
    assert 3 not in out


def rasterize_helper(
    feats_df,
    like,
    field,
    overlap_resolve_method,
    all_touched,
    mask,
    mask_invert,
):
    field_was_none = field is None
    if field_was_none:
        field = "_field_"
        feats_df[field] = np.arange(len(feats_df)) + 1

    if mask:
        field = "_touched_"
        feats_df[field] = np.uint8(1)

    expected = rts.creation.zeros_like(like, dtype=feats_df[field].dtype)

    # Vectorize the grid
    grid_pts = expected.to_points().compute()
    if all_touched:
        # Transform the points into pixel boxes
        x = grid_pts.geometry.x.to_numpy()
        y = grid_pts.geometry.y.to_numpy()
        boxes = [
            shapely.geometry.box(xi - 15, yi - 15, xi + 15, yi + 15)
            for xi, yi in zip(x, y, strict=True)
        ]
        grid_pts["geometry"] = boxes

    # Perform a spatial join to find where the features touch pixels in the
    # grid. how="left" retains the pixels that did not get touched.
    sjoined = grid_pts.sjoin(feats_df, how="left").reset_index()
    touched = sjoined[~sjoined.index_right.isna()]

    # Resolve overlaps in touched
    if overlap_resolve_method in ("first", "last"):
        # Sort on the features index so that head/tail can be used to get the
        # first/last.
        second_sort_field = "index_right"
    else:
        # Sort on the field value so that head/tail can be used to get the
        # min/max.
        second_sort_field = field
    grps = touched.sort_values(["index", second_sort_field]).groupby("index")
    # Depending on second_sort_field, either the first feature or the minimum
    # field value is at the top of each group. Use head/tail to first/min or
    # last/max.
    if overlap_resolve_method in ("first", "min"):
        touched = grps.head(1)
    else:
        touched = grps.tail(1)

    grid_data = expected.data.compute()
    grid_mask = np.ones_like(grid_data, dtype=bool)
    for _, row in touched.iterrows():
        grid_data[row.band - 1, row.row, row.col] = row[field]
        grid_mask[row.band - 1, row.row, row.col] = False

    if mask and mask_invert:
        # Swap 1's with 0 and 0's with 1
        grid_data = 1 - grid_data
        grid_mask = ~grid_mask

    expected = expected.set_null_value(0)
    expected.data[:] = da.from_array(grid_data, chunks=expected.data.chunks)
    expected.mask[:] = da.from_array(grid_mask, chunks=expected.mask.chunks)
    # Use 0 as null value when mask
    if not mask:
        # Otherwise use 0 if no field given or a default based on the field
        # dtype
        if field_was_none:
            nv = 0
        else:
            nv = get_default_null_value(feats_df[field].dtype)
        expected = expected.set_null_value(nv)
    return expected


# Partitioning/chunking variants. `use_spatial_aware=True` is only paired with
# the variant that has spatial partitions pre-calculated, since that is the
# only case where spatial-aware dispatch structurally differs from the naive
# path. The structural reduction in chunk ops from spatial awareness is
# covered separately by the partition/chunk match tests.
_PARTITIONING_CASES = [
    pytest.param(
        testdata.vector.test_circles_small,
        testdata.raster.dem_small,
        False,
        id="single-partition",
    ),
    pytest.param(
        testdata.vector.test_circles_small.data.repartition(npartitions=2),
        testdata.raster.dem_small.chunk((1, 20, 20)),
        False,
        id="multi-partition",
    ),
    pytest.param(
        calc_spatial_parts(
            testdata.vector.test_circles_small.data.repartition(npartitions=2)
        ),
        testdata.raster.dem_small.chunk((1, 20, 20)),
        True,
        id="spatial-aware",
    ),
]


@pytest.mark.parametrize(
    "features,like,use_spatial_aware", _PARTITIONING_CASES
)
@pytest.mark.parametrize("all_touched", [False, True])
@pytest.mark.parametrize(
    "overlap_resolve_method", ["first", "last", "min", "max"]
)
@pytest.mark.parametrize("field", [None, "values"])
def test_rasterize_field(
    features,
    like,
    use_spatial_aware,
    field,
    overlap_resolve_method,
    all_touched,
):
    feats = rts.vector.get_vector(features).data.compute()

    expected = rasterize_helper(
        feats,
        like,
        field,
        overlap_resolve_method,
        all_touched,
        mask=False,
        mask_invert=False,
    )

    result = rasterize.rasterize(
        features,
        like,
        field=field,
        overlap_resolve_method=overlap_resolve_method,
        all_touched=all_touched,
        use_spatial_aware=use_spatial_aware,
    )

    assert_valid_raster(result)
    assert_rasters_similar(result, like, check_nbands=False)
    assert result.null_value == expected.null_value
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "features,like,use_spatial_aware", _PARTITIONING_CASES
)
@pytest.mark.parametrize("all_touched", [False, True])
@pytest.mark.parametrize("mask_invert", [False, True])
def test_rasterize_mask(
    features, like, use_spatial_aware, mask_invert, all_touched
):
    feats = rts.vector.get_vector(features).data.compute()

    expected = rasterize_helper(
        feats,
        like,
        None,
        "first",
        all_touched,
        mask=True,
        mask_invert=mask_invert,
    )

    result = rasterize.rasterize(
        features,
        like,
        all_touched=all_touched,
        mask=True,
        mask_invert=mask_invert,
        use_spatial_aware=use_spatial_aware,
    )

    assert_valid_raster(result)
    assert_rasters_similar(result, like, check_nbands=False)
    assert result.null_value == expected.null_value
    assert np.allclose(result, expected)
    assert result.dtype == np.dtype("uint8")


def test_rasterize_no_field_uses_unique_index_across_partitions():
    like = testdata.raster.dem_small
    feats = rts.vector.get_vector(testdata.vector.test_circles_small).data
    feats = feats.compute().reset_index(drop=True)
    # Two partitions with a global RangeIndex, as the file readers produce
    dfeats = dgpd.from_geopandas(feats, npartitions=2)
    assert dfeats.npartitions == 2

    result = rasterize.rasterize(dfeats, like).load()

    values = np.unique(result.to_numpy())
    values = values[values != result.null_value]
    assert set(values) <= set(range(1, len(feats) + 1))
    assert len(values) > 1


def test_rasterize_no_field_rejects_duplicate_index():
    like = testdata.raster.dem_small
    feats = rts.vector.get_vector(testdata.vector.test_circles_small).data
    feats = feats.compute()
    feats.index = np.arange(len(feats)) // 2
    dfeats = dgpd.from_geopandas(feats, npartitions=2)
    assert dfeats.known_divisions

    result = rasterize.rasterize(dfeats, like)
    with pytest.raises(ValueError, match="not unique within a partition"):
        result.load()
    # A field sidesteps the index entirely
    rasterize.rasterize(dfeats, like, field="values").load()


@pytest.mark.parametrize(
    "field,mask",
    [(None, False), ("values", False), (None, True)],
)
def test_rasterize_null_value(field, mask):
    features = testdata.vector.test_circles_small
    like = testdata.raster.dem_small
    # For the field=None cases, null_value=99_000 exceeds the index-derived
    # value range and must widen the no-field dtype to hold it, so this also
    # exercises the null_value-widens-dtype path.
    null_value = 99_000

    feats = rts.vector.get_vector(features).data.compute()
    expected = rasterize_helper(
        feats,
        like,
        field,
        "last",
        True,
        mask=mask,
        mask_invert=False,
    )
    expected = expected.set_null_value(null_value)

    result = rasterize.rasterize(
        features,
        like,
        field=field,
        mask=mask,
        null_value=null_value,
    )

    assert result.null_value == expected.null_value
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "n_features,null_value,expected",
    [
        (255, None, np.dtype("uint8")),
        (256, None, np.dtype("uint16")),
        (2**16 - 1, None, np.dtype("uint16")),
        (2**16, None, np.dtype("uint32")),
        # An explicit null_value wider than the index range widens the dtype.
        (10, 99_000, np.dtype("uint32")),
        # A negative null_value would wrap in an unsigned dtype, so fall back.
        (10, -1, np.dtype("int64")),
    ],
)
def test_resolve_index_value_dtype_boundaries(
    n_features, null_value, expected
):
    gdf = gpd.GeoDataFrame(
        geometry=[shapely.Point(0, 0)] * n_features,
        index=pd.RangeIndex(n_features),
    )
    dgdf = dgpd.from_geopandas(gdf, npartitions=1)
    assert rasterize._resolve_index_value_dtype(dgdf, null_value) == expected


def test_resolve_index_value_dtype_negative_index_falls_back():
    gdf = gpd.GeoDataFrame(
        geometry=[shapely.Point(0, 0)] * 3, index=[-1, 0, 1]
    )
    dgdf = dgpd.from_geopandas(gdf, npartitions=1)
    assert rasterize._resolve_index_value_dtype(dgdf, None) == np.dtype(
        "int64"
    )


def test_resolve_index_value_dtype_float_index_falls_back():
    gdf = gpd.GeoDataFrame(
        geometry=[shapely.Point(0, 0)] * 3, index=[0.0, 1.0, 2.0]
    )
    dgdf = dgpd.from_geopandas(gdf, npartitions=1)
    assert rasterize._resolve_index_value_dtype(dgdf, None) == np.dtype(
        "int64"
    )


@pytest.mark.parametrize("n,dtype", [(255, "uint8"), (256, "uint16")])
def test_rasterize_no_field_dtype_boundary_end_to_end(n, dtype):
    # One unit box per cell on a 20x20 grid, so every index-plus-one value
    # is burned and the largest value sits exactly at the dtype boundary.
    boxes = [
        shapely.box(j, i, j + 1, i + 1) for i in range(16) for j in range(16)
    ]
    gdf = gpd.GeoDataFrame(geometry=boxes[:n], crs="EPSG:3857")
    result = rasterize.rasterize(gdf, _edge_like(), all_touched=False)
    assert result.dtype == np.dtype(dtype)
    arr = result.load().to_numpy()
    assert arr.dtype == np.dtype(dtype)
    assert set(np.unique(arr)) == set(range(n + 1))


@pytest.mark.parametrize(
    "overlap_resolve_method", ["first", "last", "min", "max"]
)
def test_rasterize_no_field_dtype_survives_multi_partition_reduce(
    overlap_resolve_method,
):
    # Two partitions without spatial partitions both land on every chunk,
    # so the stacked reducers run and must keep the declared small dtype.
    feats = testdata.vector.test_circles_small.data.compute()
    feats = feats.reset_index(drop=True)
    dfeats = dgpd.from_geopandas(feats, npartitions=2)
    result = rasterize.rasterize(
        dfeats,
        testdata.raster.dem_small,
        overlap_resolve_method=overlap_resolve_method,
        use_spatial_aware=False,
    )
    assert result.dtype == np.dtype("uint8")
    loaded = result.load()
    assert loaded.dtype == np.dtype("uint8")
    assert loaded.to_numpy().max() == len(feats)


def test_rasterize_no_field_uses_minimal_dtype(monkeypatch):
    like = testdata.raster.dem_small
    features = testdata.vector.test_circles_small
    n_features = len(rts.vector.get_vector(features).data)
    # The small fixture has few features, so the burned index-plus-one values
    # fit in a uint8.
    assert n_features < 256

    result = rasterize.rasterize(features, like).load()
    assert result.dtype == np.dtype("uint8")

    # Narrowing the dtype must not change which values get burned. Force the
    # prior fixed int64 behavior and compare the burned values.
    def force_i64(gdf, null_value):
        return np.dtype("int64")

    monkeypatch.setattr(rasterize, "_resolve_index_value_dtype", force_i64)
    forced = rasterize.rasterize(features, like).load()
    assert forced.dtype == np.dtype("int64")
    assert np.array_equal(result.to_numpy().astype("int64"), forced.to_numpy())
