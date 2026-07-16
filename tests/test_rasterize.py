# isort: off
# TODO(pygeos): remove this once shapely is the default backend for geopandas.
# Force raster_tools._compat to be loaded before geopandas when running tests
import raster_tools as rts

# isort: on

import dask.array as da
import dask_geopandas as dgpd
import numpy as np
import pytest
import shapely

from raster_tools import rasterize
from raster_tools.masking import get_default_null_value
from tests import testdata
from tests.utils import assert_rasters_similar, assert_valid_raster


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


def calc_spatial_parts(x):
    x.calculate_spatial_partitions()
    return x


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


@pytest.mark.parametrize(
    "field,mask",
    [(None, False), ("values", False), (None, True)],
)
def test_rasterize_null_value(field, mask):
    features = testdata.vector.test_circles_small
    like = testdata.raster.dem_small
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
