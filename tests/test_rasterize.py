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


def test_rasterize_spatial_aware_reduces_operations(mocker):
    like = testdata.raster.dem.chunk((1, 500, 500))
    features = dgpd.read_file(
        "tests/data/vector/pods.shp", chunksize=40
    ).to_crs(like.crs)

    chunk_func_spy = mocker.spy(rasterize, "_rasterize_onto_chunk")
    result = rasterize.rasterize(features, like)
    result.xdata.compute()
    # Every partiton gets mapped to every chunk
    n_naive = features.npartitions * np.prod(like.data.blocks.shape)
    assert chunk_func_spy.call_count == n_naive
    mocker.stop(chunk_func_spy)

    # Test that spatial awareness reduces number of chunk operations
    features.calculate_spatial_partitions()
    chunk_rasters = list(like.get_chunk_rasters().ravel())
    # Partitions only get mapped to chunks they touch based on bounding polygon
    n_aware = 0
    for g in features.spatial_partitions.geometry.to_numpy():
        for rast in chunk_rasters:
            bbox = shapely.geometry.box(*rast.bounds)
            n_aware += int(g.intersects(bbox))
    assert n_aware < n_naive

    chunk_func_spy = mocker.spy(rasterize, "_rasterize_onto_chunk")
    result = rasterize.rasterize(features, like)
    result.xdata.compute()
    assert chunk_func_spy.call_count == n_aware
    mocker.stop(chunk_func_spy)


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
            for xi, yi in zip(x, y)
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


@pytest.mark.parametrize("use_spatial_aware", [False, True])
@pytest.mark.parametrize("null_value", [None, 99_000])
@pytest.mark.parametrize(
    "mask,mask_invert", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("all_touched", [False, True])
@pytest.mark.parametrize(
    "overlap_resolve_method", ["first", "last", "min", "max"]
)
@pytest.mark.parametrize("field", [None, "values"])
@pytest.mark.parametrize(
    "features,like",
    [
        (testdata.vector.test_circles_small, testdata.raster.dem_small),
        (
            testdata.vector.test_circles_small.data.repartition(npartitions=2),
            testdata.raster.dem_small.chunk((1, 20, 20)),
        ),
        (
            calc_spatial_parts(
                testdata.vector.test_circles_small.data.repartition(
                    npartitions=2
                )
            ),
            testdata.raster.dem_small.chunk((1, 20, 20)),
        ),
    ],
)
def test_rasterize(
    features,
    like,
    field,
    overlap_resolve_method,
    all_touched,
    mask,
    mask_invert,
    null_value,
    use_spatial_aware,
):
    # Convert input features into GeoDataFrame
    feats = rts.vector.get_vector(features).data.compute()

    expected = rasterize_helper(
        feats,
        like,
        field,
        overlap_resolve_method,
        all_touched,
        mask,
        mask_invert,
    )
    if null_value is not None:
        expected = expected.set_null_value(null_value)

    result = rasterize.rasterize(
        features,
        like,
        field=field,
        overlap_resolve_method=overlap_resolve_method,
        all_touched=all_touched,
        mask=mask,
        mask_invert=mask_invert,
        null_value=null_value,
        use_spatial_aware=use_spatial_aware,
    )

    assert_valid_raster(result)
    assert_rasters_similar(result, like, check_nbands=False)
    assert result.null_value == expected.null_value
    assert np.allclose(result, expected)
    if mask and null_value is None:
        assert result.dtype == np.dtype("uint8")
