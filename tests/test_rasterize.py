# isort: off
# TODO(pygeos): remove this once shapely is the default backend for geopandas.
# Force raster_tools._compat to be loaded before geopandas when running tests
import raster_tools  # noqa: F401

# isort: on

import dask
import dask_geopandas as dgpd
import numpy as np
import pytest
import rasterio as rio
import shapely
import xarray as xr

from raster_tools import rasterize
from raster_tools.masking import get_default_null_value
from raster_tools.raster import Raster
from raster_tools.utils import to_chunk_dict
from raster_tools.vector import Vector, get_vector
from tests import testdata
from tests.utils import assert_rasters_similar, assert_valid_raster


def rio_rasterize(
    gdf,
    field,
    order,
    shape,
    transform,
    fill,
    all_touched,
    mask=False,
    mask_invert=False,
):
    gdf["idx"] = gdf.index.to_numpy() + 1
    if field is None:
        field = "idx"

    if order == "first":
        gdf = gdf.iloc[::-1]
    elif order in ("last", None):
        pass
    elif order == "min":
        gdf = gdf.sort_values(by=[field], ascending=False, na_position="first")
    else:
        gdf = gdf.sort_values(by=[field], ascending=True, na_position="first")

    values = gdf[field].to_numpy()
    if mask:
        values = values.astype("int8")
    geoms = gdf.geometry.to_numpy()
    return rio.features.rasterize(
        zip(geoms, values),
        out_shape=shape,
        transform=transform,
        fill=fill,
        all_touched=all_touched,
    )


def rio_rasterize_helper(
    features,
    like,
    field,
    overlap_resolve_method,
    null_value=None,
    all_touched=True,
    mask=False,
    mask_invert=False,
):
    if isinstance(features, Vector):
        gdf = features.data.compute()
    elif dask.is_dask_collection(features):
        gdf = features.compute()
    else:
        gdf = features.copy()
    shape = like.shape[1:]
    transform = like.affine
    if not mask:
        if null_value is None:
            fill = (
                0
                if field is None
                else get_default_null_value(gdf[field].dtype)
            )
        else:
            fill = null_value
    else:
        field = "mask_values"
        gdf[field] = 0 if mask_invert else 1
        fill = 1 if mask_invert else 0
    truth_array = rio_rasterize(
        gdf,
        field,
        overlap_resolve_method,
        shape,
        transform,
        fill=fill,
        all_touched=all_touched,
        mask=mask,
        mask_invert=mask_invert,
    )[None]
    xtruth = (
        xr.DataArray(
            truth_array, dims=like.xdata.dims, coords=[[1], like.y, like.x]
        )
        .rio.write_crs(like.crs)
        .rio.write_nodata(fill)
        .chunk(to_chunk_dict(like.data.chunks))
    )
    return Raster(xtruth)


DEM_CRS = testdata.raster.dem.crs


def test_rasterize_spatial_aware():
    features = dgpd.read_file(
        "tests/data/vector/test_circles_small.shp", chunksize=2
    )
    like = testdata.raster.dem_small.chunk((1, 20, 20))
    assert features.spatial_partitions is None

    rasterize_kwargs = {
        "field": "values",
        "overlap_resolve_method": "max",
        "all_touched": True,
    }
    truth = rio_rasterize_helper(features, like, **rasterize_kwargs)

    result = rasterize.rasterize(features, like, **rasterize_kwargs)
    assert_valid_raster(result)
    assert np.allclose(result, truth)

    # Use kwarg to force use of spatial aware code path
    result = rasterize.rasterize(
        features, like, use_spatial_aware=True, **rasterize_kwargs
    )
    assert_valid_raster(result)
    assert np.allclose(result, truth)

    # Use spatial aware features
    features_sp = features.copy()
    features_sp.calculate_spatial_partitions()
    result = rasterize.rasterize(features_sp, like, **rasterize_kwargs)
    assert_valid_raster(result)
    assert np.allclose(result, truth)

    # Use spatial aware features
    features_sp = features.spatial_shuffle()
    result = rasterize.rasterize(features_sp, like, **rasterize_kwargs)
    assert_valid_raster(result)
    assert np.allclose(result, truth)


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


@pytest.mark.parametrize("all_touched", [True, False])
@pytest.mark.parametrize(
    "mask,mask_invert", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize(
    "overlap_resolve_method", ["first", "last", "min", "max"]
)
@pytest.mark.parametrize("null_value", [None, 99_000])
@pytest.mark.parametrize("use_index", [True, False])
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
        (
            testdata.vector.pods.data.repartition(
                npartitions=5
            ).spatial_shuffle(),
            testdata.raster.dem_small.chunk((1, 1000, 1000)),
        ),
    ],
)
def test_rasterize(
    features,
    like,
    use_index,
    null_value,
    overlap_resolve_method,
    mask,
    mask_invert,
    all_touched,
):
    field = None
    if not use_index:
        if "values" in get_vector(features).data:
            field = "values"
        else:
            field = "OBJECTID_1"
    truth = rio_rasterize_helper(
        features,
        like,
        field,
        overlap_resolve_method,
        null_value=null_value,
        all_touched=all_touched,
        mask=mask,
        mask_invert=mask_invert,
    )
    assert_valid_raster(truth)
    assert_rasters_similar(truth, like, check_nbands=False)

    result = rasterize.rasterize(
        features,
        like,
        field=field,
        null_value=null_value,
        overlap_resolve_method=overlap_resolve_method,
        all_touched=all_touched,
        mask=mask,
        mask_invert=mask_invert,
    )
    assert_valid_raster(result)
    assert_rasters_similar(result, like, check_nbands=False)
    assert np.allclose(truth, result)
    if mask:
        assert result.dtype == np.dtype("uint8")
