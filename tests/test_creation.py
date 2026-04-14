import dask.array as da
import numpy as np
import pytest

from raster_tools import creation
from raster_tools.masking import get_default_null_value
from tests.utils import (
    assert_rasters_equal,
    assert_rasters_similar,
    assert_valid_raster,
    make_raster,
)

TEMPLATES = [
    pytest.param(
        make_raster(
            "arange",
            shape=(2, 10, 10),
            null=-1,
            null_pattern=[
                np.s_[0, :3, :3],
                np.s_[1, -3:, -3:],
            ],
            crs="EPSG:5070",
            chunksize=(1, 3, 3),
        ),
        id="masked-2band-10x10",
    ),
    pytest.param(
        make_raster(
            "arange",
            dtype=float,
            shape=(3, 100, 100),
            crs=None,
        ),
        id="unmasked-3band-100x100",
    ),
]


def _resolve_dtype(dtype, template):
    return np.dtype(dtype) if dtype is not None else template.dtype


def assert_creation_basics(result, template, nbands, dtype):
    assert_valid_raster(result)
    assert_rasters_similar(template, result, check_nbands=False)
    assert result.nbands == nbands
    assert result.dtype == np.dtype(dtype)


def assert_mask_copied(result, template, nbands, dtype, copy_mask):
    result.eval()
    if copy_mask and template._masked:
        if nbands == template.nbands:
            mask_truth = template.mask.compute()
        else:
            mask_truth = da.stack([template.mask[0]] * nbands).compute()
        assert mask_truth.ndim == 3
        assert mask_truth.shape[0] == nbands
        assert result._masked
        if np.dtype(dtype) == template.dtype:
            assert result.null_value == template.null_value
        else:
            assert result.null_value == get_default_null_value(result.dtype)
        assert np.allclose(result.mask.compute(), mask_truth)
        assert (result.data[result.mask].compute() == result.null_value).all()
    else:
        assert not result._masked
        assert result.mask.sum().compute() == 0


@pytest.mark.parametrize("copy_mask", [0, 1])
@pytest.mark.parametrize("nbands", [1, 3])
@pytest.mark.parametrize(
    "dist,params",
    [
        ("binomial", (5, 0.5)),
        ("b", (5, 0.5)),
        ("normal", (0, 20)),
        ("n", (1, 1)),
        ("uniform", ()),
        ("u", ()),
        # Make sure params are ignored
        ("u", (1, 2)),
        ("weibull", (11,)),
        ("w", (11,)),
    ],
)
@pytest.mark.parametrize("template", TEMPLATES)
def test_random_raster(template, dist, params, nbands, copy_mask):
    result = creation.random_raster(
        template,
        distribution=dist,
        bands=nbands,
        params=params,
        copy_mask=copy_mask,
    )
    assert_valid_raster(result)
    assert_rasters_similar(template, result, check_nbands=False)
    assert result.nbands == nbands
    assert_mask_copied(result, template, nbands, None, copy_mask)


@pytest.mark.parametrize("copy_mask", [0, 1])
@pytest.mark.parametrize("dtype", [None, "int32"])
@pytest.mark.parametrize("nbands", [1, 3])
@pytest.mark.parametrize("template", TEMPLATES)
def test_empty_like(template, nbands, dtype, copy_mask):
    dtype = _resolve_dtype(dtype, template)
    result = creation.empty_like(
        template, bands=nbands, dtype=dtype, copy_mask=copy_mask
    )
    assert_creation_basics(result, template, nbands, dtype)
    assert_mask_copied(result, template, nbands, dtype, copy_mask)


@pytest.mark.parametrize("copy_mask", [0, 1])
@pytest.mark.parametrize("dtype", [None, "int32"])
@pytest.mark.parametrize("nbands", [1, 3])
@pytest.mark.parametrize("value", [-10, 100])
@pytest.mark.parametrize("template", TEMPLATES)
def test_full_like(template, value, nbands, dtype, copy_mask):
    dtype = _resolve_dtype(dtype, template)
    result = creation.full_like(
        template,
        value,
        bands=nbands,
        dtype=dtype,
        copy_mask=copy_mask,
    )
    assert_creation_basics(result, template, nbands, dtype)
    assert_mask_copied(result, template, nbands, dtype, copy_mask)
    assert (result == value).all().compute()


@pytest.mark.parametrize("copy_mask", [0, 1])
@pytest.mark.parametrize("dtype", [None, "int32"])
@pytest.mark.parametrize("nbands", [1, 3])
@pytest.mark.parametrize("value", [-10, 100])
@pytest.mark.parametrize("template", TEMPLATES)
def test_constant_raster(template, value, nbands, dtype, copy_mask):
    dtype = _resolve_dtype(dtype, template)
    result = creation.constant_raster(
        template,
        value,
        bands=nbands,
        dtype=dtype,
        copy_mask=copy_mask,
    )
    assert_creation_basics(result, template, nbands, dtype)
    assert_mask_copied(result, template, nbands, dtype, copy_mask)
    assert (result == value).all().compute()


@pytest.mark.parametrize("copy_mask", [0, 1])
@pytest.mark.parametrize("dtype", [None, "int32"])
@pytest.mark.parametrize("nbands", [1, 3])
@pytest.mark.parametrize("template", TEMPLATES)
def test_zeros_like(template, nbands, dtype, copy_mask):
    dtype = _resolve_dtype(dtype, template)
    result = creation.zeros_like(
        template, bands=nbands, dtype=dtype, copy_mask=copy_mask
    )
    assert_creation_basics(result, template, nbands, dtype)
    assert_mask_copied(result, template, nbands, dtype, copy_mask)
    assert (result == 0).all().compute()


@pytest.mark.parametrize("copy_mask", [0, 1])
@pytest.mark.parametrize("dtype", [None, "int32"])
@pytest.mark.parametrize("nbands", [1, 3])
@pytest.mark.parametrize("template", TEMPLATES)
def test_ones_like(template, nbands, dtype, copy_mask):
    dtype = _resolve_dtype(dtype, template)
    result = creation.ones_like(
        template, bands=nbands, dtype=dtype, copy_mask=copy_mask
    )
    assert_creation_basics(result, template, nbands, dtype)
    assert_mask_copied(result, template, nbands, dtype, copy_mask)
    assert (result == 1).all().compute()


def test_full_like_single_chunk_result_writeable(dem_small):
    result = creation.full_like(dem_small, 0)
    data = result.data.compute()
    mask = result.mask.compute()
    assert data.flags.writeable
    assert mask.flags.writeable


def test_full_like_dataarray(dem_small):
    expected = (
        (dem_small / dem_small)
        .astype(int, warn_about_null_change=False)
        .set_null_value(None)
    )
    like = dem_small.astype(int, warn_about_null_change=False)
    result = creation.full_like(like.xdata, 1)
    assert_rasters_equal(result, expected)


def test_empty_like_dataarray(dem_small):
    result = creation.empty_like(dem_small.xdata)
    assert_rasters_similar(result, dem_small)


def test_random_raster_dataarray(dem_small):
    result = creation.random_raster(dem_small.xdata)
    assert_rasters_similar(result, dem_small)
