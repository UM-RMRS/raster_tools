import dask.array as da
import numpy as np
import pytest

from raster_tools import creation
from raster_tools.masking import get_default_null_value
from tests.utils import (
    arange_raster,
    assert_rasters_similar,
    assert_valid_raster,
)


def templates():
    ts = []
    r = arange_raster((2, 10, 10)).set_null_value(-1).set_crs("EPSG:5070")
    r._ds.mask.data[0, :3, :3] = True
    r._ds.mask.data[1, -3:, -3:] = True
    r = r.chunk((1, 3, 3))
    ts.append(r)
    ts.append(arange_raster((3, 100, 100), dtype=float))
    return ts


@pytest.mark.parametrize("copy_mask", [0, 1])
@pytest.mark.parametrize("nbands", [1, 2, 3])
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
@pytest.mark.parametrize("template", templates())
def test_random_raster(template, dist, params, nbands, copy_mask):
    if nbands == template.nbands:
        mask_truth = template.mask.compute()
    else:
        mask_truth = da.stack([template.mask[0]] * nbands).compute()
    assert mask_truth.ndim == 3
    assert mask_truth.shape[0] == nbands

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
    result.eval()
    if copy_mask and template._masked:
        assert result._masked
        assert result.null_value == get_default_null_value(result.dtype)
        assert np.allclose(result.mask.compute(), mask_truth)
        assert (result.data[result.mask].compute() == result.null_value).all()
    else:
        assert not result._masked
        assert result.mask.sum().compute() == 0


def run_constant_raster_tests(
    op_to_test, template, value, nbands, dtype, copy_mask, pass_value
):
    dtype = np.dtype(dtype) if dtype is not None else template.dtype
    if nbands == template.nbands:
        mask_truth = template.mask.compute()
    else:
        mask_truth = da.stack([template.mask[0]] * nbands).compute()
    assert mask_truth.ndim == 3
    assert mask_truth.shape[0] == nbands

    kwargs = dict(bands=nbands, dtype=dtype, copy_mask=copy_mask)
    if pass_value:
        kwargs["value"] = value
    result = op_to_test(template, **kwargs)
    assert_valid_raster(result)
    assert_rasters_similar(template, result, check_nbands=False)
    assert result.nbands == nbands
    assert result.dtype == dtype
    result.eval()
    if copy_mask and template._masked:
        assert result._masked
        if dtype == template.dtype:
            assert result.null_value == template.null_value
        else:
            assert result.null_value == get_default_null_value(result.dtype)
        assert np.allclose(result.mask.compute(), mask_truth)
        assert (result.data[result.mask].compute() == result.null_value).all()
    else:
        assert not result._masked
        assert result.mask.sum().compute() == 0
    if value is not None:
        assert (result == value).all().compute()


@pytest.mark.parametrize("copy_mask", [0, 1])
@pytest.mark.parametrize("dtype", [None, "int32"])
@pytest.mark.parametrize("nbands", [1, 2, 3])
@pytest.mark.parametrize("template", templates())
def test_empty_like(template, nbands, dtype, copy_mask):
    run_constant_raster_tests(
        creation.empty_like, template, None, nbands, dtype, copy_mask, False
    )


@pytest.mark.parametrize("copy_mask", [0, 1])
@pytest.mark.parametrize("dtype", [None, "int32"])
@pytest.mark.parametrize("nbands", [1, 2, 3])
@pytest.mark.parametrize("value", [9, 100, -10])
@pytest.mark.parametrize("template", templates())
def test_full_like(template, value, nbands, dtype, copy_mask):
    run_constant_raster_tests(
        creation.full_like, template, value, nbands, dtype, copy_mask, True
    )


@pytest.mark.parametrize("copy_mask", [0, 1])
@pytest.mark.parametrize("dtype", [None, "int32"])
@pytest.mark.parametrize("nbands", [1, 2, 3])
@pytest.mark.parametrize("template", templates())
def test_zeros_like(template, nbands, dtype, copy_mask):
    run_constant_raster_tests(
        creation.zeros_like, template, 0, nbands, dtype, copy_mask, False
    )


@pytest.mark.parametrize("copy_mask", [0, 1])
@pytest.mark.parametrize("dtype", [None, "int32"])
@pytest.mark.parametrize("nbands", [1, 2, 3])
@pytest.mark.parametrize("template", templates())
def test_ones_like(template, nbands, dtype, copy_mask):
    run_constant_raster_tests(
        creation.ones_like, template, 1, nbands, dtype, copy_mask, False
    )
