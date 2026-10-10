from unittest import TestCase

import dask
import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

import raster_tools as rts
from raster_tools import clipping
from raster_tools.exceptions import RasterNoDataError
from raster_tools.raster import Raster, grid_transform
from tests import testdata
from tests.utils import assert_rasters_similar, assert_valid_raster


class TestClipping(TestCase):
    def setUp(self):
        self.dem = testdata.raster.dem
        self.pods = testdata.vector.pods
        self.v10 = self.pods[10]
        self.v10_bounds = dask.compute(self.v10.to_crs(self.dem.crs).bounds)[0]

    def test_core_clip_out_dtype(self):
        result = clipping._clip(self.pods, self.dem)
        assert_valid_raster(result)
        self.assertTrue(result.dtype == self.dem.dtype)
        self.assertTrue(result.load().dtype == self.dem.dtype)

    def test_clip(self):
        res = clipping.clip(self.v10, self.dem)
        truth = testdata.raster.clipping_clip_pods_10
        assert_valid_raster(res)
        self.assertTrue(np.allclose(res, truth))
        assert res.crs == self.dem.crs

        res = clipping.clip(self.v10, self.dem, bounds=self.v10_bounds)
        self.assertTrue(np.allclose(res, truth))
        assert res.crs == self.dem.crs

    def test_erase(self):
        res = clipping.erase(self.v10, self.dem)
        truth = testdata.raster.clipping_erase_pods_10
        assert_valid_raster(res)
        self.assertTrue(np.allclose(res, truth))
        assert res.crs == self.dem.crs

        res = clipping.erase(self.v10, self.dem, bounds=self.v10_bounds)
        self.assertTrue(np.allclose(res, truth))
        assert res.crs == self.dem.crs

    def test_mask(self):
        res = clipping.mask(self.v10, self.dem)
        truth = testdata.raster.clipping_mask_pods_10
        assert_valid_raster(res)
        self.assertTrue(np.allclose(res, truth))
        assert res.crs == self.dem.crs

        res = clipping.mask(self.v10, self.dem, invert=True)
        truth = testdata.raster.clipping_mask_inverted_pods_10
        assert_valid_raster(res)
        self.assertTrue(np.allclose(res, truth))
        assert res.crs == self.dem.crs

    def test_envelope(self):
        res = clipping.envelope(self.v10, self.dem)
        truth = testdata.raster.clipping_envelope_pods_10
        assert_valid_raster(res)
        self.assertTrue(np.allclose(res, truth))
        assert res.crs == self.dem.crs

    def test_errors(self):
        with self.assertRaises(ValueError):
            rs = Raster(np.ones((4, 4)))
            clipping.clip(self.v10, rs)

        with self.assertRaises(ValueError):
            clipping._clip(self.v10, self.dem, bounds=(0, 3))

        with self.assertRaises(ValueError):
            clipping._clip(self.v10, self.dem, invert=True, envelope=True)

        with self.assertRaises(RuntimeError):
            clipping._clip(self.v10, self.dem, bounds=self.v10.bounds)

    def test_clip_box(self):
        self.dem = testdata.raster.dem
        rs_clipped = testdata.raster.dem_small
        bounds = [
            rs_clipped.xdata.x.min().item(),
            rs_clipped.xdata.y.min().item(),
            rs_clipped.xdata.x.max().item(),
            rs_clipped.xdata.y.max().item(),
        ]
        test = clipping.clip_box(self.dem, bounds)
        self.assertTrue(test.shape == rs_clipped.shape)
        self.assertTrue(np.allclose(test.to_numpy(), rs_clipped.to_numpy()))

        # Test that the mask is also clipped
        x = np.arange(25).reshape((1, 5, 5))
        x[x < 12] = 0
        rs = Raster(x).set_null_value(0).set_crs("epsg:3857")
        self.assertTrue(np.allclose(x == 0, rs.mask))
        rs_clipped = clipping.clip_box(rs, (1, 1, 4, 4))
        mask_truth = np.array([[[1, 1, 1], [1, 0, 0], [0, 0, 0]]], dtype=bool)
        self.assertTrue(np.allclose(rs_clipped.mask, mask_truth))

    def test_clip_out_of_bounds(self):
        with self.assertRaises(RasterNoDataError):
            clipping.clip_box(self.dem, (9e6, 9e6, 10e6, 10e6))


def test_clip_multiband():
    raster = testdata.raster.dem_small
    raster2 = rts.stack_bands([raster, raster])
    vector = testdata.vector.pods_small
    expected = rts.stack_bands([clipping.clip(vector, raster)] * 2)

    result = clipping.clip(vector, raster2)
    assert_valid_raster(result)
    assert_rasters_similar(result, expected)
    assert np.allclose(result, expected)


SQUARE_AFFINE = Affine(2, 0, 0, 0, -2, 30)
# Non-square cells, so the cell size along a length-1 axis can only come
# from the stored transform
NON_SQUARE_AFFINE = Affine(3, 0, 0, 0, -2, 30)
AFFINES = pytest.mark.parametrize(
    "affine", [SQUARE_AFFINE, NON_SQUARE_AFFINE], ids=["square", "non_square"]
)
ONE_CELL_WIDE_SHAPES = pytest.mark.parametrize(
    "shape", [(1, 4), (4, 1), (1, 1)], ids=["row", "column", "cell"]
)
MASKED = pytest.mark.parametrize(
    "masked", [False, True], ids=["unmasked", "masked"]
)


def _raster(shape, affine, masked):
    data = np.arange(1.0, np.prod(shape) + 1).reshape((1, *shape))
    if masked:
        data[0, -1, -1] = np.nan
    return rts.data_to_raster(
        data, affine=affine, crs=5070, nv=np.nan if masked else None
    )


def _box_feature(bounds):
    return rts.Vector(gpd.GeoDataFrame(geometry=[box(*bounds)], crs=5070))


def _cell_bounds(raster, row, col):
    minx, maxy = raster.xy(row, col, offset="ul")
    maxx, miny = raster.xy(row, col, offset="lr")
    return (minx, miny, maxx, maxy)


def _assert_grid(result, affine, shape):
    assert_valid_raster(result)
    assert result.shape == (1, *shape)
    assert result.affine == affine
    assert grid_transform(result.xdata) == affine
    assert grid_transform(result.xmask) == affine


@ONE_CELL_WIDE_SHAPES
@AFFINES
@MASKED
@pytest.mark.parametrize(
    "op",
    [
        lambda f, r: clipping.clip(f, r),
        lambda f, r: clipping.envelope(f, r),
        lambda f, r: clipping.clip_box(r, r.bounds),
    ],
    ids=["clip", "envelope", "clip_box"],
)
def test_clip_keeps_all_of_one_cell_wide_raster(shape, affine, masked, op):
    raster = _raster(shape, affine, masked)

    result = op(_box_feature(raster.bounds), raster)

    _assert_grid(result, affine, shape)
    mask = raster.mask.compute()
    np.testing.assert_array_equal(result.mask.compute(), mask)
    np.testing.assert_array_equal(
        result.to_numpy()[~mask], raster.to_numpy()[~mask]
    )


@ONE_CELL_WIDE_SHAPES
@AFFINES
@MASKED
def test_erase_all_of_one_cell_wide_raster(shape, affine, masked):
    raster = _raster(shape, affine, masked)

    result = clipping.erase(_box_feature(raster.bounds), raster)

    _assert_grid(result, affine, shape)
    assert result.mask.compute().all()


@ONE_CELL_WIDE_SHAPES
@AFFINES
@MASKED
def test_clip_and_erase_first_cell_of_one_cell_wide_raster(
    shape, affine, masked
):
    raster = _raster(shape, affine, masked)
    feature = _box_feature(_cell_bounds(raster, 0, 0))
    first_cell = np.zeros((1, *shape), dtype=bool)
    first_cell[0, 0, 0] = True

    clipped = clipping.clip(feature, raster, bounds=raster.bounds)
    erased = clipping.erase(feature, raster, bounds=raster.bounds)

    _assert_grid(clipped, affine, shape)
    _assert_grid(erased, affine, shape)
    mask = raster.mask.compute()
    clipped_mask = ~first_cell | mask
    erased_mask = first_cell | mask
    np.testing.assert_array_equal(clipped.mask.compute(), clipped_mask)
    np.testing.assert_array_equal(erased.mask.compute(), erased_mask)
    data = raster.to_numpy()
    np.testing.assert_array_equal(
        clipped.to_numpy()[~clipped_mask], data[~clipped_mask]
    )
    np.testing.assert_array_equal(
        erased.to_numpy()[~erased_mask], data[~erased_mask]
    )


@AFFINES
@MASKED
@pytest.mark.parametrize(
    "shape,rows,cols",
    [((1, 6), slice(0, 1), slice(2, 5)), ((6, 1), slice(2, 5), slice(0, 1))],
    ids=["row", "column"],
)
@pytest.mark.parametrize("use_feature", [False, True], ids=["box", "feature"])
def test_clip_one_cell_wide_raster_to_part(
    affine, masked, shape, rows, cols, use_feature
):
    data = np.arange(1.0, 7.0).reshape((1, *shape))
    if masked:
        # A null inside the part that is kept
        data[0, rows.stop - 1, cols.stop - 1] = np.nan
    raster = rts.data_to_raster(
        data, affine=affine, crs=5070, nv=np.nan if masked else None
    )
    minx, _, _, maxy = _cell_bounds(raster, rows.start, cols.start)
    _, miny, maxx, _ = _cell_bounds(raster, rows.stop - 1, cols.stop - 1)
    bounds = (minx, miny, maxx, maxy)

    if use_feature:
        result = clipping.clip(_box_feature(bounds), raster)
    else:
        result = clipping.clip_box(raster, bounds)

    expected_shape = (rows.stop - rows.start, cols.stop - cols.start)
    _assert_grid(
        result,
        affine * Affine.translation(cols.start, rows.start),
        expected_shape,
    )
    expected = raster.to_numpy()[:, rows, cols]
    expected_mask = raster.mask.compute()[:, rows, cols]
    np.testing.assert_array_equal(result.mask.compute(), expected_mask)
    np.testing.assert_array_equal(
        result.to_numpy()[~expected_mask], expected[~expected_mask]
    )
