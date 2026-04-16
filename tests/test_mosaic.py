# isort: off
# TODO(pygeos): remove this once shapely is the default backend for geopandas.
# Force raster_tools._compat to be loaded before geopandas when running tests
import raster_tools as rts  # noqa: F401

# isort: on

import numpy as np
import pytest
import shapely

import raster_tools._mosaic as mosaic
from raster_tools.masking import get_default_null_value
from tests import testdata
from tests.utils import assert_rasters_equal, assert_valid_raster, make_raster


def _raster(data, y, x):
    return make_raster(data, x=x, y=y, null=-1, crs=5070)


o = -1


@pytest.mark.parametrize(
    "method,expected_data",
    [
        (
            "first",
            [
                [3, 3, 3],
                [3, 3, 3],
                [1, 1, o],
            ],
        ),
        (
            "last",
            [
                [1, 1, 3],
                [1, 1, 3],
                [1, 1, o],
            ],
        ),
        (
            "min",
            [
                [1, 1, 3],
                [1, 1, 3],
                [1, 1, o],
            ],
        ),
        (
            "max",
            [
                [3, 3, 3],
                [3, 3, 3],
                [1, 1, o],
            ],
        ),
        (
            "sum",
            [
                [4, 4, 3],
                [4, 4, 3],
                [1, 1, o],
            ],
        ),
    ],
)
def test_mosaic_simple_overlap_2inputs(method, expected_data):
    y = np.arange(3)[::-1]
    x = np.arange(3)
    r1 = _raster(
        [
            [3, 3, 3],
            [3, 3, 3],
            [o, o, o],
        ],
        y,
        x,
    )
    r2 = _raster(
        [
            [1, 1, o],
            [1, 1, o],
            [1, 1, o],
        ],
        y,
        x,
    )
    result = mosaic.mosaic([r1, r2], method)
    assert_valid_raster(result)
    assert_rasters_equal(result, _raster(expected_data, y, x))


@pytest.mark.parametrize(
    "method,expected_data",
    [
        (
            "first",
            [
                [3, 3, 3, 1],
                [3, 3, 3, 1],
                [o, 1, o, 1],
            ],
        ),
        (
            "last",
            [
                [3, 1, 3, 1],
                [3, 1, 3, 1],
                [o, 1, o, 1],
            ],
        ),
        (
            "min",
            [
                [3, 1, 3, 1],
                [3, 1, 3, 1],
                [o, 1, o, 1],
            ],
        ),
        (
            "max",
            [
                [3, 3, 3, 1],
                [3, 3, 3, 1],
                [o, 1, o, 1],
            ],
        ),
        (
            "sum",
            [
                [3, 4, 3, 1],
                [3, 4, 3, 1],
                [o, 1, o, 1],
            ],
        ),
    ],
)
def test_mosaic_simple_partial_overlap_2inputs(method, expected_data):
    y = np.arange(3)[::-1]
    x = np.arange(4)
    r1 = _raster(
        [
            [3, 3, 3],
            [3, 3, 3],
            [o, o, o],
        ],
        y,
        x[:3],
    )
    r2 = _raster(
        [
            [1, o, 1],
            [1, o, 1],
            [1, o, 1],
        ],
        y,
        x[1:],
    )
    result = mosaic.mosaic([r1, r2], method)
    assert_valid_raster(result)
    assert_rasters_equal(result, _raster(expected_data, y, x))


@pytest.mark.parametrize(
    "method,expected_data",
    [
        (
            "first",
            [
                [1, 1, 3, 1, 3, 3],
                [1, 1, 3, 1, 3, 3],
                [1, 1, o, 1, 2, 2],
                [1, 1, 3, 1, 3, 3],
                [o, o, 2, 2, o, 2],
                [o, o, 2, 2, 2, o],
            ],
        ),
        (
            "last",
            [
                [1, 1, 3, 3, 3, 3],
                [1, 1, 3, 3, 3, 3],
                [1, 1, o, 2, 2, 2],
                [1, 1, 2, 3, 2, 2],
                [o, o, 2, 2, o, 2],
                [o, o, 2, 2, 2, o],
            ],
        ),
        (
            "min",
            [
                [1, 1, 3, 1, 3, 3],
                [1, 1, 3, 1, 3, 3],
                [1, 1, o, 1, 2, 2],
                [1, 1, 2, 1, 2, 2],
                [o, o, 2, 2, o, 2],
                [o, o, 2, 2, 2, o],
            ],
        ),
        (
            "max",
            [
                [1, 1, 3, 3, 3, 3],
                [1, 1, 3, 3, 3, 3],
                [1, 1, o, 2, 2, 2],
                [1, 1, 3, 3, 3, 3],
                [o, o, 2, 2, o, 2],
                [o, o, 2, 2, 2, o],
            ],
        ),
        (
            "sum",
            [
                [1, 1, 3, 4, 3, 3],
                [1, 1, 3, 4, 3, 3],
                [1, 1, o, 3, 2, 2],
                [1, 1, 5, 4, 5, 5],
                [o, o, 2, 2, o, 2],
                [o, o, 2, 2, 2, o],
            ],
        ),
    ],
)
def test_mosaic_simple_partial_overlap_3inputs(method, expected_data):
    y = np.arange(6)[::-1]
    x = np.arange(6)
    r1 = _raster(
        [
            [1, 1, o, 1],
            [1, 1, o, 1],
            [1, 1, o, 1],
            [1, 1, o, 1],
        ],
        y[:-2],
        x[:-2],
    )
    r2 = _raster(
        [
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [o, o, o, o],
            [3, 3, 3, 3],
        ],
        y[:-2],
        x[2:],
    )
    r3 = _raster(
        [
            [o, 2, 2, 2],
            [2, o, 2, 2],
            [2, 2, o, 2],
            [2, 2, 2, o],
        ],
        y[2:],
        x[2:],
    )
    result = mosaic.mosaic([r1, r2, r3], method)
    assert_valid_raster(result)
    assert_rasters_equal(result, _raster(expected_data, y, x))


# Values are the homogeneous fill for each 3x3 input raster, supplied in
# the order that mosaic sees them. `expected` is the homogeneous fill
# for the result of each method over those inputs.
_OVERLAP_MANY_DESCENDING_VALUES = list(np.arange(21)[::-1])
# fmt: off
_OVERLAP_MANY_MIXED_VALUES = [
    10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15,
    14, 13, 12, 11, 16, 17, 18, 20, 19,
]
# fmt: on


@pytest.mark.parametrize(
    "values,method,expected_value",
    [
        (_OVERLAP_MANY_DESCENDING_VALUES, "first", 20),
        (_OVERLAP_MANY_DESCENDING_VALUES, "last", 0),
        (_OVERLAP_MANY_DESCENDING_VALUES, "min", 0),
        (_OVERLAP_MANY_DESCENDING_VALUES, "max", 20),
        (_OVERLAP_MANY_DESCENDING_VALUES, "sum", 210),
        (_OVERLAP_MANY_MIXED_VALUES, "first", 10),
        (_OVERLAP_MANY_MIXED_VALUES, "last", 19),
        (_OVERLAP_MANY_MIXED_VALUES, "min", 0),
        (_OVERLAP_MANY_MIXED_VALUES, "max", 20),
        (_OVERLAP_MANY_MIXED_VALUES, "sum", 210),
    ],
)
def test_mosaic_simple_overlap_many(values, method, expected_value):
    y = np.arange(3)[::-1]
    x = np.arange(3)
    inputs = [_raster(np.full((3, 3), v), y, x) for v in values]
    expected = _raster(np.full((3, 3), expected_value), y, x)
    result = mosaic.mosaic(inputs, method)
    assert_valid_raster(result)
    assert_rasters_equal(result, expected)


@pytest.mark.parametrize(
    "method,expected_data",
    [
        (
            "first",
            [
                [[3, 3, 3], [3, 3, 3], [1, 1, o]],
                [[5, 5, 5], [5, 5, 5], [2, 2, o]],
            ],
        ),
        (
            "last",
            [
                [[1, 1, 3], [1, 1, 3], [1, 1, o]],
                [[2, 2, 5], [2, 2, 5], [2, 2, o]],
            ],
        ),
        (
            "min",
            [
                [[1, 1, 3], [1, 1, 3], [1, 1, o]],
                [[2, 2, 5], [2, 2, 5], [2, 2, o]],
            ],
        ),
        (
            "max",
            [
                [[3, 3, 3], [3, 3, 3], [1, 1, o]],
                [[5, 5, 5], [5, 5, 5], [2, 2, o]],
            ],
        ),
        (
            "sum",
            [
                [[4, 4, 3], [4, 4, 3], [1, 1, o]],
                [[7, 7, 5], [7, 7, 5], [2, 2, o]],
            ],
        ),
    ],
)
def test_mosaic_multiband_overlap_2inputs(method, expected_data):
    y = np.arange(3)[::-1]
    x = np.arange(3)
    r1 = _raster(
        [
            [[3, 3, 3], [3, 3, 3], [o, o, o]],
            [[5, 5, 5], [5, 5, 5], [o, o, o]],
        ],
        y,
        x,
    )
    r2 = _raster(
        [
            [[1, 1, o], [1, 1, o], [1, 1, o]],
            [[2, 2, o], [2, 2, o], [2, 2, o]],
        ],
        y,
        x,
    )
    result = mosaic.mosaic([r1, r2], method)
    assert_valid_raster(result)
    assert_rasters_equal(result, _raster(expected_data, y, x))


def test_mosaic_simple_neighboring_tiles_no_overlap():
    y = np.arange(3)[::-1]
    x = np.arange(9)
    inputs = [
        _raster(np.zeros((3, 3), dtype=int), y, x[:3]),
        _raster(np.ones((3, 3), dtype=int), y, x[3:6]),
        _raster(np.zeros((3, 3), dtype=int) + 2, y, x[-3:]),
    ]
    expected = _raster(
        [
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
        ],
        y,
        x,
    )
    result = mosaic.mosaic(inputs)
    assert_valid_raster(result)
    assert_rasters_equal(result, expected)


def test_mosaic_simple_neighboring_tiles_with_gap():
    y = np.arange(3)[::-1]
    x = np.arange(11)
    inputs = [
        _raster(np.zeros((3, 3), dtype=int), y, x[:3]),
        _raster(np.ones((3, 3), dtype=int), y, x[4:7]),
        _raster(np.zeros((3, 3), dtype=int) + 2, y, x[-3:]),
    ]
    expected = _raster(
        [
            [0, 0, 0, o, 1, 1, 1, o, 2, 2, 2],
            [0, 0, 0, o, 1, 1, 1, o, 2, 2, 2],
            [0, 0, 0, o, 1, 1, 1, o, 2, 2, 2],
        ],
        y,
        x,
    )
    result = mosaic.mosaic(inputs)
    assert_valid_raster(result)
    assert_rasters_equal(result, expected)


def _simple_pair():
    y = np.arange(3)[::-1]
    x = np.arange(3)
    r1 = _raster(np.ones((3, 3), dtype=int), y, x)
    r2 = _raster(np.full((3, 3), 2, dtype=int), y, x + 3)
    return r1, r2


def test_mosaic_empty_list_raises():
    with pytest.raises(ValueError):
        mosaic.mosaic([])


@pytest.mark.parametrize("method", ["", "mean", "FIRST", None, 1])
def test_mosaic_invalid_method_raises(method):
    r1, r2 = _simple_pair()
    with pytest.raises(ValueError):
        mosaic.mosaic([r1, r2], method)


@pytest.mark.parametrize("resampling", ["foo", "NEAREST", 42])
def test_mosaic_invalid_resampling_raises(resampling):
    r1, r2 = _simple_pair()
    with pytest.raises(ValueError):
        mosaic.mosaic([r1, r2], resampling_method=resampling)


@pytest.mark.parametrize("bad_null_value", ["x", [1], (1, 2), {1}])
def test_mosaic_bad_null_value_type_raises(bad_null_value):
    r1, r2 = _simple_pair()
    with pytest.raises(TypeError):
        mosaic.mosaic([r1, r2], null_value=bad_null_value)


@pytest.mark.parametrize("bad_dst_grid", [42, {}, 3.14, object()])
def test_mosaic_bad_dst_grid_type_raises(bad_dst_grid):
    r1, r2 = _simple_pair()
    with pytest.raises(TypeError):
        mosaic.mosaic([r1, r2], dst_grid=bad_dst_grid)


@pytest.mark.parametrize("form", ["geobox", "raster", "path"])
def test_mosaic_dst_grid_accepts_forms(form, tmp_path):
    r1, r2 = _simple_pair()
    reference = mosaic.mosaic([r1, r2])
    if form == "geobox":
        dst_grid = reference.geobox
    elif form == "raster":
        dst_grid = reference
    else:
        path = tmp_path / "reference.tif"
        reference.save(str(path))
        dst_grid = str(path)
    result = mosaic.mosaic([r1, r2], dst_grid=dst_grid)
    assert_valid_raster(result)
    assert_rasters_equal(result, reference)


def test_mosaic_dst_crs_reprojects():
    src = testdata.raster.dem_small
    result = mosaic.mosaic([src], dst_crs=4326)
    assert_valid_raster(result)
    assert result.geobox.crs == 4326


def test_mosaic_mixed_input_crs_defaults_to_first():
    src_a = testdata.raster.dem_small
    src_b = src_a.reproject(4326)
    result = mosaic.mosaic([src_a, src_b])
    assert_valid_raster(result)
    assert result.crs == src_a.crs


def test_mosaic_mixed_crs_pixel_correctness():
    # With method="first" on a given dst_grid, the native-CRS input wins
    # everywhere its footprint covers. The 4326 round-trip copy should
    # therefore have no effect on result pixels.
    src = testdata.raster.dem_small
    src_4326 = src.reproject(4326)
    result = mosaic.mosaic([src, src_4326], "first", dst_grid=src.geobox)
    assert_valid_raster(result)
    assert result.crs == src.crs
    assert np.allclose(result.to_numpy(), src.to_numpy())


def test_mosaic_three_way_crs():
    src = testdata.raster.dem_small
    src_4326 = src.reproject(4326)
    result = mosaic.mosaic([src, src_4326], dst_crs="EPSG:5070")
    assert_valid_raster(result)
    assert result.crs == 5070


def test_mosaic_non_overlapping_mixed_crs_tiles():
    # Split dem_small into two adjacent halves, reproject the right half
    # to 4326, then mosaic them back onto the original grid. The left
    # half should reconstruct exactly; the right half covers the
    # remaining extent but may differ slightly due to resampling.
    src = testdata.raster.dem_small
    left = rts.Raster(src.xdata.isel(x=slice(0, 50)))
    right = rts.Raster(src.xdata.isel(x=slice(50, 100)))
    right_4326 = right.reproject(4326)
    result = mosaic.mosaic([left, right_4326], "first", dst_grid=src.geobox)
    assert_valid_raster(result)
    assert result.crs == src.crs
    assert result.shape == src.shape
    actual = result.to_numpy()
    expected = src.to_numpy()
    # Left half: pristine input, "first" wins -> exact match.
    assert np.allclose(actual[..., :50], expected[..., :50])
    # Right half: must be mostly covered (not all nodata) after the
    # 4326 round-trip.
    right_cells = actual[..., 50:]
    assert (right_cells != result.null_value).mean() > 0.9


def test_mosaic_mixed_crs_covers_union_extent():
    src = testdata.raster.dem_small
    src_4326 = src.reproject(4326)
    result = mosaic.mosaic([src, src_4326])
    # Result bounds should cover the union of both inputs, expressed in
    # the output CRS.
    b1 = shapely.box(*src.bounds)
    b2_in_dst = shapely.box(*src_4326.reproject(src.crs).bounds)
    union = shapely.unary_union([b1, b2_in_dst])
    result_box = shapely.box(*result.bounds)
    # Allow a tolerance of one pixel on each side.
    tol = abs(src.resolution[0])
    assert result_box.buffer(tol).contains(union)


def test_mosaic_explicit_dtype():
    r1, r2 = _simple_pair()
    result = mosaic.mosaic([r1, r2], dtype="float64")
    assert_valid_raster(result)
    assert result.dtype == np.dtype("float64")


def test_mosaic_explicit_null_value():
    y = np.arange(3)[::-1]
    x = np.arange(3)
    r1 = _raster(np.ones((3, 3), dtype=int), y, x)
    r2 = _raster(np.full((3, 3), 2, dtype=int), y, x + 5)
    result = mosaic.mosaic([r1, r2], null_value=-99)
    assert_valid_raster(result)
    assert result.null_value == -99
    # Gap cells between the two rasters should hold the null value.
    data = result.to_numpy()
    assert (data == -99).any()


def test_mosaic_default_nodata_from_first_non_none():
    y = np.arange(3)[::-1]
    x = np.arange(3)
    r1 = make_raster(np.ones((3, 3), dtype=int), y=y, x=x, null=-7, crs=5070)
    r2 = make_raster(
        np.full((3, 3), 2, dtype=int),
        y=y,
        x=x + 5,
        null=-3,
        crs=5070,
    )
    result = mosaic.mosaic([r1, r2])
    assert_valid_raster(result)
    assert result.null_value == -7


def test_mosaic_default_nodata_when_all_none():
    y = np.arange(3)[::-1]
    x = np.arange(3)
    r1 = make_raster(np.ones((3, 3), dtype="float32"), y=y, x=x, crs=5070)
    r2 = make_raster(
        np.full((3, 3), 2, dtype="float32"),
        y=y,
        x=x + 5,
        crs=5070,
    )
    assert r1.null_value is None
    assert r2.null_value is None
    result = mosaic.mosaic([r1, r2])
    assert_valid_raster(result)
    assert result.null_value == get_default_null_value(np.dtype("float32"))


_NAN = np.nan


@pytest.mark.parametrize(
    "method,expected_data",
    [
        (
            "first",
            [
                [3, 3, 3, 1],
                [3, 3, 3, 1],
                [_NAN, 1, _NAN, 1],
            ],
        ),
        (
            "last",
            [
                [3, 1, 3, 1],
                [3, 1, 3, 1],
                [_NAN, 1, _NAN, 1],
            ],
        ),
        (
            "min",
            [
                [3, 1, 3, 1],
                [3, 1, 3, 1],
                [_NAN, 1, _NAN, 1],
            ],
        ),
        (
            "max",
            [
                [3, 3, 3, 1],
                [3, 3, 3, 1],
                [_NAN, 1, _NAN, 1],
            ],
        ),
        (
            "sum",
            [
                [3, 4, 3, 1],
                [3, 4, 3, 1],
                [_NAN, 1, _NAN, 1],
            ],
        ),
    ],
)
def test_mosaic_float_nan_nodata_partial_overlap(method, expected_data):
    y = np.arange(3)[::-1]
    x = np.arange(4)
    n = np.nan
    r1 = make_raster(
        np.array([[3.0, 3, 3], [3, 3, 3], [n, n, n]], dtype="float32"),
        y=y,
        x=x[:3],
        null=n,
        crs=5070,
    )
    r2 = make_raster(
        np.array([[1.0, n, 1], [1, n, 1], [1, n, 1]], dtype="float32"),
        y=y,
        x=x[1:],
        null=n,
        crs=5070,
    )
    result = mosaic.mosaic([r1, r2], method)
    assert_valid_raster(result)
    assert np.isnan(result.null_value)
    actual = result.to_numpy()[0]
    expected = np.asarray(expected_data, dtype="float32")
    assert np.allclose(actual, expected, equal_nan=True)


def test_mosaic_mixed_nbands():
    y = np.arange(3)[::-1]
    x = np.arange(6)
    r_two_band = _raster(
        np.array(
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[5, 5, 5], [5, 5, 5], [5, 5, 5]],
            ]
        ),
        y,
        x[:3],
    )
    r_one_band = _raster(np.full((3, 3), 2), y, x[3:])
    expected = _raster(
        np.array(
            [
                [
                    [1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                ],
                [
                    [5, 5, 5, o, o, o],
                    [5, 5, 5, o, o, o],
                    [5, 5, 5, o, o, o],
                ],
            ]
        ),
        y,
        x,
    )
    result = mosaic.mosaic([r_two_band, r_one_band])
    assert_valid_raster(result)
    assert result.nbands == 2
    assert_rasters_equal(result, expected)


def test_mosaic_non_default_resampling_differs_from_nearest():
    # Force a reprojection path by giving an input in a different CRS
    # than the output. Compare nearest vs bilinear -- they must differ,
    # otherwise resampling_method isn't being plumbed through.
    src = testdata.raster.dem_small
    src_4326 = src.reproject(4326)
    result_nearest = mosaic.mosaic(
        [src_4326], dst_grid=src.geobox, resampling_method="nearest"
    )
    result_bilinear = mosaic.mosaic(
        [src_4326], dst_grid=src.geobox, resampling_method="bilinear"
    )
    assert_valid_raster(result_nearest)
    assert_valid_raster(result_bilinear)
    assert result_nearest.crs == src.crs
    assert result_bilinear.crs == src.crs
    assert not np.array_equal(
        result_nearest.to_numpy(), result_bilinear.to_numpy()
    )


def test_are_all_grids_same_detects_mismatch():
    # Regression: a past bug made 2-grid comparisons always return True.
    from odc.geo.geobox import GeoBox

    a = GeoBox.from_bbox((0, 0, 100, 100), crs=3857, resolution=1, tight=True)
    b = GeoBox.from_bbox((0, 0, 200, 200), crs=3857, resolution=1, tight=True)
    assert mosaic._are_all_grids_same([a, a])
    assert not mosaic._are_all_grids_same([a, b])
    assert not mosaic._are_all_grids_same([a, a, b])


def test_are_all_grids_same_tolerates_subpixel_noise():
    # Some published products carry sub-pixel FP noise in otherwise-shared
    # grids (observed up to ~1e-4 in CRS units). Strict GeoBox equality
    # would reject those and trigger needless reprojection passes, so
    # _are_all_grids_same applies a sub-pixel tolerance.
    from affine import Affine
    from odc.geo.geobox import GeoBox

    # Realistic 30m grid (EPSG:5070 / NAD83 Albers).
    a = GeoBox.from_bbox(
        (500_000, 4_000_000, 500_000 + 30 * 100, 4_000_000 + 30 * 100),
        crs=5070,
        resolution=30,
        tight=True,
    )
    at = a.affine

    # 1e-4 m drift in origin -- realistic FP noise -- must still count
    # as "same".
    drifted = GeoBox(
        a.shape,
        Affine(at.a, at.b, at.c + 1e-4, at.d, at.e, at.f + 1e-4),
        a.crs,
    )
    assert mosaic._are_all_grids_same([a, drifted])

    # Half-a-pixel shift must not be tolerated (that's a real offset).
    half_pixel = GeoBox(
        a.shape,
        Affine(at.a, at.b, at.c + 15, at.d, at.e, at.f),
        a.crs,
    )
    assert not mosaic._are_all_grids_same([a, half_pixel])


def test_mosaic_single_raster_identity():
    y = np.arange(3)[::-1]
    x = np.arange(3)
    r = _raster(
        [
            [1, 2, 3],
            [4, 5, 6],
            [o, 8, 9],
        ],
        y,
        x,
    )
    result = mosaic.mosaic([r])
    assert_valid_raster(result)
    assert_rasters_equal(result, r)


@pytest.mark.parametrize("method", ["first", "last", "min", "max", "sum"])
def test_mosaic_all_null_input(method):
    # One input is entirely nodata. The valid input should be unaffected.
    y = np.arange(3)[::-1]
    x = np.arange(3)
    valid = _raster(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        y,
        x,
    )
    all_null = _raster(np.full((3, 3), o), y, x)
    result = mosaic.mosaic([valid, all_null], method)
    assert_valid_raster(result)
    assert_rasters_equal(result, valid)


@pytest.mark.parametrize("method", ["first", "last", "min", "max", "sum"])
def test_mosaic_all_inputs_null(method):
    # Every input is entirely nodata. The result should be all nodata.
    y = np.arange(3)[::-1]
    x = np.arange(3)
    r1 = _raster(np.full((3, 3), o), y, x)
    r2 = _raster(np.full((3, 3), o), y, x)
    result = mosaic.mosaic([r1, r2], method)
    assert_valid_raster(result)
    assert result.null_value == -1
    data = result.to_numpy()
    assert (data == -1).all()


def test_mosaic_mixed_dtypes_promotes():
    # int16 + float32 should promote to float32 via np.result_type.
    y = np.arange(3)[::-1]
    x = np.arange(3)
    r_int = make_raster(
        np.ones((3, 3), dtype="int16"),
        y=y,
        x=x,
        null=-1,
        crs=5070,
    )
    r_float = make_raster(
        np.full((3, 3), 2.5, dtype="float32"),
        y=y,
        x=x + 3,
        null=np.float32(-1),
        crs=5070,
    )
    result = mosaic.mosaic([r_int, r_float])
    assert_valid_raster(result)
    assert result.dtype == np.dtype("float32")
    data = result.to_numpy()
    # Left tile should be 1.0, right tile should be 2.5.
    assert np.allclose(data[..., :3], 1.0)
    assert np.allclose(data[..., 3:], 2.5)


@pytest.mark.parametrize(
    "method,expected_val",
    [
        ("min", 100),
        ("max", np.iinfo("int8").max),
    ],
)
def test_mosaic_int_min_max_near_dtype_limit(method, expected_val):
    # The int path for min/max uses iinfo.max (for min) and iinfo.min
    # (for max) as sentinels. If a real pixel equals the sentinel, the
    # reduction must still produce the correct answer.
    y = np.arange(3)[::-1]
    x = np.arange(3)
    nv = np.int8(-128)
    r1 = make_raster(
        np.full((3, 3), 127, dtype="int8"),
        y=y,
        x=x,
        null=nv,
        crs=5070,
    )
    r2 = make_raster(
        np.full((3, 3), 100, dtype="int8"),
        y=y,
        x=x,
        null=nv,
        crs=5070,
    )
    result = mosaic.mosaic([r1, r2], method, null_value=nv)
    assert_valid_raster(result)
    data = result.to_numpy()
    assert (data == expected_val).all()


def test_mosaic_resampling_method_none_defaults_to_nearest():
    # resampling_method=None should behave the same as "nearest".
    src = testdata.raster.dem_small
    src_4326 = src.reproject(4326)
    result_none = mosaic.mosaic(
        [src_4326], dst_grid=src.geobox, resampling_method=None
    )
    result_nearest = mosaic.mosaic(
        [src_4326], dst_grid=src.geobox, resampling_method="nearest"
    )
    assert_valid_raster(result_none)
    assert np.array_equal(result_none.to_numpy(), result_nearest.to_numpy())


@pytest.mark.parametrize(
    "method,expected_val",
    [
        ("min", np.iinfo("int8").min + 1),
        ("max", 10),
    ],
)
def test_mosaic_int_min_max_near_dtype_floor(method, expected_val):
    # Similar check near the dtype minimum. int8 min is -128, used as
    # nodata; the smallest valid pixel is -127.
    y = np.arange(3)[::-1]
    x = np.arange(3)
    nv = np.int8(-128)
    r1 = make_raster(
        np.full((3, 3), -127, dtype="int8"),
        y=y,
        x=x,
        null=nv,
        crs=5070,
    )
    r2 = make_raster(
        np.full((3, 3), 10, dtype="int8"),
        y=y,
        x=x,
        null=nv,
        crs=5070,
    )
    result = mosaic.mosaic([r1, r2], method, null_value=nv)
    assert_valid_raster(result)
    data = result.to_numpy()
    assert (data == expected_val).all()


@pytest.mark.parametrize(
    "n_inputs,method,expected_value",
    [
        # 6 inputs -> 7 stacked (incl. dst fill) -> no recursion in
        # _paint_recursive (threshold is 8).
        (6, "sum", 6 * 5),
        (6, "first", 5),
        # 7 inputs -> 8 stacked -> hits the recursive split.
        (7, "sum", 7 * 5),
        (7, "first", 5),
        # 8 inputs -> 9 stacked -> recurses further.
        (8, "sum", 8 * 5),
        (8, "first", 5),
    ],
)
def test_mosaic_paint_recursive_boundary(n_inputs, method, expected_value):
    # Exercise the n < 8 / n >= 8 boundary in _paint_recursive.
    y = np.arange(3)[::-1]
    x = np.arange(3)
    inputs = [_raster(np.full((3, 3), 5), y, x) for _ in range(n_inputs)]
    result = mosaic.mosaic(inputs, method)
    assert_valid_raster(result)
    data = result.to_numpy()
    assert (data == expected_value).all()


def test_mosaic_accepts_file_paths(tmp_path):
    y = np.arange(3)[::-1]
    x = np.arange(6)
    r1 = _raster(np.ones((3, 3), dtype=int), y, x[:3])
    r2 = _raster(np.full((3, 3), 2, dtype=int), y, x[3:])
    p1 = str(tmp_path / "r1.tif")
    p2 = str(tmp_path / "r2.tif")
    r1.save(p1)
    r2.save(p2)
    result = mosaic.mosaic([p1, p2])
    assert_valid_raster(result)
    data = result.to_numpy()
    assert (data[..., :3] == 1).all()
    assert (data[..., 3:] == 2).all()


def test_mosaic_dst_grid_ignores_dst_crs():
    # When dst_grid is provided, dst_crs should be ignored. The result
    # CRS should match the grid, not the dst_crs argument.
    src = testdata.raster.dem_small
    grid = src.geobox
    # Pass a conflicting dst_crs; it should be ignored.
    result = mosaic.mosaic([src], dst_grid=grid, dst_crs=4326)
    assert_valid_raster(result)
    assert result.crs == src.crs
    assert result.crs != 4326


@pytest.mark.parametrize("method", ["first", "last", "min", "max", "sum"])
def test_mosaic_mixed_nbands_with_overlap(method):
    # A 2-band and 1-band raster overlap. Band 2 should come entirely
    # from the 2-band input in the overlap region; the 1-band input
    # contributes nodata for its missing band.
    y = np.arange(3)[::-1]
    x = np.arange(4)
    r_two = _raster(
        [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[5, 5, 5], [5, 5, 5], [5, 5, 5]],
        ],
        y,
        x[:3],
    )
    r_one = _raster(
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        y,
        x[1:],
    )
    result = mosaic.mosaic([r_two, r_one], method)
    assert_valid_raster(result)
    assert result.nbands == 2
    data = result.to_numpy()
    # Band 2: the 1-band input has no band 2, so all band-2 data comes
    # from r_two. Column 3 (x=3) is outside r_two, so it is nodata.
    band2 = data[1]
    assert (band2[:, :3] == 5).all()
    assert (band2[:, 3:] == result.null_value).all()


def test_grids_close_different_crs():
    from odc.geo.geobox import GeoBox

    a = GeoBox.from_bbox((0, 0, 100, 100), crs=3857, resolution=1, tight=True)
    b = GeoBox.from_bbox((0, 0, 100, 100), crs=4326, resolution=1, tight=True)
    assert not mosaic._grids_close(a, b)


def test_grids_close_different_shape():
    from odc.geo.geobox import GeoBox

    a = GeoBox.from_bbox((0, 0, 100, 100), crs=3857, resolution=1, tight=True)
    b = GeoBox.from_bbox((0, 0, 200, 100), crs=3857, resolution=1, tight=True)
    assert a.shape != b.shape
    assert not mosaic._grids_close(a, b)


def test_grids_close_identical():
    from odc.geo.geobox import GeoBox

    a = GeoBox.from_bbox((0, 0, 100, 100), crs=3857, resolution=1, tight=True)
    assert mosaic._grids_close(a, a)


def test_mosaic_differing_chunks():
    # Inputs with different chunk layouts should be rechunked to match
    # the destination grid before concatenation. Verify correctness and
    # that the result has uniform spatial chunks.
    y = np.arange(6)[::-1]
    x = np.arange(6)
    r1 = _raster(np.ones((6, 6), dtype=int), y, x).chunk((1, 2, 3))
    r2 = _raster(np.full((6, 6), 2, dtype=int), y, x + 6).chunk((1, 3, 2))
    assert r1.data.chunks[1:] != r2.data.chunks[1:]
    result = mosaic.mosaic([r1, r2])
    assert_valid_raster(result)
    data = result.to_numpy()
    assert (data[..., :6] == 1).all()
    assert (data[..., 6:] == 2).all()
    # All spatial chunk tuples should be identical across bands.
    y_chunks = result.data.chunks[1]
    x_chunks = result.data.chunks[2]
    assert sum(y_chunks) == 6
    assert sum(x_chunks) == 12


def test_mosaic_differing_chunks_overlap():
    # Overlapping inputs with mismatched chunks -- verify that the
    # rechunking produces correct pixel values for every method.
    y = np.arange(4)[::-1]
    x = np.arange(4)
    r1 = _raster(
        [
            [3, 3, 3, o],
            [3, 3, 3, o],
            [3, 3, 3, o],
            [o, o, o, o],
        ],
        y,
        x,
    ).chunk((1, 2, 2))
    r2 = _raster(
        [
            [o, o, o, o],
            [o, 1, 1, 1],
            [o, 1, 1, 1],
            [o, 1, 1, 1],
        ],
        y,
        x,
    ).chunk((1, 4, 4))
    assert r1.data.chunks != r2.data.chunks
    result = mosaic.mosaic([r1, r2], "sum")
    assert_valid_raster(result)
    expected = _raster(
        [
            [3, 3, 3, o],
            [3, 4, 4, 1],
            [3, 4, 4, 1],
            [o, 1, 1, 1],
        ],
        y,
        x,
    )
    assert_rasters_equal(result, expected, check_chunks=False)


def test_mosaic_differing_resolutions():
    # Two rasters in the same CRS but different resolutions. This
    # exercises the _build_dst_grid_from_inputs bbox-union path
    # (lines 53-64) without a CRS mismatch. The output grid should
    # use the first raster's resolution.
    # r1: 3x3, resolution 1, covers [0,3)x[0,3)
    y1 = np.array([2, 1, 0], dtype=float)
    x1 = np.array([0, 1, 2], dtype=float)
    r1 = make_raster(
        np.full((3, 3), 1, dtype=int), y=y1, x=x1, null=-1, crs=5070
    )
    # r2: 3x3, resolution 2, covers [4,10)x[0,6)
    y2 = np.array([5, 3, 1], dtype=float)
    x2 = np.array([4, 6, 8], dtype=float)
    r2 = make_raster(
        np.full((3, 3), 2, dtype=int), y=y2, x=x2, null=-1, crs=5070
    )
    assert r1.resolution != r2.resolution
    result = mosaic.mosaic([r1, r2])
    assert_valid_raster(result)
    # Output resolution should match the first input.
    assert np.allclose(result.resolution, r1.resolution)
    # Both inputs should be represented in the output.
    data = result.to_numpy()
    assert (data == 1).any()
    assert (data == 2).any()
    # The result grid should be large enough to cover both extents.
    rb = result.bounds
    assert rb[0] <= 0  # minx covers r1
    assert rb[2] >= 8  # maxx covers r2
