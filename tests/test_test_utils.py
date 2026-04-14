import numpy as np
import pytest
import rasterio as rio
import xarray as xr
from affine import Affine

from raster_tools.dtypes import F32, F64, I16, I32, U8, U16
from raster_tools.masking import get_default_null_value
from raster_tools.raster import Raster
from tests.utils import assert_valid_raster, make_raster


def test_default_returns_valid_raster():
    raster = make_raster()

    assert_valid_raster(raster)
    assert isinstance(raster, Raster)
    assert raster.shape == (1, 6, 6)
    assert raster.dtype == np.dtype("int64")
    assert raster.crs == rio.crs.CRS.from_user_input("EPSG:3857")
    assert raster.null_value is None
    assert np.array_equal(
        raster.data.compute(), np.ones((1, 6, 6), dtype=np.int64)
    )
    assert not raster.mask.compute().any()


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((6, 6), (1, 6, 6)),
        ((1, 4, 5), (1, 4, 5)),
        ((3, 4, 5), (3, 4, 5)),
    ],
)
def test_shape_variants(shape, expected):
    raster = make_raster(shape=shape)

    assert_valid_raster(raster)
    assert raster.shape == expected


@pytest.mark.parametrize("shape", [(5,), (1, 2, 3, 4)])
def test_invalid_shape_raises(shape):
    with pytest.raises(ValueError, match="Shape must be 2D or 3D"):
        make_raster(shape=shape)


@pytest.mark.parametrize(
    "content,expected",
    [
        ("ones", np.ones((1, 6, 6), dtype=np.int64)),
        ("zeros", np.zeros((1, 6, 6), dtype=np.int64)),
        ("arange", np.arange(36, dtype=np.int64).reshape(1, 6, 6)),
    ],
)
def test_string_content(content, expected):
    raster = make_raster(content)

    assert_valid_raster(raster)
    assert np.array_equal(raster.data.compute(), expected)


def test_peak_content():
    raster = make_raster("peak", dtype="float64", shape=(1, 5, 5))

    assert_valid_raster(raster)
    data = raster.data.compute()
    # Peak at center
    assert data[0, 2, 2] == data.max()
    # Edges are ~zero (sin(0) and sin(pi))
    assert np.allclose(data[0, 0, :], 0.0, atol=1e-15)
    assert np.allclose(data[0, -1, :], 0.0, atol=1e-15)
    assert np.allclose(data[0, :, 0], 0.0, atol=1e-15)
    assert np.allclose(data[0, :, -1], 0.0, atol=1e-15)
    # Symmetric
    assert np.allclose(data[0], data[0, ::-1, :])
    assert np.allclose(data[0], data[0, :, ::-1])


def test_peak_multiband():
    raster = make_raster("peak", dtype="float64", shape=(3, 4, 4))

    assert_valid_raster(raster)
    data = raster.data.compute()
    # All bands identical
    assert np.array_equal(data[0], data[1])
    assert np.array_equal(data[0], data[2])


def test_invalid_content_string_raises():
    with pytest.raises(ValueError, match="Unknown content"):
        make_raster("nope")


def test_array_content_from_list():
    data = [[1, 2, 3], [4, 5, 6]]
    raster = make_raster(data)

    assert_valid_raster(raster)
    assert raster.shape == (1, 2, 3)
    assert np.array_equal(raster.data.compute(), np.asarray(data)[None])


def test_array_content_from_ndarray():
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    raster = make_raster(data)

    assert_valid_raster(raster)
    assert raster.shape == (2, 3, 4)
    assert raster.dtype == np.dtype(I16)
    assert np.array_equal(raster.data.compute(), data)


def test_array_content_dtype_cast():
    raster = make_raster([[1, 2], [3, 4]], dtype="float32")

    assert_valid_raster(raster)
    assert raster.dtype == np.dtype(F32)
    assert np.array_equal(
        raster.data.compute(),
        np.asarray([[1, 2], [3, 4]], dtype=np.float32)[None],
    )


def test_array_content_without_dtype_keeps_input_dtype():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int16)
    raster = make_raster(arr)

    assert raster.dtype == np.dtype(I16)


@pytest.mark.parametrize("dtype", [F32, F64, I16, I32, U8, U16])
@pytest.mark.parametrize("content", ["ones", "zeros", "arange"])
def test_dtype_applied_to_string_content(content, dtype):
    raster = make_raster(content, dtype=dtype, shape=(1, 3, 3))

    assert_valid_raster(raster)
    assert raster.dtype == np.dtype(dtype)


def test_mask_without_null_uses_default_nv():
    content = [[10, 20], [30, 40]]
    mask = [[1, 0], [0, 1]]
    raster = make_raster(content, mask=mask)

    assert_valid_raster(raster)
    expected_nv = get_default_null_value(np.dtype("int64"))
    assert raster.null_value == expected_nv
    assert np.array_equal(
        raster.mask.compute(),
        np.asarray(mask, dtype=bool)[None],
    )
    expected_data = np.asarray(
        [[expected_nv, 20], [30, expected_nv]], dtype=np.int64
    )[None]
    assert np.array_equal(raster.data.compute(), expected_data)


def test_scalar_null_sets_nv_without_mask_arg():
    content = [[1, -1], [2, -1]]
    raster = make_raster(content, null=-1)

    assert_valid_raster(raster)
    assert raster.null_value == -1
    expected_mask = np.asarray([[False, True], [False, True]])[None]
    assert np.array_equal(raster.mask.compute(), expected_mask)


def test_list_null_builds_mask_from_values():
    content = [[1, 2, 3], [4, 5, 6]]
    raster = make_raster(content, null=[2, 5])

    assert_valid_raster(raster)
    assert raster.null_value == 5
    data = raster.data.compute()
    mask = raster.mask.compute()
    expected_mask = np.isin(np.asarray(content, dtype=np.int64)[None], [2, 5])
    assert np.array_equal(mask, expected_mask)
    # Burned positions in data equal nv (5).
    assert np.all(data[mask] == 5)


def test_tuple_null_behaves_like_list():
    content = [[1, 2, 3], [4, 5, 6]]
    raster = make_raster(content, null=(2, 5))

    assert raster.null_value == 5
    expected_mask = np.isin(np.asarray(content, dtype=np.int64)[None], [2, 5])
    assert np.array_equal(raster.mask.compute(), expected_mask)


def test_null_true_uses_dtype_default():
    raster = make_raster("arange", dtype=I16, null=True, shape=(1, 3, 3))

    assert_valid_raster(raster)
    assert raster.null_value == get_default_null_value(np.dtype(I16))
    # No explicit mask supplied and arange values don't match the
    # sentinel, so the derived mask is all False.
    assert not raster.mask.compute().any()


def test_no_null_no_mask():
    raster = make_raster()

    assert raster.null_value is None
    assert not raster.mask.compute().any()


def test_null_pattern_modulo():
    raster = make_raster(
        "arange", dtype=I32, shape=(1, 4, 4), null_pattern="%3"
    )

    assert_valid_raster(raster)
    expected_nv = get_default_null_value(np.dtype(I32))
    assert raster.null_value == expected_nv
    arr = np.arange(16, dtype=np.int32).reshape(1, 4, 4)
    expected_mask = (arr % 3) == 0
    assert np.array_equal(raster.mask.compute(), expected_mask)
    data = raster.data.compute()
    assert np.all(data[expected_mask] == expected_nv)
    assert np.array_equal(data[~expected_mask], arr[~expected_mask])


def test_null_pattern_bare_slice():
    raster = make_raster(
        "arange",
        dtype=I32,
        shape=(1, 4, 4),
        null_pattern=np.s_[0:2],
    )

    assert_valid_raster(raster)
    expected_nv = get_default_null_value(np.dtype(I32))
    assert raster.null_value == expected_nv
    mask = raster.mask.compute()
    expected_mask = np.zeros((1, 4, 4), dtype=bool)
    expected_mask[0:2] = True
    assert np.array_equal(mask, expected_mask)
    data = raster.data.compute()
    assert np.all(data[expected_mask] == expected_nv)


def test_null_pattern_tuple_of_slices():
    raster = make_raster(
        "arange",
        dtype=I32,
        shape=(1, 4, 4),
        null_pattern=np.s_[:, 1:3, 1:3],
    )

    assert_valid_raster(raster)
    expected_nv = get_default_null_value(np.dtype(I32))
    mask = raster.mask.compute()
    expected_mask = np.zeros((1, 4, 4), dtype=bool)
    expected_mask[:, 1:3, 1:3] = True
    assert np.array_equal(mask, expected_mask)
    data = raster.data.compute()
    assert np.all(data[expected_mask] == expected_nv)


def test_null_pattern_mixed_int_slice():
    raster = make_raster(
        "arange",
        dtype=I32,
        shape=(1, 4, 4),
        null_pattern=np.s_[0, 2, :],
    )

    assert_valid_raster(raster)
    expected_nv = get_default_null_value(np.dtype(I32))
    mask = raster.mask.compute()
    expected_mask = np.zeros((1, 4, 4), dtype=bool)
    expected_mask[0, 2, :] = True
    assert np.array_equal(mask, expected_mask)
    data = raster.data.compute()
    assert np.all(data[expected_mask] == expected_nv)


def test_null_pattern_list_of_patterns():
    raster = make_raster(
        "arange",
        dtype=I32,
        shape=(1, 4, 4),
        null_pattern=["%4", np.s_[0, 0, 1:3]],
    )

    assert_valid_raster(raster)
    expected_nv = get_default_null_value(np.dtype(I32))
    arr = np.arange(16, dtype=np.int32).reshape(1, 4, 4)
    expected_mask = (arr % 4) == 0
    expected_mask[0, 0, 1:3] = True
    assert np.array_equal(raster.mask.compute(), expected_mask)
    data = raster.data.compute()
    assert np.all(data[expected_mask] == expected_nv)


def test_null_pattern_unknown_string_raises():
    with pytest.raises(ValueError, match="Unknown null_pattern string"):
        make_raster("arange", null_pattern="bogus")


def test_null_pattern_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown null_pattern type"):
        make_raster("arange", null_pattern=42)


def test_null_pattern_empty_list_raises():
    with pytest.raises(ValueError, match="null_pattern list was empty"):
        make_raster("arange", null_pattern=[])


def test_null_pattern_too_nested_raises():
    with pytest.raises(ValueError, match="Too much nesting"):
        make_raster("arange", null_pattern=[[["bogus"]]])


def test_default_chunking():
    raster = make_raster(shape=(3, 6, 6))

    assert raster.data.chunksize[0] == 1


def test_explicit_chunksize_honored():
    raster = make_raster(shape=(2, 8, 8), chunksize=(1, 4, 4))

    assert raster.data.chunks == ((1, 1), (4, 4), (4, 4))


@pytest.mark.parametrize(
    "crs",
    ["EPSG:3857", "EPSG:4326", None],
)
def test_crs_variants(crs):
    raster = make_raster(crs=crs)

    if crs is None:
        assert raster.crs is None
    else:
        assert raster.crs == rio.crs.CRS.from_user_input(crs)


@pytest.mark.parametrize(
    "content,offset,expected",
    [
        ("ones", 5, np.ones((1, 4, 4), dtype=np.int64) + 5),
        ("zeros", 3, np.full((1, 4, 4), 3, dtype=np.int64)),
        (
            "arange",
            10,
            np.arange(16, dtype=np.int64).reshape(1, 4, 4) + 10,
        ),
    ],
)
def test_offset_shifts_string_content(content, offset, expected):
    raster = make_raster(content, shape=(1, 4, 4), offset=offset)

    assert_valid_raster(raster)
    assert np.array_equal(raster.data.compute(), expected)


def test_offset_with_array_content():
    data = [[1, 2], [3, 4]]
    raster = make_raster(data, offset=10)

    assert_valid_raster(raster)
    expected = np.asarray(data, dtype=np.int64)[None] + 10
    assert np.array_equal(raster.data.compute(), expected)


def test_offset_cast_to_dtype():
    raster = make_raster("zeros", dtype=F32, shape=(1, 3, 3), offset=1.5)

    assert_valid_raster(raster)
    assert raster.dtype == np.dtype(F32)
    assert np.array_equal(
        raster.data.compute(),
        np.full((1, 3, 3), 1.5, dtype=np.float32),
    )


def test_offset_invalid_type_raises():
    with pytest.raises(TypeError, match="offset must be None or a scalar"):
        make_raster(offset=[1, 2])


def test_custom_x_y_coords():
    x = np.array([10.0, 20.0, 30.0, 40.0])
    y = np.array([400.0, 300.0, 200.0])
    raster = make_raster(shape=(1, 3, 4), x=x, y=y)

    assert_valid_raster(raster)
    assert np.allclose(raster.x, x)
    assert np.allclose(raster.y, y)


def test_custom_affine():
    aff = Affine(10.0, 0.0, 100.0, 0.0, -10.0, 500.0)
    raster = make_raster(shape=(1, 4, 4), affine=aff)

    assert_valid_raster(raster)
    assert raster.affine == aff


def test_affine_determines_coords():
    aff = Affine(2.0, 0.0, 100.0, 0.0, -2.0, 200.0)
    raster = make_raster(shape=(1, 3, 4), affine=aff)

    assert_valid_raster(raster)
    # Pixel-center coords: origin + (idx + 0.5) * resolution
    expected_x = [101.0, 103.0, 105.0, 107.0]
    expected_y = [199.0, 197.0, 195.0]
    assert np.allclose(raster.x, expected_x)
    assert np.allclose(raster.y, expected_y)


def test_xarray_flag_returns_dataarray():
    result = make_raster(xarray=True)

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("band", "y", "x")


def test_hills_content():
    raster = make_raster("hills", dtype="float64", shape=(1, 7, 7))

    assert_valid_raster(raster)
    data = raster.data.compute()
    # sin(3*pi*0) == sin(3*pi*1) == 0 at edges.
    assert np.allclose(data[0, 0, :], 0.0, atol=1e-14)
    assert np.allclose(data[0, -1, :], 0.0, atol=1e-14)
    assert np.allclose(data[0, :, 0], 0.0, atol=1e-14)
    assert np.allclose(data[0, :, -1], 0.0, atol=1e-14)
    # Range of sin*sin is [-1, 1]; hills has both peaks and valleys.
    assert np.isclose(data.max(), 1.0)
    assert np.isclose(data.min(), -1.0)
    # Symmetric under reflection.
    assert np.allclose(data[0], data[0, ::-1, :])
    assert np.allclose(data[0], data[0, :, ::-1])


def test_hills_has_more_extrema_than_peak():
    peak = make_raster("peak", dtype="float64", shape=(1, 31, 31))
    hills = make_raster("hills", dtype="float64", shape=(1, 31, 31))

    pd = peak.data.compute()[0]
    hd = hills.data.compute()[0]
    # peak is a single hump in [0, 1]; hills oscillates between -1 and 1.
    assert pd.min() >= 0.0
    assert hd.min() < -0.5
    assert hd.max() > 0.5


@pytest.mark.parametrize(
    "direction,expected_hi,expected_lo",
    [
        # (direction, (y, x) index of max, (y, x) index of min)
        ("E", (0, -1), (0, 0)),
        ("W", (0, 0), (0, -1)),
        ("S", (-1, 0), (0, 0)),
        ("N", (0, 0), (-1, 0)),
        ("NE", (0, -1), (-1, 0)),
        ("SE", (-1, -1), (0, 0)),
        ("NW", (0, 0), (-1, -1)),
        ("SW", (-1, 0), (0, -1)),
    ],
)
def test_grad_directions(direction, expected_hi, expected_lo):
    raster = make_raster(f"grad-{direction}", dtype="float64", shape=(1, 5, 5))

    assert_valid_raster(raster)
    band = raster.data.compute()[0]
    # Max is 1.0 toward the named direction; min 0.0 at the opposite.
    assert np.isclose(band[expected_hi], 1.0)
    assert np.isclose(band[expected_lo], 0.0)
    # Values are bounded in [0, 1].
    assert band.min() >= 0.0
    assert band.max() <= 1.0


def test_grad_direction_case_insensitive():
    upper = make_raster("grad-NE", dtype="float64", shape=(1, 4, 4))
    lower = make_raster("grad-ne", dtype="float64", shape=(1, 4, 4))

    assert np.array_equal(upper.data.compute(), lower.data.compute())


def test_grad_unknown_direction_raises():
    with pytest.raises(ValueError, match="Unknown content"):
        make_raster("grad-XX")


def test_grad_multiband_identical_bands():
    raster = make_raster("grad-E", dtype="float64", shape=(3, 4, 5))

    assert_valid_raster(raster)
    data = raster.data.compute()
    assert np.array_equal(data[0], data[1])
    assert np.array_equal(data[0], data[2])


@pytest.mark.parametrize(
    "content,scale,expected",
    [
        ("ones", 7, np.full((1, 3, 3), 7, dtype=np.int64)),
        ("zeros", 9, np.zeros((1, 3, 3), dtype=np.int64)),
        (
            "arange",
            2,
            np.arange(9, dtype=np.int64).reshape(1, 3, 3) * 2,
        ),
    ],
)
def test_scale_string_content(content, scale, expected):
    raster = make_raster(content, shape=(1, 3, 3), scale=scale)

    assert_valid_raster(raster)
    assert np.array_equal(raster.data.compute(), expected)


def test_scale_with_array_content():
    data = [[1, 2], [3, 4]]
    raster = make_raster(data, scale=3)

    assert_valid_raster(raster)
    expected = np.asarray(data, dtype=np.int64)[None] * 3
    assert np.array_equal(raster.data.compute(), expected)


def test_scale_preserves_array_input_dtype():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int16)
    raster = make_raster(arr, scale=2)

    assert raster.dtype == np.dtype(I16)
    assert np.array_equal(
        raster.data.compute(), (arr * 2).astype(np.int16)[None]
    )


def test_scale_applies_to_peak():
    unscaled = make_raster("peak", dtype="float64", shape=(1, 5, 5))
    scaled = make_raster("peak", dtype="float64", shape=(1, 5, 5), scale=4)

    assert np.allclose(scaled.data.compute(), unscaled.data.compute() * 4)


def test_scale_applies_to_grad():
    raster = make_raster("grad-E", dtype="float64", shape=(1, 3, 4), scale=6)

    band = raster.data.compute()[0]
    # First column 0, last column 6.
    assert np.allclose(band[:, 0], 0.0)
    assert np.allclose(band[:, -1], 6.0)


def test_scale_cast_to_dtype():
    raster = make_raster("ones", dtype=F32, shape=(1, 2, 2), scale=2.5)

    assert_valid_raster(raster)
    assert raster.dtype == np.dtype(F32)
    assert np.array_equal(
        raster.data.compute(), np.full((1, 2, 2), 2.5, dtype=np.float32)
    )


def test_scale_then_offset_order():
    # scale first, then offset: (arange * 10) + 1
    raster = make_raster("arange", shape=(1, 2, 2), scale=10, offset=1)

    expected = np.arange(4, dtype=np.int64).reshape(1, 2, 2) * 10 + 1
    assert np.array_equal(raster.data.compute(), expected)


def test_scale_invalid_type_raises():
    with pytest.raises(TypeError, match="scale must be None or a scalar"):
        make_raster(scale=[1, 2])


def test_scale_bool_dtype_raises():
    with pytest.raises(
        TypeError, match="scale is not compatible with bool dtype"
    ):
        make_raster("ones", dtype=bool, scale=2)


def test_null_pattern_greater_than():
    raster = make_raster(
        "arange", dtype=I32, shape=(1, 4, 4), null_pattern=">9"
    )

    assert_valid_raster(raster)
    expected_nv = get_default_null_value(np.dtype(I32))
    arr = np.arange(16, dtype=np.int32).reshape(1, 4, 4)
    expected_mask = arr > 9
    assert np.array_equal(raster.mask.compute(), expected_mask)
    data = raster.data.compute()
    assert np.all(data[expected_mask] == expected_nv)
    assert np.array_equal(data[~expected_mask], arr[~expected_mask])


def test_null_pattern_less_than():
    raster = make_raster(
        "arange", dtype=I32, shape=(1, 4, 4), null_pattern="<5"
    )

    assert_valid_raster(raster)
    arr = np.arange(16, dtype=np.int32).reshape(1, 4, 4)
    expected_mask = arr < 5
    assert np.array_equal(raster.mask.compute(), expected_mask)
