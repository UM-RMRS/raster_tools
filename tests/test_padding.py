import dask
import dask.array as da
import numpy as np
import pytest

import raster_tools as rts
from raster_tools import _padding as padding
from raster_tools.masking import get_default_null_value
from tests.utils import assert_valid_raster, make_raster


def _grow(bounds, dx=2, dy=2):
    minx, miny, maxx, maxy = bounds
    return (minx - dx, miny - dy, maxx + dx, maxy + dy)


class TestResolveBounds:
    def test_tuple(self):
        r = make_raster(shape=(1, 4, 5), dtype="int16")
        new = _grow(r.bounds)
        out = padding.pad(r, new, fill_values=0)
        assert_valid_raster(out)
        assert out.bounds == new
        assert out.shape == (1, 8, 9)

    def test_list_and_array(self):
        r = make_raster(shape=(1, 4, 5), dtype="int16")
        new = _grow(r.bounds)
        for form in (list(new), np.asarray(new)):
            out = padding.pad(r, form, fill_values=0)
            assert out.bounds == tuple(new)

    def test_raster_target(self):
        r = make_raster(shape=(1, 4, 5), dtype="int16")
        big = make_raster(shape=(1, 8, 9), dtype="int16")
        out = padding.pad(r, big, fill_values=0)
        assert out.bounds == big.bounds
        assert out.shape == (1, 8, 9)

    def test_geobox_target(self):
        r = make_raster(shape=(1, 4, 5), dtype="int16")
        big = make_raster(shape=(1, 8, 9), dtype="int16")
        out = padding.pad(r, big.geobox, fill_values=0)
        assert out.bounds == big.bounds

    def test_bounds_inside_is_noop_shape(self):
        r = make_raster(shape=(1, 4, 5), dtype="int16")
        minx, miny, maxx, maxy = r.bounds
        out = padding.pad(
            r, (minx + 1, miny + 1, maxx - 1, maxy - 1), fill_values=0
        )
        assert out.shape == r.shape

    def test_bad_bounds_length(self):
        r = make_raster(shape=(1, 4, 5))
        with pytest.raises(ValueError):
            padding.pad(r, (0, 1, 2), fill_values=0)

    def test_bad_bounds_non_numeric(self):
        r = make_raster(shape=(1, 4, 5))
        with pytest.raises(ValueError):
            padding.pad(r, (0, 1, 2, "bad"), fill_values=0)

    def test_bad_target_type(self):
        r = make_raster(shape=(1, 4, 5))
        with pytest.raises(TypeError):
            padding.pad(r, 42, fill_values=0)

    def test_crs_mismatch_raster(self):
        r = make_raster(shape=(1, 4, 5), crs="EPSG:3857")
        other = make_raster(shape=(1, 4, 5), crs="EPSG:4326")
        with pytest.raises(ValueError, match="CRS"):
            padding.pad(r, other, fill_values=0)

    def test_crs_mismatch_geobox(self):
        r = make_raster(shape=(1, 4, 5), crs="EPSG:3857")
        other = make_raster(shape=(1, 4, 5), crs="EPSG:4326")
        with pytest.raises(ValueError, match="CRS"):
            padding.pad(r, other.geobox, fill_values=0)


class TestFillValues:
    def test_none_with_existing_null(self):
        r = make_raster(
            content="arange", shape=(1, 4, 5), dtype="int16", null=-999
        )
        out = padding.pad(r, _grow(r.bounds))
        assert_valid_raster(out)
        assert out.null_value == -999
        m = out.mask.compute()
        # Corners (newly added) are masked.
        assert m[0, 0, 0] and m[0, -1, -1]
        # Center (originally present) is unmasked.
        assert not m[0, 4, 4]

    def test_none_without_null_sets_default(self):
        r = make_raster(
            content="arange", shape=(1, 4, 5), dtype="int16", null=None
        )
        assert r.null_value is None
        out = padding.pad(r, _grow(r.bounds))
        assert_valid_raster(out)
        expected_nv = get_default_null_value(r.dtype)
        assert out.null_value == expected_nv
        m = out.mask.compute()
        assert m[0, 0, 0]

    def test_scalar_not_equal_to_null(self):
        r = make_raster(
            content="arange", shape=(1, 4, 5), dtype="int16", null=-999
        )
        out = padding.pad(r, _grow(r.bounds), fill_values=42)
        assert_valid_raster(out)
        assert out.null_value == -999
        m = out.mask.compute()
        d = out.to_numpy()
        # New cells are 42 and NOT marked null.
        assert not m[0, 0, 0]
        assert d[0, 0, 0] == 42

    def test_scalar_equal_to_null(self):
        r = make_raster(
            content="arange", shape=(1, 4, 5), dtype="int16", null=-999
        )
        out = padding.pad(r, _grow(r.bounds), fill_values=-999)
        assert_valid_raster(out)
        m = out.mask.compute()
        assert m[0, 0, 0]

    def test_scalar_no_null_unmasked_result(self):
        r = make_raster(
            content="arange", shape=(1, 4, 5), dtype="int16", null=None
        )
        out = padding.pad(r, _grow(r.bounds), fill_values=7)
        assert out.null_value is None
        m = out.mask.compute()
        assert not m.any()
        d = out.to_numpy()
        assert d[0, 0, 0] == 7

    def test_per_band_list(self):
        r = make_raster(
            content="arange", shape=(2, 4, 5), dtype="int16", null=-999
        )
        out = padding.pad(r, _grow(r.bounds), fill_values=[10, 20])
        assert_valid_raster(out)
        d = out.to_numpy()
        m = out.mask.compute()
        assert d[0, 0, 0] == 10
        assert d[1, 0, 0] == 20
        # Neither equals null.
        assert not m[0, 0, 0]
        assert not m[1, 0, 0]

    def test_per_band_one_equals_null(self):
        r = make_raster(
            content="arange", shape=(2, 4, 5), dtype="int16", null=-999
        )
        out = padding.pad(r, _grow(r.bounds), fill_values=[10, -999])
        m = out.mask.compute()
        # Band 0 corner not null, band 1 corner null.
        assert not m[0, 0, 0]
        assert m[1, 0, 0]

    def test_fill_values_length_mismatch(self):
        r = make_raster(shape=(2, 4, 5))
        with pytest.raises(ValueError, match="length"):
            padding.pad(r, _grow(r.bounds), fill_values=[1, 2, 3])

    def test_fill_values_non_scalar_element(self):
        r = make_raster(shape=(2, 4, 5))
        with pytest.raises(TypeError):
            padding.pad(r, _grow(r.bounds), fill_values=[1, "x"])

    def test_fill_values_unsupported_type(self):
        r = make_raster(shape=(1, 4, 5))
        with pytest.raises(TypeError):
            padding.pad(r, _grow(r.bounds), fill_values={"a": 1})


class TestSemantics:
    def test_existing_data_preserved(self):
        r = make_raster(content="arange", shape=(1, 4, 5), dtype="int16")
        out = padding.pad(r, _grow(r.bounds, dx=2, dy=3), fill_values=0)
        d = out.to_numpy()
        # Original block is at rows 3:7, cols 2:7.
        np.testing.assert_array_equal(d[0, 3:7, 2:7], r.to_numpy()[0])

    def test_existing_mask_preserved(self):
        r = make_raster(
            content="arange",
            shape=(1, 4, 5),
            dtype="int16",
            null=-999,
            null_pattern=[(0, 1, 2)],
        )
        assert r.mask.compute()[0, 1, 2]
        out = padding.pad(r, _grow(r.bounds), fill_values=0)
        m = out.mask.compute()
        # Originally masked cell remains masked at its new offset.
        assert m[0, 3, 4]

    def test_unmasked_input_remains_unmasked_for_data_fill(self):
        r = make_raster(
            content="arange", shape=(1, 4, 5), dtype="int16", null=None
        )
        out = padding.pad(r, _grow(r.bounds), fill_values=5)
        assert out.null_value is None
        assert not out.mask.compute().any()

    def test_result_is_lazy(self):
        r = make_raster(content="arange", shape=(1, 4, 5), dtype="int16")
        out = padding.pad(r, _grow(r.bounds), fill_values=0)
        assert isinstance(out.data, da.Array)
        assert dask.is_dask_collection(out._ds)

    def test_resolution_preserved(self):
        r = make_raster(shape=(1, 4, 5), dtype="int16")
        out = padding.pad(r, _grow(r.bounds), fill_values=0)
        assert out.resolution == r.resolution

    def test_crs_preserved(self):
        r = make_raster(shape=(1, 4, 5), dtype="int16", crs="EPSG:3857")
        out = padding.pad(r, _grow(r.bounds), fill_values=0)
        assert out.crs == r.crs

    def test_dtype_preserved(self):
        r = make_raster(shape=(1, 4, 5), dtype="int16")
        out = padding.pad(r, _grow(r.bounds), fill_values=0)
        assert out.dtype == r.dtype


class TestNullValueEdgeCases:
    def test_float_nan_null_with_nan_fill_marks_null(self):
        r = make_raster(
            content="arange",
            shape=(1, 4, 5),
            dtype="float32",
            null=float("nan"),
        )
        assert np.isnan(r.null_value)
        out = padding.pad(r, _grow(r.bounds), fill_values=float("nan"))
        assert_valid_raster(out)
        m = out.mask.compute()
        d = out.to_numpy()
        assert m[0, 0, 0]
        assert np.isnan(d[0, 0, 0])

    def test_float_nan_null_with_finite_fill(self):
        r = make_raster(
            content="arange",
            shape=(1, 4, 5),
            dtype="float32",
            null=float("nan"),
        )
        out = padding.pad(r, _grow(r.bounds), fill_values=2.5)
        m = out.mask.compute()
        d = out.to_numpy()
        # Finite fill is not null - new cells stay unmasked.
        assert not m[0, 0, 0]
        assert d[0, 0, 0] == np.float32(2.5)

    def test_float_finite_null_with_nan_fill(self):
        r = make_raster(
            content="arange",
            shape=(1, 4, 5),
            dtype="float32",
            null=-9999.0,
        )
        out = padding.pad(r, _grow(r.bounds), fill_values=float("nan"))
        m = out.mask.compute()
        d = out.to_numpy()
        # NaN != -9999, so new cells are NaN data, not marked null.
        assert not m[0, 0, 0]
        assert np.isnan(d[0, 0, 0])

    def test_none_fill_on_float_no_null(self):
        r = make_raster(
            content="arange",
            shape=(1, 4, 5),
            dtype="float32",
            null=None,
        )
        out = padding.pad(r, _grow(r.bounds))
        # Default float null is NaN-or-sentinel from masking module.
        assert out.null_value is not None
        assert out.null_value == get_default_null_value(r.dtype) or (
            np.isnan(out.null_value)
            and np.isnan(get_default_null_value(r.dtype))
        )


class TestAsymmetricPadding:
    def test_pad_east_only(self):
        r = make_raster(content="arange", shape=(1, 4, 5), dtype="int16")
        minx, miny, maxx, maxy = r.bounds
        new = (minx, miny, maxx + 3, maxy)
        out = padding.pad(r, new, fill_values=0)
        assert_valid_raster(out)
        assert out.bounds == new
        assert out.shape == (1, 4, 8)
        d = out.to_numpy()
        # Original block is at the western edge.
        np.testing.assert_array_equal(d[0, :, :5], r.to_numpy()[0])
        # Eastern strip is the fill.
        assert (d[0, :, 5:] == 0).all()

    def test_pad_north_only(self):
        r = make_raster(content="arange", shape=(1, 4, 5), dtype="int16")
        minx, miny, maxx, maxy = r.bounds
        new = (minx, miny, maxx, maxy + 2)
        out = padding.pad(r, new, fill_values=0)
        assert_valid_raster(out)
        assert out.shape == (1, 6, 5)
        d = out.to_numpy()
        # Original block sits at the bottom rows; new rows on top.
        np.testing.assert_array_equal(d[0, 2:, :], r.to_numpy()[0])
        assert (d[0, :2, :] == 0).all()

    def test_pad_one_corner(self):
        r = make_raster(content="arange", shape=(1, 4, 5), dtype="int16")
        minx, miny, maxx, maxy = r.bounds
        new = (minx - 2, miny, maxx, maxy + 1)
        out = padding.pad(r, new, fill_values=0)
        assert_valid_raster(out)
        assert out.shape == (1, 5, 7)


class TestCoordinatePreservation:
    def test_x_y_grid_aligned(self):
        r = make_raster(content="arange", shape=(1, 4, 5), dtype="int16")
        out = padding.pad(r, _grow(r.bounds, dx=2, dy=3), fill_values=0)
        # Original x/y must appear verbatim in the padded x/y arrays
        # (no half-pixel drift).
        np.testing.assert_allclose(out.x[2:7], r.x)
        np.testing.assert_allclose(out.y[3:7], r.y)

    def test_resolution_unchanged_after_asymmetric_pad(self):
        r = make_raster(shape=(1, 4, 5), dtype="int16")
        minx, miny, maxx, maxy = r.bounds
        out = padding.pad(
            r, (minx - 1, miny, maxx + 4, maxy + 2), fill_values=0
        )
        assert out.resolution == r.resolution


class TestNoCRS:
    def test_pad_without_crs(self):
        r = make_raster(
            content="arange", shape=(1, 4, 5), dtype="int16", crs=None
        )
        assert r.crs is None
        out = padding.pad(r, _grow(r.bounds), fill_values=0)
        assert_valid_raster(out)
        assert out.crs is None
        assert out.shape == (1, 8, 9)

    def test_pad_without_crs_masked(self):
        r = make_raster(
            content="arange",
            shape=(1, 4, 5),
            dtype="int16",
            null=-999,
            crs=None,
        )
        out = padding.pad(r, _grow(r.bounds))
        assert_valid_raster(out)
        assert out.crs is None
        assert out.null_value == -999
        assert out.mask.compute()[0, 0, 0]


class TestRasterMethod:
    def test_method_matches_function(self):
        r = make_raster(
            content="arange", shape=(1, 4, 5), dtype="int16", null=-999
        )
        bounds = _grow(r.bounds)
        out_method = r.pad(bounds, fill_values=42)
        out_func = padding.pad(r, bounds, fill_values=42)
        np.testing.assert_array_equal(
            out_method.to_numpy(), out_func.to_numpy()
        )
        np.testing.assert_array_equal(
            out_method.mask.compute(), out_func.mask.compute()
        )

    def test_top_level_export(self):
        assert rts.pad is padding.pad
