# isort: off
# TODO(pygeos): remove this once shapely is the default backend for geopandas.
# Force raster_tools._compat to be loaded before geopandas when running tests
import raster_tools as rts  # noqa: F401

# isort: on

import numpy as np
import pytest

from raster_tools._stack import stack_bands
from raster_tools.masking import get_default_null_value
from tests import testdata
from tests.utils import assert_rasters_equal, assert_valid_raster, make_raster


def _raster(data, y, x, null=-1, crs=5070):
    return make_raster(data, x=x, y=y, null=null, crs=crs)


o = -1


def _same_grid_pair():
    y = np.arange(3)[::-1]
    x = np.arange(3)
    r1 = _raster(np.ones((3, 3), dtype=int), y, x)
    r2 = _raster(np.full((3, 3), 2, dtype=int), y, x)
    return r1, r2


# -- Stacking semantics ------------------------------------------------------


def test_single_raster_single_band_preserves_bands():
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
    result = stack_bands([r])
    assert_valid_raster(result)
    assert result.nbands == 1
    assert_rasters_equal(result, r, check_chunks=False)


def test_single_raster_multi_band_preserves_bands():
    y = np.arange(3)[::-1]
    x = np.arange(3)
    data = np.stack(
        [
            np.full((3, 3), 1, dtype=int),
            np.full((3, 3), 2, dtype=int),
            np.full((3, 3), 3, dtype=int),
        ]
    )
    r = _raster(data, y, x)
    result = stack_bands([r])
    assert_valid_raster(result)
    assert result.nbands == 3
    out = result.to_numpy()
    assert (out[0] == 1).all()
    assert (out[1] == 2).all()
    assert (out[2] == 3).all()


def test_two_single_band_same_grid_concatenates():
    r1, r2 = _same_grid_pair()
    result = stack_bands([r1, r2])
    assert_valid_raster(result)
    assert result.nbands == 2
    out = result.to_numpy()
    assert (out[0] == 1).all()
    assert (out[1] == 2).all()


def test_mixed_nbands_sums_bands():
    y = np.arange(3)[::-1]
    x = np.arange(3)
    r1 = _raster(np.full((1, 3, 3), 1, dtype=int), y, x)
    r2 = _raster(np.full((2, 3, 3), 2, dtype=int), y, x)
    r3 = _raster(np.full((3, 3, 3), 3, dtype=int), y, x)
    result = stack_bands([r1, r2, r3])
    assert_valid_raster(result)
    assert result.nbands == 6
    out = result.to_numpy()
    assert (out[0] == 1).all()
    assert (out[1:3] == 2).all()
    assert (out[3:6] == 3).all()


def test_stack_preserves_input_order():
    y = np.arange(3)[::-1]
    x = np.arange(3)
    rasters = [
        _raster(np.full((3, 3), v, dtype=int), y, x) for v in (10, 20, 30, 40)
    ]
    result = stack_bands(rasters)
    assert_valid_raster(result)
    assert result.nbands == 4
    out = result.to_numpy()
    for i, v in enumerate((10, 20, 30, 40)):
        assert (out[i] == v).all()


# -- join parameter ----------------------------------------------------------


def test_join_inner_uses_intersection():
    y = np.arange(3)[::-1]
    r1 = _raster(np.full((3, 3), 1, dtype=int), y, np.arange(3))
    r2 = _raster(np.full((3, 3), 2, dtype=int), y, np.arange(3) + 1)
    result = stack_bands([r1, r2], join="inner")
    assert_valid_raster(result)
    # Intersection is x in [1, 3) -> 2 columns wide.
    assert result.shape[2] == 2
    out = result.to_numpy()
    assert (out[0] == 1).all()
    assert (out[1] == 2).all()


def test_join_outer_uses_union():
    y = np.arange(3)[::-1]
    r1 = _raster(np.full((3, 3), 1, dtype=int), y, np.arange(3))
    r2 = _raster(np.full((3, 3), 2, dtype=int), y, np.arange(3) + 3)
    result = stack_bands([r1, r2], join="outer")
    assert_valid_raster(result)
    # Union is x in [0, 6) -> 6 columns wide.
    assert result.shape[2] == 6
    nv = result.null_value
    out = result.to_numpy()
    # r1 present in left half of band 0, right half is null.
    assert (out[0, :, :3] == 1).all()
    assert (out[0, :, 3:] == nv).all()
    # r2 present in right half of band 1, left half is null.
    assert (out[1, :, 3:] == 2).all()
    assert (out[1, :, :3] == nv).all()


def test_join_identical_bounds_equivalent():
    r1, r2 = _same_grid_pair()
    inner = stack_bands([r1, r2], join="inner")
    outer = stack_bands([r1, r2], join="outer")
    assert_rasters_equal(inner, outer, check_chunks=False)


# -- dst_grid parameter forms ------------------------------------------------


@pytest.mark.parametrize("form", ["geobox", "raster", "path"])
def test_dst_grid_accepts_forms(form, tmp_path):
    r1, r2 = _same_grid_pair()
    reference = stack_bands([r1, r2])
    if form == "geobox":
        dst_grid = reference.geobox
    elif form == "raster":
        dst_grid = reference
    else:
        path = tmp_path / "reference.tif"
        reference.save(str(path))
        dst_grid = str(path)
    result = stack_bands([r1, r2], dst_grid=dst_grid)
    assert_valid_raster(result)
    assert_rasters_equal(result, reference, check_chunks=False)


@pytest.mark.parametrize("bad_dst_grid", [42, {}, 3.14, object()])
def test_dst_grid_bad_type_raises(bad_dst_grid):
    r1, r2 = _same_grid_pair()
    with pytest.raises(TypeError):
        stack_bands([r1, r2], dst_grid=bad_dst_grid)


def test_dst_grid_overrides_join():
    # dst_grid covers a different extent than either join mode would pick.
    y = np.arange(3)[::-1]
    r1 = _raster(np.full((3, 3), 1, dtype=int), y, np.arange(3))
    r2 = _raster(np.full((3, 3), 2, dtype=int), y, np.arange(3) + 1)
    # Hand-picked grid: just the leftmost column of r1.
    custom = _raster(np.zeros((3, 1), dtype=int), y, np.array([0]))
    result = stack_bands([r1, r2], join="outer", dst_grid=custom.geobox)
    assert_valid_raster(result)
    assert result.geobox == custom.geobox


# -- dst_crs parameter -------------------------------------------------------


def test_dst_crs_default_is_first_input_crs():
    src = testdata.raster.dem_small
    src_4326 = src.reproject(4326)
    result = stack_bands([src, src_4326])
    assert_valid_raster(result)
    assert result.crs == src.crs


def test_dst_crs_reprojects_inputs():
    src = testdata.raster.dem_small
    result = stack_bands([src], dst_crs=4326)
    assert_valid_raster(result)
    assert result.crs == 4326


def test_dst_grid_matches_dst_crs_ok():
    src = testdata.raster.dem_small
    result = stack_bands([src], dst_grid=src.geobox, dst_crs=src.crs)
    assert_valid_raster(result)
    assert result.crs == src.crs


def test_dst_grid_conflicting_dst_crs_raises():
    src = testdata.raster.dem_small
    with pytest.raises(ValueError, match="dst_crs does not match"):
        stack_bands([src], dst_grid=src.geobox, dst_crs=4326)


def test_dst_grid_with_resolution_raises():
    src = testdata.raster.dem_small
    with pytest.raises(ValueError, match="resolution cannot"):
        stack_bands([src], dst_grid=src.geobox, resolution=100)


def test_resolution_changes_output_grid():
    src = testdata.raster.dem_small
    native = abs(src.resolution[0])
    coarser = stack_bands([src], resolution=native * 2)
    assert_valid_raster(coarser)
    assert np.isclose(abs(coarser.resolution[0]), native * 2)
    # Coarser grid should have roughly half as many cells on each spatial axis.
    assert coarser.shape[1] <= src.shape[1] // 2 + 2
    assert coarser.shape[2] <= src.shape[2] // 2 + 2


def test_resolution_with_multiple_inputs():
    src = testdata.raster.dem_small
    src_4326 = src.reproject(4326)
    native = abs(src.resolution[0])
    result = stack_bands([src, src_4326], resolution=native * 2)
    assert_valid_raster(result)
    assert result.nbands == 2
    assert np.isclose(abs(result.resolution[0]), native * 2)


# -- Fast path vs reproject path ---------------------------------------------


def test_single_raster_fast_path_returns_input(monkeypatch):
    r = _raster(np.ones((3, 3), dtype=int), np.arange(3)[::-1], np.arange(3))

    calls = []
    orig_reproject = rts.Raster.reproject
    orig_astype = rts.Raster.astype

    def reproject_spy(self, *a, **kw):
        calls.append(("reproject",))
        return orig_reproject(self, *a, **kw)

    def astype_spy(self, *a, **kw):
        calls.append(("astype",))
        return orig_astype(self, *a, **kw)

    monkeypatch.setattr(rts.Raster, "reproject", reproject_spy)
    monkeypatch.setattr(rts.Raster, "astype", astype_spy)
    result = stack_bands([r])
    # No transforms: same dtype, same null, no grid/CRS/res change.
    assert calls == []
    assert result is r


def test_fast_path_same_grid_no_reproject(monkeypatch):
    r1, r2 = _same_grid_pair()

    calls = []
    original = rts.Raster.reproject

    def spy(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(rts.Raster, "reproject", spy)
    result = stack_bands([r1, r2])
    assert_valid_raster(result)
    assert calls == []
    assert result.nbands == 2


def test_reproject_path_when_grids_differ():
    src = testdata.raster.dem_small
    src_4326 = src.reproject(4326)
    result = stack_bands([src, src_4326], dst_grid=src.geobox)
    assert_valid_raster(result)
    assert result.nbands == 2
    assert result.crs == src.crs
    # Band 0 is the pristine src, should match exactly.
    assert np.allclose(result.to_numpy()[0], src.to_numpy()[0])


# -- Dtype and null value handling -------------------------------------------


def test_explicit_dtype():
    r1, r2 = _same_grid_pair()
    result = stack_bands([r1, r2], dtype="float64")
    assert_valid_raster(result)
    assert result.dtype == np.dtype("float64")


def test_mixed_dtypes_promote():
    y = np.arange(3)[::-1]
    x = np.arange(3)
    r_int = make_raster(
        np.ones((3, 3), dtype="int16"), y=y, x=x, null=-1, crs=5070
    )
    r_float = make_raster(
        np.full((3, 3), 2.5, dtype="float32"),
        y=y,
        x=x,
        null=np.float32(-1),
        crs=5070,
    )
    result = stack_bands([r_int, r_float])
    assert_valid_raster(result)
    assert result.dtype == np.dtype("float32")
    out = result.to_numpy()
    assert np.allclose(out[0], 1.0)
    assert np.allclose(out[1], 2.5)


def test_explicit_null_value():
    y = np.arange(3)[::-1]
    r1 = _raster(np.full((3, 3), 1, dtype=int), y, np.arange(3))
    r2 = _raster(np.full((3, 3), 2, dtype=int), y, np.arange(3) + 3)
    result = stack_bands([r1, r2], join="outer", null_value=-99)
    assert_valid_raster(result)
    assert result.null_value == -99
    out = result.to_numpy()
    assert (out == -99).any()


def test_default_null_from_first_non_none():
    y = np.arange(3)[::-1]
    x = np.arange(3)
    r1 = make_raster(np.ones((3, 3), dtype=int), y=y, x=x, null=-7, crs=5070)
    r2 = make_raster(
        np.full((3, 3), 2, dtype=int), y=y, x=x, null=-3, crs=5070
    )
    result = stack_bands([r1, r2])
    assert_valid_raster(result)
    assert result.null_value == -7


def test_default_null_fallback_when_all_none():
    y = np.arange(3)[::-1]
    x = np.arange(3)
    r1 = make_raster(np.ones((3, 3), dtype="float32"), y=y, x=x, crs=5070)
    r2 = make_raster(np.full((3, 3), 2, dtype="float32"), y=y, x=x, crs=5070)
    assert r1.null_value is None
    assert r2.null_value is None
    result = stack_bands([r1, r2])
    assert_valid_raster(result)
    assert result.null_value == get_default_null_value(np.dtype("float32"))


def test_default_null_string_forces_dtype_default():
    y = np.arange(3)[::-1]
    x = np.arange(3)
    r1 = make_raster(np.ones((3, 3), dtype=int), y=y, x=x, null=-7, crs=5070)
    r2 = make_raster(
        np.full((3, 3), 2, dtype=int), y=y, x=x, null=-3, crs=5070
    )
    result = stack_bands([r1, r2], null_value="default")
    assert_valid_raster(result)
    assert result.null_value == get_default_null_value(result.dtype)
    assert result.null_value != -7


@pytest.mark.parametrize("bad_null_value", ["x", [1], (1, 2), {1}])
def test_bad_null_value_type_raises(bad_null_value):
    r1, r2 = _same_grid_pair()
    with pytest.raises(TypeError):
        stack_bands([r1, r2], null_value=bad_null_value)


# -- Resampling --------------------------------------------------------------


@pytest.mark.parametrize("resampling", ["foo", "NEAREST", 42])
def test_invalid_resampling_raises(resampling):
    r1, r2 = _same_grid_pair()
    with pytest.raises(ValueError):
        stack_bands([r1, r2], resampling_method=resampling)


def test_resampling_method_none_defaults_to_nearest():
    src = testdata.raster.dem_small
    src_4326 = src.reproject(4326)
    result_none = stack_bands(
        [src_4326], dst_grid=src.geobox, resampling_method=None
    )
    result_nearest = stack_bands(
        [src_4326], dst_grid=src.geobox, resampling_method="nearest"
    )
    assert_valid_raster(result_none)
    assert_valid_raster(result_nearest)
    assert np.array_equal(result_none.to_numpy(), result_nearest.to_numpy())


def test_non_default_resampling_differs_from_nearest():
    src = testdata.raster.dem_small
    src_4326 = src.reproject(4326)
    result_nearest = stack_bands(
        [src_4326], dst_grid=src.geobox, resampling_method="nearest"
    )
    result_bilinear = stack_bands(
        [src_4326], dst_grid=src.geobox, resampling_method="bilinear"
    )
    assert_valid_raster(result_nearest)
    assert_valid_raster(result_bilinear)
    assert not np.array_equal(
        result_nearest.to_numpy(), result_bilinear.to_numpy()
    )


# -- Errors and edge cases ---------------------------------------------------


def test_empty_rasters_list_raises():
    with pytest.raises(ValueError):
        stack_bands([])


@pytest.mark.parametrize(
    "join", ["union", "intersection", "INNER", "", None, 1]
)
def test_invalid_join_raises(join):
    r1, r2 = _same_grid_pair()
    with pytest.raises(ValueError):
        stack_bands([r1, r2], join=join)


def test_inputs_with_file_paths(tmp_path):
    r1, r2 = _same_grid_pair()
    p1 = str(tmp_path / "r1.tif")
    p2 = str(tmp_path / "r2.tif")
    r1.save(p1)
    r2.save(p2)
    result = stack_bands([p1, p2])
    assert_valid_raster(result)
    assert result.nbands == 2
    out = result.to_numpy()
    assert (out[0] == 1).all()
    assert (out[1] == 2).all()


# -- Chunking ----------------------------------------------------------------


def test_differing_input_chunks_rechunked_for_concat():
    y = np.arange(6)[::-1]
    x = np.arange(6)
    r1 = _raster(np.ones((6, 6), dtype=int), y, x).chunk((1, 2, 3))
    r2 = _raster(np.full((6, 6), 2, dtype=int), y, x).chunk((1, 3, 2))
    assert r1.data.chunks[1:] != r2.data.chunks[1:]
    result = stack_bands([r1, r2])
    assert_valid_raster(result)
    assert result.nbands == 2
    out = result.to_numpy()
    assert (out[0] == 1).all()
    assert (out[1] == 2).all()
    # All spatial chunks sum to full extent.
    y_chunks, x_chunks = result.data.chunks[1], result.data.chunks[2]
    assert sum(y_chunks) == 6
    assert sum(x_chunks) == 6
