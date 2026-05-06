# isort: off
# TODO(pygeos): remove this once shapely is the default backend for geopandas.
# Force raster_tools._compat to be loaded before geopandas when running tests
import raster_tools as rts  # noqa: F401

# isort: on

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine

from raster_tools.blocks import (
    GeoBlockInfo,
    geo_block_infos_as_dask,
    geo_map_blocks,
    geo_map_overlap,
    map_blocks,
    map_overlap,
)
from raster_tools.masking import get_default_null_value
from tests import testdata


def _make_block_info(
    raster,
    *,
    geobox=None,
    band_slice=None,
    row_slice=None,
    col_slice=None,
    chunk_location=(0, 0, 0),
):
    return GeoBlockInfo(
        geobox if geobox is not None else raster.geobox,
        band_slice if band_slice is not None else slice(0, 1),
        row_slice if row_slice is not None else slice(0, 100),
        col_slice if col_slice is not None else slice(0, 100),
        chunk_location,
        raster.affine,
        raster.shape,
    )


def test_geoblockinfo_creation():
    raster = testdata.raster.dem_small.chunk((1, 50, 50))
    chunk_raster = raster.get_chunk_rasters().ravel()[1]
    geobox = chunk_raster.geobox
    gbi = GeoBlockInfo(
        geobox,
        slice(0, 1),
        slice(0, 50),
        slice(50, 100),
        (0, 0, 1),
        raster.affine,
        raster.shape,
    )
    assert gbi.shape == chunk_raster.shape
    assert gbi.geobox == chunk_raster.geobox
    assert gbi.parent_affine == raster.affine
    assert gbi.parent_shape == raster.shape
    assert gbi.affine == chunk_raster.affine
    assert gbi.crs == raster.crs
    assert gbi.bbox == chunk_raster.geobox.extent.geom
    assert gbi.band_slice == slice(0, 1)
    assert gbi.row_slice == slice(0, 50)
    assert gbi.col_slice == slice(50, 100)
    assert gbi.chunk_location == (0, 0, 1)
    assert np.array_equal(gbi.band, np.arange(0, 1))
    assert np.allclose(gbi.x, chunk_raster.x)
    assert np.allclose(gbi.y, chunk_raster.y)


def test_geoblockinfo_to_dataarray():
    raster = testdata.raster.dem_small
    gbi = _make_block_info(raster)
    data = np.arange(np.prod(gbi.shape), dtype=np.float32).reshape(gbi.shape)
    da_only = gbi.to_dataarray(data, nodata=-9999.0)
    assert da_only.dims == ("band", "y", "x")
    assert np.array_equal(da_only.values, data)
    assert np.array_equal(da_only.coords["x"].values, gbi.x)
    assert np.array_equal(da_only.coords["y"].values, gbi.y)
    assert np.array_equal(da_only.coords["band"].values, gbi.band)
    assert da_only.rio.crs == raster.crs
    assert da_only.rio.nodata == -9999.0

    mask = np.zeros(gbi.shape, dtype=bool)
    mask[..., 0, 0] = True
    da2, ma2 = gbi.to_dataarray(data, mask=mask, nodata=-9999.0)
    assert ma2.dims == ("band", "y", "x")
    assert np.array_equal(ma2.values, mask)
    assert ma2.rio.crs == raster.crs


def test_geoblockinfo_to_dataarray_shape_mismatch():
    raster = testdata.raster.dem_small
    gbi = _make_block_info(raster)
    bad = np.zeros((1, 50, 50), dtype=np.float32)
    with pytest.raises(ValueError):
        gbi.to_dataarray(bad)
    good = np.zeros(gbi.shape, dtype=np.float32)
    bad_mask = np.zeros((1, 50, 50), dtype=bool)
    with pytest.raises(ValueError):
        gbi.to_dataarray(good, mask=bad_mask)


@pytest.mark.parametrize(
    "method,args,expected_shape,expected_affine,expected_row,expected_col",
    [
        # pad_y
        (
            "pad_y",
            (1, 0),
            (1, 101, 100),
            testdata.raster.dem_small.affine * Affine.translation(0, -1),
            slice(-1, 100),
            slice(0, 100),
        ),
        (
            "pad_y",
            (-1, 0),
            (1, 99, 100),
            testdata.raster.dem_small.affine * Affine.translation(0, 1),
            slice(1, 100),
            slice(0, 100),
        ),
        (
            "pad_y",
            (1, 1),
            (1, 102, 100),
            testdata.raster.dem_small.affine * Affine.translation(0, -1),
            slice(-1, 101),
            slice(0, 100),
        ),
        # pad_x
        (
            "pad_x",
            (1, 0),
            (1, 100, 101),
            testdata.raster.dem_small.affine * Affine.translation(-1, 0),
            slice(0, 100),
            slice(-1, 100),
        ),
        (
            "pad_x",
            (1, 2),
            (1, 100, 103),
            testdata.raster.dem_small.affine * Affine.translation(-1, 0),
            slice(0, 100),
            slice(-1, 102),
        ),
    ],
)
def test_geoblockinfo_pad_axes(
    method, args, expected_shape, expected_affine, expected_row, expected_col
):
    raster = testdata.raster.dem_small
    gbi = _make_block_info(raster)
    new = getattr(gbi, method)(*args)
    assert new.shape == expected_shape
    assert new.affine == expected_affine
    assert new.parent_affine == raster.affine
    assert new.parent_shape == raster.shape
    assert new.crs == raster.crs
    assert new.row_slice == expected_row
    assert new.col_slice == expected_col
    assert new.chunk_location == gbi.chunk_location


@pytest.mark.parametrize(
    "args,expected_shape,expected_affine,expected_row,expected_col",
    [
        (
            (0, 0),
            (1, 100, 100),
            testdata.raster.dem_small.affine,
            slice(0, 100),
            slice(0, 100),
        ),
        (
            (1, None),
            (1, 102, 102),
            testdata.raster.dem_small.affine * Affine.translation(-1, -1),
            slice(-1, 101),
            slice(-1, 101),
        ),
        (
            (1, 2),
            (1, 102, 104),
            testdata.raster.dem_small.affine * Affine.translation(-2, -1),
            slice(-1, 101),
            slice(-2, 102),
        ),
        (
            (-1, None),
            (1, 98, 98),
            testdata.raster.dem_small.affine * Affine.translation(1, 1),
            slice(1, 99),
            slice(1, 99),
        ),
    ],
)
def test_geoblockinfo_pad_symmetric(
    args, expected_shape, expected_affine, expected_row, expected_col
):
    raster = testdata.raster.dem_small
    gbi = _make_block_info(raster)
    new = gbi.pad(*args)
    assert new.shape == expected_shape
    assert new.affine == expected_affine
    assert new.parent_affine == raster.affine
    assert new.crs == raster.crs
    assert new.row_slice == expected_row
    assert new.col_slice == expected_col
    assert new.chunk_location == gbi.chunk_location


@pytest.mark.parametrize(
    "method,args,expected_affine,expected_row,expected_col",
    [
        (
            "shift_y",
            (1,),
            testdata.raster.dem_small.affine * Affine.translation(0, 1),
            slice(1, 101),
            slice(0, 100),
        ),
        (
            "shift_x",
            (2,),
            testdata.raster.dem_small.affine * Affine.translation(2, 0),
            slice(0, 100),
            slice(2, 102),
        ),
    ],
)
def test_geoblockinfo_shift_axes(
    method, args, expected_affine, expected_row, expected_col
):
    raster = testdata.raster.dem_small
    gbi = _make_block_info(raster)
    new = getattr(gbi, method)(*args)
    # shifts preserve overall shape
    assert new.shape == (1, 100, 100)
    assert new.affine == expected_affine
    assert new.row_slice == expected_row
    assert new.col_slice == expected_col


@pytest.mark.parametrize(
    "args,expected_affine,expected_row,expected_col",
    [
        (
            (0, 0),
            testdata.raster.dem_small.affine,
            slice(0, 100),
            slice(0, 100),
        ),
        (
            (1, None),
            testdata.raster.dem_small.affine * Affine.translation(1, 1),
            slice(1, 101),
            slice(1, 101),
        ),
        (
            (1, 2),
            testdata.raster.dem_small.affine * Affine.translation(2, 1),
            slice(1, 101),
            slice(2, 102),
        ),
    ],
)
def test_geoblockinfo_shift_symmetric(
    args, expected_affine, expected_row, expected_col
):
    raster = testdata.raster.dem_small
    gbi = _make_block_info(raster)
    new = gbi.shift(*args)
    assert new.shape == (1, 100, 100)
    assert new.affine == expected_affine
    assert new.row_slice == expected_row
    assert new.col_slice == expected_col


def test_geoblockinfo_pad_rejects_non_int():
    raster = testdata.raster.dem_small
    gbi = _make_block_info(raster)
    with pytest.raises(TypeError):
        gbi.pad_y(1.5, 0)


def test_geo_block_infos_as_dask():
    raster = testdata.raster.dem_small.chunk((1, 50, 50))
    arr = geo_block_infos_as_dask(raster)
    assert isinstance(arr, da.Array)
    assert arr.chunksize == (1, 1, 1)
    assert arr.numblocks == (1, 2, 2)
    computed = arr.compute()
    assert isinstance(computed, np.ndarray)
    assert computed.shape == (1, 2, 2)
    for gbi in computed.ravel():
        assert isinstance(gbi, GeoBlockInfo)
        assert gbi.shape == (1, 50, 50)


def test_raster_geo_block_infos_property():
    raster = testdata.raster.dem_small.chunk((1, 50, 50))
    arr = raster.geo_block_infos
    assert isinstance(arr, da.Array)
    assert arr.numblocks == raster.data.numblocks


# ---------------------------------------------------------------------------
# map_blocks
# ---------------------------------------------------------------------------


def test_map_blocks_identity():
    r = testdata.raster.dem_small
    out = map_blocks(lambda x: x.copy(), r)
    assert out.crs == r.crs
    assert out.affine == r.affine
    assert out.dtype == r.dtype
    assert out.null_value == r.null_value
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())
    np.testing.assert_array_equal(out.mask.compute(), r.mask.compute())


def test_map_blocks_arithmetic():
    r = testdata.raster.dem_small
    out = map_blocks(lambda x: x * 2, r)
    np.testing.assert_array_equal(out.data.compute(), r.data.compute() * 2)
    assert out.crs == r.crs
    assert out.affine == r.affine


def test_map_blocks_two_input_add():
    r1 = testdata.raster.dem_small
    r2 = testdata.raster.dem_small
    out = map_blocks(lambda a, b: a + b, r1, r2)
    np.testing.assert_array_equal(
        out.data.compute(), r1.data.compute() + r2.data.compute()
    )
    assert out.crs == r1.crs
    assert out.affine == r1.affine


def test_map_blocks_three_input_kwargs():
    r1 = testdata.raster.dem_small
    r2 = testdata.raster.dem_small
    r3 = testdata.raster.dem_small

    def weighted(a, b, c, w):
        return w * a + (1 - w) * (b + c)

    out = map_blocks(weighted, r1, r2, r3, w=0.5)
    a = r1.data.compute()
    expected = 0.5 * a + 0.5 * (a + a)
    np.testing.assert_array_equal(out.data.compute(), expected)


def test_map_blocks_dtype_change():
    r = testdata.raster.dem_small
    out = map_blocks(
        lambda x: x.astype(np.int32),
        r,
        dtype=np.int32,
        null_value="default",
    )
    assert out.dtype == np.int32
    np.testing.assert_array_equal(
        out.data.compute(), r.data.compute().astype(np.int32)
    )


def test_map_blocks_null_value_scalar_override():
    r = testdata.raster.dem_small
    out = map_blocks(lambda x: x.copy(), r, null_value=0.0)
    assert out.null_value == 0.0


def test_map_blocks_null_value_default_with_dtype_change():
    r = testdata.raster.dem_small
    out = map_blocks(
        lambda x: x.astype(np.int32),
        r,
        dtype=np.int32,
        null_value="default",
    )
    assert out.dtype == np.int32
    assert out.null_value == get_default_null_value(np.dtype(np.int32))


def test_map_blocks_null_value_invalid_string():
    r = testdata.raster.dem_small
    with pytest.raises(ValueError):
        map_blocks(lambda x: x.copy(), r, null_value="auto")


def test_map_blocks_pass_mask_single_input():
    r = testdata.raster.dem_clipped_small  # has masked cells
    sentinel = np.float32(-1.0)

    def fill_nulls(data, mask):
        assert data.shape == mask.shape
        return np.where(mask, sentinel, data)

    out = map_blocks(fill_nulls, r, pass_mask=True)
    data_in = r.data.compute()
    mask_in = r.mask.compute()
    expected = np.where(mask_in, sentinel, data_in)
    np.testing.assert_array_equal(out.data.compute(), expected)


def test_map_blocks_pass_mask_two_input_ordering():
    r1 = testdata.raster.dem_small
    r2 = testdata.raster.dem_small

    def check(d1, m1, d2, m2):
        # Order check: data/mask interleaved per input. d1/d2 are float
        # data, m1/m2 are bool masks. Adjacent positions belong to the
        # same raster.
        assert d1.dtype == r1.dtype and d2.dtype == r2.dtype
        assert m1.dtype == np.bool_ and m2.dtype == np.bool_
        # Same shapes across all four block args.
        assert d1.shape == d2.shape == m1.shape == m2.shape
        return d1 + d2 + m1.astype(d1.dtype) + m2.astype(d2.dtype)

    out = map_blocks(check, r1, r2, pass_mask=True)
    d1 = r1.data.compute()
    d2 = r2.data.compute()
    m1 = r1.mask.compute()
    m2 = r2.mask.compute()
    expected = d1 + d2 + m1.astype(d1.dtype) + m2.astype(d2.dtype)
    np.testing.assert_array_equal(out.data.compute(), expected)


def test_map_blocks_empty_rasters_raises():
    with pytest.raises(ValueError, match="at least one"):
        map_blocks(lambda x: x)


def test_map_blocks_shape_mismatch_raises():
    r1 = testdata.raster.dem_small  # 100x100
    r2 = testdata.raster.dem  # different shape (full DEM)
    assert r1.shape != r2.shape
    with pytest.raises(ValueError, match="raster 1 shape"):
        map_blocks(lambda a, b: a + b, r1, r2)


def test_map_blocks_preserves_first_input_grid():
    r1 = testdata.raster.dem_small
    r2 = testdata.raster.dem_small.set_null_value(99.0)
    assert r2.null_value != r1.null_value
    out = map_blocks(lambda a, b: a + b, r1, r2)
    assert out.crs == r1.crs
    assert out.affine == r1.affine
    assert out.null_value == r1.null_value


# ---------------------------------------------------------------------------
# map_overlap
# ---------------------------------------------------------------------------


def _np_3x3_mean_reflect(arr_2d):
    """Reference: per-pixel 3x3 mean with reflect padding.

    Note: dask's ``boundary='reflect'`` corresponds to NumPy's
    ``mode='symmetric'`` (edge cell repeated), not ``mode='reflect'``
    (edge not repeated).
    """
    pad = np.pad(arr_2d, 1, mode="symmetric")
    out = np.zeros_like(arr_2d, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += pad[dy : dy + arr_2d.shape[0], dx : dx + arr_2d.shape[1]]
    return out / 9.0


def _block_3x3_mean(block):
    # block is (1, ny, nx) including 1-cell overlap on each spatial side.
    # Must return same-shape block; dask trims the rim afterward.
    pad = block[0]
    out = np.zeros_like(pad, dtype=np.float32)
    inner_h = pad.shape[0] - 2
    inner_w = pad.shape[1] - 2
    if inner_h > 0 and inner_w > 0:
        s = np.zeros((inner_h, inner_w), dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                s += pad[dy : dy + inner_h, dx : dx + inner_w]
        out[1:-1, 1:-1] = s / 9.0
    return out[None]


def _block_3x3_sum(block):
    pad = block[0]
    out = np.zeros_like(pad, dtype=np.float32)
    inner_h = pad.shape[0] - 2
    inner_w = pad.shape[1] - 2
    if inner_h > 0 and inner_w > 0:
        s = np.zeros((inner_h, inner_w), dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                s += pad[dy : dy + inner_h, dx : dx + inner_w]
        out[1:-1, 1:-1] = s
    return out[None]


def _real_blocks(samples):
    """Filter out dask's zero-size meta-call blocks."""
    return [s for s in samples if all(d > 0 for d in s[0].shape)]


def test_map_overlap_identity_depth_zero():
    r = testdata.raster.dem_small
    out = map_overlap(lambda b: b, r, depth=0)
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())
    assert out.crs == r.crs
    assert out.affine == r.affine
    assert out.dtype == r.dtype


def test_map_overlap_3x3_mean_reflect():
    r = testdata.raster.dem_small.chunk((1, 50, 50))
    # The user function: assume dask provides a (1, ny+2, nx+2) block with
    # reflect padding on edges; trim=True -> dask trims our 1-cell rim back
    # off after we return a same-shape block.
    out = map_overlap(
        _block_3x3_mean, r, depth=1, boundary="reflect", dtype=np.float32
    )
    expected = _np_3x3_mean_reflect(r.data.compute()[0])[None]
    np.testing.assert_allclose(out.data.compute(), expected, rtol=1e-5)


def test_map_overlap_per_axis_depth_tuple():
    r = testdata.raster.dem_small
    out = map_overlap(lambda b: b, r, depth=(2, 1), boundary="reflect")
    # trim=True preserves shape regardless of depth
    assert out.shape == r.shape


def test_map_overlap_scalar_numeric_boundary():
    r = testdata.raster.dem_small.chunk((1, 50, 50))
    out = map_overlap(_block_3x3_sum, r, depth=1, boundary=0, dtype=np.float32)
    arr = r.data.compute()[0]
    pad = np.pad(arr, 1, mode="constant", constant_values=0)
    ref = np.zeros_like(arr, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            ref += pad[dy : dy + arr.shape[0], dx : dx + arr.shape[1]]
    np.testing.assert_allclose(out.data.compute()[0], ref, rtol=1e-5)


def test_map_overlap_boundary_null_with_set_null_value():
    r = testdata.raster.dem_small.set_null_value(-1.0).chunk((1, 50, 50))
    captured = {}

    def grab(d, m):
        # When this is called on a block, d/m include the overlap rim.
        # On an edge chunk the boundary-side rim is filled with the
        # null value in data and True in mask.
        captured.setdefault("samples", []).append((d.copy(), m.copy()))
        return d

    map_overlap(
        grab, r, depth=1, boundary="null", pass_mask=True
    ).data.compute()
    samples = _real_blocks(captured["samples"])
    # With 50x50 chunks of a 100x100 array and depth=1, the array has
    # 2x2 = 4 chunks, each grown to 52x52 (rim from neighbor or from
    # the boundary fill). The chunk at (0,0) has its top row and left
    # column filled with -1.0 and mask True.
    assert any(
        (d[0, 0, :] == -1.0).all() and m[0, 0, :].all() for d, m in samples
    )
    assert any(
        (d[0, :, 0] == -1.0).all() and m[0, :, 0].all() for d, m in samples
    )


def test_map_overlap_boundary_null_uses_default_when_unset():
    # When the input raster has no null_value, the "null" boundary
    # falls back to get_default_null_value(dtype) for the data fill.
    # Verify by inspecting the actual padded values via pass_mask.
    import raster_tools as rts_

    r = rts_.data_to_raster(
        testdata.raster.dem_small.data,
        x=testdata.raster.dem_small.x,
        y=testdata.raster.dem_small.y,
        crs=testdata.raster.dem_small.crs,
    ).chunk((1, 50, 50))
    assert r.null_value is None
    expected_fill = get_default_null_value(np.dtype(r.dtype))

    captured = {}

    def grab(d, m):
        captured.setdefault("samples", []).append((d.copy(), m.copy()))
        return d

    map_overlap(
        grab, r, depth=1, boundary="null", pass_mask=True
    ).data.compute()
    samples = _real_blocks(captured["samples"])
    # Expect at least one edge sample whose boundary rim is the
    # default null value in data and True in mask.
    assert any(
        (d[0, 0, :] == expected_fill).all() and m[0, 0, :].all()
        for d, m in samples
    )


def test_map_overlap_boundary_null_value_alias():
    r = testdata.raster.dem_small.set_null_value(-1.0)
    out1 = map_overlap(lambda b: b, r, depth=1, boundary="null")
    out2 = map_overlap(lambda b: b, r, depth=1, boundary="null_value")
    np.testing.assert_array_equal(out1.data.compute(), out2.data.compute())


@pytest.mark.parametrize("alias", ["nodata", "NODATA", "NoData"])
def test_map_overlap_boundary_nodata_aliases(alias):
    r = testdata.raster.dem_small.set_null_value(-1.0)
    out_null = map_overlap(lambda b: b, r, depth=1, boundary="null")
    out_alias = map_overlap(lambda b: b, r, depth=1, boundary=alias)
    np.testing.assert_array_equal(
        out_null.data.compute(), out_alias.data.compute()
    )


def test_map_overlap_scalar_matching_null_value_acts_like_null():
    r = testdata.raster.dem_small.set_null_value(-1.0).chunk((1, 50, 50))
    captured_null = {}
    captured_scalar = {}

    def grab_into(target):
        def _grab(d, m):
            target.setdefault("samples", []).append((d.copy(), m.copy()))
            return d

        return _grab

    map_overlap(
        grab_into(captured_null),
        r,
        depth=1,
        boundary="null",
        pass_mask=True,
    ).data.compute()
    map_overlap(
        grab_into(captured_scalar),
        r,
        depth=1,
        boundary=-1.0,
        pass_mask=True,
    ).data.compute()
    null_samples = _real_blocks(captured_null["samples"])
    scalar_samples = _real_blocks(captured_scalar["samples"])
    # Both runs should have at least one edge sample with True mask
    # rim along the boundary side.
    assert any(m[0, 0, :].all() for d, m in null_samples)
    assert any(m[0, 0, :].all() for d, m in scalar_samples)


def test_map_overlap_scalar_zero_when_null_is_not_zero():
    r = testdata.raster.dem_small.set_null_value(-1.0).chunk((1, 50, 50))
    captured = {}

    def grab(d, m):
        captured.setdefault("samples", []).append((d.copy(), m.copy()))
        return d

    map_overlap(grab, r, depth=1, boundary=0, pass_mask=True).data.compute()
    samples = _real_blocks(captured["samples"])
    # boundary=0 pads both array edges. The corner-edge chunk has a
    # boundary-rim filled with 0 in data and False in mask (because
    # 0 != null_value of -1).
    saw_false_rim = any(
        (d[0, 0, :] == 0).all() and not m[0, 0, :].any() for d, m in samples
    )
    assert saw_false_rim


def test_map_overlap_multi_input_different_null_values():
    r1 = testdata.raster.dem_small.set_null_value(-1.0).chunk((1, 50, 50))
    r2 = testdata.raster.dem_small.set_null_value(-2.0).chunk((1, 50, 50))
    captured = []

    def grab(d1, m1, d2, m2):
        if all(s > 0 for s in d1.shape):
            captured.append((d1.copy(), m1.copy(), d2.copy(), m2.copy()))
        return d1 + d2

    map_overlap(
        grab,
        r1,
        r2,
        depth=1,
        boundary="null",
        pass_mask=True,
    ).data.compute()
    # On each edge sample, each input's boundary-rim cells are filled
    # with its OWN null_value (not the other's).
    found = any(
        (d1[0, 0, :] == -1.0).all()
        and m1[0, 0, :].all()
        and (d2[0, 0, :] == -2.0).all()
        and m2[0, 0, :].all()
        for d1, m1, d2, m2 in captured
    )
    assert found


def test_map_overlap_two_input_add():
    r1 = testdata.raster.dem_small
    r2 = testdata.raster.dem_small
    out = map_overlap(lambda a, b: a + b, r1, r2, depth=1, boundary="reflect")
    np.testing.assert_allclose(
        out.data.compute(), r1.data.compute() + r2.data.compute()
    )


def test_map_overlap_pass_mask_single_input():
    r = testdata.raster.dem_small.chunk((1, 50, 50))

    def f(d, m):
        assert d.shape == m.shape
        return d + m.astype(d.dtype)

    out = map_overlap(f, r, depth=1, boundary="reflect", pass_mask=True)
    np.testing.assert_allclose(
        out.data.compute(),
        r.data.compute() + r.mask.compute().astype(r.dtype),
    )


def test_map_overlap_pass_mask_two_input_interleaved():
    r1 = testdata.raster.dem_small.chunk((1, 50, 50))
    r2 = testdata.raster.dem_small.chunk((1, 50, 50))

    def f(d1, m1, d2, m2):
        assert d1.dtype == r1.dtype and d2.dtype == r2.dtype
        assert m1.dtype == np.bool_ and m2.dtype == np.bool_
        assert d1.shape == d2.shape == m1.shape == m2.shape
        return d1 + d2

    out = map_overlap(f, r1, r2, depth=1, boundary="reflect", pass_mask=True)
    np.testing.assert_allclose(
        out.data.compute(), r1.data.compute() + r2.data.compute()
    )


def test_map_overlap_dtype_change():
    r = testdata.raster.dem_small.chunk((1, 50, 50))
    out = map_overlap(
        lambda b: b.astype(np.int32),
        r,
        depth=1,
        boundary="reflect",
        dtype=np.int32,
        null_value="default",
    )
    assert out.dtype == np.int32
    assert out.null_value == get_default_null_value(np.dtype(np.int32))


def test_map_overlap_null_value_scalar_override():
    r = testdata.raster.dem_small
    out = map_overlap(lambda b: b, r, depth=0, null_value=42.0)
    assert out.null_value == 42.0


def test_map_overlap_empty_rasters_raises():
    with pytest.raises(ValueError, match="at least one"):
        map_overlap(lambda b: b, depth=0)


def test_map_overlap_shape_mismatch_raises():
    r1 = testdata.raster.dem_small  # 100x100
    r2 = testdata.raster.dem  # full DEM
    with pytest.raises(ValueError, match="raster 1 shape"):
        map_overlap(lambda a, b: a + b, r1, r2, depth=0)


def test_map_overlap_band_axis_depth_raises():
    r = testdata.raster.dem_small
    with pytest.raises(ValueError, match="band-axis"):
        map_overlap(lambda b: b, r, depth={0: 1, 1: 0, 2: 0})


def test_map_overlap_negative_depth_raises():
    r = testdata.raster.dem_small
    with pytest.raises(ValueError, match="non-negative"):
        map_overlap(lambda b: b, r, depth=-1)
    with pytest.raises(ValueError, match="non-negative"):
        map_overlap(lambda b: b, r, depth=(2, -1))


def test_map_overlap_unrecognized_boundary_raises():
    r = testdata.raster.dem_small
    with pytest.raises(ValueError, match="unrecognized boundary"):
        map_overlap(lambda b: b, r, depth=1, boundary="bogus")


def test_map_overlap_asymmetric_depth_with_reflect_propagates_dask_error():
    # TODO: pre-validate this combo with a friendlier error.
    r = testdata.raster.dem_small.chunk((1, 50, 50))
    with pytest.raises(NotImplementedError, match="[Aa]symmetric"):
        map_overlap(
            lambda b: b,
            r,
            depth={1: (2, 1), 2: 0},
            boundary="reflect",
        ).data.compute()


# ---------------------------------------------------------------------------
# geo_map_blocks
# ---------------------------------------------------------------------------


def test_geo_map_blocks_identity():
    r = testdata.raster.dem_small.chunk((1, 50, 50))

    def f(xda, **kw):
        return xda

    out = geo_map_blocks(f, r)
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())
    assert out.crs == r.crs
    assert out.affine == r.affine
    assert out.dtype == r.dtype
    assert out.null_value == r.null_value


def test_geo_map_blocks_func_sees_coords_crs_nodata():
    r = testdata.raster.dem_small.chunk((1, 50, 50))

    def f(xda, geo_block_info=None, **kw):
        # Skip dask's dtype-inference meta call (geo_block_info is None).
        if geo_block_info is None:
            return xda
        assert isinstance(xda, xr.DataArray)
        assert xda.dims == ("band", "y", "x")
        assert xda.rio.crs == r.crs
        assert xda.rio.nodata == r.null_value
        assert isinstance(geo_block_info, GeoBlockInfo)
        np.testing.assert_array_equal(xda.coords["x"].values, geo_block_info.x)
        np.testing.assert_array_equal(xda.coords["y"].values, geo_block_info.y)
        return xda

    geo_map_blocks(f, r).data.compute()


def test_geo_map_blocks_nodata_visible_in_meta_call():
    # The DataArray passed to the user's func during dask's dtype-
    # inference meta call should carry the same nodata as the real
    # per-block call.
    r = testdata.raster.dem_small.set_null_value(-1.0)
    seen_meta = []
    seen_real = []

    def f(xda, geo_block_info=None, **kw):
        (seen_meta if geo_block_info is None else seen_real).append(
            xda.rio.nodata
        )
        return xda

    geo_map_blocks(f, r).data.compute()
    assert seen_meta and all(nv == -1.0 for nv in seen_meta)
    assert seen_real and all(nv == -1.0 for nv in seen_real)


def test_geo_map_blocks_returns_dataarray():
    r = testdata.raster.dem_small.chunk((1, 50, 50))

    def f(xda, **kw):
        return xda * 2

    out = geo_map_blocks(f, r)
    np.testing.assert_allclose(out.data.compute(), r.data.compute() * 2)


def test_geo_map_blocks_returns_ndarray():
    r = testdata.raster.dem_small.chunk((1, 50, 50))

    def f(xda, **kw):
        return xda.values * 3

    out = geo_map_blocks(f, r)
    np.testing.assert_allclose(out.data.compute(), r.data.compute() * 3)


def test_geo_map_blocks_geo_block_info_kwarg_present():
    r = testdata.raster.dem_small.chunk((1, 50, 50))
    seen = []

    def f(xda, geo_block_info=None, **kw):
        # geo_block_info is None during dask's meta inference; only
        # collect chunk locations from the real per-chunk calls.
        if geo_block_info is not None:
            seen.append(geo_block_info.chunk_location)
        return xda

    geo_map_blocks(f, r).data.compute()
    # 2x2 spatial chunks -> 4 unique chunk locations.
    assert sorted(set(seen)) == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
    ]


def test_geo_map_blocks_two_input_add():
    r = testdata.raster.dem_small.chunk((1, 50, 50))

    def f(a, b, **kw):
        return a + b

    out = geo_map_blocks(f, r, r)
    np.testing.assert_allclose(out.data.compute(), r.data.compute() * 2)


def test_geo_map_blocks_kwargs_forwarded():
    r = testdata.raster.dem_small.chunk((1, 50, 50))

    def f(xda, *, factor, **kw):
        return xda * factor

    out = geo_map_blocks(f, r, factor=4)
    np.testing.assert_allclose(out.data.compute(), r.data.compute() * 4)


def test_geo_map_blocks_pass_mask_single_input():
    r = testdata.raster.dem_small.chunk((1, 50, 50))

    def f(xda, xma, **kw):
        assert isinstance(xda, xr.DataArray)
        assert isinstance(xma, xr.DataArray)
        assert xma.dtype == bool
        # Coords aligned between data and mask DataArrays.
        np.testing.assert_array_equal(
            xda.coords["x"].values, xma.coords["x"].values
        )
        return xda + xma.astype(xda.dtype)

    out = geo_map_blocks(f, r, pass_mask=True)
    np.testing.assert_allclose(
        out.data.compute(),
        r.data.compute() + r.mask.compute().astype(r.dtype),
    )


def test_geo_map_blocks_pass_mask_two_input_interleaved():
    r1 = testdata.raster.dem_small.chunk((1, 50, 50))
    r2 = testdata.raster.dem_small.chunk((1, 50, 50))

    def f(xda1, xma1, xda2, xma2, **kw):
        assert xda1.dtype == r1.dtype and xda2.dtype == r2.dtype
        assert xma1.dtype == bool and xma2.dtype == bool
        return xda1 + xda2

    out = geo_map_blocks(f, r1, r2, pass_mask=True)
    np.testing.assert_allclose(
        out.data.compute(), r1.data.compute() + r2.data.compute()
    )


def test_geo_map_blocks_dtype_change():
    r = testdata.raster.dem_small.chunk((1, 50, 50))

    def f(xda, **kw):
        return xda.astype(np.int32)

    out = geo_map_blocks(f, r, dtype=np.int32, null_value="default")
    assert out.dtype == np.int32
    assert out.null_value == get_default_null_value(np.dtype(np.int32))


def test_geo_map_blocks_null_value_scalar_override():
    r = testdata.raster.dem_small

    def f(xda, **kw):
        return xda

    out = geo_map_blocks(f, r, null_value=42.0)
    assert out.null_value == 42.0


def test_geo_map_blocks_empty_rasters_raises():
    with pytest.raises(ValueError, match="at least one"):
        geo_map_blocks(lambda xda, **kw: xda)


def test_geo_map_blocks_shape_mismatch_raises():
    r1 = testdata.raster.dem_small
    r2 = testdata.raster.dem
    with pytest.raises(ValueError, match="raster 1 shape"):
        geo_map_blocks(lambda a, b, **kw: a + b, r1, r2)


def test_geo_map_blocks_preserves_first_input_grid():
    r1 = testdata.raster.dem_small
    r2 = testdata.raster.dem_small.set_null_value(99.0)
    assert r2.null_value != r1.null_value
    out = geo_map_blocks(lambda a, b, **kw: a + b, r1, r2)
    assert out.crs == r1.crs
    assert out.affine == r1.affine
    assert out.null_value == r1.null_value


def test_geo_map_blocks_different_crs_raises():
    import raster_tools as rts_

    r1 = testdata.raster.dem_small
    # Same shape and coords as r1 but a different CRS.
    r2 = rts_.data_to_raster(r1.data, x=r1.x, y=r1.y, crs="EPSG:4326")
    assert r1.shape == r2.shape
    assert r1.crs != r2.crs
    with pytest.raises(ValueError, match="same grid"):
        geo_map_blocks(lambda a, b, **kw: a + b, r1, r2)


def test_geo_map_blocks_different_affine_raises():
    import raster_tools as rts_

    r1 = testdata.raster.dem_small
    # Same shape and CRS but coords shifted by a full pixel: a
    # different affine.
    pixel = abs(r1.affine.a)
    shifted_x = r1.x + pixel
    r2 = rts_.data_to_raster(r1.data, x=shifted_x, y=r1.y, crs=r1.crs)
    assert r1.shape == r2.shape
    assert r1.crs == r2.crs
    assert r1.affine != r2.affine
    with pytest.raises(ValueError, match="same grid"):
        geo_map_blocks(lambda a, b, **kw: a + b, r1, r2)


def test_geo_map_blocks_aligned_succeeds_regression():
    # Trivial alignment: two copies of the same raster succeed.
    r = testdata.raster.dem_small
    out = geo_map_blocks(lambda a, b, **kw: a + b, r, r)
    np.testing.assert_allclose(out.data.compute(), r.data.compute() * 2)


def test_geo_map_blocks_sub_pixel_fp_noise_tolerated():
    import raster_tools as rts_
    from raster_tools._grids import GRID_PIXEL_TOLERANCE

    r1 = testdata.raster.dem_small
    # Perturb the x coords by well under one pixel of FP noise. The
    # grids_close tolerance should let this through.
    pixel = abs(r1.affine.a)
    noise = pixel * GRID_PIXEL_TOLERANCE * 0.1
    perturbed_x = r1.x + noise
    r2 = rts_.data_to_raster(r1.data, x=perturbed_x, y=r1.y, crs=r1.crs)
    # Sanity: affines aren't bit-equal but are within tolerance.
    assert r1.affine != r2.affine
    out = geo_map_blocks(lambda a, b, **kw: a + b, r1, r2)
    np.testing.assert_allclose(out.data.compute(), r1.data.compute() * 2)


# ---------------------------------------------------------------------------
# dtype inference (mirror dask)
# ---------------------------------------------------------------------------


def _int16_copy(r):
    """Return a copy of `r` cast to int16, on the same grid."""
    import raster_tools as rts_

    return rts_.data_to_raster(
        r.data.astype(np.int16), x=r.x, y=r.y, crs=r.crs
    )


def test_map_blocks_dtype_inferred_for_mixed_inputs():
    r1 = testdata.raster.dem_small  # float32
    r2 = _int16_copy(r1)
    out = map_blocks(lambda a, b: a + b, r1, r2, null_value="default")
    # NumPy promotion: float32 + int16 -> float32. dask infers it.
    assert out.dtype == np.float32


def test_map_blocks_dtype_inferred_from_in_func_cast():
    r = testdata.raster.dem_small  # float32
    out = map_blocks(lambda x: x.astype(np.int32), r, null_value="default")
    # If we used np.result_type on inputs we'd get float32.
    # Inferring via the actual function call yields int32.
    assert out.dtype == np.int32


def test_map_blocks_explicit_dtype_overrides_inference():
    r = _int16_copy(testdata.raster.dem_small)  # int16
    out = map_blocks(
        lambda x: x.astype(np.float64),
        r,
        dtype=np.float64,
        null_value="default",
    )
    assert out.dtype == np.float64


def test_map_overlap_dtype_inferred_for_mixed_inputs():
    r1 = testdata.raster.dem_small  # float32
    r2 = _int16_copy(r1)
    out = map_overlap(
        lambda a, b: a + b,
        r1,
        r2,
        depth=1,
        boundary="reflect",
        null_value="default",
    )
    assert out.dtype == np.float32


def test_geo_map_blocks_dtype_inferred_for_mixed_inputs():
    r1 = testdata.raster.dem_small  # float32
    r2 = _int16_copy(r1)
    out = geo_map_blocks(
        lambda a, b, **kw: a + b, r1, r2, null_value="default"
    )
    assert out.dtype == np.float32


def test_geo_map_blocks_dtype_inferred_from_in_func_cast():
    r = testdata.raster.dem_small  # float32
    out = geo_map_blocks(
        lambda xda, **kw: xda.astype(np.int32),
        r,
        null_value="default",
    )
    assert out.dtype == np.int32


def test_map_blocks_null_value_default_uses_inferred_dtype():
    # When dtype is inferred to float32 (the promoted result), the
    # "default" null should be the float32 default, NOT int16's.
    r1 = testdata.raster.dem_small  # float32
    r2 = _int16_copy(r1)
    out = map_blocks(lambda a, b: a + b, r1, r2, null_value="default")
    assert out.dtype == np.float32
    assert out.null_value == get_default_null_value(np.dtype(np.float32))


# ---------------------------------------------------------------------------
# geo_map_overlap
# ---------------------------------------------------------------------------


def _identity_or_meta(xda, geo_block_info=None, **kw):
    """Identity that tolerates dask's 0-shape meta call."""
    return xda


def test_geo_map_overlap_identity_depth_zero():
    r = testdata.raster.dem_small.chunk((1, 50, 50))
    out = geo_map_overlap(_identity_or_meta, r, depth=0)
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())
    assert out.crs == r.crs
    assert out.affine == r.affine
    assert out.dtype == r.dtype


def test_geo_map_overlap_3x3_mean_reflect():
    r = testdata.raster.dem_small.chunk((1, 50, 50))

    def block_3x3_mean(xda, **kw):
        if not all(s > 0 for s in xda.shape):
            return xda  # meta call
        pad = xda.values[0]
        out = np.zeros_like(pad, dtype=np.float32)
        ih = pad.shape[0] - 2
        iw = pad.shape[1] - 2
        if ih > 0 and iw > 0:
            s = np.zeros((ih, iw), dtype=np.float32)
            for dy in range(3):
                for dx in range(3):
                    s += pad[dy : dy + ih, dx : dx + iw]
            out[1:-1, 1:-1] = s / 9.0
        return out[None]

    out = geo_map_overlap(
        block_3x3_mean, r, depth=1, boundary="reflect", dtype=np.float32
    )
    arr = r.data.compute()[0]
    pad = np.pad(arr, 1, mode="symmetric")
    expected = np.zeros_like(arr, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            expected += pad[dy : dy + arr.shape[0], dx : dx + arr.shape[1]]
    expected /= 9.0
    np.testing.assert_allclose(out.data.compute()[0], expected, rtol=1e-5)


def test_geo_map_overlap_coords_reflect_overlap():
    """The DataArray's coords cover the overlapped extent, not just
    the chunk's original extent."""
    r = testdata.raster.dem_small.chunk((1, 50, 50))
    seen = []

    def f(xda, geo_block_info=None, **kw):
        if geo_block_info is not None and all(s > 0 for s in xda.shape):
            seen.append(
                (
                    xda.shape,
                    geo_block_info.shape,
                    xda.coords["x"].values[0],
                    xda.coords["x"].values[-1],
                )
            )
        return xda

    geo_map_overlap(f, r, depth=1, boundary="reflect").data.compute()
    # All chunks under boundary='reflect' get padded by 1 on each side
    # in both spatial axes -> 52x52.
    for xda_shape, gbi_shape, _x_first, _x_last in seen:
        assert xda_shape == gbi_shape == (1, 52, 52)


def test_geo_map_overlap_geo_block_info_shape_matches_block():
    r = testdata.raster.dem_small.chunk((1, 50, 50))

    def f(xda, geo_block_info=None, **kw):
        if geo_block_info is not None and all(s > 0 for s in xda.shape):
            assert geo_block_info.shape == xda.shape
            # row_slice / col_slice may have negative starts at edges.
            assert isinstance(geo_block_info.row_slice, slice)
            assert isinstance(geo_block_info.col_slice, slice)
        return xda

    geo_map_overlap(f, r, depth=1, boundary="reflect").data.compute()


def test_geo_map_overlap_nodata_visible_in_meta_call():
    r = testdata.raster.dem_small.set_null_value(-1.0)
    seen_meta = []
    seen_real = []

    def f(xda, geo_block_info=None, **kw):
        (seen_meta if geo_block_info is None else seen_real).append(
            xda.rio.nodata
        )
        return xda

    geo_map_overlap(f, r, depth=1, boundary="reflect").data.compute()
    assert seen_meta and all(nv == -1.0 for nv in seen_meta)
    assert seen_real and all(nv == -1.0 for nv in seen_real)


def test_geo_map_overlap_boundary_null_with_pass_mask():
    r = testdata.raster.dem_small.set_null_value(-1.0).chunk((1, 50, 50))
    captured = []

    def grab(xda, xma, geo_block_info=None, **kw):
        if all(s > 0 for s in xda.shape):
            captured.append((xda.values.copy(), xma.values.copy()))
        return xda

    geo_map_overlap(
        grab, r, depth=1, boundary="null", pass_mask=True
    ).data.compute()
    # On the (0, 0) corner chunk: top row and left column of the
    # padded block are null fill (-1.0) and mask True.
    found_top = any(
        (d[0, 0, :] == -1.0).all() and m[0, 0, :].all() for d, m in captured
    )
    found_left = any(
        (d[0, :, 0] == -1.0).all() and m[0, :, 0].all() for d, m in captured
    )
    assert found_top and found_left


def test_geo_map_overlap_two_input_add():
    r1 = testdata.raster.dem_small.chunk((1, 50, 50))
    r2 = testdata.raster.dem_small.chunk((1, 50, 50))
    out = geo_map_overlap(
        lambda a, b, **kw: a + b,
        r1,
        r2,
        depth=1,
        boundary="reflect",
    )
    np.testing.assert_allclose(
        out.data.compute(), r1.data.compute() + r2.data.compute()
    )


def test_geo_map_overlap_pass_mask_two_input_interleaved():
    r1 = testdata.raster.dem_small.chunk((1, 50, 50))
    r2 = testdata.raster.dem_small.chunk((1, 50, 50))

    def f(xda1, xma1, xda2, xma2, **kw):
        if not all(s > 0 for s in xda1.shape):
            return xda1
        assert xda1.dtype == r1.dtype and xda2.dtype == r2.dtype
        assert xma1.dtype == bool and xma2.dtype == bool
        assert xda1.shape == xma1.shape == xda2.shape == xma2.shape
        return xda1 + xda2

    out = geo_map_overlap(
        f, r1, r2, depth=1, boundary="reflect", pass_mask=True
    )
    np.testing.assert_allclose(
        out.data.compute(), r1.data.compute() + r2.data.compute()
    )


def test_geo_map_overlap_dtype_inferred_for_mixed_inputs():
    r1 = testdata.raster.dem_small  # float32
    r2 = _int16_copy(r1)
    out = geo_map_overlap(
        lambda a, b, **kw: a + b,
        r1,
        r2,
        depth=1,
        boundary="reflect",
        null_value="default",
    )
    assert out.dtype == np.float32


def test_geo_map_overlap_dtype_inferred_from_in_func_cast():
    r = testdata.raster.dem_small  # float32
    out = geo_map_overlap(
        lambda xda, **kw: xda.astype(np.int32),
        r,
        depth=1,
        boundary="reflect",
        null_value="default",
    )
    assert out.dtype == np.int32


def test_geo_map_overlap_null_value_default_uses_inferred_dtype():
    r1 = testdata.raster.dem_small  # float32
    r2 = _int16_copy(r1)
    out = geo_map_overlap(
        lambda a, b, **kw: a + b,
        r1,
        r2,
        depth=1,
        boundary="reflect",
        null_value="default",
    )
    assert out.dtype == np.float32
    assert out.null_value == get_default_null_value(np.dtype(np.float32))


def test_geo_map_overlap_different_crs_raises():
    import raster_tools as rts_

    r1 = testdata.raster.dem_small
    r2 = rts_.data_to_raster(r1.data, x=r1.x, y=r1.y, crs="EPSG:4326")
    with pytest.raises(ValueError, match="same grid"):
        geo_map_overlap(
            lambda a, b, **kw: a + b, r1, r2, depth=1, boundary="reflect"
        )


def test_geo_map_overlap_different_affine_raises():
    import raster_tools as rts_

    r1 = testdata.raster.dem_small
    pixel = abs(r1.affine.a)
    r2 = rts_.data_to_raster(r1.data, x=r1.x + pixel, y=r1.y, crs=r1.crs)
    with pytest.raises(ValueError, match="same grid"):
        geo_map_overlap(
            lambda a, b, **kw: a + b, r1, r2, depth=1, boundary="reflect"
        )


def test_geo_map_overlap_empty_rasters_raises():
    with pytest.raises(ValueError, match="at least one"):
        geo_map_overlap(_identity_or_meta, depth=1, boundary="reflect")


def test_geo_map_overlap_rechunks_when_chunks_smaller_than_depth():
    # Input chunks (10) are smaller than depth (20). The wrapper
    # should pre-rechunk so the in-wrapper GeoBlockInfo lookup keys
    # match the per-call chunk-location after dask's overlap call.
    r = testdata.raster.dem_small.chunk((1, 10, 10))
    assert all(c < 20 for c in r.data.chunks[1])

    out = geo_map_overlap(_identity_or_meta, r, depth=20, boundary="reflect")
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())
    # Verify the rechunk actually happened: spatial chunks are now
    # >= depth.
    assert all(c >= 20 for c in out.data.chunks[1])


def test_geo_map_overlap_rechunk_with_pass_mask():
    # The mask should be rechunked alongside the data so pass_mask
    # callbacks see matching shapes.
    r = testdata.raster.dem_small.chunk((1, 10, 10))

    def f(xda, xma, **kw):
        if not all(s > 0 for s in xda.shape):
            return xda
        assert xda.shape == xma.shape
        return xda + xma.astype(xda.dtype)

    out = geo_map_overlap(f, r, depth=20, boundary="reflect", pass_mask=True)
    np.testing.assert_allclose(out.data.compute(), r.data.compute())
