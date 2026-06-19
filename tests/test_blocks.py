# isort: off
# TODO(pygeos): remove this once shapely is the default backend for geopandas.
# Force raster_tools._compat to be loaded before geopandas when running tests
import raster_tools as rts  # noqa: F401

# isort: on

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from dask.utils import parse_bytes

from raster_tools.blocks import (
    GeoBlockInfo,
    _align_chunks_single_band,
    geo_block_infos_as_dask,
    geo_map_blocks,
    geo_map_overlap,
    map_blocks,
    map_overlap,
)
from raster_tools.masking import get_default_null_value
from tests.utils import make_raster

# Module-level reference affine for the parametrize decorators
# (Affine is immutable; it's safe to cache across tests. The
# project's "fresh fixture per test" rule applies to Raster objects,
# which we still build inside each test body.)
_BLOCKS_AFFINE = make_raster(shape=(1, 100, 100), dtype=np.float32).affine


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
    raster = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
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
    raster = make_raster(shape=(1, 100, 100), dtype=np.float32)
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
    raster = make_raster(shape=(1, 100, 100), dtype=np.float32)
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
            _BLOCKS_AFFINE * Affine.translation(0, -1),
            slice(-1, 100),
            slice(0, 100),
        ),
        (
            "pad_y",
            (-1, 0),
            (1, 99, 100),
            _BLOCKS_AFFINE * Affine.translation(0, 1),
            slice(1, 100),
            slice(0, 100),
        ),
        (
            "pad_y",
            (1, 1),
            (1, 102, 100),
            _BLOCKS_AFFINE * Affine.translation(0, -1),
            slice(-1, 101),
            slice(0, 100),
        ),
        # pad_x
        (
            "pad_x",
            (1, 0),
            (1, 100, 101),
            _BLOCKS_AFFINE * Affine.translation(-1, 0),
            slice(0, 100),
            slice(-1, 100),
        ),
        (
            "pad_x",
            (1, 2),
            (1, 100, 103),
            _BLOCKS_AFFINE * Affine.translation(-1, 0),
            slice(0, 100),
            slice(-1, 102),
        ),
    ],
)
def test_geoblockinfo_pad_axes(
    method, args, expected_shape, expected_affine, expected_row, expected_col
):
    raster = make_raster(shape=(1, 100, 100), dtype=np.float32)
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
            _BLOCKS_AFFINE,
            slice(0, 100),
            slice(0, 100),
        ),
        (
            (1, None),
            (1, 102, 102),
            _BLOCKS_AFFINE * Affine.translation(-1, -1),
            slice(-1, 101),
            slice(-1, 101),
        ),
        (
            (1, 2),
            (1, 102, 104),
            _BLOCKS_AFFINE * Affine.translation(-2, -1),
            slice(-1, 101),
            slice(-2, 102),
        ),
        (
            (-1, None),
            (1, 98, 98),
            _BLOCKS_AFFINE * Affine.translation(1, 1),
            slice(1, 99),
            slice(1, 99),
        ),
    ],
)
def test_geoblockinfo_pad_symmetric(
    args, expected_shape, expected_affine, expected_row, expected_col
):
    raster = make_raster(shape=(1, 100, 100), dtype=np.float32)
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
            _BLOCKS_AFFINE * Affine.translation(0, 1),
            slice(1, 101),
            slice(0, 100),
        ),
        (
            "shift_x",
            (2,),
            _BLOCKS_AFFINE * Affine.translation(2, 0),
            slice(0, 100),
            slice(2, 102),
        ),
    ],
)
def test_geoblockinfo_shift_axes(
    method, args, expected_affine, expected_row, expected_col
):
    raster = make_raster(shape=(1, 100, 100), dtype=np.float32)
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
            _BLOCKS_AFFINE,
            slice(0, 100),
            slice(0, 100),
        ),
        (
            (1, None),
            _BLOCKS_AFFINE * Affine.translation(1, 1),
            slice(1, 101),
            slice(1, 101),
        ),
        (
            (1, 2),
            _BLOCKS_AFFINE * Affine.translation(2, 1),
            slice(1, 101),
            slice(2, 102),
        ),
    ],
)
def test_geoblockinfo_shift_symmetric(
    args, expected_affine, expected_row, expected_col
):
    raster = make_raster(shape=(1, 100, 100), dtype=np.float32)
    gbi = _make_block_info(raster)
    new = gbi.shift(*args)
    assert new.shape == (1, 100, 100)
    assert new.affine == expected_affine
    assert new.row_slice == expected_row
    assert new.col_slice == expected_col


def test_geoblockinfo_pad_rejects_non_int():
    raster = make_raster(shape=(1, 100, 100), dtype=np.float32)
    gbi = _make_block_info(raster)
    with pytest.raises(TypeError):
        gbi.pad_y(1.5, 0)


def test_geo_block_infos_as_dask():
    raster = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
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
    raster = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    arr = raster.geo_block_infos
    assert isinstance(arr, da.Array)
    assert arr.numblocks == raster.data.numblocks


# ---------------------------------------------------------------------------
# map_blocks
# ---------------------------------------------------------------------------


def test_map_blocks_identity():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    # Single input with unchanged dtype: null_value inherits from r.
    out = map_blocks(lambda x: x.copy(), r)
    assert out.crs == r.crs
    assert out.affine == r.affine
    assert out.dtype == r.dtype
    assert out.null_value == r.null_value
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())
    np.testing.assert_array_equal(out.mask.compute(), r.mask.compute())


def test_map_blocks_arithmetic():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    out = map_blocks(lambda x: x * 2, r)
    np.testing.assert_array_equal(out.data.compute(), r.data.compute() * 2)
    assert out.crs == r.crs
    assert out.affine == r.affine


def test_map_blocks_two_input_add():
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)
    r2 = make_raster(shape=(1, 100, 100), dtype=np.float32)
    out = map_blocks(lambda a, b: a + b, r1, r2)
    np.testing.assert_array_equal(
        out.data.compute(), r1.data.compute() + r2.data.compute()
    )
    assert out.crs == r1.crs
    assert out.affine == r1.affine


def test_map_blocks_three_input_kwargs():
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)
    r2 = make_raster(shape=(1, 100, 100), dtype=np.float32)
    r3 = make_raster(shape=(1, 100, 100), dtype=np.float32)

    def weighted(a, b, c, w):
        return w * a + (1 - w) * (b + c)

    out = map_blocks(weighted, r1, r2, r3, w=0.5)
    a = r1.data.compute()
    expected = 0.5 * a + 0.5 * (a + a)
    np.testing.assert_array_equal(out.data.compute(), expected)


def test_map_blocks_dtype_change():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    # null_value=None now picks the dtype-default automatically.
    out = map_blocks(lambda x: x.astype(np.int32), r, dtype=np.int32)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(
        out.data.compute(), r.data.compute().astype(np.int32)
    )


def test_map_blocks_null_value_scalar_override():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    out = map_blocks(lambda x: x.copy(), r, null_value=0.0)
    assert out.null_value == 0.0


def test_map_blocks_null_value_none_is_dtype_default_when_dtype_changes():
    # Single input but the dtype changes -> the input's null value
    # may not be representable, so fall back to the dtype default.
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    out = map_blocks(lambda x: x.astype(np.int32), r, dtype=np.int32)
    assert out.dtype == np.int32
    assert out.null_value == get_default_null_value(np.dtype(np.int32))


def test_map_blocks_null_value_inherits_for_single_input_unchanged_dtype():
    # Single input with unchanged dtype -> output inherits the input's
    # null value (preserves the sentinel for identity-like ops).
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    out = map_blocks(lambda x: x * 2, r)
    assert out.dtype == r.dtype
    assert out.null_value == r.null_value


def test_map_blocks_null_value_inherits_when_explicit_dtype_matches_input():
    # The inherit-on-single-input rule fires when the output dtype
    # ends up matching the input's, whether dask inferred it or the
    # user specified it explicitly via dtype=.
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    out = map_blocks(lambda x: x.astype(r.dtype), r, dtype=r.dtype)
    assert out.dtype == r.dtype
    assert out.null_value == r.null_value


def test_map_blocks_explicit_null_value_overrides_inherit_rule():
    # Even when the inherit rule would fire (single input, unchanged
    # dtype), an explicit null_value scalar wins.
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    out = map_blocks(lambda x: x * 2, r, null_value=42.0)
    assert out.null_value == 42.0


def test_map_blocks_null_value_none_uses_default_for_multi_input():
    # Multi-input with no explicit null_value -> use dtype default,
    # not "first input's null" (the inherit-on-single-input rule does
    # not extend to multiple inputs).
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    r2 = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-2.0)
    out = map_blocks(lambda a, b: a + b, r1, r2)
    assert out.null_value == get_default_null_value(np.dtype(out.dtype))
    assert out.null_value not in (-1.0, -2.0)


def test_map_blocks_null_value_invalid_string():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError):
        map_blocks(lambda x: x.copy(), r, null_value="auto")


def test_map_blocks_input_masks_single_input():
    # User opts in to mask injection by naming `input_masks` in func.
    r = make_raster(
        shape=(1, 100, 100),
        dtype=np.float32,
        null=True,
        null_pattern=np.s_[:, :10, :10],
    )  # has masked cells
    sentinel = np.float32(-1.0)

    def fill_nulls(data, *, input_masks):
        (mask,) = input_masks
        assert data.shape == mask.shape
        return np.where(mask, sentinel, data)

    out = map_blocks(fill_nulls, r)
    data_in = r.data.compute()
    mask_in = r.mask.compute()
    expected = np.where(mask_in, sentinel, data_in)
    np.testing.assert_array_equal(out.data.compute(), expected)


def test_map_blocks_input_masks_two_input_ordering():
    # input_masks is a tuple parallel to *input_data: input_masks[i] is
    # the mask block for input i.
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)
    r2 = make_raster(shape=(1, 100, 100), dtype=np.float32)

    def check(d1, d2, *, input_masks):
        m1, m2 = input_masks
        assert d1.dtype == r1.dtype and d2.dtype == r2.dtype
        assert m1.dtype == np.bool_ and m2.dtype == np.bool_
        assert d1.shape == d2.shape == m1.shape == m2.shape
        return d1 + d2 + m1.astype(d1.dtype) + m2.astype(d2.dtype)

    out = map_blocks(check, r1, r2)
    d1 = r1.data.compute()
    d2 = r2.data.compute()
    m1 = r1.mask.compute()
    m2 = r2.mask.compute()
    expected = d1 + d2 + m1.astype(d1.dtype) + m2.astype(d2.dtype)
    np.testing.assert_array_equal(out.data.compute(), expected)


def test_map_blocks_plain_func_gets_no_injection():
    # No special params named -> none of the reserved injection
    # kwargs should appear in **kwargs.
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)

    seen = []

    def f(d, **kw):
        seen.append(set(kw))
        return d

    map_blocks(f, r).data.compute()
    for kw_keys in seen:
        assert "input_masks" not in kw_keys
        assert "input_null_values" not in kw_keys
        assert "block_info" not in kw_keys
        assert "block_id" not in kw_keys


def test_map_blocks_input_null_values_injected():
    # input_null_values is a tuple parallel to inputs.
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    r2 = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-2.0)

    seen = []

    def f(d1, d2, *, input_null_values):
        seen.append(input_null_values)
        return d1 + d2

    map_blocks(f, r1, r2).data.compute()
    assert seen
    for nvs in seen:
        assert nvs == (np.float32(-1.0), np.float32(-2.0))


def test_map_blocks_block_info_injected():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    seen = []

    def f(d, *, block_info):
        if block_info is not None:
            seen.append(block_info[None]["chunk-location"])
        return d

    map_blocks(f, r).data.compute()
    # 2x2 spatial chunks -> 4 unique chunk locations.
    assert sorted(set(seen)) == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
    ]


def test_map_blocks_block_id_injected():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    seen = []

    def f(d, *, block_id):
        # None during the meta inference call; a chunk-location tuple
        # on real per-block calls.
        if block_id is not None:
            seen.append(block_id)
        return d

    map_blocks(f, r).data.compute()
    # 2x2 spatial chunks -> 4 unique chunk locations (band axis 0).
    assert sorted(set(seen)) == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
    ]


def test_map_blocks_block_id_matches_block_info_chunk_location():
    # block_id is exactly block_info[None]["chunk-location"].
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    seen = []

    def f(d, *, block_id, block_info):
        if block_info is not None:
            seen.append((block_id, block_info[None]["chunk-location"]))
        return d

    map_blocks(f, r).data.compute()
    assert seen
    assert all(bid == loc for bid, loc in seen)


def test_map_blocks_multiple_specials_simultaneously():
    r1 = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-1.0, chunksize=(1, 50, 50)
    )
    r2 = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-2.0, chunksize=(1, 50, 50)
    )

    seen = []

    def f(d1, d2, *, input_masks, input_null_values, block_info, factor):
        # All three reserved + a user-passed kwarg `factor`.
        if block_info is not None:
            seen.append(
                {
                    "n_masks": len(input_masks),
                    "nvs": input_null_values,
                    "factor": factor,
                    "has_block_info": True,
                }
            )
        return (d1 + d2) * factor

    map_blocks(f, r1, r2, factor=3).data.compute()
    assert seen and all(
        s["n_masks"] == 2
        and s["nvs"] == (np.float32(-1.0), np.float32(-2.0))
        and s["factor"] == 3
        and s["has_block_info"]
        for s in seen
    )


def test_map_blocks_kwargs_only_func_gets_no_injection():
    # A function whose only kwargs absorber is **kwargs (no explicit
    # named params) does not trigger introspection-based injection.
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)

    seen = []

    def f(*args, **kwargs):
        seen.append(set(kwargs))
        return args[0]

    map_blocks(f, r).data.compute()
    for kw_keys in seen:
        assert "input_masks" not in kw_keys
        assert "input_null_values" not in kw_keys
        assert "block_info" not in kw_keys
        assert "block_id" not in kw_keys


@pytest.mark.parametrize(
    "name",
    [
        "input_masks",
        "input_null_values",
        "block_info",
        "block_id",
        "out_null_value",
    ],
)
def test_map_blocks_reserved_name_collision_raises(name):
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="reserved"):
        map_blocks(lambda d, **kw: d, r, **{name: "anything"})


def test_map_blocks_out_null_value_inherits_for_single_input():
    # Single input + dtype unchanged -> out_null_value matches the
    # input's null_value (the inherit-on-single-input rule).
    # NOTE: when dtype is None the meta call sees a typed-zero
    # placeholder, so `seen` may include 0.0 in addition to the
    # resolved value. Check for membership rather than uniformity.
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    seen = []

    def f(d, *, out_null_value):
        seen.append(out_null_value)
        return d

    map_blocks(f, r).data.compute()
    assert -1.0 in seen


def test_map_blocks_out_null_value_dtype_default_when_dtype_changes():
    # Explicit dtype: real per-chunk calls see the resolved value
    # (the int32 default). dask may still invoke the wrapper once
    # with block_info=None for meta inspection -- that call sees the
    # typed-zero placeholder. Check membership.
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    seen = []

    def f(d, *, out_null_value):
        seen.append(out_null_value)
        return d.astype(np.int32)

    map_blocks(f, r, dtype=np.int32).data.compute()
    expected = get_default_null_value(np.dtype(np.int32))
    assert expected in seen


def test_map_blocks_out_null_value_dtype_default_inferred():
    # No explicit dtype -- the wrapper runs apply_infer_dtype to
    # resolve out_null_value upfront. Real per-chunk calls see the
    # resolved value; the meta call sees the placeholder.
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    seen = []

    def f(d, *, out_null_value):
        seen.append(out_null_value)
        return d.astype(np.int32)

    map_blocks(f, r).data.compute()
    expected = get_default_null_value(np.dtype(np.int32))
    assert expected in seen


def test_map_blocks_out_null_value_scalar_override():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    seen = []

    def f(d, *, out_null_value):
        seen.append(out_null_value)
        return d

    map_blocks(f, r, null_value=42.0).data.compute()
    assert 42.0 in seen


def test_map_blocks_no_injection_when_func_does_not_name_out_null_value():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)

    def f(d, **kw):
        assert "out_null_value" not in kw
        return d

    map_blocks(f, r).data.compute()


def test_map_blocks_mask_round_trip():
    # User opts in to input_masks AND out_null_value; writes the
    # resolved null at the masked cells; verify those land in the
    # output mask. No explicit null_value is needed -- the
    # single-input-dtype-match inherit rule resolves out_null_value
    # to the input's -1.0.
    r = make_raster(
        shape=(1, 100, 100),
        dtype=np.float32,
        null=-1.0,
        null_pattern=np.s_[:, :10, :10],
    )

    def fill_nulls(d, *, input_masks, out_null_value):
        (m,) = input_masks
        return np.where(m, out_null_value, d)

    out = map_blocks(fill_nulls, r)
    np.testing.assert_array_equal(out.mask.compute(), r.mask.compute())


def test_map_blocks_empty_rasters_raises():
    with pytest.raises(ValueError, match="at least one"):
        map_blocks(lambda x: x)


def test_map_blocks_shape_mismatch_raises():
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)  # 100x100
    r2 = make_raster(shape=(1, 50, 50), dtype=np.float32)  # different shape
    assert r1.shape != r2.shape
    with pytest.raises(ValueError, match="raster 1 shape"):
        map_blocks(lambda a, b: a + b, r1, r2)


def test_map_blocks_preserves_first_input_grid():
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    r2 = make_raster(shape=(1, 100, 100), dtype=np.float32, null=99.0)
    assert r2.null_value != r1.null_value
    out = map_blocks(lambda a, b: a + b, r1, r2, null_value=r1.null_value)
    assert out.crs == r1.crs
    assert out.affine == r1.affine
    assert out.null_value == r1.null_value


def test_map_blocks_meta_skips_zero_shape_call():
    # When meta= is provided, dask uses it as the output meta and
    # skips the 0-shape sample call it would otherwise make. A func
    # that raises on 0-shape input must not be invoked.
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)

    def f(d):
        if 0 in d.shape:
            raise RuntimeError("0-shape call should not happen")
        return d

    out = map_blocks(f, r, meta=np.empty((), dtype=np.float32))
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())


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
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    out = map_overlap(lambda b: b, r, depth=0)
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())
    assert out.crs == r.crs
    assert out.affine == r.affine
    assert out.dtype == r.dtype


def test_map_overlap_3x3_mean_reflect():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    # The user function: assume dask provides a (1, ny+2, nx+2) block with
    # reflect padding on edges; trim=True -> dask trims our 1-cell rim back
    # off after we return a same-shape block.
    out = map_overlap(
        _block_3x3_mean, r, depth=1, boundary="reflect", dtype=np.float32
    )
    expected = _np_3x3_mean_reflect(r.data.compute()[0])[None]
    np.testing.assert_allclose(out.data.compute(), expected, rtol=1e-5)


def test_map_overlap_per_axis_depth_tuple():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    out = map_overlap(lambda b: b, r, depth=(2, 1), boundary="reflect")
    # trim=True preserves shape regardless of depth
    assert out.shape == r.shape


def test_map_overlap_scalar_numeric_boundary():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    out = map_overlap(_block_3x3_sum, r, depth=1, boundary=0, dtype=np.float32)
    arr = r.data.compute()[0]
    pad = np.pad(arr, 1, mode="constant", constant_values=0)
    ref = np.zeros_like(arr, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            ref += pad[dy : dy + arr.shape[0], dx : dx + arr.shape[1]]
    np.testing.assert_allclose(out.data.compute()[0], ref, rtol=1e-5)


def test_map_overlap_boundary_null_with_set_null_value():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-1.0, chunksize=(1, 50, 50)
    )
    captured = {}

    def grab(d, *, input_masks):
        # When this is called on a block, d/m include the overlap rim.
        # On an edge chunk the boundary-side rim is filled with the
        # null value in data and True in mask.
        m = input_masks[0]
        captured.setdefault("samples", []).append((d.copy(), m.copy()))
        return d

    map_overlap(grab, r, depth=1, boundary="null").data.compute()
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
        make_raster(shape=(1, 100, 100), dtype=np.float32).data,
        x=make_raster(shape=(1, 100, 100), dtype=np.float32).x,
        y=make_raster(shape=(1, 100, 100), dtype=np.float32).y,
        crs=make_raster(shape=(1, 100, 100), dtype=np.float32).crs,
    ).chunk((1, 50, 50))
    assert r.null_value is None
    expected_fill = get_default_null_value(np.dtype(r.dtype))

    captured = {}

    def grab(d, *, input_masks):
        m = input_masks[0]
        captured.setdefault("samples", []).append((d.copy(), m.copy()))
        return d

    map_overlap(grab, r, depth=1, boundary="null").data.compute()
    samples = _real_blocks(captured["samples"])
    # Expect at least one edge sample whose boundary rim is the
    # default null value in data and True in mask.
    assert any(
        (d[0, 0, :] == expected_fill).all() and m[0, 0, :].all()
        for d, m in samples
    )


def test_map_overlap_boundary_null_value_alias():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    out1 = map_overlap(lambda b: b, r, depth=1, boundary="null")
    out2 = map_overlap(lambda b: b, r, depth=1, boundary="null_value")
    np.testing.assert_array_equal(out1.data.compute(), out2.data.compute())


@pytest.mark.parametrize("alias", ["nodata", "NODATA", "NoData"])
def test_map_overlap_boundary_nodata_aliases(alias):
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    out_null = map_overlap(lambda b: b, r, depth=1, boundary="null")
    out_alias = map_overlap(lambda b: b, r, depth=1, boundary=alias)
    np.testing.assert_array_equal(
        out_null.data.compute(), out_alias.data.compute()
    )


def test_map_overlap_scalar_matching_null_value_acts_like_null():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-1.0, chunksize=(1, 50, 50)
    )
    captured_null = {}
    captured_scalar = {}

    def grab_into(target):
        def _grab(d, *, input_masks):
            m = input_masks[0]
            target.setdefault("samples", []).append((d.copy(), m.copy()))
            return d

        return _grab

    map_overlap(
        grab_into(captured_null), r, depth=1, boundary="null"
    ).data.compute()
    map_overlap(
        grab_into(captured_scalar), r, depth=1, boundary=-1.0
    ).data.compute()
    null_samples = _real_blocks(captured_null["samples"])
    scalar_samples = _real_blocks(captured_scalar["samples"])
    # Both runs should have at least one edge sample with True mask
    # rim along the boundary side.
    assert any(m[0, 0, :].all() for d, m in null_samples)
    assert any(m[0, 0, :].all() for d, m in scalar_samples)


def test_map_overlap_scalar_zero_when_null_is_not_zero():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-1.0, chunksize=(1, 50, 50)
    )
    captured = {}

    def grab(d, *, input_masks):
        m = input_masks[0]
        captured.setdefault("samples", []).append((d.copy(), m.copy()))
        return d

    map_overlap(grab, r, depth=1, boundary=0).data.compute()
    samples = _real_blocks(captured["samples"])
    # boundary=0 pads both array edges. The corner-edge chunk has a
    # boundary-rim filled with 0 in data and False in mask (because
    # 0 != null_value of -1).
    saw_false_rim = any(
        (d[0, 0, :] == 0).all() and not m[0, 0, :].any() for d, m in samples
    )
    assert saw_false_rim


def test_map_overlap_multi_input_different_null_values():
    r1 = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-1.0, chunksize=(1, 50, 50)
    )
    r2 = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-2.0, chunksize=(1, 50, 50)
    )
    captured = []

    def grab(d1, d2, *, input_masks):
        m1, m2 = input_masks
        if all(s > 0 for s in d1.shape):
            captured.append((d1.copy(), m1.copy(), d2.copy(), m2.copy()))
        return d1 + d2

    map_overlap(grab, r1, r2, depth=1, boundary="null").data.compute()
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
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)
    r2 = make_raster(shape=(1, 100, 100), dtype=np.float32)
    out = map_overlap(lambda a, b: a + b, r1, r2, depth=1, boundary="reflect")
    np.testing.assert_allclose(
        out.data.compute(), r1.data.compute() + r2.data.compute()
    )


def test_map_overlap_input_masks_single_input():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(d, *, input_masks):
        m = input_masks[0]
        assert d.shape == m.shape
        return d + m.astype(d.dtype)

    out = map_overlap(f, r, depth=1, boundary="reflect")
    np.testing.assert_allclose(
        out.data.compute(),
        r.data.compute() + r.mask.compute().astype(r.dtype),
    )


def test_map_overlap_input_masks_two_input():
    r1 = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    r2 = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(d1, d2, *, input_masks):
        m1, m2 = input_masks
        assert d1.dtype == r1.dtype and d2.dtype == r2.dtype
        assert m1.dtype == np.bool_ and m2.dtype == np.bool_
        assert d1.shape == d2.shape == m1.shape == m2.shape
        return d1 + d2

    out = map_overlap(f, r1, r2, depth=1, boundary="reflect")
    np.testing.assert_allclose(
        out.data.compute(), r1.data.compute() + r2.data.compute()
    )


def test_map_overlap_dtype_change():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    out = map_overlap(
        lambda b: b.astype(np.int32),
        r,
        depth=1,
        boundary="reflect",
        dtype=np.int32,
    )
    assert out.dtype == np.int32
    # null_value=None + dtype change -> dtype default.
    assert out.null_value == get_default_null_value(np.dtype(np.int32))


def test_map_overlap_null_value_scalar_override():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    out = map_overlap(lambda b: b, r, depth=0, null_value=42.0)
    assert out.null_value == 42.0


def test_map_overlap_empty_rasters_raises():
    with pytest.raises(ValueError, match="at least one"):
        map_overlap(lambda b: b, depth=0)


def test_map_overlap_shape_mismatch_raises():
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)  # 100x100
    r2 = make_raster(shape=(1, 50, 50), dtype=np.float32)  # different shape
    with pytest.raises(ValueError, match="raster 1 shape"):
        map_overlap(lambda a, b: a + b, r1, r2, depth=0)


def test_map_overlap_band_axis_depth_raises():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="band-axis"):
        map_overlap(lambda b: b, r, depth={0: 1, 1: 0, 2: 0})


def test_map_overlap_negative_depth_raises():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="non-negative"):
        map_overlap(lambda b: b, r, depth=-1)
    with pytest.raises(ValueError, match="non-negative"):
        map_overlap(lambda b: b, r, depth=(2, -1))


def test_map_overlap_unrecognized_boundary_raises():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="unrecognized boundary"):
        map_overlap(lambda b: b, r, depth=1, boundary="bogus")


@pytest.mark.parametrize(
    "boundary",
    [
        "Reflect",
        "REFLECT",
        "Periodic",
        "Nearest",
        "None",
        "NONE",
    ],
)
def test_map_overlap_named_boundary_case_insensitive(boundary):
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    # Just confirm it doesn't raise and the round-trip computes.
    out = map_overlap(lambda b: b, r, depth=1, boundary=boundary)
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())


@pytest.mark.parametrize(
    "boundary",
    [
        "Reflect",
        "REFLECT",
        "Periodic",
        "Nearest",
        "None",
        "NONE",
    ],
)
def test_geo_map_overlap_named_boundary_case_insensitive(boundary):
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(xda, **kw):
        return xda

    out = geo_map_overlap(f, r, depth=1, boundary=boundary)
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())


def test_map_overlap_asymmetric_depth_with_capital_none_boundary_ok():
    # The asymmetric-depth pre-check is now case-insensitive: "None"
    # is treated as the no-pad sentinel and asymmetric depth is
    # accepted.
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    out = map_overlap(lambda b: b, r, depth={1: (2, 1), 2: 0}, boundary="None")
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())


@pytest.mark.parametrize(
    ("depth", "boundary"),
    [
        ({1: (2, 1), 2: 0}, "reflect"),
        ({1: (1, 2), 2: 0}, 0),
        ({1: 0, 2: (0, 1)}, "null"),
        # Even a "symmetric" tuple counts -- dask checks the type.
        ({1: (2, 2), 2: 0}, "reflect"),
    ],
)
def test_map_overlap_asymmetric_depth_with_non_none_boundary_raises(
    depth, boundary
):
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    with pytest.raises(ValueError, match="asymmetric depth"):
        map_overlap(lambda b: b, r, depth=depth, boundary=boundary)


@pytest.mark.parametrize("boundary", [None, "none"])
def test_map_overlap_asymmetric_depth_with_none_boundary_ok(boundary):
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    out = map_overlap(
        lambda b: b, r, depth={1: (2, 1), 2: 0}, boundary=boundary
    )
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())


def test_map_overlap_input_null_values_injected():
    r1 = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-1.0, chunksize=(1, 50, 50)
    )
    r2 = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-2.0, chunksize=(1, 50, 50)
    )
    seen = []

    def f(d1, d2, *, input_null_values):
        seen.append(input_null_values)
        return d1 + d2

    map_overlap(f, r1, r2, depth=1, boundary="reflect").data.compute()
    assert any(s == (-1.0, -2.0) for s in seen)


def test_map_overlap_block_info_injected():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    seen = []

    def f(d, *, block_info):
        if block_info is not None:
            seen.append(block_info[0]["chunk-location"])
        return d

    map_overlap(f, r, depth=1, boundary="reflect").data.compute()
    # 2x2 spatial chunks -> 4 distinct chunk locations.
    assert len(set(seen)) == 4


def test_map_overlap_out_null_value_inherits_for_single_input():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-1.0, chunksize=(1, 50, 50)
    )
    seen = []

    def f(d, *, out_null_value):
        seen.append(out_null_value)
        return d

    out = map_overlap(f, r, depth=1, boundary="reflect")
    out.data.compute()
    assert -1.0 in seen
    assert out.null_value == -1.0


def test_map_overlap_out_null_value_dtype_default_when_dtype_changes():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-1.0, chunksize=(1, 50, 50)
    )

    def f(d, *, out_null_value):
        # Use out_null_value at a typed location so dtype inference
        # produces int32 from the cast inside func.
        return d.astype(np.int32)

    seen = []

    def grab(d, *, out_null_value):
        seen.append(out_null_value)
        return d.astype(np.int32)

    out = map_overlap(grab, r, depth=1, boundary="reflect", dtype=np.int32)
    out.data.compute()
    expected = get_default_null_value(np.dtype(np.int32))
    assert expected in seen


def test_map_overlap_no_injection_when_func_does_not_name_kwargs():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    seen = []

    def f(d, **kw):
        seen.append(set(kw.keys()))
        return d

    map_overlap(f, r, depth=1, boundary="reflect").data.compute()
    for kw_keys in seen:
        assert "input_masks" not in kw_keys
        assert "input_null_values" not in kw_keys
        assert "block_info" not in kw_keys
        assert "out_null_value" not in kw_keys


@pytest.mark.parametrize(
    "name",
    [
        "input_masks",
        "input_null_values",
        "block_info",
        "out_null_value",
    ],
)
def test_map_overlap_reserved_kwargs_collide(name):
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="reserved"):
        map_overlap(lambda d: d, r, depth=0, **{name: object()})


def test_map_overlap_null_value_inherits_for_single_input_unchanged_dtype():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    out = map_overlap(lambda b: b, r, depth=0)
    assert out.null_value == -1.0


def test_map_overlap_null_value_dtype_default_for_multi_input():
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    r2 = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-2.0)
    out = map_overlap(lambda a, b: a + b, r1, r2, depth=0)
    assert out.null_value == get_default_null_value(np.dtype(out.dtype))
    assert out.null_value not in (-1.0, -2.0)


def test_map_overlap_null_value_invalid_string():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="must be None or a scalar"):
        map_overlap(lambda b: b, r, depth=0, null_value="default")


def test_map_overlap_meta_skips_zero_shape_call():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(d):
        if 0 in d.shape:
            raise RuntimeError("0-shape call should not happen")
        return d

    out = map_overlap(
        f,
        r,
        depth=1,
        boundary="reflect",
        meta=np.empty((), dtype=np.float32),
    )
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.data.compute(), r.data.compute())


# ---------------------------------------------------------------------------
# geo_map_blocks
# ---------------------------------------------------------------------------


def test_geo_map_blocks_identity():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(xda, **kw):
        return xda

    out = geo_map_blocks(f, r)
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())
    assert out.crs == r.crs
    assert out.affine == r.affine
    assert out.dtype == r.dtype
    assert out.null_value == r.null_value


def test_geo_map_blocks_func_sees_coords_crs_nodata():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

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
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
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
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(xda, **kw):
        return xda * 2

    out = geo_map_blocks(f, r)
    np.testing.assert_allclose(out.data.compute(), r.data.compute() * 2)


def test_geo_map_blocks_returns_ndarray():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(xda, **kw):
        return xda.values * 3

    out = geo_map_blocks(f, r)
    np.testing.assert_allclose(out.data.compute(), r.data.compute() * 3)


def test_geo_map_blocks_geo_block_info_kwarg_present():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
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
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(a, b, **kw):
        return a + b

    out = geo_map_blocks(f, r, r)
    np.testing.assert_allclose(out.data.compute(), r.data.compute() * 2)


def test_geo_map_blocks_kwargs_forwarded():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(xda, *, factor, **kw):
        return xda * factor

    out = geo_map_blocks(f, r, factor=4)
    np.testing.assert_allclose(out.data.compute(), r.data.compute() * 4)


def test_geo_map_blocks_input_masks_single_input():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(xda, *, input_masks):
        xma = input_masks[0]
        assert isinstance(xda, xr.DataArray)
        assert isinstance(xma, xr.DataArray)
        assert xma.dtype == bool
        # Coords aligned between data and mask DataArrays.
        np.testing.assert_array_equal(
            xda.coords["x"].values, xma.coords["x"].values
        )
        return xda + xma.astype(xda.dtype)

    out = geo_map_blocks(f, r)
    np.testing.assert_allclose(
        out.data.compute(),
        r.data.compute() + r.mask.compute().astype(r.dtype),
    )


def test_geo_map_blocks_input_masks_two_input():
    r1 = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    r2 = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(xda1, xda2, *, input_masks):
        xma1, xma2 = input_masks
        assert xda1.dtype == r1.dtype and xda2.dtype == r2.dtype
        assert xma1.dtype == bool and xma2.dtype == bool
        return xda1 + xda2

    out = geo_map_blocks(f, r1, r2)
    np.testing.assert_allclose(
        out.data.compute(), r1.data.compute() + r2.data.compute()
    )


def test_geo_map_blocks_dtype_change():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(xda, **kw):
        return xda.astype(np.int32)

    out = geo_map_blocks(f, r, dtype=np.int32)
    assert out.dtype == np.int32
    # null_value=None + dtype change -> dtype default.
    assert out.null_value == get_default_null_value(np.dtype(np.int32))


def test_geo_map_blocks_null_value_scalar_override():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)

    def f(xda, **kw):
        return xda

    out = geo_map_blocks(f, r, null_value=42.0)
    assert out.null_value == 42.0


def test_geo_map_blocks_empty_rasters_raises():
    with pytest.raises(ValueError, match="at least one"):
        geo_map_blocks(lambda xda, **kw: xda)


def test_geo_map_blocks_shape_mismatch_raises():
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)
    r2 = make_raster(shape=(1, 50, 50), dtype=np.float32)
    with pytest.raises(ValueError, match="raster 1 shape"):
        geo_map_blocks(lambda a, b, **kw: a + b, r1, r2)


def test_geo_map_blocks_preserves_first_input_grid():
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    r2 = make_raster(shape=(1, 100, 100), dtype=np.float32, null=99.0)
    assert r2.null_value != r1.null_value
    # Force the inherit by passing r1's null_value explicitly; the
    # multi-input default rule would otherwise pick the dtype default.
    out = geo_map_blocks(
        lambda a, b, **kw: a + b, r1, r2, null_value=r1.null_value
    )
    assert out.crs == r1.crs
    assert out.affine == r1.affine
    assert out.null_value == r1.null_value


def test_geo_map_blocks_different_crs_raises():
    import raster_tools as rts_

    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)
    # Same shape and coords as r1 but a different CRS.
    r2 = rts_.data_to_raster(r1.data, x=r1.x, y=r1.y, crs="EPSG:4326")
    assert r1.shape == r2.shape
    assert r1.crs != r2.crs
    with pytest.raises(ValueError, match="same grid"):
        geo_map_blocks(lambda a, b, **kw: a + b, r1, r2)


def test_geo_map_blocks_different_affine_raises():
    import raster_tools as rts_

    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)
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
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    out = geo_map_blocks(lambda a, b, **kw: a + b, r, r)
    np.testing.assert_allclose(out.data.compute(), r.data.compute() * 2)


def test_geo_map_blocks_sub_pixel_fp_noise_tolerated():
    import raster_tools as rts_
    from raster_tools._grids import GRID_PIXEL_TOLERANCE

    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)
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


def test_geo_map_blocks_input_null_values_injected():
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    r2 = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-2.0)
    seen = []

    def f(xda1, xda2, *, input_null_values):
        seen.append(input_null_values)
        return xda1 + xda2

    geo_map_blocks(f, r1, r2).data.compute()
    assert any(s == (-1.0, -2.0) for s in seen)


def test_geo_map_blocks_block_info_injected():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    seen = []

    def f(xda, *, block_info):
        if block_info is not None:
            seen.append(block_info[0]["chunk-location"])
        return xda

    geo_map_blocks(f, r).data.compute()
    assert len(set(seen)) == 4


def test_geo_map_blocks_block_id_injected():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    seen = []

    def f(xda, *, block_id, block_info):
        if block_info is not None:
            seen.append((block_id, block_info[None]["chunk-location"]))
        return xda

    geo_map_blocks(f, r).data.compute()
    assert len(seen) == 4
    assert all(bid == loc for bid, loc in seen)


def test_geo_map_blocks_geo_block_info_optin():
    # geo_block_info is opt-in: a func that doesn't name it doesn't
    # receive it. Bare 1-arg signature is now valid.
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(xda):
        # No **kw absorber; geo_block_info MUST not be passed.
        return xda

    out = geo_map_blocks(f, r)
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())


def test_geo_map_blocks_out_null_value_inherits_for_single_input():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    seen = []

    def f(xda, *, out_null_value):
        seen.append(out_null_value)
        return xda

    out = geo_map_blocks(f, r)
    out.data.compute()
    assert -1.0 in seen
    assert out.null_value == -1.0


def test_geo_map_blocks_out_null_value_dtype_default_when_dtype_changes():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    seen = []

    def f(xda, *, out_null_value):
        seen.append(out_null_value)
        return xda.astype(np.int32)

    out = geo_map_blocks(f, r, dtype=np.int32)
    out.data.compute()
    expected = get_default_null_value(np.dtype(np.int32))
    assert expected in seen


def test_geo_map_blocks_no_injection_when_func_does_not_name_kwargs():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    seen = []

    def f(xda, **kw):
        seen.append(set(kw.keys()))
        return xda

    geo_map_blocks(f, r).data.compute()
    for kw_keys in seen:
        assert "input_masks" not in kw_keys
        assert "input_null_values" not in kw_keys
        assert "block_info" not in kw_keys
        assert "out_null_value" not in kw_keys
        assert "geo_block_info" not in kw_keys


@pytest.mark.parametrize(
    "name",
    [
        "input_masks",
        "input_null_values",
        "block_info",
        "out_null_value",
        "geo_block_info",
    ],
)
def test_geo_map_blocks_reserved_kwargs_collide(name):
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="reserved"):
        geo_map_blocks(lambda xda, **kw: xda, r, **{name: object()})


def test_geo_map_blocks_null_value_inherits_for_single_input_unchanged_dtype():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    out = geo_map_blocks(lambda xda, **kw: xda, r)
    assert out.null_value == -1.0


def test_geo_map_blocks_null_value_dtype_default_for_multi_input():
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    r2 = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-2.0)
    out = geo_map_blocks(lambda a, b, **kw: a + b, r1, r2)
    assert out.null_value == get_default_null_value(np.dtype(out.dtype))
    assert out.null_value not in (-1.0, -2.0)


def test_geo_map_blocks_null_value_invalid_string():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="must be None or a scalar"):
        geo_map_blocks(lambda xda, **kw: xda, r, null_value="default")


@pytest.mark.parametrize(
    "fn",
    [
        lambda r, **kw: map_blocks(lambda d: d, r, **kw),
        lambda r, **kw: map_overlap(
            lambda d: d, r, depth=0, boundary="reflect", **kw
        ),
        lambda r, **kw: geo_map_blocks(lambda xda, **k: xda, r, **kw),
        lambda r, **kw: geo_map_overlap(
            lambda xda, **k: xda, r, depth=0, boundary="reflect", **kw
        ),
    ],
)
def test_meta_dtype_mismatch_raises(fn):
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="conflicts with meta"):
        fn(r, dtype=np.int64, meta=np.empty((), dtype=np.float32))


# ---------------------------------------------------------------------------
# dask-kwargs leakage rejection
# ---------------------------------------------------------------------------


_DASK_LEAK_NAMES = [
    "chunks",
    "name",
    "token",
    "drop_axis",
    "new_axis",
    "enforce_ndim",
    "concatenate",
    "align_arrays",
    "trim",
    "allow_rechunk",
]


@pytest.mark.parametrize(
    "fn",
    [
        lambda r, **kw: map_blocks(lambda d: d, r, **kw),
        lambda r, **kw: map_overlap(
            lambda d: d, r, depth=0, boundary="reflect", **kw
        ),
        lambda r, **kw: geo_map_blocks(lambda xda, **k: xda, r, **kw),
        lambda r, **kw: geo_map_overlap(
            lambda xda, **k: xda, r, depth=0, boundary="reflect", **kw
        ),
    ],
)
@pytest.mark.parametrize("name", _DASK_LEAK_NAMES)
def test_dask_kwargs_leakage_rejected(fn, name):
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="dask graph-construction options"):
        fn(r, **{name: object()})


# ---------------------------------------------------------------------------
# DataArray return preservation
# ---------------------------------------------------------------------------


def test_map_blocks_dataarray_return_extracted():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)

    def f(d):
        # Wrap in a DataArray; wrapper should extract .data.
        return xr.DataArray(d * 2, dims=("band", "y", "x"))

    out = map_blocks(f, r)
    np.testing.assert_allclose(out.data.compute(), r.data.compute() * 2)


def test_map_overlap_dataarray_return_extracted():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(d):
        return xr.DataArray(d * 3, dims=("band", "y", "x"))

    out = map_overlap(f, r, depth=0, boundary="reflect")
    np.testing.assert_allclose(out.data.compute(), r.data.compute() * 3)


def test_geo_map_blocks_meta_skips_zero_shape_call():
    # meta= forwards to dask's map_blocks. The user's func must not
    # be invoked with 0-shape DataArrays.
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)

    def f(xda, **kw):
        if 0 in xda.shape:
            raise RuntimeError("0-shape call should not happen")
        return xda

    out = geo_map_blocks(f, r, meta=np.empty((), dtype=np.float32))
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())


def test_geo_map_blocks_out_null_value_meta_call_typed_zero():
    # During the 0-shape meta call (no meta= passed), the wrapper
    # supplies out_null_value as a typed zero of the first input's
    # dtype. Real per-chunk calls see the resolved scalar.
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    seen_meta = []
    seen_real = []

    def f(xda, *, out_null_value, geo_block_info):
        if geo_block_info is None:
            seen_meta.append((type(out_null_value), out_null_value))
        else:
            seen_real.append(out_null_value)
        return xda

    geo_map_blocks(f, r).data.compute()
    assert any(t is np.float32 for t, _ in seen_meta)
    assert any(v == np.float32(0.0) for _, v in seen_meta)
    assert -1.0 in seen_real


def test_geo_map_blocks_out_null_value_with_meta_skips_meta_call():
    # When meta= is passed, the meta call is skipped entirely; the
    # func only sees real-chunk values for out_null_value.
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    seen = []

    def f(xda, *, out_null_value, geo_block_info):
        if geo_block_info is None:
            raise RuntimeError("0-shape meta call should not happen")
        seen.append(out_null_value)
        return xda

    geo_map_blocks(f, r, meta=np.empty((), dtype=np.float32)).data.compute()
    assert -1.0 in seen
    assert all(v == -1.0 for v in seen)


# ---------------------------------------------------------------------------
# dtype inference (mirror dask)
# ---------------------------------------------------------------------------


def test_map_blocks_dtype_inferred_for_mixed_inputs():
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)  # float32
    r2 = make_raster(shape=(1, 100, 100), dtype=np.int16)
    out = map_blocks(lambda a, b: a + b, r1, r2)
    # NumPy promotion: float32 + int16 -> float32. dask infers it.
    assert out.dtype == np.float32


def test_map_blocks_dtype_inferred_from_in_func_cast():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)  # float32
    out = map_blocks(lambda x: x.astype(np.int32), r)
    # If we used np.result_type on inputs we'd get float32.
    # Inferring via the actual function call yields int32.
    assert out.dtype == np.int32


def test_map_blocks_explicit_dtype_overrides_inference():
    r = make_raster(shape=(1, 100, 100), dtype=np.int16)
    out = map_blocks(
        lambda x: x.astype(np.float64),
        r,
        dtype=np.float64,
    )
    assert out.dtype == np.float64


def test_map_overlap_dtype_inferred_for_mixed_inputs():
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)  # float32
    r2 = make_raster(shape=(1, 100, 100), dtype=np.int16)
    out = map_overlap(
        lambda a, b: a + b,
        r1,
        r2,
        depth=1,
        boundary="reflect",
    )
    assert out.dtype == np.float32


def test_geo_map_blocks_dtype_inferred_for_mixed_inputs():
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)  # float32
    r2 = make_raster(shape=(1, 100, 100), dtype=np.int16)
    out = geo_map_blocks(lambda a, b, **kw: a + b, r1, r2)
    assert out.dtype == np.float32


def test_geo_map_blocks_dtype_inferred_from_in_func_cast():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)  # float32
    out = geo_map_blocks(lambda xda, **kw: xda.astype(np.int32), r)
    assert out.dtype == np.int32


def test_map_blocks_null_value_uses_inferred_dtype():
    # When dtype is inferred to float32 (the promoted result), the
    # default null is the float32 default, NOT int16's.
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)  # float32
    r2 = make_raster(shape=(1, 100, 100), dtype=np.int16)
    out = map_blocks(lambda a, b: a + b, r1, r2)
    assert out.dtype == np.float32
    assert out.null_value == get_default_null_value(np.dtype(np.float32))


# ---------------------------------------------------------------------------
# geo_map_overlap
# ---------------------------------------------------------------------------


def _identity_or_meta(xda, geo_block_info=None, **kw):
    """Identity that tolerates dask's 0-shape meta call."""
    return xda


def test_geo_map_overlap_identity_depth_zero():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    out = geo_map_overlap(_identity_or_meta, r, depth=0)
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())
    assert out.crs == r.crs
    assert out.affine == r.affine
    assert out.dtype == r.dtype


def test_geo_map_overlap_3x3_mean_reflect():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

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
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
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
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(xda, geo_block_info=None, **kw):
        if geo_block_info is not None and all(s > 0 for s in xda.shape):
            assert geo_block_info.shape == xda.shape
            # row_slice / col_slice may have negative starts at edges.
            assert isinstance(geo_block_info.row_slice, slice)
            assert isinstance(geo_block_info.col_slice, slice)
        return xda

    geo_map_overlap(f, r, depth=1, boundary="reflect").data.compute()


def test_geo_map_overlap_nodata_visible_in_meta_call():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
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


def test_geo_map_overlap_boundary_null_with_input_masks():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-1.0, chunksize=(1, 50, 50)
    )
    captured = []

    def grab(xda, *, input_masks, geo_block_info=None):
        xma = input_masks[0]
        if all(s > 0 for s in xda.shape):
            captured.append((xda.values.copy(), xma.values.copy()))
        return xda

    geo_map_overlap(grab, r, depth=1, boundary="null").data.compute()
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
    r1 = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    r2 = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
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


def test_geo_map_overlap_input_masks_two_input():
    r1 = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    r2 = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(xda1, xda2, *, input_masks):
        xma1, xma2 = input_masks
        if not all(s > 0 for s in xda1.shape):
            return xda1
        assert xda1.dtype == r1.dtype and xda2.dtype == r2.dtype
        assert xma1.dtype == bool and xma2.dtype == bool
        assert xda1.shape == xma1.shape == xda2.shape == xma2.shape
        return xda1 + xda2

    out = geo_map_overlap(f, r1, r2, depth=1, boundary="reflect")
    np.testing.assert_allclose(
        out.data.compute(), r1.data.compute() + r2.data.compute()
    )


def test_geo_map_overlap_dtype_inferred_for_mixed_inputs():
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)  # float32
    r2 = make_raster(shape=(1, 100, 100), dtype=np.int16)
    out = geo_map_overlap(
        lambda a, b, **kw: a + b, r1, r2, depth=1, boundary="reflect"
    )
    assert out.dtype == np.float32


def test_geo_map_overlap_dtype_inferred_from_in_func_cast():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)  # float32
    out = geo_map_overlap(
        lambda xda, **kw: xda.astype(np.int32),
        r,
        depth=1,
        boundary="reflect",
    )
    assert out.dtype == np.int32


def test_geo_map_overlap_different_crs_raises():
    import raster_tools as rts_

    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)
    r2 = rts_.data_to_raster(r1.data, x=r1.x, y=r1.y, crs="EPSG:4326")
    with pytest.raises(ValueError, match="same grid"):
        geo_map_overlap(
            lambda a, b, **kw: a + b, r1, r2, depth=1, boundary="reflect"
        )


def test_geo_map_overlap_different_affine_raises():
    import raster_tools as rts_

    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32)
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
    r = make_raster(shape=(1, 100, 100), dtype=np.float32).chunk((1, 10, 10))
    assert all(c < 20 for c in r.data.chunks[1])

    out = geo_map_overlap(_identity_or_meta, r, depth=20, boundary="reflect")
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())
    # Verify the rechunk actually happened: spatial chunks are now
    # >= depth.
    assert all(c >= 20 for c in out.data.chunks[1])


def test_geo_map_overlap_rechunk_with_input_masks():
    # The mask should be rechunked alongside the data so input_masks
    # callbacks see matching shapes.
    r = make_raster(shape=(1, 100, 100), dtype=np.float32).chunk((1, 10, 10))

    def f(xda, *, input_masks):
        xma = input_masks[0]
        if not all(s > 0 for s in xda.shape):
            return xda
        assert xda.shape == xma.shape
        return xda + xma.astype(xda.dtype)

    out = geo_map_overlap(f, r, depth=20, boundary="reflect")
    np.testing.assert_allclose(out.data.compute(), r.data.compute())


@pytest.mark.parametrize(
    ("depth", "boundary"),
    [
        ({1: (2, 1), 2: 0}, "reflect"),
        ({1: (1, 2), 2: 0}, 0),
        ({1: 0, 2: (0, 1)}, "null"),
        ({1: (2, 2), 2: 0}, "reflect"),
    ],
)
def test_geo_map_overlap_asymmetric_depth_with_non_none_boundary_raises(
    depth, boundary
):
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    with pytest.raises(ValueError, match="asymmetric depth"):
        geo_map_overlap(
            lambda xda, **kw: xda, r, depth=depth, boundary=boundary
        )


@pytest.mark.parametrize("boundary", [None, "none"])
def test_geo_map_overlap_asymmetric_depth_with_none_boundary_ok(boundary):
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(xda, **kw):
        return xda

    out = geo_map_overlap(f, r, depth={1: (2, 1), 2: 0}, boundary=boundary)
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())


# ---------------------------------------------------------------------------
# geo_map_overlap -- introspection contract
# ---------------------------------------------------------------------------


def test_geo_map_overlap_input_null_values_injected():
    r1 = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-1.0, chunksize=(1, 50, 50)
    )
    r2 = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-2.0, chunksize=(1, 50, 50)
    )
    seen = []

    def f(xda1, xda2, *, input_null_values):
        seen.append(input_null_values)
        return xda1 + xda2

    geo_map_overlap(f, r1, r2, depth=1, boundary="reflect").data.compute()
    assert any(s == (-1.0, -2.0) for s in seen)


def test_geo_map_overlap_block_info_injected():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    seen = []

    def f(xda, *, block_info):
        if block_info is not None:
            seen.append(block_info[0]["chunk-location"])
        return xda

    geo_map_overlap(f, r, depth=1, boundary="reflect").data.compute()
    # 2x2 spatial chunks -> 4 distinct chunk locations.
    assert len(set(seen)) == 4


def test_geo_map_overlap_geo_block_info_optin():
    # geo_block_info is opt-in: a func that doesn't name it doesn't
    # receive it. Bare 1-arg signature is now valid.
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(xda):
        # No **kw absorber; geo_block_info MUST not be passed.
        return xda

    out = geo_map_overlap(f, r, depth=1, boundary="reflect")
    np.testing.assert_array_equal(out.data.compute(), r.data.compute())


def test_geo_map_overlap_geo_block_info_reflects_overlap_extent():
    # When the user names geo_block_info, it should reflect the
    # overlap-included extent: shape matches the data block, and
    # coords align with the (possibly negative-start) row/col slices.
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    seen = []

    def f(xda, *, geo_block_info):
        if geo_block_info is None:
            return xda
        assert xda.shape == geo_block_info.shape
        np.testing.assert_array_equal(xda.coords["x"].values, geo_block_info.x)
        np.testing.assert_array_equal(xda.coords["y"].values, geo_block_info.y)
        seen.append(geo_block_info.chunk_location)
        return xda

    geo_map_overlap(f, r, depth=1, boundary="reflect").data.compute()
    assert len(set(seen)) == 4


def test_geo_map_overlap_out_null_value_inherits_for_single_input():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-1.0, chunksize=(1, 50, 50)
    )
    seen = []

    def f(xda, *, out_null_value):
        seen.append(out_null_value)
        return xda

    out = geo_map_overlap(f, r, depth=1, boundary="reflect")
    out.data.compute()
    assert -1.0 in seen
    assert out.null_value == -1.0


def test_geo_map_overlap_out_null_value_dtype_default_when_dtype_changes():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-1.0, chunksize=(1, 50, 50)
    )
    seen = []

    def f(xda, *, out_null_value):
        seen.append(out_null_value)
        return xda.astype(np.int32)

    out = geo_map_overlap(f, r, depth=1, boundary="reflect", dtype=np.int32)
    out.data.compute()
    expected = get_default_null_value(np.dtype(np.int32))
    assert expected in seen


def test_geo_map_overlap_no_injection_when_func_does_not_name_kwargs():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )
    seen = []

    def f(xda, **kw):
        seen.append(set(kw.keys()))
        return xda

    geo_map_overlap(f, r, depth=1, boundary="reflect").data.compute()
    for kw_keys in seen:
        assert "input_masks" not in kw_keys
        assert "input_null_values" not in kw_keys
        assert "block_info" not in kw_keys
        assert "out_null_value" not in kw_keys
        assert "geo_block_info" not in kw_keys


@pytest.mark.parametrize(
    "name",
    [
        "input_masks",
        "input_null_values",
        "block_info",
        "out_null_value",
        "geo_block_info",
    ],
)
def test_geo_map_overlap_reserved_kwargs_collide(name):
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="reserved"):
        geo_map_overlap(lambda xda, **kw: xda, r, depth=0, **{name: object()})


def test_geo_map_overlap_null_value_inherits_single_input_unchanged_dtype():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    out = geo_map_overlap(lambda xda, **kw: xda, r, depth=0)
    assert out.null_value == -1.0


def test_geo_map_overlap_null_value_dtype_default_for_multi_input():
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-1.0)
    r2 = make_raster(shape=(1, 100, 100), dtype=np.float32, null=-2.0)
    out = geo_map_overlap(lambda a, b, **kw: a + b, r1, r2, depth=0)
    assert out.null_value == get_default_null_value(np.dtype(out.dtype))
    assert out.null_value not in (-1.0, -2.0)


def test_geo_map_overlap_null_value_invalid_string():
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="must be None or a scalar"):
        geo_map_overlap(
            lambda xda, **kw: xda, r, depth=0, null_value="default"
        )


def test_geo_map_overlap_meta_skips_zero_shape_call():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, chunksize=(1, 50, 50)
    )

    def f(xda, **kw):
        if 0 in xda.shape:
            raise RuntimeError("0-shape call should not happen")
        return xda

    out = geo_map_overlap(
        f,
        r,
        depth=1,
        boundary="reflect",
        meta=np.empty((), dtype=np.float32),
    )
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.data.compute(), r.data.compute())


def test_geo_map_overlap_out_null_value_meta_call_typed_zero():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-1.0, chunksize=(1, 50, 50)
    )
    seen_meta = []
    seen_real = []

    def f(xda, *, out_null_value, geo_block_info):
        if geo_block_info is None:
            seen_meta.append((type(out_null_value), out_null_value))
        else:
            seen_real.append(out_null_value)
        return xda

    geo_map_overlap(f, r, depth=1, boundary="reflect").data.compute()
    assert any(t is np.float32 for t, _ in seen_meta)
    assert any(v == np.float32(0.0) for _, v in seen_meta)
    assert -1.0 in seen_real


def test_geo_map_overlap_out_null_value_with_meta_skips_meta_call():
    r = make_raster(
        shape=(1, 100, 100), dtype=np.float32, null=-1.0, chunksize=(1, 50, 50)
    )
    seen = []

    def f(xda, *, out_null_value, geo_block_info):
        if geo_block_info is None:
            raise RuntimeError("0-shape meta call should not happen")
        seen.append(out_null_value)
        return xda

    geo_map_overlap(
        f,
        r,
        depth=1,
        boundary="reflect",
        meta=np.empty((), dtype=np.float32),
    ).data.compute()
    assert -1.0 in seen
    assert all(v == -1.0 for v in seen)


# ---------------------------------------------------------------------------
# mismatched input chunking is aligned to the first input
#
# Same-grid inputs with different chunking used to crash with an opaque
# dask IndexError (map_blocks / geo_map_blocks / geo_map_overlap). They
# are now rechunked to the first input's chunk structure.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        lambda r1, r2: map_blocks(lambda a, b: a + b, r1, r2),
        lambda r1, r2: map_overlap(
            lambda a, b: a + b, r1, r2, depth=1, boundary="reflect"
        ),
        lambda r1, r2: geo_map_blocks(lambda a, b, **k: a + b, r1, r2),
        lambda r1, r2: geo_map_overlap(
            lambda a, b, **k: a + b, r1, r2, depth=1, boundary="reflect"
        ),
    ],
)
def test_mismatched_input_chunks_aligned_to_first(fn):
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32).chunk((1, 50, 50))
    # Same data and grid as r1 but a different chunk structure.
    r2 = rts.data_to_raster(
        r1.data.compute(), x=r1.x, y=r1.y, crs=r1.crs
    ).chunk((1, 25, 25))
    assert r1.data.chunks != r2.data.chunks
    out = fn(r1, r2)
    np.testing.assert_allclose(
        out.data.compute(), r1.data.compute() * 2, rtol=1e-5
    )
    # Output lands on the first input's chunking (depth=1 -> no
    # min-chunk growth for the overlap variants).
    assert out.data.chunks == r1.data.chunks


def test_reproject_workflow_aligns_chunks_regression():
    # The documented alignment recipe: r2.reproject(r1.geobox). reproject
    # does not adopt the target's chunking, which used to crash
    # geo_map_blocks with an opaque IndexError.
    r1 = make_raster(shape=(1, 100, 100), dtype=np.float32).chunk((1, 50, 50))
    # Same grid/data as r1 but oddly chunked; reproject onto r1's geobox
    # keeps values (identity) while retaining the odd chunking.
    r2 = rts.data_to_raster(
        r1.data.compute(), x=r1.x, y=r1.y, crs=r1.crs
    ).chunk((1, 32, 32))
    r2 = r2.reproject(r1.geobox)
    assert r2.data.chunks != r1.data.chunks
    assert r1.geobox == r2.geobox
    out = geo_map_blocks(lambda a, b, **k: a + b, r1, r2)
    np.testing.assert_allclose(
        out.data.compute(), r1.data.compute() * 2, rtol=1e-5
    )
    assert out.data.chunks == r1.data.chunks


# ---------------------------------------------------------------------------
# Multi-band / multi-band-chunk coverage
# ---------------------------------------------------------------------------


def test_map_blocks_multiband_multichunk():
    r = make_raster(shape=(4, 60, 60), dtype=np.float32).chunk((2, 30, 30))
    out = map_blocks(lambda d: d * 2, r)
    np.testing.assert_allclose(out.data.compute(), r.data.compute() * 2)
    assert out.shape == (4, 60, 60)


def test_geo_map_blocks_multiband_multichunk():
    r = make_raster(shape=(4, 60, 60), dtype=np.float32).chunk((2, 30, 30))
    seen = []

    def f(xda, geo_block_info=None, **kw):
        if geo_block_info is not None:
            # The block's band coords match the gbi's band slice.
            np.testing.assert_array_equal(
                xda.coords["band"].values, geo_block_info.band
            )
            assert xda.shape == geo_block_info.shape
            seen.append(geo_block_info.chunk_location)
        return xda * 2

    out = geo_map_blocks(f, r)
    np.testing.assert_allclose(out.data.compute(), r.data.compute() * 2)
    assert out.shape == (4, 60, 60)
    # 2 band-chunks x 2 y-chunks x 2 x-chunks = 8 blocks.
    assert len(set(seen)) == 8


# ---------------------------------------------------------------------------
# out_bands: band-count change (map_blocks / geo_map_blocks)
# ---------------------------------------------------------------------------


def test_map_blocks_out_bands_expansion():
    # 1-band input -> 3-band output (per-pixel stats pattern).
    r = make_raster(content="arange", shape=(1, 100, 100), dtype=np.float32)

    def expand(d):
        return np.concatenate([d, d * 2, d * 3], axis=0)

    out = map_blocks(expand, r, out_bands=3)
    assert out.nbands == 3
    np.testing.assert_array_equal(np.asarray(out.band), [1, 2, 3])
    src = r.data.compute()
    res = out.data.compute()
    np.testing.assert_allclose(res[0], src[0])
    np.testing.assert_allclose(res[1], src[0] * 2)
    np.testing.assert_allclose(res[2], src[0] * 3)
    # y/x grid preserved.
    assert out.affine == r.affine
    assert out.crs == r.crs
    np.testing.assert_array_equal(out.x, r.x)
    np.testing.assert_array_equal(out.y, r.y)


def test_map_blocks_out_bands_reduction_sees_all_bands():
    # 4-band input -> 1-band output: func reduces over the band axis,
    # which only works if it sees all input bands at once (the
    # single-band-chunk requirement).
    r = make_raster(content="arange", shape=(4, 100, 100), dtype=np.float32)

    def band_mean(d):
        return d.mean(axis=0, keepdims=True)

    out = map_blocks(band_mean, r, out_bands=1)
    assert out.nbands == 1
    np.testing.assert_array_equal(np.asarray(out.band), [1])
    np.testing.assert_allclose(
        out.data.compute()[0], r.data.compute().mean(axis=0)
    )


def test_map_blocks_out_bands_multichunk_spatial():
    # Single-band-chunk + multiple spatial blocks together.
    r = make_raster(
        content="arange",
        shape=(2, 100, 100),
        dtype=np.float32,
        chunksize=(1, 50, 50),
    )

    def collapse(d):
        return d.sum(axis=0, keepdims=True)

    out = map_blocks(collapse, r, out_bands=1)
    assert out.nbands == 1
    np.testing.assert_allclose(
        out.data.compute()[0], r.data.compute().sum(axis=0)
    )


def test_map_blocks_out_bands_restores_original_chunking():
    # The transient single-band-chunk / re-tiled grid is normalized away:
    # the output lands on the caller's original y/x chunking with
    # per-band band chunks.
    r = make_raster(
        content="arange",
        shape=(3, 100, 100),
        dtype=np.float32,
        chunksize=(1, 50, 50),
    )

    def three_to_two(d):
        return np.stack([d[0] + d[1], d[2]], axis=0)

    # Band-indexing func -> pass dtype= so dask skips the (1,1,1)-shaped
    # dtype-inference sample call it would otherwise make.
    out = map_blocks(three_to_two, r, out_bands=2, dtype=np.float32)
    assert out.data.chunks == ((1, 1), (50, 50), (50, 50))
    assert out.mask.chunks == ((1, 1), (50, 50), (50, 50))


def test_align_chunks_single_band_reduction_is_memory_bounded():
    # Reduction (out_bands <= nbands): single band-chunk and each block
    # stays within array.chunk-size (equivalent to (nbands,auto,auto)).
    r = make_raster(content="zeros", shape=(8, 4000, 4000), dtype=np.float32)
    r = r.chunk((8, 4000, 4000))
    chunks = _align_chunks_single_band([r], out_bands=1)[0].data.chunks
    assert len(chunks[0]) == 1 and chunks[0][0] == 8  # single band-chunk
    block_bytes = chunks[0][0] * chunks[1][0] * chunks[2][0] * 4
    assert block_bytes <= parse_bytes(dask.config.get("array.chunk-size"))


def test_align_chunks_single_band_expansion_bounds_output_block():
    # Expansion (out_bands > nbands): the tile is shrunk so the OUTPUT
    # block (out_bands x tile) also fits array.chunk-size.
    r = make_raster(content="zeros", shape=(1, 8000, 8000), dtype=np.float32)
    r = r.chunk((1, 8000, 8000))
    out_bands = 64
    chunks = _align_chunks_single_band([r], out_bands=out_bands)[0].data.chunks
    out_block_bytes = out_bands * chunks[1][0] * chunks[2][0] * 4
    assert out_block_bytes <= parse_bytes(dask.config.get("array.chunk-size"))


def test_map_blocks_out_bands_null_value_inherits_single_input():
    # Single input, dtype unchanged -> inherits the input's null value.
    r = make_raster(
        content="arange", shape=(3, 100, 100), dtype=np.float32, null=-1.0
    )

    def collapse(d):
        return d.max(axis=0, keepdims=True)

    out = map_blocks(collapse, r, out_bands=1)
    assert out.null_value == np.float32(-1.0)


def test_map_blocks_out_bands_null_value_default_on_dtype_change():
    # dtype change -> dtype-appropriate default null value.
    r = make_raster(content="arange", shape=(3, 100, 100), dtype=np.float32)

    def collapse_to_int(d):
        return d.argmax(axis=0, keepdims=True).astype(np.int16)

    out = map_blocks(collapse_to_int, r, out_bands=1, dtype=np.int16)
    assert out.dtype == np.int16
    assert out.null_value == get_default_null_value(np.int16)


def test_geo_map_blocks_out_bands_expansion():
    # geo variant: gbi describes the INPUT band range; output has
    # out_bands bands.
    r = make_raster(content="arange", shape=(1, 100, 100), dtype=np.float32)
    seen = []

    def gexpand(xda, *, geo_block_info=None):
        if geo_block_info is not None:
            seen.append(geo_block_info.shape[0])  # input band count
        return np.concatenate([xda.data, xda.data * 2], axis=0)

    out = geo_map_blocks(gexpand, r, out_bands=2)
    assert out.nbands == 2
    np.testing.assert_array_equal(np.asarray(out.band), [1, 2])
    assert out.data.compute().shape == (2, 100, 100)
    # gbi described the single input band on every real block.
    assert seen and all(n == 1 for n in seen)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "3", 1.0])
def test_map_blocks_out_bands_validation(bad):
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="out_bands"):
        map_blocks(lambda d: d, r, out_bands=bad)


@pytest.mark.parametrize(
    "fn",
    [
        lambda r, ob: map_blocks(
            lambda d: np.concatenate([d, d, d], axis=0), r, out_bands=ob
        ),
        lambda r, ob: geo_map_blocks(
            lambda xda: np.concatenate([xda.data] * 3, axis=0),
            r,
            out_bands=ob,
        ),
    ],
)
def test_out_bands_wrong_count_func_raises(fn):
    # func returns 3 bands but out_bands=2: the wrapper guard catches
    # dask's silent shape mismatch at compute time.
    r = make_raster(content="arange", shape=(1, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="out_bands"):
        fn(r, 2).data.compute()


def test_out_bands_correct_count_3_to_2_passes():
    # The passing 3 -> 2 counterpart to the wrong-count test.
    r = make_raster(content="arange", shape=(3, 100, 100), dtype=np.float32)

    def three_to_two(d):
        return np.stack([d[0] + d[1], d[2]], axis=0)

    out = map_blocks(three_to_two, r, out_bands=2, dtype=np.float32)
    assert out.nbands == 2
    src = r.data.compute()
    res = out.data.compute()
    np.testing.assert_allclose(res[0], src[0] + src[1])
    np.testing.assert_allclose(res[1], src[2])


@pytest.mark.parametrize(
    "fn",
    [
        lambda r: map_overlap(lambda d: d, r, depth=1, out_bands=2),
        lambda r: geo_map_overlap(
            lambda xda, **k: xda, r, depth=1, out_bands=2
        ),
    ],
)
def test_overlap_variants_reject_out_bands(fn):
    r = make_raster(shape=(1, 100, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="out_bands"):
        fn(r)


def test_map_blocks_out_bands_none_is_shape_preserving():
    # Regression guard: omitting out_bands (and out_bands=None) keep the
    # existing shape-preserving behavior.
    r = make_raster(
        content="arange",
        shape=(2, 100, 100),
        dtype=np.float32,
        chunksize=(1, 50, 50),
    )
    out_default = map_blocks(lambda d: d * 2, r)
    out_none = map_blocks(lambda d: d * 2, r, out_bands=None)
    assert out_default.shape == r.shape == out_none.shape
    assert out_default.data.chunks == r.data.chunks
    np.testing.assert_allclose(
        out_default.data.compute(), r.data.compute() * 2
    )
