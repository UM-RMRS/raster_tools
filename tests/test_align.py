import numpy as np
import pytest
import rasterio as rio
from affine import Affine
from odc.geo.geobox import GeoBox

import raster_tools as rts
from raster_tools import align
from raster_tools.dtypes import F64, I16, U8
from raster_tools.masking import get_default_null_value
from raster_tools.warp import SUPPORTED_RESAMPLE_METHODS
from tests import testdata
from tests.utils import assert_valid_raster, make_raster


def _grid_raster(
    x0, y0, shape=(4, 4), res=1, crs="EPSG:3857", null=None, dtype=F64
):
    """Raster with values 1..N whose upper-left corner is at (x0, y0)."""
    data = np.arange(1, np.prod(shape) + 1).reshape(shape).astype(dtype)
    rs = make_raster(data, affine=Affine(res, 0, x0, 0, -res, y0), crs=crs)
    if null is not None:
        rs = rs.set_null_value(null)
    return rs


def _assert_on_common_grid(rasters):
    ref = rasters[0]
    for r in rasters:
        assert_valid_raster(r)
        assert r.crs == ref.crs
        assert r.affine == ref.affine
        assert np.array_equal(r.x, ref.x)
        assert np.array_equal(r.y, ref.y)


# -- Shared lattice: no-op, cut, and pad -------------------------------------


def test_identical_grids_are_new_wrappers():
    a = _grid_raster(0, 4)
    b = _grid_raster(0, 4, null=3)
    out = align([a, b])
    assert isinstance(out, tuple)
    assert len(out) == 2
    for result, src in zip(out, (a, b), strict=True):
        assert result is not src
        assert result._ds is src._ds
    _assert_on_common_grid(out)


@pytest.mark.parametrize("null", [None, 6])
def test_inner_join_cuts_to_overlap(null):
    # b is shifted one cell right and one cell down from a
    a = _grid_raster(0, 4, null=null)
    b = _grid_raster(1, 3)
    out_a, out_b = align([a, b])
    _assert_on_common_grid([out_a, out_b])
    assert out_a.shape == (1, 3, 3)
    assert out_a.bounds == (1.0, 0.0, 4.0, 3.0)
    assert np.array_equal(out_a.to_numpy(), a.to_numpy()[:, 1:, 1:])
    assert np.array_equal(out_b.to_numpy(), b.to_numpy()[:, :3, :3])
    assert np.array_equal(out_a.mask.compute(), a.mask.compute()[:, 1:, 1:])
    # Cutting adds no null cells, so null values are left alone
    assert out_a.null_value == a.null_value
    assert out_b.null_value is None


def test_outer_join_pads_with_null():
    a = _grid_raster(0, 4, null=6)
    b = _grid_raster(1, 3)
    out_a, out_b = align([a, b], join="outer")
    _assert_on_common_grid([out_a, out_b])
    assert out_a.shape == (1, 5, 5)
    assert out_a.bounds == (0.0, -1.0, 5.0, 4.0)

    # Masked inputs keep their null value
    assert out_a.null_value == 6
    mask_a = np.ones((1, 5, 5), dtype=bool)
    mask_a[:, :4, :4] = a.mask.compute()
    assert np.array_equal(out_a.mask.compute(), mask_a)
    data_a = out_a.to_numpy()
    assert np.array_equal(data_a[:, :4, :4], a.to_numpy())
    assert (data_a[mask_a] == 6).all()

    # Unmasked inputs get the default null value for their dtype
    nv = get_default_null_value(b.dtype)
    assert out_b.null_value == nv
    mask_b = np.ones((1, 5, 5), dtype=bool)
    mask_b[:, 1:, 1:] = False
    assert np.array_equal(out_b.mask.compute(), mask_b)
    data_b = out_b.to_numpy()
    assert np.array_equal(data_b[:, 1:, 1:], b.to_numpy())
    assert (data_b[mask_b] == nv).all()


def test_outer_join_leaves_covering_raster_unchanged():
    big = _grid_raster(0, 4)
    small = _grid_raster(1, 3, shape=(2, 2))
    out_big, out_small = align([small, big], join="outer")[::-1]
    assert out_big._ds is big._ds
    _assert_on_common_grid([out_big, out_small])
    expected = np.ones((1, 4, 4), dtype=bool)
    expected[:, 1:3, 1:3] = False
    assert np.array_equal(out_small.mask.compute(), expected)


def test_bands_and_dtype_are_kept():
    a = _grid_raster(0, 4, dtype=U8)
    data = np.arange(2 * 4 * 4, dtype=I16).reshape((2, 4, 4))
    b = make_raster(data, affine=Affine(1, 0, 2, 0, -1, 4)).set_null_value(-1)
    out_a, out_b = align([a, b], join="outer")
    assert (out_a.nbands, out_a.dtype) == (1, np.dtype(U8))
    assert (out_b.nbands, out_b.dtype) == (2, np.dtype(I16))
    assert out_a.null_value == get_default_null_value(U8)
    assert out_b.null_value == -1
    assert np.array_equal(out_b.to_numpy()[:, :, 2:], data)


def test_inner_join_of_touching_rasters_raises():
    a = _grid_raster(0, 4)
    b = _grid_raster(4, 4)
    with pytest.raises(ValueError, match="intersection.*empty"):
        align([a, b])
    out = align([a, b], join="outer")
    assert out[0].shape == (1, 4, 8)


def test_raster_outside_dst_grid_is_all_null():
    a = _grid_raster(0, 4)
    far = _grid_raster(100, 4)
    (out,) = align([far], dst_grid=a.geobox)
    assert out.shape == a.shape
    assert out.mask.compute().all()
    assert out.null_value == get_default_null_value(far.dtype)


def test_align_does_not_resample_on_shared_lattice(monkeypatch):
    calls = []
    original = rts.warp.reproject

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr("raster_tools._align.reproject", spy)
    a = _grid_raster(0, 4)
    b = _grid_raster(2, 1, null=0)
    align([a, b], join="outer")
    align([a, b], join="inner")
    align([b], dst_grid=a)
    assert not calls


def _masked_float_source():
    data = np.arange(1, 37, dtype=F64).reshape((6, 6))
    mask = np.zeros_like(data, dtype=bool)
    mask[2, 3] = mask[4, 1] = True
    return make_raster(
        data,
        mask=mask,
        affine=Affine(10, 0, 1000, 0, -10, 2000),
        crs="EPSG:5070",
        null=-1.0,
    )


def _unmasked_int_source():
    # Unmasked, but holding the default null value in a cell that survives
    # the cut
    data = np.arange(1, 37, dtype=I16).reshape((6, 6))
    data[1, 3] = get_default_null_value(I16)
    src = make_raster(
        data, affine=Affine(10, 0, 1000, 0, -10, 2000), crs="EPSG:5070"
    )
    assert src.null_value is None
    return src


@pytest.mark.parametrize(
    "make_src", [_masked_float_source, _unmasked_int_source]
)
@pytest.mark.parametrize("method", sorted(SUPPORTED_RESAMPLE_METHODS))
def test_cut_and_pad_matches_reproject(method, make_src):
    # On a whole-cell offset, cutting and padding must give exactly what
    # reprojecting would.
    src = make_src()
    # Overlaps src on its last 4 columns and first 4 rows and extends past
    # it, so both cutting and padding are needed.
    dst = make_raster(
        np.zeros((6, 6)),
        affine=Affine(10, 0, 1020, 0, -10, 2020),
        crs="EPSG:5070",
    ).geobox
    (aligned,) = align([src], dst_grid=dst, resampling_method=method)
    reprojected = rts.reproject(src, dst, resample_method=method)
    assert aligned.null_value == reprojected.null_value
    mask = aligned.mask.compute()
    data = aligned.to_numpy()
    expected_mask = reprojected.mask.compute()
    expected_data = reprojected.to_numpy()
    if src.null_value is None:
        # src cell (1, 3) holds the null value the output is given and
        # lands on cell (3, 1). Cutting masks it. Reprojecting nudges its
        # value off the null value instead, so it stays valid.
        held = np.zeros_like(mask)
        held[0, 3, 1] = True
        assert mask[held].all()
        assert not expected_mask[held].any()
        expected_mask |= held
        expected_data[held] = aligned.null_value
        # Any cell holding the null value is masked
        assert mask[data == aligned.null_value].all()
    assert np.array_equal(mask, expected_mask)
    assert np.array_equal(data, expected_data)
    assert np.array_equal(aligned.x, reprojected.x)
    assert np.array_equal(aligned.y, reprojected.y)


# -- Length-1 axes ------------------------------------------------------------


def test_one_row_raster_from_coordinates_has_square_cells():
    a = _grid_raster(0, 40, res=10)
    row = make_raster(
        np.arange(1, 5, dtype=F64).reshape((1, 4)), x=a.x, y=a.y[2:3]
    )
    out_a, out_row = align([a, row])
    _assert_on_common_grid([out_a, out_row])
    assert out_a.shape == (1, 1, 4)
    assert np.array_equal(out_a.to_numpy(), a.to_numpy()[:, 2:3])
    assert np.array_equal(out_row.to_numpy(), row.to_numpy())
    assert out_a.resolution == (10.0, -10.0)
    assert out_row.resolution == (10.0, -10.0)

    out_a, out_row = align([a, row], join="outer")
    _assert_on_common_grid([out_a, out_row])
    assert out_row.shape == (1, 4, 4)
    expected = np.ones((1, 4, 4), dtype=bool)
    expected[:, 2] = False
    assert np.array_equal(out_row.mask.compute(), expected)
    assert np.array_equal(out_row.to_numpy()[:, 2], row.to_numpy()[:, 0])


def test_one_column_raster_placed_by_its_coordinates():
    a = _grid_raster(0, 40, res=10)
    col = make_raster(
        np.arange(1, 5, dtype=F64).reshape((4, 1)), x=np.array([55.0]), y=a.y
    )
    out_a, out_col = align([a, col], join="outer")
    _assert_on_common_grid([out_a, out_col])
    assert out_a.shape == (1, 4, 6)
    assert np.array_equal(out_col.to_numpy()[:, :, 5], col.to_numpy()[:, :, 0])
    assert out_col.mask.compute()[:, :, :5].all()


def test_one_row_raster_alongside_reprojected_input():
    src = testdata.raster.dem_small
    row = rts.data_to_raster(
        src.data[:, 40:41],
        x=src.x,
        y=src.y[40:41],
        crs=src.crs,
        nv=src.null_value,
    )
    other = src.reproject(4326)
    out_src, out_row, out_other = align([src, row, other])
    _assert_on_common_grid([out_src, out_row, out_other])
    assert np.array_equal(out_row.y, row.y)
    assert np.array_equal(out_row.to_numpy(), row.to_numpy())
    assert np.array_equal(out_src.to_numpy(), src.to_numpy()[:, 40:41])


def _write_tif(path, data, transform, crs):
    data = np.asarray(data, dtype="float64")
    with rio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)
    return rts.Raster(str(path))


@pytest.mark.parametrize("combine", ["align", "stack_bands", "mosaic"])
def test_one_cell_raster_keeps_its_cell_size_on_dst_grid(combine, tmp_path):
    # A 3x3 cell whose center falls on the destination lattice must still
    # be reprojected, not placed as a single destination cell. The cell
    # size of a 1x1 raster is only recorded in a file's transform.
    cell = _write_tif(
        tmp_path / "cell.tif",
        np.full((1, 1, 1), 7.0),
        Affine(3, 0, 0, 0, -3, 4),
        "EPSG:32611",
    )
    dst = make_raster(
        np.zeros((4, 4)), affine=Affine(1, 0, 0, 0, -1, 4), crs="EPSG:32611"
    ).geobox
    if combine == "align":
        (out,) = align([cell], dst_grid=dst)
    else:
        out = getattr(rts, combine)([cell], dst_grid=dst)
    expected = rts.reproject(cell, dst)
    assert out.geobox == dst
    assert np.array_equal(out.mask.compute(), expected.mask.compute())
    assert np.array_equal(out.to_numpy(), expected.to_numpy(), equal_nan=True)
    assert not out.mask.compute()[0, :3, :3].any()


def _row_slice_pair():
    a = rts.data_to_raster(
        np.arange(100).reshape((1, 10, 10)),
        x=np.arange(10) * 30 + 15,
        y=np.arange(10) * -30 + 285,
        crs=5070,
    )
    return a, rts.Raster(a.xdata.isel(y=slice(3, 4)))


@pytest.mark.parametrize("method", ["nearest", "average", "max", "bilinear"])
def test_row_slice_lands_on_its_parent_row(method):
    a, one = _row_slice_pair()
    stacked = rts.stack_bands([a, one], dst_grid=a, resampling_method=method)
    (aligned,) = align([one], dst_grid=a, resampling_method=method)
    outputs = [
        (stacked.to_numpy()[1], stacked.mask.compute()[1]),
        (aligned.to_numpy()[0], aligned.mask.compute()[0]),
    ]
    for data, mask in outputs:
        assert (~mask).sum() == 10
        assert not mask[3].any()
        assert np.array_equal(data[3], a.to_numpy()[0, 3])


def _row_and_square():
    x = np.array([5.0, 15, 25, 35])
    y = x[::-1]
    a = rts.data_to_raster(
        np.arange(16).reshape((1, 4, 4)), x=x, y=y, crs=5070
    )
    row = rts.data_to_raster(np.full((1, 1, 4), 7), x=x, y=y[2:3], crs=5070)
    return a, row


def _assert_affine_matches_coords(raster):
    xres, yres = raster.affine.a, raster.affine.e
    expected = Affine(
        xres, 0, raster.x[0] - xres / 2, 0, yres, raster.y[0] - yres / 2
    )
    assert raster.affine == expected
    assert raster.geobox.affine == expected
    assert raster._ds.rio.transform() == expected


@pytest.mark.parametrize("row_first", [True, False])
def test_one_row_raster_in_either_input_position(row_first):
    a, row = _row_and_square()
    order = [row, a] if row_first else [a, row]
    ia = 1 if row_first else 0
    expected_row = a.to_numpy()[:, 2:3]

    out = align(order)
    _assert_on_common_grid(out)
    assert np.array_equal(out[ia].to_numpy(), expected_row)
    assert np.array_equal(out[1 - ia].to_numpy(), row.to_numpy())
    for r in out:
        assert r.affine == Affine(10, 0, 0, 0, -10, 20)
        _assert_affine_matches_coords(r)

    stacked = rts.stack_bands(order)
    assert stacked.shape == (2, 1, 4)
    assert np.array_equal(stacked.to_numpy()[ia], expected_row[0])
    _assert_affine_matches_coords(stacked)

    expected = a.to_numpy()
    if row_first:
        expected[0, 2] = 7
    mosaicked = rts.mosaic(order, "first")
    assert mosaicked.shape == (1, 4, 4)
    assert np.array_equal(mosaicked.to_numpy(), expected)
    _assert_affine_matches_coords(mosaicked)


def test_one_row_file_keeps_its_cell_size_next_to_finer_raster(tmp_path):
    # The file's 3-unit cells cover the top 3 rows of a's 1-unit cells
    r = _write_tif(
        tmp_path / "row.tif",
        np.arange(1, 6).reshape((1, 1, 5)),
        Affine(3, 0, 0, 0, -3, 30),
        "EPSG:5070",
    )
    a = rts.data_to_raster(
        np.arange(450.0).reshape((1, 30, 15)),
        affine=Affine(1, 0, 0, 0, -1, 30),
        crs=5070,
    )
    expected = rts.reproject(r, a.geobox)
    assert not expected.mask.compute()[0, :3].any()
    assert expected.mask.compute()[0, 3:].all()

    out_a, out_r = align([a, r], join="outer")
    assert out_r.geobox == a.geobox
    assert np.array_equal(out_r.mask.compute(), expected.mask.compute())
    assert np.array_equal(
        out_r.to_numpy(), expected.to_numpy(), equal_nan=True
    )

    stacked = rts.stack_bands([a, r], join="outer")
    assert stacked.geobox == a.geobox
    assert np.array_equal(
        stacked.mask.compute()[1], expected.mask.compute()[0]
    )
    assert np.array_equal(
        stacked.to_numpy()[1], expected.to_numpy()[0], equal_nan=True
    )

    out_a, out_r = align([a, r])
    _assert_on_common_grid([out_a, out_r])
    assert out_a.shape == (1, 3, 15)
    assert np.array_equal(out_a.to_numpy(), a.to_numpy()[:, :3])
    assert not out_r.mask.compute().any()


def test_one_row_file_next_to_non_square_cells(tmp_path):
    # b shares the file's 3-unit x lattice but has 1-unit rows, so the
    # file's single 3-unit row spans 3 of b's rows.
    r = _write_tif(
        tmp_path / "row.tif",
        np.arange(1, 6).reshape((1, 1, 5)),
        Affine(3, 0, 0, 0, -3, 30),
        "EPSG:5070",
    )
    b = rts.data_to_raster(
        np.arange(150.0).reshape((1, 30, 5)),
        affine=Affine(3, 0, 0, 0, -1, 30),
        crs=5070,
    )
    out_b, out_r = align([b, r])
    _assert_on_common_grid([out_b, out_r])
    assert out_b.affine == Affine(3, 0, 0, 0, -1, 30)
    assert out_b.shape == (1, 3, 5)
    assert np.array_equal(out_b.to_numpy(), b.to_numpy()[:, :3])
    assert np.array_equal(
        out_r.to_numpy(), np.broadcast_to(r.to_numpy(), (1, 3, 5))
    )
    assert rts.stack_bands([b, r]).shape == (2, 3, 5)


# -- CRS handling -------------------------------------------------------------


def test_missing_crs_on_shared_lattice():
    a = _grid_raster(0, 4, crs=None)
    b = _grid_raster(2, 2, crs=None)
    out = align([a, b], join="outer")
    _assert_on_common_grid(out)
    assert out[0].crs is None
    assert out[0].shape == (1, 6, 6)


def test_missing_crs_matches_any_crs():
    a = _grid_raster(0, 4)
    b = _grid_raster(1, 3, crs=None)
    out = align([b, a])
    _assert_on_common_grid(out)
    assert out[0].crs == a.crs


def test_missing_crs_with_dst_crs_takes_dst_crs(monkeypatch):
    calls = []
    original = rts.warp.reproject

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr("raster_tools._align.reproject", spy)
    a = _grid_raster(0, 4, crs=None)
    b = _grid_raster(1, 3, crs=None)
    out_a, out_b = align([a, b], join="outer", dst_crs="EPSG:5070")
    _assert_on_common_grid([out_a, out_b])
    assert out_a.crs == "EPSG:5070"
    assert out_a.affine == Affine(1, 0, 0, 0, -1, 4)
    assert np.array_equal(out_a.to_numpy()[:, :4, :4], a.to_numpy())
    assert np.array_equal(out_b.to_numpy()[:, 1:, 1:], b.to_numpy())
    assert not calls


@pytest.mark.parametrize(
    "other", [{"x0": 0.5, "y0": 4}, {"x0": 0, "y0": 4, "res": 2}]
)
def test_missing_crs_needing_reprojection_raises(other):
    a = _grid_raster(0, 4, crs=None)
    b = _grid_raster(**other, crs=None)
    with pytest.raises(ValueError, match="requires a CRS"):
        align([a, b])
    with pytest.raises(ValueError, match="requires a CRS"):
        align([b], dst_grid=_grid_raster(0, 4))


def test_dst_grid_without_crs_takes_raster_crs():
    a = _grid_raster(0, 4, crs="EPSG:5070")
    b = _grid_raster(1, 4, crs="EPSG:5070")
    dst = GeoBox(a.geobox.shape, a.geobox.affine, None)
    out = align([a, b], dst_grid=dst)
    _assert_on_common_grid(out)
    assert all(r.crs == a.crs for r in out)
    assert rts.stack_bands([a, b], dst_grid=dst).crs == a.crs


@pytest.mark.parametrize("combine", ["align", "stack_bands", "mosaic"])
@pytest.mark.parametrize("shape", ["row", "column"])
def test_one_cell_wide_raster_without_crs_as_dst_grid(combine, shape):
    x = np.array([5.0, 15, 25, 35])
    a = rts.data_to_raster(np.arange(16).reshape(1, 4, 4), x=x, y=x[::-1])
    if shape == "row":
        dst = rts.data_to_raster(np.full((1, 1, 4), 7), x=x, y=x[2:3])
        expected = [[[4, 5, 6, 7]]]
    else:
        dst = rts.data_to_raster(np.full((1, 4, 1), 7), x=x[1:2], y=x[::-1])
        expected = [[[1], [5], [9], [13]]]
    if combine == "align":
        (out,) = align([a], dst_grid=dst)
    elif combine == "stack_bands":
        out = rts.stack_bands([a], dst_grid=dst)
    else:
        out = rts.mosaic([a], dst_grid=dst)
    assert out.crs is None
    assert out.affine == dst.affine
    np.testing.assert_array_equal(out.values, expected)
    assert not out.mask.compute().any()


def test_dst_grid_without_crs_and_mixed_raster_crs_raises():
    a = _grid_raster(0, 4, crs="EPSG:5070")
    b = _grid_raster(0, 4, crs="EPSG:3857")
    dst = GeoBox(a.geobox.shape, a.geobox.affine, None)
    with pytest.raises(ValueError, match="different CRSs"):
        align([a, b], dst_grid=dst)


def test_dst_grid_without_crs_needing_reprojection_raises():
    a = _grid_raster(0.5, 4, crs="EPSG:5070")
    dst = GeoBox((4, 4), Affine(1, 0, 0, 0, -1, 4), None)
    with pytest.raises(ValueError, match="requires a CRS"):
        align([a], dst_grid=dst)


def test_different_crs_is_reprojected():
    src = testdata.raster.dem_small
    other = src.reproject(4326)
    out_src, out_other = align([src, other])
    _assert_on_common_grid([out_src, out_other])
    assert out_src.crs == src.crs
    assert np.allclose(abs(out_src.resolution[0]), abs(src.resolution[0]))
    # src shares the destination lattice, so its values are not resampled
    xmin, _, _, ymax = out_src.bounds
    col = int(round((xmin - src.bounds[0]) / src.resolution[0]))
    row = int(round((src.bounds[3] - ymax) / src.resolution[1]))
    ny, nx = out_src.shape[1:]
    assert np.array_equal(
        out_src.to_numpy(), src.to_numpy()[:, row : row + ny, col : col + nx]
    )


def test_non_whole_cell_offset_is_reprojected():
    a = _grid_raster(0, 4)
    b = _grid_raster(0.5, 4)
    out = align([a, b])
    _assert_on_common_grid(out)
    assert out[0].shape == out[1].shape

    (out,) = align([b], dst_grid=a)
    assert out.geobox == a.geobox
    # Reprojection can leave cells uncovered, so the unmasked input gets a
    # null value
    assert out.null_value == get_default_null_value(b.dtype)


def test_outputs_share_transform_when_cell_sizes_differ_by_float_noise():
    y = 0.95 - np.arange(10) * 0.1
    a = rts.data_to_raster(
        np.ones((1, 10, 100)), x=0.05 + np.arange(100) * 0.1, y=y, crs=5070
    )
    b = rts.data_to_raster(
        np.ones((1, 10, 50)),
        x=np.linspace(1.05, 5.95, 50) * 1.0000000001,
        y=y,
        crs=5070,
    )
    out = align([a, b])
    _assert_on_common_grid(out)
    assert out[0].geobox == out[1].geobox
    transforms = [r._ds.rio.transform() for r in out]
    assert transforms[0] == transforms[1]


def test_different_resolution_uses_first_raster_resolution():
    a = _grid_raster(0, 8, shape=(8, 8))
    b = _grid_raster(0, 8, res=2)
    out = align([a, b])
    _assert_on_common_grid(out)
    assert out[1].resolution == (1.0, -1.0)
    assert out[1].shape == (1, 8, 8)


# -- Destination grid arguments ----------------------------------------------


@pytest.mark.parametrize("form", ["geobox", "raster", "path"])
def test_dst_grid_forms(form, tmp_path):
    ref = _grid_raster(1, 3, shape=(2, 2))
    if form == "geobox":
        dst_grid = ref.geobox
    elif form == "raster":
        dst_grid = ref
    else:
        dst_grid = str(tmp_path / "ref.tif")
        ref.save(dst_grid)
    a = _grid_raster(0, 4, null=0)
    (out,) = align([a], dst_grid=dst_grid)
    assert out.geobox == ref.geobox
    assert np.array_equal(out.to_numpy(), a.to_numpy()[:, 1:3, 1:3])


def test_dst_grid_ignores_join():
    a = _grid_raster(0, 4)
    b = _grid_raster(1, 3)
    dst = _grid_raster(-1, 5, shape=(6, 6))
    for join in ("inner", "outer"):
        out = align([a, b], join=join, dst_grid=dst)
        assert all(r.geobox == dst.geobox for r in out)


def test_dst_grid_with_matching_dst_crs():
    a = _grid_raster(0, 4)
    (out,) = align([a], dst_grid=a.geobox, dst_crs=a.crs)
    assert out._ds is a._ds


def test_dst_grid_with_conflicting_dst_crs_raises():
    a = _grid_raster(0, 4)
    with pytest.raises(ValueError, match="dst_crs does not match"):
        align([a], dst_grid=a.geobox, dst_crs=4326)


def test_dst_grid_with_resolution_raises():
    a = _grid_raster(0, 4)
    with pytest.raises(ValueError, match="resolution cannot"):
        align([a], dst_grid=a.geobox, resolution=2)


@pytest.mark.parametrize(
    "affine", [Affine(1, 0.5, 0, 0, -1, 4), Affine(1, 0, 0, 0.5, -1, 4)]
)
def test_rotated_or_sheared_dst_grid_raises(affine):
    dst = GeoBox((4, 4), affine, "EPSG:3857")
    with pytest.raises(ValueError, match="rotated or sheared"):
        align([_grid_raster(0, 4)], dst_grid=dst)
    with pytest.raises(ValueError, match="rotated or sheared"):
        rts.stack_bands([_grid_raster(0, 4)], dst_grid=dst)


@pytest.mark.parametrize("bad", [42, {}, 3.14, object()])
def test_dst_grid_bad_type_raises(bad):
    with pytest.raises(TypeError, match="dst_grid"):
        align([_grid_raster(0, 4)], dst_grid=bad)


def test_dst_crs_reprojects():
    src = testdata.raster.dem_small
    out = align([src, src], dst_crs=4326)
    _assert_on_common_grid(out)
    assert out[0].crs == 4326


def test_resolution_changes_grid():
    a = _grid_raster(0, 8, shape=(8, 8))
    b = _grid_raster(2, 6, shape=(4, 4))
    out = align([a, b], join="outer", resolution=2)
    _assert_on_common_grid(out)
    assert out[0].resolution == (2.0, -2.0)
    assert out[0].shape == (1, 4, 4)


def test_matching_resolution_keeps_lattice():
    a = _grid_raster(0, 4)
    (out,) = align([a], resolution=1)
    assert out._ds is a._ds


# -- Argument validation ------------------------------------------------------


def test_accepts_paths(tmp_path):
    a = _grid_raster(0, 4, null=0)
    path = str(tmp_path / "a.tif")
    a.save(path)
    (out,) = align([path])
    assert np.array_equal(out.to_numpy(), a.to_numpy())


def test_empty_input_raises():
    with pytest.raises(ValueError, match="No rasters"):
        align([])


@pytest.mark.parametrize("rasters", ["a.tif", _grid_raster(0, 4)])
def test_single_raster_instead_of_list_raises(rasters):
    with pytest.raises(TypeError, match="list of rasters"):
        align(rasters)


@pytest.mark.parametrize("join", ["left", "union", None])
def test_invalid_join_raises(join):
    with pytest.raises(ValueError, match="join"):
        align([_grid_raster(0, 4)], join=join)


@pytest.mark.parametrize("method", ["foo", "NEAREST", "gauss"])
def test_invalid_resampling_method_raises(method):
    with pytest.raises(ValueError, match="resampling"):
        align([_grid_raster(0, 4)], resampling_method=method)


def test_is_exported():
    assert rts.align is align
    assert "align" in rts.__all__


def test_aligned_rasters_combine_cell_by_cell():
    a = _grid_raster(0, 4)
    b = _grid_raster(0.5, 4)
    with pytest.raises(ValueError, match=r"align\(\[raster, other\]\)"):
        a + b
    a2, b2 = align([a, b])
    result = a2 + b2
    assert result.shape == a2.shape
