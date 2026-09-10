import os
import struct

import numpy as np
import pytest
import rasterio
from affine import Affine

import raster_tools as rts
from raster_tools._mosaic import mosaic
from raster_tools.batch import _batch_parse_save, _BatchScripParserState
from raster_tools.dtypes import U8, U16
from raster_tools.exceptions import RasterIOError
from raster_tools.io import (
    _auto_overview_factors,
    _supports_native_int64,
    normalize_color_table,
    read_color_table,
    write_raster,
)
from raster_tools.masking import get_default_null_value
from raster_tools.raster import _ext_for_driver
from tests.utils import make_raster


@pytest.fixture
def out_tif(tmp_path):
    return str(tmp_path / "out.tif")


def _open(path):
    return rasterio.open(path)


# --- Round-trip correctness -------------------------------------------------


@pytest.mark.parametrize(
    "dtype", ["float32", "float64", "int16", "uint8", "int32"]
)
def test_save_roundtrip_basic(out_tif, dtype):
    src = make_raster("arange", dtype=dtype, shape=(1, 8, 8))
    src.save(out_tif)
    reloaded = rts.Raster(out_tif)
    assert reloaded.dtype == np.dtype(dtype)
    assert reloaded.crs == src.crs
    assert np.array_equal(reloaded.to_numpy(), src.to_numpy())


def test_save_roundtrip_multiband(out_tif):
    src = make_raster("arange", dtype="float32", shape=(3, 8, 8))
    src.save(out_tif)
    reloaded = rts.Raster(out_tif)
    assert reloaded.nbands == 3
    assert np.array_equal(reloaded.to_numpy(), src.to_numpy())


def test_save_roundtrip_bool(out_tif):
    src = make_raster("ones", dtype="float32", shape=(1, 6, 6)) > 0
    assert src.dtype == np.dtype(bool)
    src.save(out_tif)
    with _open(out_tif) as ds:
        assert ds.dtypes[0] == "uint8"
        # nbits=1 is recorded in the IMAGE_STRUCTURE namespace
        tags = ds.tags(1, ns="IMAGE_STRUCTURE")
        assert tags.get("NBITS") == "1"
    reloaded = rts.Raster(out_tif)
    assert (reloaded.to_numpy() != 0).all()


def test_save_roundtrip_i64_native(out_tif):
    if not _supports_native_int64("GTiff"):
        pytest.skip("GDAL build lacks native int64 raster support")
    sentinel = get_default_null_value(np.dtype("int64"))
    assert sentinel == -(2**63)
    # A value above 2**53 and the int64 min sentinel cannot survive a
    # float64 detour; the native write must preserve every cell exactly.
    data = np.array([[[2**53 + 1, 5, sentinel, 7]]], dtype="int64")
    src = make_raster(data, dtype="int64")
    src.save(out_tif)
    with _open(out_tif) as ds:
        assert ds.dtypes[0] == "int64"
    reloaded = rts.Raster(out_tif)
    assert reloaded.dtype == np.dtype("int64")
    assert reloaded.to_numpy()[0, 0, 0] == 2**53 + 1
    assert np.array_equal(reloaded.to_numpy(), data)


def test_save_roundtrip_i64_native_preserves_null(out_tif):
    if not _supports_native_int64("GTiff"):
        pytest.skip("GDAL build lacks native int64 raster support")
    # A value above 2**53 alongside a masked cell: the native write keeps
    # both the large value and the null mask, which a float64 cast could
    # not do for the large value.
    data = np.array([[[2**53 + 1, 5, -9999, 7]]], dtype="int64")
    src = make_raster(data, dtype="int64", null=-9999)
    src.save(out_tif)
    with _open(out_tif) as ds:
        assert ds.dtypes[0] == "int64"
        assert ds.nodata == -9999
    reloaded = rts.Raster(out_tif)
    assert reloaded.dtype == np.dtype("int64")
    assert reloaded.null_value == -9999
    assert reloaded.to_numpy()[0, 0, 0] == 2**53 + 1
    assert np.array_equal(reloaded.mask.compute(), src.mask.compute())


def test_save_roundtrip_i64_default_sentinel_null_preserves_mask(out_tif):
    if not _supports_native_int64("GTiff"):
        pytest.skip("GDAL build lacks native int64 raster support")
    # int64's default null (-2**63) cannot survive GeoTIFF's float nodata
    # field, so a native write would silently drop the mask. The writer
    # must fall back to float64 (with a warning) and keep the mask intact.
    sentinel = get_default_null_value(np.dtype("int64"))
    assert sentinel == -(2**63)
    data = np.array([[[10, 5, sentinel, 7]]], dtype="int64")
    src = make_raster(data, dtype="int64", null=sentinel)
    assert np.array_equal(
        src.mask.compute(), np.array([[[False, False, True, False]]])
    )
    with pytest.warns(UserWarning, match="float64"):
        src.save(out_tif)
    with _open(out_tif) as ds:
        assert ds.dtypes[0] == "float64"
    reloaded = rts.Raster(out_tif)
    assert np.array_equal(reloaded.mask.compute(), src.mask.compute())


def test_save_roundtrip_i64_fallback(out_tif, monkeypatch):
    monkeypatch.setattr(
        "raster_tools.io._supports_native_int64", lambda driver: False
    )
    src = make_raster("arange", dtype="int64", shape=(1, 4, 4))
    with pytest.warns(UserWarning, match="int64"):
        src.save(out_tif)
    with _open(out_tif) as ds:
        assert ds.dtypes[0] == "float64"
    reloaded = rts.Raster(out_tif)
    assert np.allclose(reloaded.to_numpy(), src.to_numpy().astype("float64"))


def test_save_roundtrip_u64(out_tif):
    if not rasterio.dtypes.check_dtype("uint64"):
        pytest.skip("GDAL build lacks native uint64 raster support")
    src = make_raster("arange", dtype="uint64", shape=(1, 4, 4))
    src.save(out_tif)
    with _open(out_tif) as ds:
        assert ds.dtypes[0] == "uint64"
    reloaded = rts.Raster(out_tif)
    assert reloaded.dtype == np.dtype("uint64")
    assert np.array_equal(reloaded.to_numpy(), src.to_numpy())


def test_save_roundtrip_with_nulls(out_tif):
    src = make_raster(
        "arange",
        dtype="float32",
        shape=(1, 6, 6),
        null_pattern=np.s_[:, :2, :2],
    )
    src.save(out_tif)
    reloaded = rts.Raster(out_tif)
    assert reloaded.null_value == src.null_value
    assert np.array_equal(reloaded.mask.compute(), src.mask.compute())


# --- Default creation options (TIFF) ----------------------------------------


def test_default_tiled_uncompressed(out_tif):
    make_raster("arange", dtype="float32", shape=(1, 8, 8)).save(out_tif)
    with _open(out_tif) as ds:
        profile = ds.profile
        assert profile["tiled"] is True
        # Default is no compression; GDAL omits the COMPRESS tag.
        assert profile.get("compress") is None
        assert profile.get("blockxsize") is not None
        assert profile.get("blockysize") is not None


def test_compress_lzw_opt_in(out_tif):
    make_raster("arange", dtype="float32", shape=(1, 8, 8)).save(
        out_tif, compress="lzw"
    )
    with _open(out_tif) as ds:
        assert ds.profile["compress"] == "lzw"


def test_compress_deflate_with_level(out_tif):
    make_raster("arange", dtype="float32", shape=(1, 64, 64)).save(
        out_tif, compress="deflate", compress_level=9
    )
    with _open(out_tif) as ds:
        assert ds.profile["compress"] == "deflate"


@pytest.mark.parametrize(
    "blocksize,expected",
    [(128, (128, 128)), ((64, 256), (64, 256))],
)
def test_blocksize_int_and_tuple(out_tif, blocksize, expected):
    make_raster("arange", dtype="float32", shape=(1, 512, 512)).save(
        out_tif, blocksize=blocksize
    )
    with _open(out_tif) as ds:
        assert ds.profile["blockysize"] == expected[0]
        assert ds.profile["blockxsize"] == expected[1]


def test_predictor_horizontal_int(out_tif):
    make_raster("arange", dtype="int16", shape=(1, 64, 64)).save(
        out_tif, compress="deflate", predictor="horizontal"
    )
    with _open(out_tif) as ds:
        # Predictor lives in IMAGE_STRUCTURE tags
        tags = ds.tags(ns="IMAGE_STRUCTURE")
        assert tags.get("PREDICTOR") == "2"


def test_predictor_float(out_tif):
    make_raster("arange", dtype="float32", shape=(1, 64, 64)).save(
        out_tif, compress="deflate", predictor="float"
    )
    with _open(out_tif) as ds:
        tags = ds.tags(ns="IMAGE_STRUCTURE")
        assert tags.get("PREDICTOR") == "3"


def test_predictor_invalid_raises(out_tif):
    with pytest.raises(ValueError, match="predictor"):
        make_raster("arange", dtype="int16", shape=(1, 8, 8)).save(
            out_tif, predictor="bogus"
        )


@pytest.mark.parametrize("predictor", [2, 3])
def test_predictor_int_backward_compat(out_tif, predictor):
    # rasterio-style integer predictor values are accepted unchanged for
    # backward compatibility with workflows that pre-date the logical kwargs.
    dtype = "int16" if predictor == 2 else "float32"
    make_raster("arange", dtype=dtype, shape=(1, 64, 64)).save(
        out_tif, compress="deflate", predictor=predictor
    )
    with _open(out_tif) as ds:
        tags = ds.tags(ns="IMAGE_STRUCTURE")
        assert tags.get("PREDICTOR") == str(predictor)


# --- Null value plumbing ----------------------------------------------------


def test_null_value_kwarg_overrides_raster_null(out_tif):
    src = make_raster("arange", dtype="float32", shape=(1, 6, 6))
    src.save(out_tif, null_value=-9999.0)
    with _open(out_tif) as ds:
        assert ds.nodata == -9999.0
    reloaded = rts.Raster(out_tif)
    assert reloaded.null_value == -9999.0


def test_no_data_value_emits_deprecation_warning_and_works(out_tif):
    src = make_raster("arange", dtype="float32", shape=(1, 6, 6))
    with pytest.warns(DeprecationWarning, match="no_data_value"):
        src.save(out_tif, no_data_value=-1.0)
    with _open(out_tif) as ds:
        assert ds.nodata == -1.0


def test_passing_both_null_and_no_data_raises(out_tif):
    src = make_raster("arange", dtype="float32", shape=(1, 6, 6))
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(TypeError, match="Cannot specify both"),
    ):
        src.save(out_tif, null_value=-1.0, no_data_value=-2.0)


# --- Overviews --------------------------------------------------------------


def test_overviews_explicit_list(out_tif):
    make_raster("arange", dtype="float32", shape=(1, 256, 256)).save(
        out_tif, overviews=[2, 4]
    )
    with _open(out_tif) as ds:
        assert ds.overviews(1) == [2, 4]


def test_overviews_true_builds_auto_chain(out_tif):
    make_raster("arange", dtype="float32", shape=(1, 1024, 1024)).save(
        out_tif, overviews=True
    )
    expected = _auto_overview_factors(1024, 1024)
    with _open(out_tif) as ds:
        assert ds.overviews(1) == expected


@pytest.mark.parametrize("overviews", [None, False])
def test_overviews_skip(out_tif, overviews):
    make_raster("arange", dtype="float32", shape=(1, 256, 256)).save(
        out_tif, overviews=overviews
    )
    with _open(out_tif) as ds:
        assert ds.overviews(1) == []


def test_overview_resampling_tag(out_tif):
    make_raster("arange", dtype="float32", shape=(1, 512, 512)).save(
        out_tif, overviews=[2], overview_resampling="nearest"
    )
    with _open(out_tif) as ds:
        tags = ds.tags(ns="rio_overview")
        assert tags.get("resampling") == "nearest"


# --- Escape hatch -----------------------------------------------------------


def test_gdal_kwargs_with_unmodelled_option(out_tif):
    # 'sparse_ok' is a GTiff option we don't model; should pass through.
    make_raster("arange", dtype="float32", shape=(1, 64, 64)).save(
        out_tif, sparse_ok=True
    )
    # If the file opens cleanly it accepted the option; nothing more to check.
    with _open(out_tif) as ds:
        assert ds.count == 1


# --- COG driver -------------------------------------------------------------


def test_cog_driver_basic_roundtrip(out_tif):
    src = make_raster("arange", dtype="float32", shape=(1, 256, 256))
    src.save(out_tif, driver="COG")
    with _open(out_tif) as ds:
        assert ds.profile["tiled"] is True
        # Default overviews=None means no overviews are built, even for COG.
        assert ds.overviews(1) == []
    reloaded = rts.Raster(out_tif)
    assert np.array_equal(reloaded.to_numpy(), src.to_numpy())


def test_cog_overviews_true_builds_internal(out_tif):
    make_raster("arange", dtype="float32", shape=(1, 1024, 1024)).save(
        out_tif, driver="COG", overviews=True
    )
    with _open(out_tif) as ds:
        assert ds.overviews(1) != []


def test_cog_blocksize(out_tif):
    make_raster("arange", dtype="float32", shape=(1, 512, 512)).save(
        out_tif, driver="COG", blocksize=256
    )
    with _open(out_tif) as ds:
        assert ds.profile["blockxsize"] == 256
        assert ds.profile["blockysize"] == 256


def test_cog_non_square_blocksize_raises(out_tif):
    with pytest.raises(ValueError, match="square blocksize"):
        make_raster("arange", dtype="float32", shape=(1, 256, 256)).save(
            out_tif, driver="COG", blocksize=(128, 256)
        )


def test_cog_compress_lzw(out_tif):
    make_raster("arange", dtype="float32", shape=(1, 256, 256)).save(
        out_tif, driver="COG", compress="lzw"
    )
    with _open(out_tif) as ds:
        assert ds.profile["compress"] == "lzw"


def test_cog_compress_deflate_with_level(out_tif):
    make_raster("arange", dtype="float32", shape=(1, 256, 256)).save(
        out_tif, driver="COG", compress="deflate", compress_level=9
    )
    with _open(out_tif) as ds:
        assert ds.profile["compress"] == "deflate"


def test_cog_overviews_disabled(out_tif):
    make_raster("arange", dtype="float32", shape=(1, 1024, 1024)).save(
        out_tif, driver="COG", overviews=False
    )
    with _open(out_tif) as ds:
        assert ds.overviews(1) == []


@pytest.mark.parametrize("predictor", [2, 3])
def test_cog_predictor_int_backward_compat(out_tif, predictor):
    dtype = "int16" if predictor == 2 else "float32"
    make_raster("arange", dtype=dtype, shape=(1, 256, 256)).save(
        out_tif,
        driver="COG",
        compress="deflate",
        predictor=predictor,
    )
    with _open(out_tif) as ds:
        tags = ds.tags(ns="IMAGE_STRUCTURE")
        assert tags.get("PREDICTOR") == str(predictor)


def test_cog_explicit_overview_list_warns(out_tif):
    with pytest.warns(UserWarning, match="COG driver builds overviews"):
        make_raster("arange", dtype="float32", shape=(1, 1024, 1024)).save(
            out_tif, driver="COG", overviews=[2, 4]
        )
    with _open(out_tif) as ds:
        # COG picks its own factors; just confirm overviews exist.
        assert ds.overviews(1) != []


# --- Color tables -----------------------------------------------------------

# TIFF tag numbers and PhotometricInterpretation values used below.
_TAG_PHOTOMETRIC = 262
_PHOTOMETRIC_MINISBLACK = 1
_PHOTOMETRIC_PALETTE = 3

COLOR_TABLE = {
    0: (0, 0, 0),
    1: (34, 139, 34),
    2: (70, 130, 180),
    3: (210, 180, 140),
}


def _tiff_photometric(path):
    """Read a classic TIFF's PhotometricInterpretation tag directly.

    rasterio exposes neither ``dataset.photometric`` nor
    ``profile["photometric"]`` for these files (both come back ``None``), and
    ``colorinterp`` reports ``palette`` even when the tag says min-is-black.
    The tag itself is what decides whether a strict TIFF reader honors the
    palette, so it is read out of the first IFD by hand.
    """
    with open(path, "rb") as fd:
        buf = fd.read()
    byte_order = "<" if buf[:2] == b"II" else ">"
    (ifd_offset,) = struct.unpack(byte_order + "I", buf[4:8])
    (n_entries,) = struct.unpack(
        byte_order + "H", buf[ifd_offset : ifd_offset + 2]
    )
    for i in range(n_entries):
        entry = ifd_offset + 2 + (i * 12)
        (tag,) = struct.unpack(byte_order + "H", buf[entry : entry + 2])
        if tag == _TAG_PHOTOMETRIC:
            # A single SHORT is left justified in the 4 byte value field.
            (value,) = struct.unpack(
                byte_order + "H", buf[entry + 8 : entry + 10]
            )
            return value
    return None


def _indexed_raster(dtype="uint8", shape=(1, 8, 8)):
    """A raster whose only values are the four indices in ``COLOR_TABLE``."""
    data = (np.arange(int(np.prod(shape))).reshape(shape) % 4).astype(dtype)
    return make_raster(data, dtype=dtype)


def test_save_color_table_dict(out_tif):
    src = _indexed_raster()
    src.save(out_tif, color_table=COLOR_TABLE)
    with _open(out_tif) as ds:
        written = ds.colormap(1)
        # GDAL pads the table out to the full index range of the dtype.
        assert len(written) == 256
        for value, (r, g, b) in COLOR_TABLE.items():
            assert written[value] == (r, g, b, 255)
    assert _tiff_photometric(out_tif) == _PHOTOMETRIC_PALETTE
    # The palette must not disturb the cell values.
    assert np.array_equal(rts.Raster(out_tif).to_numpy(), src.to_numpy())


def test_save_color_table_array(out_tif):
    table = np.array([COLOR_TABLE[i] for i in range(4)])
    _indexed_raster().save(out_tif, color_table=table)
    with _open(out_tif) as ds:
        for value, (r, g, b) in COLOR_TABLE.items():
            assert ds.colormap(1)[value] == (r, g, b, 255)


def test_save_color_table_array_with_alpha_column(out_tif):
    table = np.array([(*COLOR_TABLE[i], 255) for i in range(4)])
    _indexed_raster().save(out_tif, color_table=table)
    with _open(out_tif) as ds:
        assert ds.colormap(1)[2] == (70, 130, 180, 255)


def test_save_color_table_from_path(tmp_path):
    source = str(tmp_path / "source.tif")
    target = str(tmp_path / "target.tif")
    _indexed_raster().save(source, color_table=COLOR_TABLE)
    _indexed_raster().save(target, color_table=source)
    with _open(source) as src_ds, _open(target) as target_ds:
        assert src_ds.colormap(1) == target_ds.colormap(1)
    assert _tiff_photometric(target) == _PHOTOMETRIC_PALETTE


def test_save_color_table_from_pathlib_path(tmp_path):
    source = tmp_path / "source.tif"
    target = tmp_path / "target.tif"
    _indexed_raster().save(str(source), color_table=COLOR_TABLE)
    _indexed_raster().save(str(target), color_table=source)
    with _open(str(target)) as ds:
        assert ds.colormap(1)[1] == (34, 139, 34, 255)


def test_save_color_table_uint16(out_tif):
    _indexed_raster(dtype="uint16").save(out_tif, color_table=COLOR_TABLE)
    with _open(out_tif) as ds:
        assert ds.dtypes[0] == "uint16"
        assert len(ds.colormap(1)) == 65536
        assert ds.colormap(1)[1] == (34, 139, 34, 255)
    assert _tiff_photometric(out_tif) == _PHOTOMETRIC_PALETTE


def test_save_color_table_cog(out_tif):
    src = _indexed_raster(shape=(1, 256, 256))
    src.save(out_tif, driver="COG", color_table=COLOR_TABLE)
    with _open(out_tif) as ds:
        assert ds.profile["tiled"] is True
        assert ds.colormap(1)[1] == (34, 139, 34, 255)
    # The COG driver has no palette creation option; the tag has to survive
    # the copy from the staging file.
    assert _tiff_photometric(out_tif) == _PHOTOMETRIC_PALETTE
    assert np.array_equal(rts.Raster(out_tif).to_numpy(), src.to_numpy())


def test_save_color_table_rgba_drops_alpha_with_warning(out_tif):
    with pytest.warns(UserWarning, match="cannot store alpha"):
        _indexed_raster().save(
            out_tif, color_table={0: (0, 0, 0, 0), 1: (1, 2, 3, 7)}
        )
    with _open(out_tif) as ds:
        assert ds.colormap(1)[0] == (0, 0, 0, 255)
        assert ds.colormap(1)[1] == (1, 2, 3, 255)


def test_save_color_table_opaque_alpha_does_not_warn(out_tif, recwarn):
    _indexed_raster().save(out_tif, color_table={1: (1, 2, 3, 255)})
    assert not [w for w in recwarn if "alpha" in str(w.message)]


def test_save_without_color_table_writes_no_palette(out_tif):
    _indexed_raster().save(out_tif)
    with _open(out_tif) as ds, pytest.raises(ValueError):
        ds.colormap(1)
    assert _tiff_photometric(out_tif) == _PHOTOMETRIC_MINISBLACK


def test_save_color_table_bool_keeps_nbits(out_tif):
    src = make_raster("ones", dtype="float32", shape=(1, 6, 6)) > 0
    src.save(out_tif, color_table={0: (0, 0, 0), 1: (255, 255, 255)})
    with _open(out_tif) as ds:
        assert ds.dtypes[0] == "uint8"
        # A palette costs a bool raster nothing: nbits=1 is kept and GDAL
        # sizes the table to the two indices a single bit can hold.
        assert ds.tags(1, ns="IMAGE_STRUCTURE").get("NBITS") == "1"
        assert len(ds.colormap(1)) == 2
        assert ds.colormap(1)[0] == (0, 0, 0, 255)
        assert ds.colormap(1)[1] == (255, 255, 255, 255)
    assert _tiff_photometric(out_tif) == _PHOTOMETRIC_PALETTE


@pytest.mark.parametrize("dtype", ["float32", "float64", "int16", "int32"])
def test_save_color_table_unsupported_dtype_raises(out_tif, dtype):
    with pytest.raises(ValueError, match="only supported for uint8 and"):
        _indexed_raster(dtype=dtype).save(out_tif, color_table=COLOR_TABLE)


def test_save_color_table_multiband_raises(out_tif):
    src = make_raster("arange", dtype="uint8", shape=(3, 8, 8))
    with pytest.raises(ValueError, match="single band"):
        src.save(out_tif, color_table=COLOR_TABLE)


@pytest.mark.parametrize(
    "color_table,match",
    [
        ({}, "was empty"),
        ({300: (1, 2, 3)}, r"integers in the range \[0, 255\]"),
        ({-1: (1, 2, 3)}, r"integers in the range \[0, 255\]"),
        ({1.5: (1, 2, 3)}, r"integers in the range \[0, 255\]"),
        ({1: (1, 2)}, "3 \\(RGB\\) or 4 \\(RGBA\\)"),
        ({1: (1, 2, 3, 4, 5)}, "3 \\(RGB\\) or 4 \\(RGBA\\)"),
        ({1: (1, 2, 999)}, "components must be integers"),
        ({1: (1, 2, -1)}, "components must be integers"),
        ({1: (1.0, 2.0, 3.5)}, "components must be integers"),
        (np.zeros((4, 5)), r"shape \(N, 3\) or \(N, 4\)"),
        (np.zeros(4), r"shape \(N, 3\) or \(N, 4\)"),
    ],
)
def test_save_color_table_invalid_spec_raises(out_tif, color_table, match):
    with pytest.raises(ValueError, match=match):
        _indexed_raster().save(out_tif, color_table=color_table)


def test_save_color_table_source_without_palette_raises(tmp_path):
    source = str(tmp_path / "source.tif")
    _indexed_raster().save(source)
    with pytest.raises(RasterIOError, match="does not have a color table"):
        _indexed_raster().save(str(tmp_path / "out.tif"), color_table=source)


def test_save_color_table_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _indexed_raster().save(
            str(tmp_path / "out.tif"), color_table=str(tmp_path / "nope.tif")
        )


def test_save_color_table_interpolating_overviews_warns(out_tif):
    src = _indexed_raster(shape=(1, 1024, 1024))
    with pytest.warns(UserWarning, match="blends cell values"):
        src.save(out_tif, color_table=COLOR_TABLE, overviews=True)


def test_save_color_table_nearest_overviews_does_not_warn(out_tif, recwarn):
    src = _indexed_raster(shape=(1, 1024, 1024))
    src.save(
        out_tif,
        color_table=COLOR_TABLE,
        overviews=True,
        overview_resampling="nearest",
    )
    assert not [w for w in recwarn if "blends cell values" in str(w.message)]
    with _open(out_tif) as ds:
        assert ds.overviews(1) != []
        assert ds.colormap(1)[1] == (34, 139, 34, 255)


def test_save_color_table_without_overviews_does_not_warn(out_tif, recwarn):
    _indexed_raster().save(out_tif, color_table=COLOR_TABLE)
    assert not [w for w in recwarn if "blends cell values" in str(w.message)]


def test_write_raster_direct_color_table(out_tif):
    src = _indexed_raster()
    write_raster(src.xdata, out_tif, color_table=COLOR_TABLE)
    with _open(out_tif) as ds:
        assert ds.colormap(1)[1] == (34, 139, 34, 255)


def test_write_raster_2d_dataarray_color_table(out_tif):
    # A 2D (y, x) DataArray is a single band. Counting bands off shape[0]
    # would read the row count and reject it.
    src = _indexed_raster(shape=(1, 8, 8))
    write_raster(src.xdata[0], out_tif, color_table=COLOR_TABLE)
    with _open(out_tif) as ds:
        assert ds.count == 1
        assert ds.colormap(1)[1] == (34, 139, 34, 255)


def test_save_color_table_multiband_error_reports_band_count(out_tif):
    src = make_raster("arange", dtype="uint8", shape=(3, 8, 8))
    with pytest.raises(ValueError, match="has 3 bands"):
        src.save(out_tif, color_table=COLOR_TABLE)


def test_save_color_table_from_uint16_source_onto_uint8(tmp_path):
    # GDAL pads a table it reads back out to the source dtype's full index
    # range, so a uint16 source hands over 65536 entries. The ones a uint8
    # raster cannot index are padding and must not be treated as an error.
    source = str(tmp_path / "source16.tif")
    target = str(tmp_path / "target8.tif")
    _indexed_raster(dtype="uint16").save(source, color_table=COLOR_TABLE)
    with _open(source) as ds:
        assert len(ds.colormap(1)) == 65536
    _indexed_raster(dtype="uint8").save(target, color_table=source)
    with _open(target) as ds:
        assert len(ds.colormap(1)) == 256
        for value, (r, g, b) in COLOR_TABLE.items():
            assert ds.colormap(1)[value] == (r, g, b, 255)


def test_save_color_table_png_keeps_alpha(tmp_path, recwarn):
    # A PNG palette stores alpha, so it must not be forced opaque. Every
    # value present needs an entry; PNG palettes are not padded.
    path = str(tmp_path / "out.png")
    color_table = {
        0: (0, 0, 0, 0),
        1: (1, 2, 3, 128),
        2: (4, 5, 6, 255),
        3: (7, 8, 9, 64),
    }
    _indexed_raster().save(path, color_table=color_table)
    with _open(path) as ds:
        assert ds.colormap(1) == color_table
    assert not [w for w in recwarn if "alpha" in str(w.message)]


def test_save_color_table_gtiff_alpha_warning_names_gtiff(out_tif):
    with pytest.warns(
        UserWarning, match="GTiff color table cannot store alpha"
    ):
        _indexed_raster().save(out_tif, color_table={1: (1, 2, 3, 128)})


def test_save_color_table_small_raster_overviews_does_not_warn(
    out_tif, recwarn
):
    # The auto chain is empty at this size, so there are no overviews to be
    # wrong about and nothing to warn on.
    _indexed_raster(shape=(1, 8, 8)).save(
        out_tif, color_table=COLOR_TABLE, overviews=True
    )
    assert not [w for w in recwarn if "blends cell values" in str(w.message)]
    with _open(out_tif) as ds:
        assert ds.overviews(1) == []


def test_save_color_table_explicit_factors_small_raster_warns(out_tif):
    # An explicit factor list is built regardless of size, so it does warn.
    with pytest.warns(UserWarning, match="blends cell values"):
        _indexed_raster(shape=(1, 8, 8)).save(
            out_tif, color_table=COLOR_TABLE, overviews=[2]
        )


def test_save_color_table_uppercase_resampling_warns(out_tif):
    # The COG translator lowercases the method before GDAL sees it, so the
    # check cannot be case sensitive.
    with pytest.warns(UserWarning, match="blends cell values"):
        _indexed_raster(shape=(1, 2048, 2048)).save(
            out_tif,
            driver="COG",
            color_table=COLOR_TABLE,
            overviews=True,
            overview_resampling="AVERAGE",
        )


@pytest.mark.parametrize("color", [5, None, 3.5])
def test_save_color_table_non_sequence_color_raises_value_error(
    out_tif, color
):
    with pytest.raises(ValueError, match="must be a sequence"):
        _indexed_raster().save(out_tif, color_table={1: color})


# --- read_color_table -------------------------------------------------------


def test_read_color_table_returns_written_colors(out_tif):
    _indexed_raster().save(out_tif, color_table=COLOR_TABLE)
    result = read_color_table(out_tif)
    for value, (r, g, b) in COLOR_TABLE.items():
        assert result[value] == (r, g, b, 255)


def test_read_color_table_is_exported_at_top_level():
    assert rts.read_color_table is read_color_table
    assert "read_color_table" in rts.__all__


@pytest.mark.parametrize("dtype,expected", [("uint8", 256), ("uint16", 65536)])
def test_read_color_table_includes_gdal_padding(tmp_path, dtype, expected):
    # GDAL pads the stored table out to the band dtype's full index range.
    # The padding cannot be filtered by value, because an undefined entry is
    # opaque black and so is COLOR_TABLE[0]; this pins the documented shape.
    path = str(tmp_path / f"out_{dtype}.tif")
    _indexed_raster(dtype=dtype).save(path, color_table=COLOR_TABLE)
    result = read_color_table(path)
    assert len(result) == expected
    assert result[max(result)] == (0, 0, 0, 255)
    assert result[0] == (*COLOR_TABLE[0], 255)


def test_read_color_table_result_round_trips_through_save(tmp_path):
    # The docstring promises the result can be handed straight back to save.
    source = str(tmp_path / "source.tif")
    target = str(tmp_path / "target.tif")
    _indexed_raster().save(source, color_table=COLOR_TABLE)
    _indexed_raster().save(target, color_table=read_color_table(source))
    assert read_color_table(target) == read_color_table(source)


def test_read_color_table_after_editing_an_entry(tmp_path):
    source = str(tmp_path / "source.tif")
    target = str(tmp_path / "target.tif")
    _indexed_raster().save(source, color_table=COLOR_TABLE)
    edited = read_color_table(source)
    edited[2] = (1, 2, 3, 255)
    _indexed_raster().save(target, color_table=edited)
    result = read_color_table(target)
    assert result[2] == (1, 2, 3, 255)
    assert result[1] == (*COLOR_TABLE[1], 255)


def test_read_color_table_keeps_alpha_from_png(tmp_path):
    path = str(tmp_path / "out.png")
    color_table = {
        0: (0, 0, 0, 0),
        1: (1, 2, 3, 128),
        2: (4, 5, 6, 255),
        3: (7, 8, 9, 64),
    }
    _indexed_raster().save(path, color_table=color_table)
    result = read_color_table(path)
    for value, color in color_table.items():
        assert result[value] == color


def test_read_color_table_accepts_pathlib_path(tmp_path):
    path = tmp_path / "out.tif"
    _indexed_raster().save(str(path), color_table=COLOR_TABLE)
    assert read_color_table(path)[1] == (*COLOR_TABLE[1], 255)


def test_read_color_table_explicit_band(out_tif):
    _indexed_raster().save(out_tif, color_table=COLOR_TABLE)
    assert read_color_table(out_tif, band=1) == read_color_table(out_tif)


def test_read_color_table_without_table_raises(out_tif):
    _indexed_raster().save(out_tif)
    with pytest.raises(RasterIOError, match="does not have a color table"):
        read_color_table(out_tif)


def test_read_color_table_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_color_table(str(tmp_path / "nope.tif"))


def test_read_color_table_bad_band_raises(out_tif):
    _indexed_raster().save(out_tif, color_table=COLOR_TABLE)
    with pytest.raises(IndexError, match="No such band index"):
        read_color_table(out_tif, band=2)


# --- normalize_color_table (unit) -------------------------------------------


def test_normalize_color_table_returns_4_tuples():
    result = normalize_color_table(
        {1: (1, 2, 3), 2: (4, 5, 6, 7)}, np.dtype(U8)
    )
    assert result == {1: (1, 2, 3, 255), 2: (4, 5, 6, 7)}


def test_normalize_color_table_gtiff_forces_opaque_alpha():
    with pytest.warns(
        UserWarning, match="GTiff color table cannot store alpha"
    ):
        result = normalize_color_table(
            {1: (1, 2, 3), 2: (4, 5, 6, 0)}, np.dtype(U8), "GTiff"
        )
    assert result == {1: (1, 2, 3, 255), 2: (4, 5, 6, 255)}


def test_normalize_color_table_gtiff_opaque_input_does_not_warn(recwarn):
    result = normalize_color_table({1: (1, 2, 3, 255)}, np.dtype(U8), "GTiff")
    assert result == {1: (1, 2, 3, 255)}
    assert not [w for w in recwarn if "alpha" in str(w.message)]


def test_normalize_color_table_accepts_dtype_like():
    assert normalize_color_table({1: (1, 2, 3)}, "uint8") == {
        1: (1, 2, 3, 255)
    }


@pytest.mark.parametrize("value", [5, None, 3.5])
def test_normalize_color_table_non_sequence_color_raises_value_error(value):
    # A bare TypeError here would escape the ValueError contract that every
    # other malformed spec follows.
    with pytest.raises(ValueError, match="must be a sequence"):
        normalize_color_table({1: value}, np.dtype(U8))


def test_normalize_color_table_accepts_numpy_scalars():
    color_table = {np.uint8(1): (np.uint8(1), np.uint8(2), np.uint8(3))}
    result = normalize_color_table(color_table, np.dtype(U8))
    assert result == {1: (1, 2, 3, 255)}
    assert all(isinstance(c, int) for c in result[1])


def test_normalize_color_table_uint16_allows_wide_indices():
    result = normalize_color_table({65535: (1, 2, 3)}, np.dtype(U16))
    assert result == {65535: (1, 2, 3, 255)}


def test_normalize_color_table_uint8_rejects_uint16_index():
    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        normalize_color_table({256: (1, 2, 3)}, np.dtype(U8))


def test_normalize_color_table_array_rows_are_indices():
    result = normalize_color_table(
        np.array([[1, 2, 3], [4, 5, 6]]), np.dtype(U8)
    )
    assert result == {0: (1, 2, 3, 255), 1: (4, 5, 6, 255)}


# --- Auto chain helper (unit) ----------------------------------------------


@pytest.mark.parametrize(
    "h,w,min_size,expected",
    [
        (1024, 1024, 256, [2, 4]),
        (1000, 1000, 256, [2]),
        (100, 100, 256, []),
        (157000, 90000, 256, [2, 4, 8, 16, 32, 64, 128, 256]),
        (512, 512, 128, [2, 4]),
    ],
)
def test_auto_overview_factors(h, w, min_size, expected):
    assert _auto_overview_factors(h, w, min_size=min_size) == expected


# --- write_raster direct ---------------------------------------------------


def test_write_raster_direct_call(out_tif):
    src = make_raster("arange", dtype="float32", shape=(1, 8, 8))
    write_raster(src.xdata, out_tif)
    reloaded = rts.Raster(out_tif)
    assert np.array_equal(reloaded.to_numpy(), src.to_numpy())


def test_write_raster_unsupported_extension_raises(tmp_path):
    src = make_raster("arange", dtype="float32", shape=(1, 4, 4))
    with pytest.raises(NotImplementedError):
        write_raster(src.xdata, str(tmp_path / "x.nc"))


# --- Batch save (regression for batch.py SAVEFUNCTIONRASTER) ---------------


def test_batch_save_uses_keyword_args(tmp_path):
    src = make_raster("arange", dtype="float32", shape=(1, 32, 32))
    state = _BatchScripParserState.__new__(_BatchScripParserState)
    state.rasters = {"r1": src}
    state.location = str(tmp_path)
    # SAVEFUNCTIONRASTER args: inRaster;outName;outWorkspace;type;nodata;
    #                          blockwidth;blockheight
    args_str = f"r1;out;{tmp_path};TIFF;-1;16;16"
    _batch_parse_save(state, args_str, line_no=1)
    out_path = str(tmp_path / "out.tif")
    with _open(out_path) as ds:
        assert ds.profile["blockxsize"] == 16
        assert ds.profile["blockysize"] == 16
        assert ds.nodata == -1


# --- save_chunks ============================================================


def _multiband_raster():
    return make_raster("arange", dtype="float32", shape=(3, 12, 12)).chunk(
        (1, 4, 4)
    )


def _singleband_raster():
    return make_raster("arange", dtype="float32", shape=(1, 12, 12)).chunk(
        (1, 4, 4)
    )


def test_save_chunks_default_multiband(tmp_path):
    src = _multiband_raster()
    out = src.save_chunks(str(tmp_path / "tile"))

    assert out.shape == (3, 3)
    src_np = src.to_numpy()
    chunk_size = 4
    for r, c in np.ndindex(*out.shape):
        tile = out[r, c]
        assert isinstance(tile, rts.Raster)
        assert tile.nbands == 3
        ys = slice(r * chunk_size, (r + 1) * chunk_size)
        xs = slice(c * chunk_size, (c + 1) * chunk_size)
        assert np.array_equal(tile.to_numpy(), src_np[:, ys, xs])


def test_save_chunks_single_band(tmp_path):
    src = _singleband_raster()
    out = src.save_chunks(str(tmp_path / "tile"))

    assert out.shape == (3, 3)
    for tile in out.ravel():
        assert isinstance(tile, rts.Raster)
        assert tile.nbands == 1


def test_save_chunks_per_band(tmp_path):
    src = _multiband_raster()
    out = src.save_chunks(str(tmp_path / "tile"), per_band=True)

    assert out.shape == (3, 3, 3)
    src_np = src.to_numpy()
    chunk_size = 4
    for b, r, c in np.ndindex(*out.shape):
        tile = out[b, r, c]
        assert isinstance(tile, rts.Raster)
        assert tile.nbands == 1
        ys = slice(r * chunk_size, (r + 1) * chunk_size)
        xs = slice(c * chunk_size, (c + 1) * chunk_size)
        assert np.array_equal(tile.to_numpy()[0], src_np[b, ys, xs])


def test_save_chunks_filenames_zero_padded(tmp_path):
    src = _singleband_raster()
    src.save_chunks(str(tmp_path / "tile"))
    files = sorted(p.name for p in tmp_path.glob("*.tif"))
    # 3 row chunks and 3 col chunks -> single-digit indices, no padding.
    assert files[0] == "tile_0_0.tif"
    assert files[-1] == "tile_2_2.tif"


def test_save_chunks_filenames_padded_for_large_grid(tmp_path):
    # 12 row chunks forces 2-digit zero-padding so files sort correctly.
    src = make_raster("arange", dtype="float32", shape=(1, 144, 12)).chunk(
        (1, 12, 12)
    )
    src.save_chunks(str(tmp_path / "tile"))
    files = sorted(p.name for p in tmp_path.glob("*.tif"))
    assert files[0] == "tile_00_0.tif"
    assert files[-1] == "tile_11_0.tif"


def test_save_chunks_creates_parent_dirs(tmp_path):
    src = _singleband_raster()
    src.save_chunks(str(tmp_path / "nested" / "deep" / "tile"))
    assert (tmp_path / "nested" / "deep").is_dir()
    assert len(list((tmp_path / "nested" / "deep").glob("*.tif"))) == 9


def test_save_chunks_pathlike_prefix_accepted(tmp_path):
    src = _singleband_raster()
    out = src.save_chunks(tmp_path / "tile")
    assert out.shape == (3, 3)
    assert (tmp_path / "tile_0_0.tif").is_file()


@pytest.mark.parametrize("ext", [".tiff", "tiff"])
def test_save_chunks_custom_ext(tmp_path, ext):
    # Verify the leading-dot is optional and the requested extension lands
    # on every file.
    src = _singleband_raster()
    src.save_chunks(str(tmp_path / "tile"), ext=ext)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert all(f.endswith(".tiff") for f in files)


def test_save_chunks_ext_default_picks_tif_without_driver(tmp_path):
    src = _singleband_raster()
    src.save_chunks(str(tmp_path / "tile"))
    files = sorted(p.name for p in tmp_path.iterdir())
    assert all(f.endswith(".tif") for f in files)


@pytest.mark.parametrize(
    "driver,expected_ext",
    [
        ("GTiff", ".tif"),
        ("COG", ".tif"),
        ("HFA", ".img"),
        ("PNG", ".png"),
        ("JPEG", ".jpg"),
    ],
)
def test_ext_for_driver_table_hits(driver, expected_ext):
    assert _ext_for_driver(driver) == expected_ext


@pytest.mark.parametrize(
    "driver,expected_ext",
    [("gtiff", ".tif"), ("hfa", ".img"), ("Png", ".png")],
)
def test_ext_for_driver_case_insensitive(driver, expected_ext):
    assert _ext_for_driver(driver) == expected_ext


def test_ext_for_driver_unlisted_falls_through_to_rasterio_map():
    assert _ext_for_driver("BMP") == ".bmp"


def test_ext_for_driver_unknown_driver_defaults_to_tif():
    assert _ext_for_driver("NotARealDriver") == ".tif"


def test_ext_for_driver_none_defaults_to_tif():
    assert _ext_for_driver(None) == ".tif"


@pytest.mark.parametrize(
    "driver,expected_ext",
    [("GTiff", ".tif"), ("COG", ".tif"), ("HFA", ".img")],
)
def test_save_chunks_ext_auto_from_driver(tmp_path, driver, expected_ext):
    src = make_raster("arange", dtype="float32", shape=(1, 256, 256)).chunk(
        (1, 128, 128)
    )
    src.save_chunks(str(tmp_path / "tile"), driver=driver)
    # Some drivers (e.g. HFA) write sidecar files like .aux.xml; only
    # check the primary data files.
    primary = [
        p.name for p in tmp_path.iterdir() if p.name.endswith(expected_ext)
    ]
    assert len(primary) == 4


def test_save_chunks_empty_ext_means_no_extension(tmp_path):
    src = _singleband_raster()
    # ext="" requires an explicit driver since GDAL can't infer from the
    # missing extension.
    src.save_chunks(str(tmp_path / "tile"), ext="", driver="GTiff")
    files = sorted(p.name for p in tmp_path.iterdir())
    # No trailing dot, no extension at all.
    assert files[0] == "tile_0_0"
    assert "." not in files[0]


def test_save_chunks_ext_and_driver_compose(tmp_path):
    # ext controls the filename; driver="COG" controls the writer. Both
    # are orthogonal -- the explicit driver wins regardless of extension
    # (handy because .cog isn't in GDAL's extension table).
    src = make_raster("arange", dtype="float32", shape=(1, 256, 256)).chunk(
        (1, 128, 128)
    )
    src.save_chunks(str(tmp_path / "tile"), ext=".cog", driver="COG")
    files = sorted(p.name for p in tmp_path.iterdir())
    assert all(f.endswith(".cog") for f in files)
    with rasterio.open(tmp_path / files[0]) as ds:
        # COG always writes tiled files.
        assert ds.profile["tiled"] is True


def test_save_chunks_rejects_path_in_save_kwargs(tmp_path):
    src = _singleband_raster()
    with pytest.raises(TypeError, match="do not pass 'path'"):
        src.save_chunks(str(tmp_path / "tile"), path="elsewhere.tif")


def test_save_chunks_no_data_value_warns_once(tmp_path):
    src = _multiband_raster()
    with pytest.warns(DeprecationWarning, match="no_data_value") as record:
        src.save_chunks(str(tmp_path / "tile"), no_data_value=-9999.0)
    # The deprecation should fire once at the top of save_chunks rather than
    # once per tile (9 tiles otherwise).
    assert sum("no_data_value" in str(w.message) for w in record) == 1
    one = next(tmp_path.glob("*.tif"))
    with rasterio.open(one) as ds:
        assert ds.nodata == -9999.0


def test_save_chunks_save_kwargs_forwarded(tmp_path):
    src = make_raster("arange", dtype="float32", shape=(1, 64, 64)).chunk(
        (1, 32, 32)
    )
    src.save_chunks(str(tmp_path / "tile"), compress="lzw", blocksize=16)
    one = next(tmp_path.glob("*.tif"))
    with rasterio.open(one) as ds:
        assert ds.profile["compress"] == "lzw"
        assert ds.profile["blockxsize"] == 16
        assert ds.profile["blockysize"] == 16


def test_save_chunks_color_table_forwarded(tmp_path):
    src = _indexed_raster(shape=(1, 64, 64)).chunk((1, 32, 32))
    src.save_chunks(str(tmp_path / "tile"), color_table=COLOR_TABLE)
    tiles = sorted(tmp_path.glob("*.tif"))
    assert len(tiles) == 4
    for tile in tiles:
        with rasterio.open(tile) as ds:
            assert ds.colormap(1)[1] == (34, 139, 34, 255)


def test_save_chunks_affine_per_tile(tmp_path):
    src = _multiband_raster()
    out = src.save_chunks(str(tmp_path / "tile"))

    chunk_size = 4
    for r, c in np.ndindex(*out.shape):
        expected = src.affine * Affine.translation(
            c * chunk_size, r * chunk_size
        )
        assert out[r, c].affine == expected


def test_save_chunks_null_value_preserved(tmp_path):
    src = make_raster(
        "arange",
        dtype="float32",
        shape=(1, 12, 12),
        null_pattern=np.s_[:, :2, :2],
    ).chunk((1, 4, 4))
    out = src.save_chunks(str(tmp_path / "tile"))
    for tile in out.ravel():
        assert tile.null_value == src.null_value


def test_save_chunks_return_shape_and_files_exist(tmp_path):
    src = _multiband_raster()
    out = src.save_chunks(str(tmp_path / "tile"))
    assert isinstance(out, np.ndarray)
    assert out.dtype == object
    assert out.shape == (3, 3)
    files = list(tmp_path.glob("*.tif"))
    assert len(files) == 9
    for f in files:
        assert f.is_file() and os.path.getsize(f) > 0


def test_save_chunks_mosaic_round_trip(tmp_path):
    # End-to-end: split a raster into chunk files, then mosaic them back
    # together and verify pixel-equality with the original.
    src = _multiband_raster()
    tiles = src.save_chunks(str(tmp_path / "tile"))
    rebuilt = mosaic(list(tiles.ravel()), dst_grid=src)
    assert rebuilt.shape == src.shape
    assert rebuilt.crs == src.crs
    assert rebuilt.affine == src.affine
    assert np.array_equal(rebuilt.to_numpy(), src.to_numpy())


def test_save_chunks_mosaic_round_trip_per_band(tmp_path):
    src = _multiband_raster()
    tiles = src.save_chunks(str(tmp_path / "tile"), per_band=True)
    # Mosaic per-band single-band tiles spatially, then stack the per-band
    # results back into a multi-band raster.
    rebuilt_bands = [
        mosaic(list(tiles[b].ravel()), dst_grid=src.get_bands(b + 1))
        for b in range(src.nbands)
    ]
    rebuilt = rts.stack_bands(rebuilt_bands)
    assert rebuilt.shape == src.shape
    assert np.array_equal(rebuilt.to_numpy(), src.to_numpy())
