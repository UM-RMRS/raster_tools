import numpy as np
import pytest
import rasterio

import raster_tools as rts
from raster_tools.batch import _batch_parse_save, _BatchScripParserState
from raster_tools.io import _auto_overview_factors, write_raster
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


def test_save_roundtrip_i64(out_tif):
    src = make_raster("arange", dtype="int64", shape=(1, 4, 4))
    src.save(out_tif)
    with _open(out_tif) as ds:
        assert ds.dtypes[0] == "float64"
    reloaded = rts.Raster(out_tif)
    assert np.allclose(reloaded.to_numpy(), src.to_numpy().astype("float64"))


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
