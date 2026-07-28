import os
import urllib
import warnings

import dask
import numpy as np
import rasterio as rio
import rioxarray as xrio
import xarray as xr
from affine import Affine
from dask.array.core import normalize_chunks as dask_chunks

from raster_tools.dtypes import (
    F32,
    F64,
    I64,
    U8,
    U16,
    is_bool,
    is_float,
    is_int,
)
from raster_tools.exceptions import (
    AffineEncodingError,
    DimensionsError,
    RasterDataError,
    RasterIOError,
)
from raster_tools.masking import get_default_null_value
from raster_tools.utils import to_chunk_dict, validate_path


def _get_extension(path):
    return os.path.splitext(path)[-1].lower()


def _get_chunking_info_from_file(src_file):
    with rio.open(src_file) as src:
        tile_shape = (1, *src.block_shapes[0])
        shape = (src.count, *src.shape)
        dtype = np.dtype(src.dtypes[0])
        return tile_shape, shape, dtype


def _get_chunks(data=None, src_file=None):
    chunks = (1, "auto", "auto")
    if data is None:
        if src_file is None:
            return chunks
        tile_shape, shape, dtype = _get_chunking_info_from_file(src_file)
    else:
        shape = data.shape
        dtype = data.dtype
        tile_shape = None
        if dask.is_dask_collection(data):
            tile_shape = data.chunks
        elif src_file is not None:
            _, tile_shape, _ = _get_chunking_info_from_file(src_file)
    return dask_chunks(chunks, shape, dtype=dtype, previous_chunks=tile_shape)


def chunk(xrs, src_file=None):
    chunks = to_chunk_dict(
        _get_chunks(
            xrs.raster if isinstance(xrs, xr.Dataset) else xrs, src_file
        )
    )
    return xrs.chunk(chunks)


TIFF_EXTS = frozenset((".tif", ".tiff"))
NC_EXTS = frozenset((".cdf", ".nc", ".nc4"))
HDF_EXTS = frozenset((".hdf", ".h4", ".hdf4", ".he2", ".h5", ".hdf5", ".he5"))
GRIB_EXTS = frozenset((".grib", ".grib2", ".grb", ".grb2", ".gb", ".gb2"))
BATCH_EXTS = frozenset((".bch",))

# File extenstions that can't be read in yet
READ_NOT_IMPLEMENTED_EXTS = NC_EXTS | HDF_EXTS | GRIB_EXTS
# File extenstions that can't be written out yet
WRITE_NOT_IMPLEMENTED_EXTS = NC_EXTS | HDF_EXTS | GRIB_EXTS


def is_batch_file(path):
    return _get_extension(path) in BATCH_EXTS


def _require_backend(import_name, package, ext, extra):
    import importlib.util

    if importlib.util.find_spec(import_name) is None:
        raise ImportError(
            f"Reading {ext} files requires the '{package}' package, which is"
            " not installed. Install it with 'pip install"
            f" raster-tools[{extra}]' or 'conda install -c conda-forge"
            f" {package}'."
        )


ESRI_DEFAULT_F32_NV = np.finfo(F32).min


def normalize_null_value(nv, dtype):
    # Make sure that ESRI's default F32 null value is properly
    # registered as F32
    if dtype == F32 and nv is not None and np.isclose(nv, ESRI_DEFAULT_F32_NV):
        nv = F32.type(nv)
    # Some rasters have (u)int dtype and a null value that is a whole number
    # but it gets read in as a float. This can cause a lot of accidental type
    # promotions down the pipeline. Check for this case and correct it.
    if is_int(dtype) and is_float(nv) and float(nv).is_integer():
        nv = int(nv)
    return nv


def open_raster_from_path_or_url(path):
    from raster_tools.raster import (
        _try_to_get_null_value_xarray,
        normalize_xarray_data,
    )

    if isinstance(path, os.PathLike):
        ext = path.suffix
    elif isinstance(path, str):
        if urllib.parse.urlparse(path) == "":
            # Assume file path
            validate_path(path)
            ext = _get_extension(path)
        else:
            # Could be a URL or path
            ext = ""
    else:
        raise RasterIOError(
            f"Could not resolve input to a raster path or URL: '{path}'"
        )

    xrs = None
    # Try to let gdal open anything but NC, HDF, GRIB files
    if ext in READ_NOT_IMPLEMENTED_EXTS:
        raise NotImplementedError(
            "Reading of NetCDF, HDF, and GRIB files is not supported at this"
            " time. Try 'raster_tools.open_dataset'."
        )
    else:
        try:
            xrs = xrio.open_rasterio(
                path, chunks=to_chunk_dict(_get_chunks()), lock=False
            )
        except rio.errors.RasterioIOError as e:
            raise RasterIOError(
                "Could not open given path as a raster."
            ) from e
    if isinstance(xrs, xr.Dataset):
        raise RasterDataError("Too many data variables in input data")
    assert isinstance(
        xrs, xr.DataArray
    ), "Resulting data structure must be a DataArray"
    if not dask.is_dask_collection(xrs):
        xrs = chunk(xrs, path)

    xrs = normalize_xarray_data(xrs)

    nv = _try_to_get_null_value_xarray(xrs)
    nv = normalize_null_value(nv, xrs.dtype)
    xrs = xrs.rio.write_nodata(nv)
    return xrs


_EXT_TO_DRIVER = {".tif": "GTiff", ".tiff": "GTiff"}

# Drivers that build overviews as part of the write itself; the post-write
# build_overviews pass is skipped for these.
_DRIVERS_WITH_INTERNAL_OVERVIEWS = frozenset({"COG"})


def _resolve_driver(path, driver):
    if driver is not None:
        return driver
    return _EXT_TO_DRIVER.get(_get_extension(path))


# Drivers that support native Int64/UInt64 rasters on a capable GDAL.
# The COG writer stages through a temporary GTiff, so it inherits GTiff's
# support.
_DRIVERS_WITH_INT64 = frozenset({"GTiff", "COG"})


def _supports_native_int64(driver):
    """Whether native int64 writes work for the resolved ``driver``.

    GDAL added native Int64/UInt64 GeoTIFF support in 3.5;
    ``check_dtype`` reflects the running GDAL's global dtype support.
    A resolved driver outside the known-good set (including ``None``,
    which means GDAL would infer the driver from an unmapped extension)
    is treated conservatively as unsupported.
    """
    from rasterio.dtypes import check_dtype

    return driver in _DRIVERS_WITH_INT64 and check_dtype("int64")


def _gtiff_translate(opts):
    out = {}
    tiled = opts.get("tiled")
    if tiled is not None:
        out["tiled"] = bool(tiled)
    bs = opts.get("blocksize")
    if bs is not None:
        if isinstance(bs, int):
            h = w = bs
        else:
            h, w = bs
        out["blockxsize"] = int(w)
        out["blockysize"] = int(h)
    compress = opts.get("compress")
    if compress is None:
        out["compress"] = "none"
    else:
        out["compress"] = str(compress).lower()
    level = opts.get("compress_level")
    if level is not None:
        c = out["compress"]
        if c == "deflate":
            out["zlevel"] = int(level)
        elif c == "zstd":
            out["zstd_level"] = int(level)
        elif c == "jpeg":
            out["jpeg_quality"] = int(level)
        else:
            warnings.warn(
                f"compress_level has no effect with compress={compress!r}",
                stacklevel=4,
            )
    predictor = opts.get("predictor")
    if predictor is not None:
        if out["compress"] == "jpeg":
            warnings.warn(
                "predictor is not valid with compress='jpeg'; ignoring",
                stacklevel=4,
            )
        elif isinstance(predictor, int):
            # Backward compat with rasterio-style integer predictor values.
            out["predictor"] = predictor
        else:
            mapping = {"horizontal": 2, "float": 3}
            if predictor not in mapping:
                raise ValueError(
                    "predictor must be 'horizontal' or 'float', got "
                    f"{predictor!r}"
                )
            out["predictor"] = mapping[predictor]
    bigtiff = opts.get("bigtiff")
    if isinstance(bigtiff, bool):
        out["bigtiff"] = "yes" if bigtiff else "no"
    elif bigtiff is not None:
        out["bigtiff"] = str(bigtiff).lower()
    return out


def _cog_translate(opts):
    out = {}
    # COG is always tiled; the tiled kwarg is intentionally ignored.
    bs = opts.get("blocksize")
    if bs is not None:
        if isinstance(bs, int):
            size = bs
        else:
            h, w = bs
            if h != w:
                raise ValueError(
                    f"COG driver requires a square blocksize; got {bs!r}"
                )
            size = h
        out["blocksize"] = int(size)
    compress = opts.get("compress")
    if compress is None:
        out["compress"] = "none"
    else:
        out["compress"] = str(compress).lower()
    level = opts.get("compress_level")
    if level is not None:
        c = out["compress"]
        if c in (
            "deflate",
            "zstd",
            "lzw",
            "lerc",
            "lerc_deflate",
            "lerc_zstd",
        ):
            out["level"] = int(level)
        elif c == "jpeg":
            out["quality"] = int(level)
        else:
            warnings.warn(
                f"compress_level has no effect with compress={compress!r}",
                stacklevel=4,
            )
    predictor = opts.get("predictor")
    if predictor is not None:
        if out["compress"] == "jpeg":
            warnings.warn(
                "predictor is not valid with compress='jpeg'; ignoring",
                stacklevel=4,
            )
        elif isinstance(predictor, int):
            # Backward compat with rasterio-style integer predictor values.
            out["predictor"] = predictor
        else:
            mapping = {"horizontal": "STANDARD", "float": "FLOATING_POINT"}
            if predictor not in mapping:
                raise ValueError(
                    "predictor must be 'horizontal' or 'float', got "
                    f"{predictor!r}"
                )
            out["predictor"] = mapping[predictor]
    bigtiff = opts.get("bigtiff")
    if isinstance(bigtiff, bool):
        out["bigtiff"] = "yes" if bigtiff else "no"
    elif bigtiff is not None:
        out["bigtiff"] = str(bigtiff).lower()
    overviews = opts.get("overviews")
    if overviews is None or overviews is False:
        out["overviews"] = "none"
    elif isinstance(overviews, (list, tuple)):
        warnings.warn(
            "COG driver builds overviews with auto-selected factors; "
            "explicit overview list is ignored.",
            stacklevel=4,
        )
        out["overviews"] = "auto"
    else:
        out["overviews"] = "auto"
    overview_resampling = opts.get("overview_resampling")
    if overview_resampling is not None:
        out["overview_resampling"] = str(overview_resampling).lower()
    return out


_DRIVER_TRANSLATORS = {"GTiff": _gtiff_translate, "COG": _cog_translate}


def _auto_overview_factors(height, width, min_size=256):
    factors = []
    f = 2
    while min(height, width) / f >= min_size:
        factors.append(f)
        f *= 2
    return factors


# GDAL only supports color tables on Byte and UInt16 rasters. Other dtypes
# accept the write call without error but store nothing, so the dtype is
# checked up front rather than letting the palette be dropped silently.
COLOR_TABLE_DTYPES = frozenset((U8, U16))

# A GeoTIFF color table is three 16-bit planes of red, green, and blue with
# no room for alpha. Other palette formats, such as PNG and GIF, do store it.
_COLOR_TABLE_DRIVERS_WITHOUT_ALPHA = frozenset({"GTiff", "COG"})

# Overview resampling methods that average or interpolate cell values. Applied
# to palette indices they produce indices that do not correspond to the
# original colors, so the overviews render as noise.
_INTERPOLATING_RESAMPLING = frozenset(
    {
        "average",
        "average_magphase",
        "bilinear",
        "cubic",
        "cubic_spline",
        "gauss",
        "lanczos",
        "rms",
    }
)


def _read_color_table_from_file(path, band=1):
    validate_path(path)
    try:
        with rio.open(path) as src:
            return src.colormap(band)
    except rio.errors.RasterioIOError as e:
        raise RasterIOError(
            f"Could not open {str(path)!r} to read a color table from."
        ) from e
    except ValueError as e:
        raise RasterIOError(
            f"Band {band} of {str(path)!r} does not have a color table."
        ) from e


def _color_table_spec_to_entries(color_table):
    """Coerce a color table spec to an ``{index: components}`` mapping.

    Also reports whether the entries were read from a file, since a table
    GDAL hands back carries padding that a caller never asked for.
    """
    if isinstance(color_table, (str, os.PathLike)):
        return _read_color_table_from_file(color_table), True
    if isinstance(color_table, dict):
        return color_table, False
    entries = np.asarray(color_table)
    if entries.ndim != 2 or entries.shape[-1] not in (3, 4):
        raise ValueError(
            "An array-like color table must have shape (N, 3) or (N, 4). Got"
            f" shape {entries.shape}."
        )
    return dict(enumerate(entries.tolist())), False


def normalize_color_table(color_table, dtype, driver=None):
    """Validate a color table spec and return a mapping GDAL can write.

    Parameters
    ----------
    color_table : dict, array-like, str, pathlib.Path
        The color table to normalize. A ``dict`` maps raster values to
        ``(r, g, b)`` or ``(r, g, b, a)`` components in the range 0-255. An
        array-like of shape ``(N, 3)`` or ``(N, 4)`` is treated as a lookup
        table where row ``i`` is the color for value ``i``. A path is opened
        and the color table on its first band is used.
    dtype : numpy.dtype
        The dtype of the data being written. Used to bound the valid value
        range.
    driver : str, optional
        The GDAL driver the color table will be written with. Used to decide
        whether alpha components can be kept. The default assumes they can.

    Returns
    -------
    dict
        Maps each value to a 4-tuple of ints. Alpha is forced to 255 for
        drivers whose palette format has no room for it.

    """
    dtype = np.dtype(dtype)
    if dtype not in COLOR_TABLE_DTYPES:
        raise ValueError(
            "Color tables are only supported for uint8 and uint16 rasters."
            f" Got dtype {str(dtype)!r}."
        )
    entries, from_file = _color_table_spec_to_entries(color_table)
    if not entries:
        raise ValueError("The given color table was empty.")
    keeps_alpha = driver not in _COLOR_TABLE_DRIVERS_WITHOUT_ALPHA

    max_value = np.iinfo(dtype).max
    normalized = {}
    alpha_dropped = False
    for value, components in entries.items():
        if not is_int(value) or not 0 <= value <= max_value:
            if from_file:
                # GDAL pads a table it reads out to the full index range of
                # the source dtype, so a wider source carries entries this
                # dtype cannot index. Those are padding, not intent.
                continue
            raise ValueError(
                "Color table values must be integers in the range"
                f" [0, {max_value}] for dtype {str(dtype)!r}. Got {value!r}."
            )
        try:
            components = tuple(components)
        except TypeError as e:
            raise ValueError(
                "Color table colors must be a sequence of 3 (RGB) or 4 (RGBA)"
                f" components. Got {components!r} for value {value}."
            ) from e
        if len(components) not in (3, 4):
            raise ValueError(
                "Color table colors must have 3 (RGB) or 4 (RGBA) components."
                f" Got {components!r} for value {value}."
            )
        if not all(is_int(c) and 0 <= c <= 255 for c in components):
            raise ValueError(
                "Color table color components must be integers in the range"
                f" [0, 255]. Got {components!r} for value {value}."
            )
        alpha = components[3] if len(components) == 4 else 255
        if not keeps_alpha and alpha != 255:
            alpha_dropped = True
            alpha = 255
        normalized[int(value)] = (
            *(int(c) for c in components[:3]),
            int(alpha),
        )

    if not normalized:
        raise ValueError(
            "The given color table had no entries that a"
            f" {str(dtype)!r} raster can index."
        )
    if alpha_dropped:
        warnings.warn(
            f"The {driver} color table cannot store alpha; the alpha"
            " components of the given color table were dropped. Use the null"
            " value to mark cells that should not be rendered.",
            UserWarning,
            stacklevel=4,
        )
    return normalized


def _blends_palette_overviews(overviews, resampling, yx_shape):
    """Whether overviews would be built by blending palette indices.

    Averaging or interpolating palette indices yields indices that no longer
    correspond to the original colors. Only reports ``True`` when overviews
    will actually be built, so a raster too small for the auto chain does not
    draw a warning about output that will not exist.
    """
    if not overviews:
        return False
    if str(resampling).lower() not in _INTERPOLATING_RESAMPLING:
        return False
    if overviews is True:
        return bool(_auto_overview_factors(*yx_shape))
    return bool(list(overviews))


def _attach_color_table(path, color_table):
    """Write `color_table` to the first band of an already written raster.

    Writing the color table also switches the file's photometric
    interpretation to palette, which is what makes strict readers honor it.
    Requesting the palette photometric as a creation option on top of this
    only reserves a second color table block that nothing reads, so the tag
    is left to this call and pinned by the tests instead.
    """
    with rio.open(path, "r+") as ds:
        ds.write_colormap(1, color_table)


def write_raster(
    xrs,
    path,
    *,
    color_table=None,
    driver=None,
    tiled=True,
    blocksize=None,
    compress=None,
    compress_level=None,
    predictor=None,
    bigtiff="if_safer",
    overviews=None,
    overview_resampling="average",
    overview_num_threads="all_cpus",
    **gdal_kwargs,
):
    ext = _get_extension(path)
    if ext and ext in WRITE_NOT_IMPLEMENTED_EXTS:
        raise NotImplementedError(
            f"Writing files with extension {ext!r} is not supported"
        )

    resolved_driver = _resolve_driver(path, driver)

    rio_is_bool = False
    if xrs.dtype == I64:
        nv = xrs.rio.nodata
        # GDAL stores a GeoTIFF nodata value as a C double. Its Int64
        # nodata handling is only reliable for magnitudes up to 2**53;
        # larger sentinels (including int64's default null, -2**63) are
        # mangled on write, which would silently drop the null mask on
        # reload. Treat such a null as unpreservable by a native write.
        null_preservable = nv is None or -(2**53) <= nv <= 2**53
        if not _supports_native_int64(resolved_driver):
            # Older GDAL builds lack native int64 raster support; cast up
            # to F64 so to_raster won't reject the array.
            warnings.warn(
                "This GDAL build lacks native int64 raster support; the"
                " array will be written as float64. Values above 2**53"
                " may lose precision, and the file will read back as"
                " float64 (cast it back with .astype('int64') after"
                " loading if needed).",
                UserWarning,
                stacklevel=2,
            )
            xrs = xrs.astype(F64)
        elif not null_preservable:
            # Native int64 is available, but GeoTIFF's float nodata
            # field cannot hold this null value, so a native write would
            # lose the mask. Cast to F64, which stores the null exactly
            # as a double and keeps the mask intact.
            warnings.warn(
                "The int64 null value cannot be represented in a"
                " GeoTIFF nodata field; the array will be written as"
                " float64 to preserve the null mask. Values above 2**53"
                " may lose precision, and the file will read back as"
                " float64 (cast it back with .astype('int64') after"
                " loading if needed).",
                UserWarning,
                stacklevel=2,
            )
            xrs = xrs.astype(F64)
    elif is_bool(xrs.dtype):
        # GDAL doesn't support bool; encode as uint8.
        rio_is_bool = True
        xrs = xrs.astype(U8)

    if color_table is not None:
        # 2D input is a single band; only a real band dim can hold more.
        nbands = xrs.sizes.get("band", xrs.shape[0] if xrs.ndim == 3 else 1)
        if nbands > 1:
            raise ValueError(
                "A color table can only be written for a single band raster."
                f" This raster has {nbands} bands."
            )
        color_table = normalize_color_table(
            color_table, xrs.dtype, resolved_driver
        )
        if _blends_palette_overviews(
            overviews, overview_resampling, xrs.shape[-2:]
        ):
            warnings.warn(
                f"overview_resampling={overview_resampling!r} blends cell"
                " values together, which is not meaningful for values that"
                " index a color table. The overviews will not match the"
                " colors of the full resolution data. Use 'nearest' or"
                " 'mode' instead.",
                UserWarning,
                stacklevel=3,
            )

    translator = _DRIVER_TRANSLATORS.get(resolved_driver)
    creation_opts = {}
    if translator is not None:
        creation_opts = translator(
            {
                "tiled": tiled,
                "blocksize": blocksize,
                "compress": compress,
                "compress_level": compress_level,
                "predictor": predictor,
                "bigtiff": bigtiff,
                "overviews": overviews,
                "overview_resampling": overview_resampling,
            }
        )
        if rio_is_bool and resolved_driver == "GTiff":
            creation_opts["nbits"] = 1
    # Escape hatch wins on collisions.
    creation_opts.update(gdal_kwargs)

    if resolved_driver == "COG":
        # rioxarray streams dask chunks by reopening the file in "r+", but
        # COG forbids updates after creation (it would break the layout).
        # Stage to a temporary GTiff, then translate to COG.
        import tempfile

        from rasterio.shutil import copy as rio_copy

        out_dir = os.path.dirname(os.path.abspath(path))
        with tempfile.NamedTemporaryFile(
            suffix=".tif", dir=out_dir, delete=False
        ) as tmpf:
            tmp_path = tmpf.name
        try:
            xrs.rio.to_raster(tmp_path, lock=True, compute=True)
            if color_table is not None:
                # The COG driver has no palette creation option, but the copy
                # carries the staged file's color table and photometric tag
                # over, so the palette has to be in place before it runs.
                _attach_color_table(tmp_path, color_table)
            rio_copy(tmp_path, path, driver="COG", **creation_opts)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        to_raster_kwargs = {"lock": True, "compute": True, **creation_opts}
        if driver is not None:
            to_raster_kwargs["driver"] = driver
        xrs.rio.to_raster(path, **to_raster_kwargs)
        if color_table is not None:
            _attach_color_table(path, color_table)

    if overviews and resolved_driver not in _DRIVERS_WITH_INTERNAL_OVERVIEWS:
        factors = (
            _auto_overview_factors(*xrs.shape[-2:])
            if overviews is True
            else list(overviews)
        )
        if factors:
            from rasterio.enums import Resampling

            resampling = Resampling[overview_resampling]
            env_kwargs = {}
            if overview_num_threads is not None:
                env_kwargs["GDAL_NUM_THREADS"] = str(
                    overview_num_threads
                ).upper()
            with rio.Env(**env_kwargs), rio.open(path, "r+") as ds:
                ds.build_overviews(factors, resampling)
                ds.update_tags(
                    ns="rio_overview", resampling=overview_resampling
                )


def _get_valid_variables(meta, ignore_too_many_dims):
    data_vars = list(meta.data_vars)
    valid = []
    for v in data_vars:
        n = meta[v].squeeze().ndim
        if n > 3:
            if ignore_too_many_dims:
                continue
            else:
                raise DimensionsError(
                    f"Too many dimensions for variable {v!r} with "
                    f"{meta[v].ndim}."
                )
        elif n in (2, 3):
            valid.append(v)
        else:
            raise DimensionsError(
                f"Too few dimensions for variable {v!r} with {n}."
            )
    if not valid:
        raise ValueError("No valid raster variables found")
    return valid


def _build_raster(path, variable, affine, crs, xarray_kwargs):
    from raster_tools.raster import data_to_raster

    if affine is None:
        affine = Affine(1, 0, 0, 0, -1, 0, 0)
    kwargs = xarray_kwargs.copy()
    kwargs["chunks"] = "auto"
    var = xr.open_dataset(path, **kwargs)[variable].squeeze()
    x = var[var.rio.x_dim].to_numpy()
    y = var[var.rio.y_dim].to_numpy()
    nv = var._FillValue if "_FillValue" in var.attrs else var.rio.nodata
    raster = data_to_raster(var.data, x=x, y=y, affine=affine, crs=crs, nv=nv)
    if nv is None or np.isnan(nv):
        raster = raster.set_null_value(get_default_null_value(raster.dtype))
    return raster


def _get_affine(ds):
    try:
        affine = ds.rio.transform()
    except TypeError as err:
        # Some datasets like gridMET improperly encode the transform.
        raise AffineEncodingError(
            "Error reading GeoTransform data:"
            f"{ds.coords[ds.rio.grid_mapping].attrs['GeoTransform']!r}"
        ) from err
    return affine


def open_dataset(
    path,
    crs=None,
    ignore_extra_dim_errors=False,
    xarray_kwargs=None,
):
    """Open a netCDF or GRIB dataset.

    This function opens a netCDF or GRIB dataset file and returns a dictionary
    of Raster objectds where each raster corrersponds to the variables in the
    the file. netCDF/GRIB files can be N-dimensional, while rasters only
    comprehend 2 to 3 dimensions (band, y, x), so it may not be possible to map
    all variables in a file to a raster. See the `ignore_extra_dim_errors`
    option below for more information.

    Parameters
    ----------
    path : str
        THe path to the netCDF or GRIB dataset file.
    crs : str, rasterio.crs.CRS, optional
        A coordinate reference system definition to attach to the dataset. This
        can be an EPSG, PROJ, or WKT string. It can also be a
        `rasterio.crs.CRS` object. netCDF/GRIB files do not always encode a
        CRS. This option allows a CRS to be supplied, if known ahead of time.
        It can also be used to override the CRS encoded in the file.
    ignore_extra_dim_errors : bool, optional
        If ``True``, ignore dataset variables that cannot be mapped to a
        raster. An error is raised, otherwise. netCDF/GRIB files allow
        N-dimensional. Rasters only comprehend 2 or 3 dimensional data so it is
        not always possible to map a variable to a raster. The default is
        ``False``.
    xarray_kwargs : dict, optional
        Keyword arguments to supply to `xarray.open_dataset` when opening the
        file.

    Raises
    ------
    raster_tools.io.AffineEncodingError
        Raised if the affine matrix is improperly encoded.
    ra

    Returns
    -------
    dataset : dict of Raster
        A ``dict`` of Raster objects. The keys are the variable names in the
        dataset file and the values are the corresponding variable data as a
        raster.

    """
    if xarray_kwargs is None:
        xarray_kwargs = {}
    xarray_kwargs["decode_coords"] = "all"
    ext = _get_extension(path)
    if ext in NC_EXTS:
        _require_backend("netCDF4", "netcdf4", ext, extra="io")
    elif ext in GRIB_EXTS:
        _require_backend("cfgrib", "cfgrib", ext, extra="io")
    tmp_ds = xr.open_dataset(path, **xarray_kwargs)
    data_vars = _get_valid_variables(tmp_ds, ignore_extra_dim_errors)
    crs = crs or tmp_ds.rio.crs
    affine = _get_affine(tmp_ds)
    tmp_ds = None
    ds = {}
    for v in data_vars:
        ds[v] = _build_raster(path, v, affine, crs, xarray_kwargs)
    return ds
