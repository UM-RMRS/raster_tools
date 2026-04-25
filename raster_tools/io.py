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

from raster_tools.dtypes import F32, F64, I64, U8, is_bool, is_float, is_int
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


def write_raster(
    xrs,
    path,
    *,
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

    rio_is_bool = False
    if xrs.dtype == I64:
        # GDAL doesn't support I64; cast up to F64 so to_raster won't reject.
        xrs = xrs.astype(F64)
    elif is_bool(xrs.dtype):
        # GDAL doesn't support bool; encode as uint8.
        rio_is_bool = True
        xrs = xrs.astype(U8)

    resolved_driver = _resolve_driver(path, driver)
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
            rio_copy(tmp_path, path, driver="COG", **creation_opts)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        to_raster_kwargs = {"lock": True, "compute": True, **creation_opts}
        if driver is not None:
            to_raster_kwargs["driver"] = driver
        xrs.rio.to_raster(path, **to_raster_kwargs)

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
