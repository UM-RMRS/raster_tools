"""
   Description: general module used to perform common spatial analyses
   on Raster objects

   * `ESRI Generalization Tools <https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/an-overview-of-the-generalization-tools.htm>`_
   * `ESRI Local Tools <https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/an-overview-of-the-local-tools.htm>`_

"""  # noqa: E501
import os
import re
from collections.abc import Iterable, Sequence
from functools import partial

import dask.array as da
import numba as nb
import numpy as np
import xarray as xr
from dask_image import ndmeasure as ndm
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    grey_dilation,
    grey_erosion,
)

from raster_tools.creation import empty_like, zeros_like
from raster_tools.dtypes import (
    BOOL,
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    get_common_dtype,
    is_bool,
    is_int,
    is_scalar,
    is_str,
)
from raster_tools.focal import _get_offsets
from raster_tools.masking import (
    create_null_mask,
    get_default_null_value,
    reconcile_nullvalue_with_dtype,
)
from raster_tools.raster import Raster, get_raster
from raster_tools.stat_common import (
    nan_unique_count_jit,
    nanargmax_jit,
    nanargmin_jit,
    nanasm_jit,
    nanentropy_jit,
    nanmode_jit,
)
from raster_tools.utils import make_raster_ds

__all__ = [
    "aggregate",
    "band_concat",
    "dilate",
    "erode",
    "local_stats",
    "predict_model",
    "reclassify",
    "regions",
    "remap_range",
]

# TODO: mosaic


def _create_labels(xarr, wd, uarr=None):
    if uarr is None:
        uarr = da.unique(xarr).compute()
    # Drop 0 so they are skipped later
    uarr = uarr[uarr != 0]

    cum_num = 0
    result = da.zeros_like(xarr)
    for v in uarr:
        labeled_array, num_features = ndm.label((xarr == v), structure=wd)
        result += da.where(
            labeled_array > 0, labeled_array + cum_num, 0
        ).astype(result.dtype)
        cum_num += num_features
    return result


def regions(raster, neighbors=4, unique_values=None):
    """Calculates the unique regions (patches) within a raster band.

    The approach is based on ESRI's region group calculation.

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on. All unique non zero values
        will be used to define unique regions.
    neighbors : int, optional
        The neighborhood connectivity value. Valid values are 4 and 8. If
        4, a rook pattern is used, e.g. the neighbors along the horizontal and
        vertical directions are used. if 8, then all of the 8 neighbors are
        used. Default is 4.
    unique_values : array or list, optional
        Values that represent zones from which regions will be made. Values not
        included will be grouped together to form one large zone. If `None`,
        each unique value in the raster will be considered a zone and will be
        calculated up front.

    Returns
    -------
    Raster
        The resulting raster of unique regions values. The bands will have the
        same shape as the original Raster. The null value mask from the origial
        raster will be applied to the result.

    References
    ----------
    * `ESRI: Region Group <https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/region-group.htm>`_

    """  # noqa: E501
    raster = get_raster(raster)
    rs_out = zeros_like(raster, dtype=U64)

    if not is_int(neighbors):
        raise TypeError(
            f"neighbors argument must be an int. Got {type(neighbors)}"
        )
    if neighbors == 4:
        wd = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    elif neighbors == 8:
        wd = np.ones((3, 3), dtype=int)
    else:
        raise ValueError(
            f"Invalid value for neighbors parameter: {repr(neighbors)}"
        )

    if unique_values is not None:
        if isinstance(unique_values, (np.ndarray, da.Array, Sequence)):
            unique_values = np.asarray(unique_values)
        else:
            raise TypeError("Invalid type for unique_values parameter")

    data = raster.data
    dout = rs_out.data
    if raster._masked:
        # Set null values to 0 to skip them in the labeling phase
        data = da.where(raster._ds.mask.data, 0, data)

    for bnd in range(data.shape[0]):
        dout[bnd] = _create_labels(data[bnd], wd, unique_values)

    if raster._masked:
        nv = reconcile_nullvalue_with_dtype(raster.null_value, rs_out.dtype)
        dout = da.where(raster._ds.mask.data, nv, dout)
    rs_out._ds.raster.data = dout
    return rs_out


@nb.jit(nopython=True, nogil=True, parallel=True)
def _coarsen_chunk(x, axis, func, out_dtype, check_nan):
    dims = sorted(set(range(len(x.shape))) - set(axis))
    shape = (x.shape[dims[0]], x.shape[dims[1]], x.shape[dims[2]])
    out = np.empty(shape, out_dtype)
    for i in range(shape[0]):
        for j in nb.prange(shape[1]):
            for k in nb.prange(shape[2]):
                v = func(x[i, j, :, k, :])
                if check_nan and np.isnan(v):
                    # It doesn't matter what value is swapped with nan values
                    # since those cells will be masked out later.
                    v = 0
                out[i, j, k] = v
    return out


def _coarsen_block_map(x, axis, agg_func, out_dtype, check_nan):
    dims = sorted(set(range(len(x.shape))) - set(axis))
    chunks = tuple(x.chunks[d] for d in dims)
    return da.map_blocks(
        partial(
            _coarsen_chunk,
            axis=axis,
            func=agg_func,
            out_dtype=out_dtype,
            check_nan=check_nan,
        ),
        x,
        chunks=chunks,
        drop_axis=axis,
        meta=np.array((), dtype=out_dtype),
    )


def _get_coarsen_dtype(stat, window_size, input_dtype):
    if stat == "unique":
        return np.min_scalar_type(window_size)
    if stat in ("mode", "min", "max"):
        return input_dtype
    if input_dtype == F32:
        return F32
    return F64


_COARSEN_STYPE_TO_FUNC = {
    "max": lambda x: x.max(),
    "mean": lambda x: x.mean(),
    "median": lambda x: x.median(),
    "min": lambda x: x.min(),
    "prod": lambda x: x.prod(),
    "std": lambda x: x.std(),
    "sum": lambda x: x.sum(),
    "var": lambda x: x.var(),
}
_COARSEN_STYPE_TO_CUSTOM_FUNC = {
    "asm": nanasm_jit,
    "entropy": nanentropy_jit,
    "mode": nanmode_jit,
    "unique": nan_unique_count_jit,
}


def _get_unique_dtype(cur_dtype):
    if cur_dtype in (I8, U8):
        return I16
    if cur_dtype == U16:
        return I32
    if cur_dtype == U32:
        return I64
    return cur_dtype


def aggregate(raster, expand_cells, stype):
    """Creates a Raster of aggregated cell values for a new resolution.

    The approach is based on ESRI's aggregate and majority filter functions.

    Parameters
    ----------
    raster : Raster or path str
        Input Raster object or path string
    expand_cells : 2-tuple, list, array-like
        Tuple, array, or list of the number of cells to expand in y and x
        directions. The first element corresponds to the y dimension and the
        second to the x dimension.
    stype : str
        Summarization type. Valid opition are mean, std, var, max, min,
        unique prod, median, mode, sum, unique, entropy, asm.

    Returns
    -------
    Raster
        The resulting raster of aggregated values.

    References
    ----------
    * `ESRI aggregate <https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/aggregate.htm>`_
    * `ESRI majority filter <https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/majority-filter.htm>`_

    """  # noqa: E501
    expand_cells = np.atleast_1d(expand_cells)
    if not is_int(expand_cells.dtype):
        raise TypeError("expand_cells must contain integers")
    if expand_cells.shape != (2,):
        raise ValueError("expand_cells must contain 2 elements")
    if (expand_cells == 1).all():
        raise ValueError("expand_cells values cannont both be one")
    if not (expand_cells >= 1).all():
        raise ValueError("All expand_cells values must be >= 1")
    if not is_str(stype):
        raise TypeError("stype argument must be a string")
    stype = stype.lower()
    if (
        stype not in _COARSEN_STYPE_TO_FUNC
        and stype not in _COARSEN_STYPE_TO_CUSTOM_FUNC
    ):
        raise ValueError(f"Invalid stype argument: {repr(stype)}")
    orig_dtype = get_raster(raster).dtype
    rs = get_raster(raster, null_to_nan=True)

    xda = rs.xdata
    xmask = rs.xmask
    dim_map = {"y": expand_cells[0], "x": expand_cells[1]}
    xdac = xda.coarsen(dim=dim_map, boundary="trim")
    if stype in _COARSEN_STYPE_TO_FUNC:
        xda = _COARSEN_STYPE_TO_FUNC[stype](xdac)
    else:
        custom_stat_func = _COARSEN_STYPE_TO_CUSTOM_FUNC[stype]
        check_nan = stype == "mode"
        out_dtype = _get_coarsen_dtype(
            stype, np.prod(expand_cells), orig_dtype
        )
        # Use partial because reduce seems to be bugged and grabs kwargs that
        # it shouldn't.
        xda = xdac.reduce(
            partial(
                _coarsen_block_map,
                agg_func=custom_stat_func,
                out_dtype=out_dtype,
                check_nan=check_nan,
            )
        )
    # Coarsen mask as well
    if rs._masked:
        xmask = rs.xmask.coarsen(dim=dim_map, boundary="trim").all()

    ds_out = make_raster_ds(xda, xmask)
    if rs._masked:
        if stype == "unique":
            ds_out["raster"] = ds_out.raster.astype(
                _get_unique_dtype(ds_out.raster.dtype)
            )
        elif stype == "mode":
            # Cast back to original dtype. Original null value will work
            ds_out["raster"] = ds_out.raster.astype(orig_dtype)
        # Replace null cells with null value acording to mask
        nv = get_default_null_value(ds_out.raster.dtype)
        ds_out["raster"] = xr.where(xmask, nv, ds_out.raster).rio.write_nodata(
            nv
        )
    else:
        ds_out["mask"] = xr.zeros_like(rs.get_bands(1).xmask)
    if rs.crs is not None:
        ds_out = ds_out.rio.write_crs(rs.crs)
    return Raster(ds_out, _fast_path=True)


def predict_model(raster, model):
    """
    Predict cell estimates from a model using raster band values as predictors.

    Predictor bands correspond to the order of the predictor variables within
    the model. Outputs are raster surfaces with bands cell values depending on
    the type of model.

    The function uses the `model` class' predict method to estimate a new
    raster surface.

    Parameters
    ----------
    raster : Raster or path str
        Raster of predictor variables where the bands correspond to variables
        in the model (one band for each variable in the model).
    model : object
        The model used to estimate new values. Must have a `predict` method
        that takes an `xarray.DataArray` object. The provided DataArray will
        have null cells marked with ``NaN``.

    Returns
    -------
    Raster
        The resulting raster of estmated values.

    """
    rs = get_raster(raster, null_to_nan=True)
    xarr = rs.xdata
    xarrout = model.predict(xarr)

    return Raster(xarrout)


@nb.jit(nopython=True, nogil=True)
def _local_chunk(x, func, out):
    dshape = x.shape
    rws = dshape[1]
    clms = dshape[2]
    for r in range(rws):
        for c in range(clms):
            farr = x[:, r, c]
            out[0, r, c] = func(farr)
    return out


def _get_local_chunk_func(func, out_dtype):
    def wrapped(x):
        return _local_chunk(
            x, func, np.empty((1, *x.shape[1:]), dtype=out_dtype)
        )

    return wrapped


_LOCAL_STYPE_TO_NUMPY_FUNC = {
    "mean": np.nanmean,
    "std": np.nanstd,
    "var": np.nanvar,
    "max": np.nanmax,
    "min": np.nanmin,
    "prod": np.nanprod,
    "sum": np.nansum,
    "median": np.nanmedian,
}
# Use custom min/max band because numpy versions throw errors if all values are
# nan.
_LOCAL_STYPE_TO_CUSTOM_FUNC = {
    "asm": nanasm_jit,
    "entropy": nanentropy_jit,
    "minband": nanargmin_jit,
    "maxband": nanargmax_jit,
    "mode": nanmode_jit,
    "unique": nan_unique_count_jit,
}


def _local(data, stype):
    bnds = data.shape[0]
    orig_chunks = data.chunks
    # Rechunk so band dim is contiguous and chunk sizes are reasonable
    data = da.rechunk(data, chunks=(bnds, "auto", "auto"))

    ffun = _LOCAL_STYPE_TO_CUSTOM_FUNC[stype]
    if stype == "unique":
        out_dtype = np.min_scalar_type(data.shape[0])
    elif stype == "mode":
        out_dtype = data.dtype
    elif stype in ("minband", "maxband"):
        # min/max band inner funcs will return < 0 if all values are nan which
        # will cause overflow for the unsigned type returned by
        # min_scalar_type. This is fine since we can mask them out later.
        out_dtype = np.min_scalar_type(data.shape[0] - 1)
    elif data.dtype == F32:
        out_dtype = F32
    else:
        out_dtype = F64
    ffun = _get_local_chunk_func(ffun, out_dtype)

    out_chunks = ((1,), *data.chunks[1:])
    data_out = da.map_blocks(
        ffun,
        data,
        chunks=out_chunks,
        dtype=out_dtype,
        meta=np.array((), dtype=out_dtype),
    )
    # Rechunk again to expand/contract chunks to a reasonable size and split
    # bands apart
    return da.rechunk(data_out, chunks=(1, *orig_chunks[1:]))


def local_stats(raster, stype):
    """Creates a Raster of summarized values across bands.

    The approach is based on ESRI's local function.

    Parameters
    ----------
    raster : Raster or path str
        Input Raster object or path string
    stype : str
        Summarization type. Valid opition are mean, std, var, max, min,
        maxband, minband, prod, sum, mode, median, unique, entropy, asm

    Returns
    -------
    Raster
        The resulting raster of values aggregated along the band dimension.

    References
    ----------
    * `ESRI local <https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/an-overview-of-the-local-tools.htm>`_

    """  # noqa: E501
    if not is_str(stype):
        raise TypeError("stype argument must be a string")
    stype = stype.lower()
    if (
        stype not in _LOCAL_STYPE_TO_NUMPY_FUNC
        and stype not in _LOCAL_STYPE_TO_CUSTOM_FUNC
    ):
        raise ValueError(f"Invalid stype aregument: {repr(stype)}")
    orig_dtype = get_raster(raster).dtype
    rs = get_raster(raster, null_to_nan=True)

    xda = rs.xdata.copy()
    xmask = rs.xmask
    if stype in _LOCAL_STYPE_TO_NUMPY_FUNC:
        xndata = xda.reduce(
            _LOCAL_STYPE_TO_NUMPY_FUNC[stype], dim="band", keepdims=True
        )
    else:
        data = xda.data
        xndata = empty_like(rs.get_bands(1)).xdata
        xndata.data = _local(data, stype)
    if rs._masked:
        xmask = xmask.reduce(
            np.all,
            dim="band",
            keepdims=True,
        )
    else:
        xmask = xr.zeros_like(xmask, dtype=bool)

    ds_out = make_raster_ds(xndata, xmask)
    if rs._masked:
        if stype in ("unique", "minband", "maxband"):
            ds_out["raster"] = ds_out.raster.astype(
                _get_unique_dtype(ds_out.raster.dtype)
            )
        elif stype == "mode":
            # Cast back to original dtype. Original null value will work
            ds_out["raster"] = ds_out.raster.astype(orig_dtype)
        nv = get_default_null_value(ds_out.raster.dtype)
        ds_out["raster"] = ds_out.raster.rio.write_nodata(nv)
        ds_out.raster.data = da.where(ds_out.mask.data, nv, ds_out.raster.data)
    if rs.crs is not None:
        ds_out = ds_out.rio.write_crs(rs.crs)
    return Raster(ds_out, _fast_path=True)


def _morph_op_chunk(x, footprint, cval, morph_op, binary=False):
    if x.ndim > 2:
        x = x[0]
    if not binary:
        if morph_op == "dilation":
            morph_func = grey_dilation
        else:
            morph_func = grey_erosion
        out = morph_func(x, footprint=footprint, mode="constant", cval=cval)
    else:
        if morph_op == "dilation":
            morph_func = binary_dilation
        else:
            morph_func = binary_erosion
        out = morph_func(x, structure=footprint)
    return out[None]


def _get_footprint(size):
    if isinstance(size, Iterable):
        size = tuple(size)
        if len(size) != 2:
            raise ValueError("size sequence must have lenght 2.")
        if not all(is_int(s) for s in size):
            raise TypeError("size sequence must only contain ints.")
    elif is_int(size):
        size = (size, size)
    else:
        raise TypeError("size input must be an int or sequence of ints.")
    if not all(s > 0 for s in size):
        raise ValueError("size values must be greater than 0.")
    if all(s == 1 for s in size):
        raise ValueError("At least one size value must be greater than 1.")
    footprint = np.ones(size) > 0
    return footprint


def _get_fill(dtype, op):
    if is_int(dtype):
        type_info = np.iinfo(dtype)
    else:
        type_info = np.finfo(dtype)
    if op == "erosion":
        fill = type_info.max
    else:
        # dilation
        fill = type_info.min
    return fill


def _erosion_or_dilation_filter(rs, footprint, op):
    data = rs.data
    fill = _get_fill(rs.dtype, op)
    if rs._masked:
        data = da.where(rs.mask, fill, data)
    rpad, cpad = _get_offsets(footprint.shape)
    # Take max because map_overlap does not support asymmetrical overlaps when
    # a boundary value is given
    depth = {0: 0, 1: max(rpad), 2: max(cpad)}
    data = da.map_overlap(
        partial(
            _morph_op_chunk,
            footprint=footprint,
            cval=fill,
            morph_op=op,
        ),
        data,
        depth=depth,
        boundary=fill,
        dtype=rs.dtype,
        meta=np.array((), dtype=rs.dtype),
    )
    mask = rs.mask
    if rs._masked:
        mask = da.map_overlap(
            partial(
                _morph_op_chunk,
                footprint=footprint,
                cval=fill,
                morph_op=op,
                binary=True,
            ),
            ~mask,
            depth=depth,
            boundary=False,
            dtype=BOOL,
            meta=np.array((), dtype=BOOL),
        )
        mask = ~mask
        data = da.where(mask, rs.null_value, data)
    else:
        mask = mask.copy()
    xrs_out = xr.zeros_like(rs.xdata).rio.write_nodata(rs.null_value)
    xrs_out.data = data
    xmask = xr.DataArray(mask, coords=xrs_out.coords, dims=xrs_out.dims)
    ds_out = make_raster_ds(xrs_out, xmask)
    if rs.crs is not None:
        ds_out = ds_out.rio.write_crs(rs.crs)
    return Raster(ds_out, _fast_path=True)


def dilate(raster, size):
    """Perform dilation on a raster

    Dilation increases the thickness of raster features. Features with higher
    values will cover features with lower values. At each cell, the miximum
    value within a window, defined by `size`, is stored in the output location.
    This is very similar to the max focal filter except that raster features
    are dilated (expanded) into null regions. Dilation is performed on each
    band separately.

    Parameters
    ----------
    raster : Raster or path str
        The raster to dilate
    size : int or 2-tuple of ints
        The shape of the window to use when dilating. A Single int is
        interpreted as the size of a square window. A tuple of 2 ints is used
        as the dimensions of rectangular window. At least one value must be
        greater than 1. Values cannot be less than 1.

    Returns
    -------
    Raster
        The resulting raster with eroded features. This raster will have the
        same shape and meta data as the original

    See also
    --------
    erode, raster_tools.focal.focal

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Dilation_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    """
    raster = get_raster(raster)
    footprint = _get_footprint(size)

    return _erosion_or_dilation_filter(raster, footprint, "dilation")


def erode(raster, size):
    """Perform erosion on a raster

    Erosion increases the thickness of raster features. Features with higher
    values will cover features with lower values. At each cell, the miximum
    value within a window, defined by `size`, is stored in the output location.
    This is very similar to the max focal filter except that raster features
    are eroded (contracted) into null regions. Erosion is performed on each
    band separately.

    Parameters
    ----------
    raster : Raster or path str
        The raster to erode
    size : int or 2-tuple of ints
        The shape of the window to use when eroding. A Single int is
        interpreted as the size of a square window. A tuple of 2 ints is used
        as the dimensions of rectangular window. At least one value must be
        greater than 1. Values cannot be less than 1.

    Returns
    -------
    Raster
        The resulting raster with eroded features. This raster will have the
        same shape and meta data as the original

    See also
    --------
    dilate, raster_tools.focal.focal

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Erosion_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    """
    raster = get_raster(raster)
    footprint = _get_footprint(size)

    return _erosion_or_dilation_filter(raster, footprint, "erosion")


def band_concat(rasters):
    """Join a sequence of rasters along the band dimension.

    Parameters
    ----------
    rasters : sequence of Rasters and/or paths
        The rasters to concatenate. These can be a mix of Rasters and paths.
        All rasters must have the same shape in the x and y dimensions.

    Returns
    -------
    Raster
        The resulting concatenated Raster.

    """
    rasters = [get_raster(raster) for raster in rasters]
    if not rasters:
        raise ValueError("No rasters provided")
    if len(rasters) == 1:
        return rasters[0]

    # TODO: make join a user option or set join="exact"?
    ds = xr.concat([r._ds for r in rasters], dim="band", join="inner")
    ds["band"] = np.arange(np.sum([r.nbands for r in rasters])) + 1
    if any(r._masked for r in rasters):
        nv = get_default_null_value(ds.raster.dtype)
        ds["raster"] = xr.where(ds.mask, nv, ds.raster).rio.write_nodata(nv)
    crs = ([None] + [r.crs for r in rasters if r.crs is not None]).pop()
    if ds.rio.crs is None and crs is not None:
        ds = ds.rio.write_crs(crs)

    return Raster(ds, _fast_path=True)


@nb.jit(nopython=True, nogil=True)
def _remap_values(x, mask, mappings, inclusivity):
    outx = np.zeros_like(x)
    bands, rows, columns = x.shape
    rngs = mappings.shape[0]
    for bnd in range(bands):
        for rw in range(rows):
            for cl in range(columns):
                if mask[bnd, rw, cl]:
                    continue
                vl = int(x[bnd, rw, cl])
                remap = False
                for imap in range(rngs):
                    left, right, new = mappings[imap]
                    if inclusivity == 0:
                        remap = left <= vl < right
                    elif inclusivity == 1:
                        remap = left < vl <= right
                    elif inclusivity == 2:
                        remap = left <= vl <= right
                    elif inclusivity == 3:
                        remap = left < vl < right
                    if remap:
                        outx[bnd, rw, cl] = new
                        break
                if not remap:
                    outx[bnd, rw, cl] = x[bnd, rw, cl]
    return outx


def _normalize_mappings(mappings):
    if not isinstance(mappings, (list, tuple)):
        raise TypeError(
            "Mappings must be either single 3-tuple or list of 3-tuples of "
            "scalars"
        )
    if not len(mappings):
        raise ValueError("No mappings provided")
    if len(mappings) and is_scalar(mappings[0]):
        mappings = [mappings]
    try:
        mappings = [list(m) for m in mappings]
    except TypeError:
        raise TypeError(
            "Mappings must be either single 3-tuple or list of 3-tuples of "
            "scalars"
        )
    for m in mappings:
        if len(m) != 3:
            raise ValueError(
                "Mappings must be either single 3-tuple or list of 3-tuples of"
                " scalars"
            )
        if not all(is_scalar(mi) for mi in m):
            raise TypeError("Mappings values must be scalars")
        if any(np.isnan(mi) for mi in m[:2]):
            raise ValueError("Mapping min and max values cannot be NaN")
        if m[0] >= m[1]:
            raise ValueError(
                "Mapping min value must be strictly less than max value:"
                f" {m[0]}, {m[1]}"
            )
    return mappings


def remap_range(raster, mapping, inclusivity="left"):
    """Remaps values based on a mapping or list of mappings.

    Mappings are applied all at once with earlier mappings taking
    precedence.

    Parameters
    ----------
    raster : Raster or str
        Path string or Raster to perform remap on.
    mapping : 3-tuple of scalars or list of 3-tuples of scalars
        A tuple or list of tuples containing ``(min, max, new_value)``
        scalars. The mappiing(s) map values between the min and max to the
        ``new_value``. If `mapping` is a list and there are mappings that
        conflict or overlap, earlier mappings take precedence. `inclusivity`
        determines which sides of the range are inclusive and exclusive.
    inclusivity : str, optional
        Determines whether to be inclusive or exclusive on either end of the
        range. Default is `'left'`.

        'left' [min, max)
            Left (min) side is inclusive and right (max) side is exclusive.
        'right' (min, max]
            Left (min) side is exclusive and right (max) side is inclusive.
        'both' [min, max]
            Both sides are inclusive.
        'none' (min, max)
            Both sides are exclusive.

    Returns
    -------
    Raster
        The resulting Raster.

    """
    raster = get_raster(raster)
    mappings = _normalize_mappings(mapping)
    if not is_str(inclusivity):
        raise TypeError(
            f"inclusivity must be a str. Got type: {type(inclusivity)}"
        )
    inc_map = {
        name: value
        for name, value in zip(("left", "right", "both", "none"), range(4))
    }
    if inclusivity not in inc_map:
        raise ValueError(f"Invalid inclusivity value. Got: {inclusivity!r}")
    mappings_common_dtype = get_common_dtype([m[-1] for m in mappings])
    out_dtype = np.promote_types(raster.dtype, mappings_common_dtype)
    # numba doesn't understand f16 so use f32 and then downcast
    f16_workaround = out_dtype == F16
    mappings = np.atleast_2d(mappings)

    outrs = raster.copy()
    if out_dtype != outrs.dtype:
        if not f16_workaround:
            outrs = outrs.astype(out_dtype)
        else:
            outrs = outrs.astype(F32)
    elif f16_workaround:
        outrs = outrs.astype(F32)
    data = outrs.data
    func = partial(
        _remap_values, mappings=mappings, inclusivity=inc_map[inclusivity]
    )
    outrs.xdata.data = data.map_blocks(
        func,
        raster.mask,
        dtype=data.dtype,
        meta=np.array((), dtype=data.dtype),
    )
    if f16_workaround:
        outrs = outrs.astype(F16)
    return outrs


def where(condition, true_rast, false_rast):
    """
    Return elements chosen from `true_rast` or `false_rast` depending on
    `condition`.

    Parameters
    ----------
    condition : str or Raster
        A boolean or int raster that indicates where elements in the result
        should be selected from. If the condition is an int raster, it is
        coerced to bool using `condition > 0`.  ``True`` cells pull values from
        `true_rast` and ``False`` cells pull from `y`. *str* is treated as a
        path to a raster.
    true_rast : scalar, Raster, str
        Raster or scalar to pull from if the corresponding location in
        `condition` is ``True``.
    false_rast : scalar, Raster, str
        Raster or scalar to pull from if the corresponding location in
        `condition` is ``False``.

    Returns
    -------
    Raster
        The resulting Raster.

    """
    condition = get_raster(condition)
    if not is_bool(condition.dtype) and not is_int(condition.dtype):
        raise TypeError(
            "Condition argument must be a boolean or integer raster"
        )
    args = []
    for r, name in [(true_rast, "true_rast"), (false_rast, "false_rast")]:
        if not is_scalar(r):
            try:
                r = get_raster(r)
            except TypeError:
                raise TypeError(
                    f"Could not understand {name} argument. Got: {r!r}"
                )
        args.append(r)
    true_rast, false_rast = args
    out_crs = None
    for r in [condition, true_rast, false_rast]:
        crs = getattr(r, "crs", None)
        if crs is not None:
            out_crs = crs
            break

    xtrue, xfalse = [r.xdata if isinstance(r, Raster) else r for r in args]
    masked = any(r._masked if isinstance(r, Raster) else False for r in args)
    scalar_and_nan = all(is_scalar(r) for r in args) and np.isnan(args).any()
    masked |= scalar_and_nan
    xcondition = condition.xdata
    if is_int(condition.dtype):
        # if condition.dtype is not bool then must be an int raster so
        # assume that condition is raster of 0 and 1 values.
        # condition > 0 will grab all 1/True values.
        xcondition = xcondition > 0

    out_xrs = xr.where(xcondition, xtrue, xfalse)
    if masked and not scalar_and_nan:
        xtrue_mask, xfalse_mask = [
            r.xmask
            if isinstance(r, Raster)
            else xr.DataArray(
                create_null_mask(condition.xdata, None),
                dims=condition.xdata.dims,
                coords=condition.xdata.coords,
            )
            for r in args
        ]
        xmask = xr.where(xcondition, xtrue_mask, xfalse_mask)
    elif scalar_and_nan:
        xmask = np.isnan(out_xrs)
    else:
        xmask = xr.DataArray(
            create_null_mask(condition.xdata, None),
            dims=condition.xdata.dims,
            coords=condition.xdata.coords,
        )
    if masked or scalar_and_nan:
        nv = get_default_null_value(out_xrs.dtype)
        out_xrs = xr.where(xmask, nv, out_xrs).rio.write_nodata(nv)
    out_ds = make_raster_ds(out_xrs, xmask)
    if out_crs is not None:
        out_ds = out_ds.rio.write_crs(out_crs)
    return Raster(out_ds, _fast_path=True)


@nb.jit(nopython=True, nogil=True)
def _reclassify_chunk(x, mask, mapping_array, unmapped_to_null, null):
    mapping = dict()
    for i in range(mapping_array.shape[0]):
        mapping[mapping_array[i, 0]] = mapping_array[i, 1]

    out = np.empty_like(x)
    nb, ny, nx = x.shape
    for b in range(nb):
        for i in range(ny):
            for j in range(nx):
                if mask[b, i, j]:
                    out[b, i, j] = null
                    break
                v = x[b, i, j]
                reclass = v in mapping
                if reclass:
                    out[b, i, j] = mapping[v]
                elif unmapped_to_null:
                    out[b, i, j] = null
                else:
                    out[b, i, j] = v
    return out


class RemapFileParseError(Exception):
    pass


_REMAPPING_LINE_PATTERN = re.compile(r"^(?P<from>[+-]*\d+):(?P<to>[+-]*\d+)$")


def _parse_ascii_remap_file(path):
    mapping = {}
    with open(path) as fd:
        for line in fd:
            line = line.strip()
            m = _REMAPPING_LINE_PATTERN.match(line)
            if m is None:
                raise RemapFileParseError(f"Invalid remap line: {line!r}")
            k = int(m.group("from"))
            v = int(m.group("to"))
            if k in mapping:
                raise ValueError("Found duplicate mapping: '{k}:{v}'.")
            mapping[k] = v
    return mapping


def _get_remapping(mapping):
    if is_str(mapping):
        if os.path.exists(mapping):
            mapping = _parse_ascii_remap_file(mapping)
        else:
            raise IOError(f"No such file: {mapping!r}")
    elif not isinstance(mapping, dict):
        raise TypeError(
            f"Remapping must be a str or dict. Got: {type(mapping)!r}"
        )
    if not all(is_int(k) for k in mapping.keys()) or not all(
        is_int(v) for v in mapping.values()
    ):
        raise TypeError("Remapping values must all be integer types")
    return mapping


def reclassify(raster, remapping, unmapped_to_null=False):
    """Reclassify the input raster values based on a mapping.

    This function only works with integer type rasters.

    Parameters
    ----------
    raster : str, Raster
        The input raster to reclassify. Can be a path string or Raster object.
        The raster dtype must be integer.
    remapping : str, dict
        Can be either a ``dict`` or a path string. If a ``dict`` is provided,
        the keys will be reclassified to the corresponding values. If a path
        string, it is treated as an ASCII remap file where each line looks like
        ``a:b`` and ``a`` and ``b`` are integers. All remap values (both from
        and to) must be integers.
    unmapped_to_null : bool, optional
        If ``True``, values not included in the mapping are instead mapped to
        the null value. Default is ``False``.

    Returns
    -------
    Raster
        The remapped raster.

    """
    raster = get_raster(raster)
    remapping = _get_remapping(remapping)
    if not is_int(raster.dtype):
        raise TypeError(
            f"Input raster must be an integer type. Got {raster.dtype!r}"
        )

    out_dtype = raster.dtype
    to_values = list(remapping.values())
    if any(not np.can_cast(v, out_dtype) for v in to_values):
        extra_min = raster.dtype.type(np.iinfo(raster.dtype).min)
        extra_max = raster.dtype.type(np.iinfo(raster.dtype).max)
        out_dtype = get_common_dtype(to_values + [extra_min, extra_max, -1])
    if unmapped_to_null:
        if raster._masked:
            nv = raster.null_value
        else:
            nv = get_default_null_value(out_dtype)
    else:
        nv = 0

    mapping = np.array(list(remapping.items()))
    data = da.map_blocks(
        _reclassify_chunk,
        raster.data.copy().astype(out_dtype),
        raster.mask,
        mapping_array=mapping,
        unmapped_to_null=unmapped_to_null,
        null=nv,
        dtype=out_dtype,
        meta=np.array((), dtype=out_dtype),
    )
    xdata = xr.DataArray(
        data, coords=raster.xdata.coords, dims=raster.xdata.dims
    ).rio.write_nodata(nv)
    if raster.crs is not None:
        xdata = xdata.rio.write_crs(raster.crs)
    return Raster(xdata)
