"""
   Description: general module used to perform common spatial analyses
   on Raster objects

   ref: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/an-overview-of-the-generalization-tools.htm
   ref: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/an-overview-of-the-local-tools.htm
"""  # noqa: E501
import warnings
from collections.abc import Sequence
from functools import partial

import dask.array as da
import numba as nb
import numpy as np
import xarray as xr
from dask_image import ndmeasure as ndm

from raster_tools.creation import zeros_like
from raster_tools.dtypes import (
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
    get_default_null_value,
    is_bool,
    is_float,
    is_int,
    is_scalar,
    is_str,
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

__all__ = [
    "aggregate",
    "band_concat",
    "local_stats",
    "predict_model",
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

    data = raster._data
    dout = rs_out._data
    if raster._masked:
        # Set null values to 0 to skip them in the labeling phase
        data = da.where(raster._mask, 0, data)

    for bnd in range(data.shape[0]):
        dout[bnd] = _create_labels(data[bnd], wd, unique_values)

    if raster._masked:
        nv = raster.null_value
        if raster.dtype == rs_out.dtype or np.can_cast(
            raster.dtype, rs_out.dtype
        ):
            dout = da.where(raster._mask, raster.null_value, dout)
        else:
            nv = get_default_null_value(rs_out.dtype)
            dout = da.where(raster._mask, nv, dout)
    rs_out._data = dout
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
        return I16, get_default_null_value(I16)
    if cur_dtype == U16:
        return I32, get_default_null_value(I32)
    if cur_dtype == U32:
        return I64, get_default_null_value(I64)
    return cur_dtype, get_default_null_value(cur_dtype)


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

    xda = rs.xrs
    mask = rs._mask
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
        xmask = xr.DataArray(rs._mask, dims=rs.xrs.dims, coords=rs.xrs.coords)
        mask = xmask.coarsen(dim=dim_map, boundary="trim").all().data

    rs_out = rs.copy()
    rs_out._rs = xda
    if rs._masked:
        rs_out._mask = mask
        nv = rs.null_value
        if stype == "unique":
            dt, nv = _get_unique_dtype(rs_out.dtype)
            rs_out._rs = rs_out.xrs.astype(dt)
        elif stype == "mode":
            # Cast back to original dtype. Original null value will work
            rs_out._rs = rs_out.xrs.astype(orig_dtype)
        # asm/entropy both result in F64 which should hold any value
        rs_out._null_value = nv
        # Replace null cells with null value acording to mask
        rs_out._data = da.where(rs_out._mask, nv, rs_out._data)
    else:
        rs_out._mask = da.zeros_like(rs_out._data, dtype=bool)
    return rs_out


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
    xarr = rs.xrs
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

    xda = rs.xrs
    mask = rs._mask
    if stype in _LOCAL_STYPE_TO_NUMPY_FUNC:
        xndata = xda.reduce(
            _LOCAL_STYPE_TO_NUMPY_FUNC[stype], dim="band", keepdims=True
        )
    else:
        data = xda.data
        # Destination dataarary with correct dims and copied meta data
        xndata = rs.get_bands(1).xrs
        xndata.data = _local(data, stype)
    if rs._masked:
        xmask = xr.DataArray(mask, dims=rs.xrs.dims, coords=rs.xrs.coords)
        mask = xmask.reduce(
            np.all,
            dim="band",
            keepdims=True,
        ).data

    rs_out = rs.copy()
    rs_out._rs = xndata
    if rs._masked:
        rs_out._mask = mask
        nv = rs.null_value
        if stype in ("unique", "minband", "maxband"):
            dt, nv = _get_unique_dtype(rs_out.dtype)
            rs_out._rs = rs_out.xrs.astype(dt)
        elif stype == "mode":
            # Cast back to original dtype. Original null value will work
            rs_out._rs = rs_out.xrs.astype(orig_dtype)
        # asm/entropy both result in F64 which should hold any value
        rs_out._null_value = nv
        # Replace null cells with null value acording to mask
        rs_out._data = da.where(rs_out._mask, nv, rs_out._data)
    else:
        rs_out._mask = da.zeros_like(rs_out._data, dtype=bool)
    return rs_out


def band_concat(rasters, null_value=None):
    """Join a sequence of rasters along the band dimension.

    Parameters
    ----------
    rasters : sequence of Rasters and/or paths
        The rasters to concatenate. These can be a mix of Rasters and paths.
        All rasters must have the same shape in the x and y dimensions.
    null_value : scalar, optional
        A new null value for the resulting raster. If not set, a null value
        will be selected from the input rasters. The logic for selecting a null
        value is this:
        * If none of the inputs have null values, no null value is set.
        * If all of the null values are NaN, NaN is set as the null value.
        * Finally, the first non-NaN null value is selected.

    Returns
    -------
    Raster
        The resulting concatenated Raster.

    """
    rasters = [Raster(other) for other in rasters]
    if not rasters:
        raise ValueError("No rasters provided")
    if len(rasters) == 1:
        return rasters[0]
    if null_value is not None and not is_scalar(null_value):
        raise TypeError("Null value must be a scalar")
    shapes = [r.shape for r in rasters]
    # NOTE: xarray.concat allows for arrays to be missing the first
    # dimension, e.g. concat([(2, 3, 3), (3, 3)]) works. This
    # differs from numpy.
    shapes = set([s[-2:] for s in shapes])
    if len(shapes) != 1:
        raise ValueError("X and Y dimensions must match for input rasters")

    xrs = [r.xrs for r in rasters]
    masks = [r._mask for r in rasters]

    # Choose a null value from the supplied rasters
    # The logic is:
    #   - Supplied null value arg
    #   - If all are nan, then nan
    #   - First non-nan null value
    new_nv = None
    if null_value is not None:
        new_nv = null_value
    else:
        nvs = [r._null_value for r in rasters if r._masked]
        if len(nvs):
            if np.isnan(nvs).all():
                new_nv = np.nan
            else:
                # Take first non-nan null value
                for nv in nvs:
                    if not np.isnan(nv):
                        new_nv = nv
                        break

    # TODO: make sure band dim is "band"
    rs = xr.concat(xrs, "band")
    mask = da.concatenate(masks)

    # Only mess with null values if result is not boolean or one was explicitly
    # specified.
    if not is_bool(rs.dtype) or null_value is not None:
        # Make sure that null value type is consistent with resulting dtype
        if new_nv is not None:
            if is_float(rs.dtype):
                new_nv = float(new_nv)
            elif is_int(rs.dtype):
                if np.isnan(new_nv):
                    new_nv = get_default_null_value(rs.dtype)
                    warnings.warn(
                        f"Null value is NaN but new dtype is {rs.dtype},"
                        f" using default null value for that dtype: {new_nv}",
                        RuntimeWarning,
                    )
                else:
                    new_nv = int(new_nv)
        if new_nv is not None:
            # TODO: remove this if unnecessary
            rs.data = da.where(~mask, rs.data, new_nv)
    # Make sure that band is now an increasing list starting at 1 and
    # incrementing by 1. For xrs1 (1, N, M) and xrs2 (1, N, M),
    # concat([xrs1, xrs2]) sets the band dim to [1, 1], which causes errors
    # in other operations, so this fixes that. It also keeps the band dim
    # values in line with what open_rasterio() returns for multiband
    # rasters.
    rs["band"] = list(range(1, rs.shape[0] + 1))
    return rasters[0]._replace(rs, mask=mask, null_value=new_nv)


@nb.jit(nopython=True, nogil=True)
def _remap_values(x, mappings):
    outx = np.zeros_like(x)
    bands, rows, columns = x.shape
    rngs = mappings.shape[0]
    for bnd in range(bands):
        for rw in range(rows):
            for cl in range(columns):
                vl = int(x[bnd, rw, cl])
                remapped = False
                for imap in range(rngs):
                    if mappings[imap, 0] <= vl < mappings[imap, 1]:
                        outx[bnd, rw, cl] = mappings[imap, 2]
                        remapped = True
                        break
                if not remapped:
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


def remap_range(raster, mapping):
    """Remaps values in a range [`min`, `max`) to a `new_value`.

    Mappings are applied all at once with earlier mappings taking
    precedence.

    Parameters
    ----------
    raster : Raster or str
        Path string or Raster to perform remap on.
    mapping : 3-tuple of scalars or list of 3-tuples of scalars
        A tuple or list of tuples containing ``(min, max, new_value)``
        scalars. The mappiing(s) map values between the min (inclusive) and
        max (exclusive) to the ``new_value``. If `mapping` is a list and
        there are mappings that conflict or overlap, earlier mappings take
        precedence.

    Returns
    -------
    Raster
        The resulting Raster.

    """
    raster = get_raster(raster)
    mappings = _normalize_mappings(mapping)
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
    data = outrs._data
    func = partial(_remap_values, mappings=mappings)
    outrs._data = data.map_blocks(
        func, dtype=data.dtype, meta=np.array((), dtype=data.dtype)
    )
    if f16_workaround:
        outrs = outrs.astype(F16)
    return outrs
