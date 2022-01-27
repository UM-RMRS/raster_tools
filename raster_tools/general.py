"""
   Description: general module used to perform common spatial analyses
   on Raster objects

   ref: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/an-overview-of-the-generalization-tools.htm
   ref: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/an-overview-of-the-local-tools.htm
"""  # noqa: E501
from collections.abc import Sequence

import dask.array as da
import numba as nb
import numpy as np
import xarray as xr
from dask_image import ndmeasure as ndm

from raster_tools import Raster
from raster_tools._types import F64, U64, get_default_null_value
from raster_tools._utils import is_float, is_int, is_str
from raster_tools.creation import zeros_like
from raster_tools.rsv_utils import get_raster
from raster_tools.stat_common import (
    nan_unique_count_jit,
    nanargmax_jit,
    nanargmin_jit,
    nanasm_jit,
    nanentropy_jit,
    nanmode_jit,
)

__all__ = [
    "regions",
    "aggregate",
    "predict_model",
    "local_stats",
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
    """Calculates the unique regions (patches) within a raser band.

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

    data = raster._rs.data
    dout = rs_out._rs.data
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
    rs_out._rs.data = dout
    return rs_out


@nb.jit(nopython=True, nogil=True)
def _coarsen_chunk(x, ycells, xcells, func, out):
    xshape = x.shape
    bnds = xshape[0]
    rws = xshape[1]
    clms = xshape[2]
    for b in range(bnds):
        for r in range(0, rws, ycells):
            nr = r // ycells
            sr = r + ycells
            for c in range(0, clms, xcells):
                nc = c // xcells
                sc = c + xcells
                farr = x[b, r:sr, c:sc]
                out[b, nr, nc] = func(farr)
    return out


def _get_coarsen_chunk_func(func, ycells, xcells, out_shape, out_dtype):
    def wrapped(x):
        return _coarsen_chunk(
            x, ycells, xcells, func, np.empty(out_shape, dtype=out_dtype)
        )

    return wrapped


def _get_chunk_array(bnds, rws, clms, tcy, tcx):
    fcr = [tcy] * (rws // tcy)
    fcc = [tcx] * (clms // tcx)
    tcl = len(fcr) * tcy
    trl = len(fcc) * tcx
    if not (tcl == rws):
        fcr = fcr + [rws - tcl]
    if not (trl == clms):
        fcc = fcc + [clms - trl]
    return ((bnds), tuple(fcr), tuple(fcc))


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


def _coarsen(xarr, e_cells, stype):
    data = xarr.data
    dshape = data.shape
    bnds, rws, clms = dshape
    bnds = dshape[0]
    rws = dshape[1]
    clms = dshape[2]

    # trim extent if necesary
    trim = False
    if rws % e_cells[0] > 0:
        rws = (rws // e_cells[0]) * e_cells[0]
        trim = True
    if clms % e_cells[0] > 0:
        clms = (clms // e_cells[1]) * e_cells[1]
        trim = True
    if trim:
        data = data[:, :rws, :clms]
        dshape = data.shape

    # change chunksize if necessary to chunk along aggregation boundaries
    dchc = data.chunksize
    _, tcy, tcx = dchc
    rechunk = False
    if tcy % e_cells[0] > 0:
        tcy = (tcy // e_cells[0]) * e_cells[0]
        rechunk = True
    if tcx % e_cells[1] > 0:
        tcx = (tcx // e_cells[1]) * e_cells[1]
        rechunk = True
    nchunks = _get_chunk_array(dchc[0], rws, clms, tcy, tcx)
    if rechunk:
        data = data.rechunk(nchunks)
        dchc = data.chunksize

    # create new chunksize and coords for aggregated raster
    ychunk = tuple(int(y / e_cells[0]) for y in nchunks[1])
    xchunk = tuple(int(x / e_cells[1]) for x in nchunks[2])
    out_chunks = (dchc[0], ychunk, xchunk)

    dres = xarr.rio.resolution()
    nres = (dres[0] * e_cells[0], dres[1] * e_cells[1])
    ycells = rws // e_cells[0]
    xcells = clms // e_cells[1]

    syloc = float(xarr.coords["y"][e_cells[0] // 2])
    sxloc = float(xarr.coords["x"][e_cells[0] // 2])
    ycorarr = np.arange(ycells) * nres[0] + syloc
    xcorarr = np.arange(xcells) * nres[1] + sxloc
    tcoords = {"band": np.arange(bnds) + 1, "y": ycorarr, "x": xcorarr}

    # select function to perform
    if stype == "unique":
        out_dtype = np.min_scalar_type(np.prod(e_cells))
    elif stype == "mode":
        out_dtype = data.dtype
    else:
        out_dtype = F64
    ffun = _COARSEN_STYPE_TO_CUSTOM_FUNC[stype]
    out_shape = [np.sum(d) for d in out_chunks]
    ffun = _get_coarsen_chunk_func(ffun, ycells, xcells, out_shape, out_dtype)
    # map function to blocks and create xarray
    out_data = da.map_blocks(
        ffun,
        data,
        chunks=out_chunks,
        dtype=out_dtype,
        meta=np.array((), dtype=out_dtype),
    )
    out_data = da.rechunk(out_data, chunks=(1, "auto", "auto"))

    outxr = xr.DataArray(out_data, tcoords, dims=xarr.dims, attrs=xarr.attrs)
    if xarr.rio.crs is not None:
        outxr = outxr.rio.write_crs(xarr.rio.crs)
    return outxr


def aggregate(raster, expand_cells, stype):
    """Creates a Raster of aggregated cell values for a new resolution.

    The approach is based ESRI's aggregate and majority filter functions.

    Parameters
    ----------
    raster : Raster or path str
        Input Raster object or path string
    expand_cells : 2-tuple, list, array-like
        Tuple, array, or list of the number of cells to expand in y and x
        directions.
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
    orig_dtype = get_raster(raster).dtype
    rs = get_raster(raster, null_to_nan=True)
    expand_cells = np.asarray(expand_cells)
    if not is_int(expand_cells.dtype):
        raise TypeError("expand_cells must contain integers")
    if not (expand_cells > 1).all():
        raise ValueError("expand_cells values must be greater than 1")
    if expand_cells.shape != (2,):
        raise ValueError("expand_cells must contain 2 elements")
    if not is_str(stype):
        raise TypeError("stype argument must be a string")
    stype = stype.lower()
    if (
        stype not in _COARSEN_STYPE_TO_FUNC
        and stype not in _COARSEN_STYPE_TO_CUSTOM_FUNC
    ):
        raise ValueError(f"Invalid stype argument: {repr(stype)}")

    xda = rs._rs
    mask = rs._mask
    dim_map = {"y": expand_cells[0], "x": expand_cells[1]}
    if stype in _COARSEN_STYPE_TO_FUNC:
        xda = _COARSEN_STYPE_TO_FUNC[stype](
            xda.coarsen(dim=dim_map, boundary="trim")
        )
    else:
        xda = _coarsen(xda, expand_cells, stype)
    # Coarsen mask as well
    if rs._masked:
        xmask = xr.DataArray(rs._mask, dims=rs._rs.dims, coords=rs._rs.coords)
        mask = xmask.coarsen(dim=dim_map, boundary="trim").all().data

    rs_out = rs.copy()
    rs_out._rs = xda
    if rs._masked:
        rs_out._mask = mask
        nv = rs.null_value
        if stype == "unique":
            # Find null value that will work for uint type
            if not np.can_cast(nv, rs_out.dtype):
                if (
                    is_float(nv)
                    and hasattr(nv, "is_integer")
                    and nv.is_integer()
                ):
                    nvi = int(nv)
                    if np.can_cast(nvi, rs_out.dtype):
                        nv = nvi
                    else:
                        nv = get_default_null_value(rs_out.dtype)
                else:
                    nv = get_default_null_value(rs_out.dtype)
        elif stype == "mode":
            # Cast back to original dtype. Original null value will work
            rs_out._rs = rs_out._rs.astype(orig_dtype)
        # asm/entropy both result in F64 which should hold any value
        rs_out._null_value = nv
        # Replace null cells with null value acording to mask
        rs_out._rs.data = da.where(rs_out._mask, nv, rs_out._rs.data)
    else:
        rs_out._mask = da.zeros_like(rs_out._rs.data, dtype=bool)
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
    xarr = rs._rs
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
    return da.rechunk(data_out, chunks=(1, "auto", "auto"))


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
    orig_dtype = get_raster(raster).dtype
    rs = get_raster(raster, null_to_nan=True)
    if not is_str(stype):
        raise TypeError("stype argument must be a string")
    stype = stype.lower()
    if (
        stype not in _LOCAL_STYPE_TO_NUMPY_FUNC
        and stype not in _LOCAL_STYPE_TO_CUSTOM_FUNC
    ):
        raise ValueError(f"Invalid stype aregument: {repr(stype)}")

    xda = rs._rs
    mask = rs._mask
    if stype in _LOCAL_STYPE_TO_NUMPY_FUNC:
        xndata = xda.reduce(
            _LOCAL_STYPE_TO_NUMPY_FUNC[stype], dim="band", keepdims=True
        )
    else:
        data = xda.data
        # Destination dataarary with correct dims and copied meta data
        xndata = rs.get_bands(1)._rs
        xndata.data = _local(data, stype)
    if rs._masked:
        xmask = xr.DataArray(mask, dims=rs._rs.dims, coords=rs._rs.coords)
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
            # Find null value that will work for uint type
            if not np.can_cast(nv, rs_out.dtype):
                if (
                    is_float(nv)
                    and hasattr(nv, "is_integer")
                    and nv.is_integer()
                ):
                    nvi = int(nv)
                    if np.can_cast(nvi, rs_out.dtype):
                        nv = nvi
                    else:
                        nv = get_default_null_value(rs_out.dtype)
                else:
                    nv = get_default_null_value(rs_out.dtype)
        elif stype == "mode":
            # Cast back to original dtype. Original null value will work
            rs_out._rs = rs_out._rs.astype(orig_dtype)
        # asm/entropy both result in F64 which should hold any value
        rs_out._null_value = nv
        # Replace null cells with null value acording to mask
        rs_out._rs.data = da.where(rs_out._mask, nv, rs_out._rs.data)
    else:
        rs_out._mask = da.zeros_like(rs_out._rs.data, dtype=bool)
    return rs_out
