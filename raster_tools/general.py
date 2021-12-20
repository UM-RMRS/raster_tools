###############################################################################
"""
   Description: general module used to perform common spatial analyses
   on Raster objects

   ref: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/an-overview-of-the-generalization-tools.htm
   ref: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/an-overview-of-the-local-tools.htm
"""  # noqa: E501
###############################################################################
from functools import partial

import dask.array as da
import numba as nb
import numpy as np
import xarray as xr
from dask_image import ndmeasure as ndm

from raster_tools import Raster
from raster_tools._types import U64
from raster_tools._utils import is_str
from raster_tools.focal import (
    _agg_nan_asm,
    _agg_nan_entropy,
    _agg_nan_mode,
    _agg_nan_unique,
)
from raster_tools.raster import is_raster_class
from raster_tools.surface import _get_rs

__all__ = [
    "regions",
    "aggregate",
    "random_raster",
    "constant_raster",
    "predict_model",
    "local_stats",
]


def _create_labels(xarr, wd, uarr=None):
    if uarr is None:
        uarr = da.unique(xarr).compute()

    cum_num = 0
    result = da.zeros_like(xarr)
    for v in uarr[1:]:
        labeled_array, num_features = ndm.label((xarr == v), structure=wd)
        result += da.where(
            labeled_array > 0, labeled_array + cum_num, 0
        ).astype(result.dtype)
        cum_num += num_features
    return result


@nb.jit(nopython=True, nogil=True)
def _mode(xarr):
    dshape = xarr.shape
    rws = dshape[1]
    clms = dshape[2]
    outx = np.empty((1, rws, clms))
    for r in range(rws):
        for c in range(clms):
            farr = xarr[:, r, c]
            vl = _agg_nan_mode(farr)
            outx[0, r, c] = vl

    return outx


@nb.jit(nopython=True, nogil=True)
def _asm(xarr):
    dshape = xarr.shape
    rws = dshape[1]
    clms = dshape[2]
    outx = np.empty((1, rws, clms))
    for r in range(rws):
        for c in range(clms):
            farr = xarr[:, r, c]
            vl = _agg_nan_asm(farr)
            outx[0, r, c] = vl

    return outx


@nb.jit(nopython=True, nogil=True)
def _entropy(xarr):
    dshape = xarr.shape
    rws = dshape[1]
    clms = dshape[2]
    outx = np.empty((1, rws, clms))
    for r in range(rws):
        for c in range(clms):
            farr = xarr[:, r, c]
            vl = _agg_nan_entropy(farr)
            outx[0, r, c] = vl

    return outx


@nb.jit(nopython=True, nogil=True)
def _unique(xarr):
    dshape = xarr.shape
    rws = dshape[1]
    clms = dshape[2]
    outx = np.empty((1, rws, clms))
    for r in range(rws):
        for c in range(clms):
            farr = xarr[:, r, c]
            vl = _agg_nan_unique(farr)
            outx[0, r, c] = vl

    return outx


@nb.jit(nopython=True, nogil=True)
def _mode_coarsenf(xarr, ycells, xcells):
    xshape = xarr.shape
    bnds = xshape[0]
    rws = xshape[1]
    clms = xshape[2]
    outx = np.empty((bnds, rws // ycells, clms // xcells))
    for b in range(bnds):
        for r in range(0, rws, ycells):
            nr = r // ycells
            sr = r + ycells
            for c in range(0, clms, xcells):
                nc = c // xcells
                sc = c + xcells
                farr = xarr[b, r:sr, c:sc]
                vl = _agg_nan_mode(farr)
                outx[b, nr, nc] = vl

    return outx


@nb.jit(nopython=True, nogil=True)
def _unique_coarsenf(xarr, ycells, xcells):
    xshape = xarr.shape
    bnds = xshape[0]
    rws = xshape[1]
    clms = xshape[2]
    outx = np.empty((bnds, rws // ycells, clms // xcells))
    for b in range(bnds):
        for r in range(0, rws, ycells):
            nr = r // ycells
            sr = r + ycells
            for c in range(0, clms, xcells):
                nc = c // xcells
                sc = c + xcells
                farr = xarr[b, r:sr, c:sc]
                vl = _agg_nan_unique(farr)
                outx[b, nr, nc] = vl

    return outx


@nb.jit(nopython=True, nogil=True)
def _entropy_coarsenf(xarr, ycells, xcells):
    xshape = xarr.shape
    bnds = xshape[0]
    rws = xshape[1]
    clms = xshape[2]
    outx = np.empty((bnds, rws // ycells, clms // xcells))
    for b in range(bnds):
        for r in range(0, rws, ycells):
            nr = r // ycells
            sr = r + ycells
            for c in range(0, clms, xcells):
                nc = c // xcells
                sc = c + xcells
                farr = xarr[b, r:sr, c:sc]
                vl = _agg_nan_entropy(farr)
                outx[b, nr, nc] = vl

    return outx


@nb.jit(nopython=True, nogil=True)
def _asm_coarsenf(xarr, ycells, xcells):
    xshape = xarr.shape
    bnds = xshape[0]
    rws = xshape[1]
    clms = xshape[2]
    outx = np.empty((bnds, rws // ycells, clms // xcells))
    for b in range(bnds):
        for r in range(0, rws, ycells):
            nr = r // ycells
            sr = r + ycells
            for c in range(0, clms, xcells):
                nc = c // xcells
                sc = c + xcells
                farr = xarr[b, r:sr, c:sc]
                vl = _agg_nan_asm(farr)
                outx[b, nr, nc] = vl

    return outx


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


def _coarsen(xarr, e_cells, stype):
    # trim extent if necesary
    data = xarr.data
    dshape = data.shape
    bnds = dshape[0]
    rws = dshape[1]
    clms = dshape[2]
    chg_shape = False
    if rws % e_cells[0] > 0:
        rws = (rws // e_cells[0]) * e_cells[0]
        chg_shape = True
    if clms % e_cells[0] > 0:
        clms = (clms // e_cells[1]) * e_cells[1]
        chg_shape = True
    if chg_shape:
        data = data[:, :rws, :clms]
        dshape = data.shape

    # change chunksize if necessary
    dchc = data.chunksize
    tcy = dchc[1]
    tcx = dchc[2]
    test_chunk = False
    if tcy % e_cells[0] > 0:
        tcy = (tcy // e_cells[0]) * e_cells[0]
        test_chunk = True
    if tcx % e_cells[1] > 0:
        tcx = (tcx // e_cells[1]) * e_cells[1]
        test_chunk = True

    nchunks = _get_chunk_array(dchc[0], rws, clms, tcy, tcx)

    if test_chunk:
        data = data.rechunk(nchunks)
        dchc = data.chunksize

    # create new chunksize and coords for aggregated raster
    ychunk = tuple(int(y / e_cells[0]) for y in nchunks[1])
    xchunk = tuple(int(x / e_cells[1]) for x in nchunks[2])
    schunks = (dchc[0], ychunk, xchunk)

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
    if stype.lower() == "mode":
        ffun = partial(_mode_coarsenf, ycells=e_cells[0], xcells=e_cells[1])
    elif stype.lower() == "asm":
        ffun = partial(_asm_coarsenf, ycells=e_cells[0], xcells=e_cells[1])
    elif stype.lower() == "entropy":
        ffun = partial(_entropy_coarsenf, ycells=e_cells[0], xcells=e_cells[1])
    else:
        ffun = partial(_unique_coarsenf, ycells=e_cells[0], xcells=e_cells[1])

    # map function to blocks and create xarray
    outx = da.map_blocks(
        ffun,
        data,
        chunks=schunks,
        dtype=data.dtype,
        meta=np.array((), dtype=data.dtype),
    )

    outxr = xr.DataArray(outx, tcoords, dims=xarr.dims)
    outxr = outxr.rio.write_crs(xarr.rio.crs)
    return outxr


def _local(xarr, stype):
    data = xarr.data
    dshape = data.shape
    dchunks = data.chunksize
    bnds = dshape[0]
    rws = dshape[1]
    clms = dshape[2]
    d = int(np.sqrt(bnds))
    nchunky = dchunks[1] // d
    nchunkx = dchunks[2] // d
    nchunks = _get_chunk_array(bnds, rws, clms, tcy=nchunky, tcx=nchunkx)
    outchunks = (1, (nchunks[1]), (nchunks[2]))
    xcoords = xarr.coords
    outcoords = {
        "band": [1],
        "y": xcoords["y"].values,
        "x": xcoords["x"].values,
    }
    data = data.rechunk(nchunks)
    if stype.lower() == "mode":
        ffun = _mode
    elif stype.lower() == "asm":
        ffun = _asm
    elif stype.lower() == "entropy":
        ffun = _entropy
    else:
        ffun = _unique

    outx = da.map_blocks(
        ffun,
        data,
        chunks=outchunks,
        dtype=data.dtype,
        meta=np.array((), dtype=data.dtype),
    )
    outxr = xr.DataArray(outx, outcoords, dims=xarr.dims)
    outxr = outxr.rio.write_crs(xarr.rio.crs)

    return outxr


def regions(raster, neighbors=4, unique_values=None):
    """Calculates the unique regions (patches) within a raser band.

    The approach is based ESRI's regions calculation:
    https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/region-group.htm

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on. All unique
        non zero values will be used to define unique regions.

    neighbors : connectivity value; 4 (rook) or 8 (queen)

                rook: False, True, False
                      True,  True, True
                      False, True, False

                Queen:True,  True, True
                      True,  True, True
                      True,  True, True

    unique_values :  array or list of values that represent
    zones from which regions will be made.Values not in the
    array or list will be grouped together to form one large
    zone. If unique_values = None, each value in the raster
    will be considered a zone and will be calculated up front.

    Returns
    -------
    Raster
        The resulting raster of unique regions values.
        The bands will have the
        same shape as the original Raster.

    """  # noqa: E501

    if not is_raster_class(raster) and not is_str(raster):
        raise TypeError(
            "First argument must be a Raster or path string to a raster"
        )
    elif is_str(raster):
        raster = Raster(raster)

    if raster.dtype == U64:
        rs = raster.copy()

    else:
        rs = raster.astype(U64)

    wd = np.array(
        [[False, True, False], [True, True, True], [False, True, False]]
    )
    if neighbors == 8:
        wd = np.array(
            [[True, True, True], [True, True, True], [True, True, True]]
        )

    data = rs._rs.data

    for bnd in range(data.shape[0]):
        data[bnd] = _create_labels(data[bnd], wd, unique_values)

    rs._rs.data = data
    return rs


def aggregate(raster, expand_cells, stype):
    """Creates a Raster of aggregated cell values
    for a new resolution.

    The approach is based ESRI's aggregate:
    https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/aggregate.htm
    and majority filter:
    https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/majority-filter.htm
    functions.



    Parameters
    ----------
    raster : input Raster object or path string

    expand_cells : array or list of the number of cells to expand in
    y and x directions

    stype : Summarization type (str). Valid opition are
    mean, std, var, max, min, unique prod, median,
    mode, sum, unique, entropy, asm

    Returns
    -------
    Raster
        The resulting raster of aggregated values.
    """  # noqa: E501

    rs = _get_rs(raster)
    stype = stype.lower()
    xda = rs._rs
    if stype == "mean":
        xda = xda.coarsen(
            dim={"y": expand_cells[0], "x": expand_cells[1]}, boundary="trim"
        ).mean()
    elif stype == "std":
        xda = xda.coarsen(
            dim={"y": expand_cells[0], "x": expand_cells[1]}, boundary="trim"
        ).std()
    elif stype == "var":
        xda = xda.coarsen(
            dim={"y": expand_cells[0], "x": expand_cells[1]}, boundary="trim"
        ).var()
    elif stype == "max":
        xda = xda.coarsen(
            dim={"y": expand_cells[0], "x": expand_cells[1]}, boundary="trim"
        ).max()
    elif stype == "min":
        xda = xda.coarsen(
            dim={"y": expand_cells[0], "x": expand_cells[1]}, boundary="trim"
        ).min()
    elif stype == "prod":
        xda = xda.coarsen(
            dim={"y": expand_cells[0], "x": expand_cells[1]}, boundary="trim"
        ).prod()
    elif stype == "median":
        xda = xda.coarsen(
            dim={"y": expand_cells[0], "x": expand_cells[1]}, boundary="trim"
        ).median()
    elif stype == "sum":
        xda = xda.coarsen(
            dim={"x": expand_cells[0], "y": expand_cells[1]}, boundary="trim"
        ).sum()
    else:
        xda = _coarsen(xda, expand_cells, stype)

    return Raster(xda)


def random_raster(
    raster_template,
    bands=1,
    rtype="normal",
    vls=[1, 0.5],
):
    """Creates a Raster of random values
    (normal, binomial, possion, weibull).

    The approach uses dask.arrays random functionality

    Parameters
    ----------
    raster_template : a template raster used to define rows,
    columns, crs, resolution, etc
    bands : integer of the number of bands needed
    rtype : random type valid values are normal (default),
    poisson, binomial, or weibull
    vls : additional parameter for distributions

    Returns
    -------
    Raster
        The resulting raster of random values.
    """
    rst = _get_rs(raster_template)

    if not isinstance(bands, int):
        try:
            bands = int(bands)
        except Exception as ex:
            print(ex)
            bands = 1

    bndLst = [1] * bands
    *_, rows, columns = rst.shape
    outrs = rst.get_bands(bndLst)
    data = outrs._rs.data
    t = rtype.lower()

    if t.startswith("p"):
        ndata = da.random.poisson(vls[0], size=[bands, rows, columns])
    elif t.startswith("b"):
        ndata = da.random.binomial(vls[0], vls[1], size=[bands, rows, columns])
    elif t.startswith("w"):
        ndata = da.random.weibull(vls[0], size=[bands, rows, columns])
    else:
        ndata = da.random.random(size=[bands, rows, columns])

    for b in range(bands):
        data[b] = ndata[b]

    outrs._rs.data = data
    return outrs


def constant_raster(raster_template, bands=1, value=1):
    """Creates a Raster with contant values.

    The approach uses dask.arrays full functionality

    Parameters
    ----------
    raster_template : a template raster used to define rows,
    columns, crs, resolution, etc
    bands : integer of the number of bands needed
    value : the constant value

    Returns
    -------
    Raster
        The resulting raster of constant values.
    """
    rst = _get_rs(raster_template)

    if not isinstance(bands, int):
        try:
            bands = int(bands)
        except Exception as ex:
            print(ex)
            bands = 1

    bndLst = [1] * bands
    *_, r, c = rst.shape
    outrs = rst.get_bands(bndLst)
    data = outrs._rs.data
    ndata = da.full([bands, r, c], value)
    for b in range(bands):
        data[b] = ndata[b]

    outrs._rs.data = data
    return outrs


def predict_model(raster, mdl):
    """Predicts cell estimates from a model using
    raster band values as predictors. Predictor bands
    correspond to the order of the predictor variables within the model.
    Outputs are raster surfaces with bands cell values depending
    on the type of model.


    The approach uses the models class predict funtion to
    estimate a new raster surface

    Parameters
    ----------
    raster : raster object of predictor variables
    where the (one band for each variable in the model)
    mdl - the model used to estimate new values

    Returns
    -------
    Raster
        The resulting raster of estmated values.
    """
    rs = _get_rs(raster)
    xarr = rs._rs
    xarrout = mdl.predict(xarr, mdl)

    return Raster(xarrout)


def local_stats(raster, stype):
    """Creates a Raster of summarrized values across bands.

    The approach is based on ESRI's local function:
    https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/an-overview-of-the-local-tools.htm

    Parameters
    ----------
    raster : input Raster object or path string

    expand_cells : array or list of the number of cells to expand in
    y and x directions

    stype : Summarization type (str). Valid opition are
    mean, std, var, max, min, maxband, minband, prod,
    sum, mode, median, unique, entropy, asm

    Returns
    -------
    Raster
        The resulting raster of aggregated values.
    """  # noqa: E501

    rs = _get_rs(raster)
    xda = rs._rs
    if stype == "mean":
        ndata = xda.reduce(np.mean, dim="band", keepdims=True)
    elif stype == "std":
        ndata = xda.reduce(np.std, dim="band", keepdims=True)
    elif stype == "var":
        ndata = xda.reduce(np.var, dim="band", keepdims=True)
    elif stype == "max":
        ndata = xda.reduce(np.max, dim="band", keepdims=True)
    elif stype == "min":
        ndata = xda.reduce(np.min, dim="band", keepdims=True)
    elif stype == "minband":
        ndata = xda.reduce(np.argmin, dim="band", keepdims=True)
    elif stype == "maxband":
        ndata = xda.reduce(np.argmax, dim="band", keepdims=True)
    elif stype == "prod":
        ndata = xda.reduce(np.prod, dim="band", keepdims=True)
    elif stype == "sum":
        ndata = xda.reduce(np.sum, dim="band", keepdims=True)
    elif stype == "median":
        ndata = xda.reduce(np.median, dim="band", keepdims=True)
    else:
        ndata = _local(xda, stype)

    return Raster(ndata)
