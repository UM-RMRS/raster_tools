###############################################################################
"""Author : John Hogland 11/2021
   contact info : john.s.hogland@usda.gov

   Description: surface module used to perform common surface analyses
   on Raster objects

   Requirements: functools, dask.array, numpy, numba
"""
###############################################################################
from functools import partial

import dask.array as da
import numba as nb
import numpy as np

from raster_tools import Raster
from raster_tools.raster import is_raster_class

from ._types import F32, U8, promote_dtype_to_float
from ._utils import is_str


@nb.jit(nopython=True, nogil=True)
def _surface_area_3d(xarr, res):  # 2d array of values
    dd = (res ** 2) * 2
    sd = res ** 2
    outx = np.empty_like(xarr)
    rws, cols = xarr.shape
    for rw in range(1, rws - 1):
        for cl in range(1, cols - 1):
            ta = 0
            e = xarr[rw, cl]
            a = xarr[rw + 1, cl - 1]
            b = xarr[rw - 1, cl + 1]
            c = xarr[rw - 1, cl + 1]
            d = xarr[rw, cl - 1]
            f = xarr[rw - 1, cl + 1]
            g = xarr[rw - 1, cl - 1]
            h = xarr[rw - 1, cl + 1]
            i = xarr[rw - 1, cl + 1]
            ea = ((dd + (e - a) ** 2) ** 0.5) * 0.5
            eb = ((sd + (e - b) ** 2) ** 0.5) * 0.5
            ab = ((sd + (a - b) ** 2) ** 0.5) * 0.5
            si = (ea + eb + ab) * 0.5
            ta += (si * (si - ea) * (si - eb) * (si - ab)) ** 0.5
            ec = ((dd + (e - c) ** 2) ** 0.5) * 0.5
            bc = ((sd + (b - c) ** 2) ** 0.5) * 0.5
            si = (ec + eb + bc) * 0.5
            ta += (si * (si - ec) * (si - eb) * (si - bc)) ** 0.5
            ef = ((sd + (e - f) ** 2) ** 0.5) * 0.5
            cf = ((sd + (c - f) ** 2) ** 0.5) * 0.5
            si = (ec + ef + cf) * 0.5
            ta += (si * (si - ec) * (si - ef) * (si - cf)) ** 0.5
            ei = ((dd + (e - i) ** 2) ** 0.5) * 0.5
            fi = ((sd + (f - i) ** 2) ** 0.5) * 0.5
            si = (ei + ef + fi) * 0.5
            ta += (si * (si - ei) * (si - ef) * (si - fi)) ** 0.5
            eh = ((sd + (e - h) ** 2) ** 0.5) * 0.5
            hi = ((sd + (h - i) ** 2) ** 0.5) * 0.5
            si = (ei + eh + hi) * 0.5
            ta += (si * (si - ei) * (si - eh) * (si - hi)) ** 0.5
            eg = ((dd + (e - g) ** 2) ** 0.5) * 0.5
            gh = ((sd + (g - h) ** 2) ** 0.5) * 0.5
            si = (eg + eh + gh) * 0.5
            ta += (si * (si - eg) * (si - eh) * (si - gh)) ** 0.5
            ed = ((sd + (e - d) ** 2) ** 0.5) * 0.5
            dg = ((sd + (d - g) ** 2) ** 0.5) * 0.5
            si = (eg + dg + ed) * 0.5
            ta += (si * (si - eg) * (si - ed) * (si - dg)) ** 0.5
            ad = ((sd + (a - d) ** 2) ** 0.5) * 0.5
            si = (ea + ed + ad) * 0.5
            ta += (si * (si - ea) * (si - ed) * (si - ad)) ** 0.5
            outx[rw, cl] = ta
    return outx


@nb.jit(nopython=True, nogil=True)
def _slope(xarr, c_x, c_y):
    outx = np.empty_like(xarr, dtype=F32)
    rws, cols = xarr.shape
    dcx = c_x * 8
    dcy = c_y * 8
    for y in range(1, rws - 1):
        for x in range(1, cols - 1):
            a = xarr[y + 1, x - 1]
            b = xarr[y + 1, x]
            c = xarr[y + 1, x + 1]
            d = xarr[y, x - 1]
            f = xarr[y, x + 1]
            g = xarr[y - 1, x - 1]
            h = xarr[y - 1, x]
            i = xarr[y - 1, x + 1]
            dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / dcx
            dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / dcy
            p = (dz_dx * dz_dx + dz_dy * dz_dy) ** 0.5
            outx[y, x] = np.arctan(p) * 57.29578
    return outx


@nb.jit(nopython=True, nogil=True)
def _slope_p(xarr, c_x, c_y):
    outx = np.empty_like(xarr, dtype=F32)
    rws, cols = xarr.shape
    dcx = c_x * 8
    dcy = c_y * 8
    for y in range(1, rws - 1):
        for x in range(1, cols - 1):
            a = xarr[y + 1, x - 1]
            b = xarr[y + 1, x]
            c = xarr[y + 1, x + 1]
            d = xarr[y, x - 1]
            f = xarr[y, x + 1]
            g = xarr[y - 1, x - 1]
            h = xarr[y - 1, x]
            i = xarr[y - 1, x + 1]
            dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / dcx
            dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / dcy
            p = (dz_dx * dz_dx + dz_dy * dz_dy) ** 0.5
            outx[y, x] = p
    return outx


@nb.jit(nopython=True, nogil=True)
def _aspect(xarr):
    outx = np.empty_like(xarr, dtype=F32)
    rws, cols = xarr.shape
    rd = 180 / np.pi
    for y in range(1, rws - 1):
        for x in range(1, cols - 1):
            g = xarr[y - 1, x - 1]
            h = xarr[y - 1, x]
            i = xarr[y - 1, x + 1]
            d = xarr[y, x - 1]
            f = xarr[y, x + 1]
            a = xarr[y + 1, x - 1]
            b = xarr[y + 1, x]
            c = xarr[y + 1, x + 1]
            dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / 8
            dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / 8

            if dz_dx == 0 and dz_dy == 0:
                outx[y, x] = -1.0
            else:
                aspect = np.arctan2(dz_dy, -dz_dx) * rd
                if aspect <= 90:
                    outx[y, x] = 90.0 - aspect
                else:
                    outx[y, x] = 450 - aspect

    return outx


@nb.jit(nopython=True, nogil=True)
def _curv(xarr, c_x, c_y):
    outx = np.empty_like(xarr, F32)
    rws, cols = xarr.shape
    ca = c_x * c_y
    for y in range(1, rws - 1):
        for x in range(1, cols - 1):
            d = (xarr[y + 1, x] + xarr[y - 1, x]) / 2 - xarr[y, x]
            e = (xarr[y, x + 1] + xarr[y, x - 1]) / 2 - xarr[y, x]
            outx[y, x] = -2 * (d + e) * 100 / ca
    return outx


def _northing(xarr):
    return np.cos(np.radians(xarr))


def _easting(xarr):
    return np.sin(np.radians(xarr))


def _get_rs(raster):
    if not is_raster_class(raster) and not is_str(raster):
        raise TypeError(
            "First argument must be a Raster or path string to a raster"
        )
    elif is_str(raster):
        raster = Raster(raster)

    rs = raster.copy()

    # Convert to float and fill nulls with nan, if needed
    upcast = False
    if raster._masked:
        data = rs._rs.data
        new_dtype = promote_dtype_to_float(raster.dtype)
        upcast = new_dtype != data.dtype
        if upcast:
            data = data.astype(new_dtype)
        data = da.where(~raster._mask, data, np.nan)
        rs._rs.data = data
    return rs


@nb.jit(nopython=True, nogil=True)
def _hillshade(xarr, c_x, c_y, azimuth=315, altitude=45):
    rad_c = np.pi / 180.0
    a_rad = (360.0 - azimuth + 90.0) * rad_c
    z_rad = (90 - altitude) * rad_c
    outx = np.empty_like(xarr, dtype=F32)
    rws, cols = xarr.shape
    dcx = c_x * 8
    dcy = c_y * 8
    for y in range(1, rws - 1):
        for x in range(1, cols - 1):
            a = xarr[y + 1, x - 1]
            b = xarr[y + 1, x]
            c = xarr[y + 1, x + 1]
            d = xarr[y, x - 1]
            f = xarr[y, x + 1]
            g = xarr[y - 1, x - 1]
            h = xarr[y - 1, x]
            i = xarr[y - 1, x + 1]
            dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / dcx
            dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / dcy
            slpr = np.arctan((dz_dx * dz_dx + dz_dy * dz_dy) ** 0.5)
            asr = asr = np.arctan2(dz_dy, -dz_dx)
            if not dz_dx == 0:
                if asr < 0:
                    asr = 2 * np.pi + asr
            else:
                if dz_dy > 0:
                    asr = np.pi / 2
                elif dz_dy < 0:
                    asr = 2 * np.pi - np.pi / 2
                else:
                    pass
            hs = 255.0 * (
                (np.cos(z_rad) * np.cos(slpr))
                + (np.sin(z_rad) * np.sin(slpr) * np.cos(a_rad - asr))
            )
            if hs < 0:
                hs = 0
            outx[y, x] = hs
    return outx


def surface_area_3d(raster):
    """Calculates the 3d surface area of each raster cell
    for each raster band.

    The approach is based on Jense 2004 description:
    Jenness 2004 (https://www.fs.usda.gov/treesearch/pubs/20437).

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on
        (typically an elevation surface).

    Returns
    -------
    Raster
        The resulting raster of 3d surface area. The
        bands will have the same shape as the original Raster.

    """
    rs = _get_rs(raster)
    data = rs._rs.data
    ffun = partial(_surface_area_3d, res=rs.resolution[0])
    for bnd in range(data.shape[0]):
        data[bnd] = data[bnd].map_overlap(
            ffun,
            depth={0: 1, 1: 1},
            boundary=np.nan,
            dtype=data.dtype,
            meta=np.array((), dtype=data.dtype),
        )
    rs._rs.data = data
    return rs


def slope(raster, degrees=True):
    """Calculates the slope (degrees) of each raster cell
    for each raster band.

    The approach is based ESRI's degree slope calculation:
    https://pro.arcgis.com/en/pro-app/latest/tool-reference/3d-analyst
    /how-slope-works.htm

    and

    xarray-spatial's:
    https://github.com/makepath/xarray-spatial/blob/master/xrspatial/slope.py

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on
        (typically an elevation surface).

    degrees : boolean to select degrees or percent slope values.
    Default is True (degrees)

    Returns
    -------
    Raster
        The resulting raster of slope values (degrees or percent). The
        bands will have the same shape as the original Raster.

    """
    rs = _get_rs(raster)
    data = rs._rs.data

    c_x, c_y = rs.resolution
    if degrees:
        ffun = partial(_slope, c_x=c_x, c_y=c_y)
    else:
        ffun = partial(_slope_p, c_x=c_x, c_y=c_y)
    for bnd in range(data.shape[0]):
        data[bnd] = data[bnd].map_overlap(
            ffun,
            depth={0: 1, 1: 1},
            boundary=np.nan,
            dtype=data.dtype,
            meta=np.array((), dtype=data.dtype),
        )

    rs._rs.data = data
    return rs


def aspect(raster):
    """Calculates the aspect of each raster cell for each raster band.

    The approach is based ESRI's Aspect calculation:
    https://pro.arcgis.com/en/pro-app/latest/tool-reference/3d-analyst
    /how-aspect-works.htm

    and

    xarray-spatial's: https://github.com/makepath/xarray-spatial/blob/master
    /xrspatial/aspect.py

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on
        (typically an elevation surface).

    Returns
    -------
    Raster
        The resulting raster of slope (degrees). The
        bands will have the same shape as the original Raster.

    """
    rs = _get_rs(raster)
    data = rs._rs.data

    for bnd in range(data.shape[0]):
        data[bnd] = data[bnd].map_overlap(
            _aspect,
            depth={0: 1, 1: 1},
            boundary=np.nan,
            dtype=data.dtype,
            meta=np.array((), dtype=data.dtype),
        )

    rs._rs.data = data
    return rs


def curvature(raster):
    """Calculates the curvature of each raster cell for each raster band.

    The approach is based ESRI's curvature calculation:
    https://pro.arcgis.com/en/pro-app/latest/tool-reference
    /3d-analyst/how-curvature-works.htm.

    and

    xarray-spatial's: https://github.com/makepath/xarray-spatial/blob
    /master/xrspatial/curvature.py

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on
        (typically an elevation surface).

    Returns
    -------
    Raster
        The resulting raster of curvature. The
        bands will have the same shape as the original Raster.

    """
    rs = _get_rs(raster)
    data = rs._rs.data

    c_x, c_y = rs.resolution
    ffun = partial(_curv, c_x=c_x, c_y=c_y)
    for bnd in range(data.shape[0]):
        data[bnd] = data[bnd].map_overlap(
            ffun,
            depth={0: 1, 1: 1},
            boundary=np.nan,
            dtype=data.dtype,
            meta=np.array((), dtype=data.dtype),
        )

    rs._rs.data = data
    return rs


def northing(raster, isAspect=False):
    """Calculates the nothing component of each raster cell
    for each raster band.

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on
        (typically an aspect or elevation surface).

    isAspect : Boolean to determine if a aspect raster is specified.
    Default is false and assumes that a elevation raster is used

    Returns
    -------
    Raster
        The resulting raster of northing (-1 to 1). The
        bands will have the same shape as the original Raster.

    """
    if not isAspect:
        raster = aspect(raster)

    rs = _get_rs(raster)
    data = rs._rs.data

    for bnd in range(data.shape[0]):
        data[bnd] = data[bnd].map_blocks(_northing)

    rs._rs.data = data
    return rs


def easting(raster, isAspect=False):
    """Calculates the easting component of each raster cell
    for each raster band.

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on
        (typically an aspect or elevation surface).

    isAspect : Boolean to determine if a aspect raster is specified.
    Default is false and assumes that a elevation raster is used

    Returns
    -------
    Raster
        The resulting raster of easting (-1 to 1). The
        bands will have the same shape as the original Raster.

    """
    if not isAspect:
        raster = aspect(raster)

    rs = _get_rs(raster)
    data = rs._rs.data

    for bnd in range(data.shape[0]):
        data[bnd] = data[bnd].map_blocks(_easting)

    rs._rs.data = data
    return rs


def hillshade(raster, azimuth=315, altitude=45):
    """Calculates the hillshade component of each raster cell
    for each raster band.
    based on ESRI's:
    https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst
    /how-hillshade-works.htm

    and

    xarray-spatial's: https://github.com/makepath/xarray-spatial/blob
    /master/xrspatial/hillshade.py

    hillshade algorithm

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on
        (typically a elevation surface).
    azimuth :  the azimuth of the sun (degrees)
    altitude : the altitude of the sun (degrees)

    Returns
    -------
    Raster
        The resulting raster of hillshade values (0-255, U8). The
        bands will have the same shape as the original Raster.

    """
    rs = _get_rs(raster)
    data = rs._rs.data
    c_x, c_y = rs.resolution
    ffun = partial(
        _hillshade, c_x=c_x, c_y=c_y, azimuth=azimuth, altitude=altitude
    )
    for bnd in range(data.shape[0]):
        data[bnd] = data[bnd].map_overlap(
            ffun,
            depth={0: 1, 1: 1},
            boundary=np.nan,
            dtype=data.dtype,
            meta=np.array((), dtype=data.dtype),
        )

    rs._rs.data = data
    rs = rs.replace_null(255)
    rs = rs.set_null_value(255)
    rs = rs.astype(U8)
    return rs
