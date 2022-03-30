"""
Surface module used to perform common surface analyses on Raster objects.

ref: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/an-overview-of-the-surface-tools.htm
ref: https://github.com/makepath/xarray-spatial
"""  # noqa: E501
from functools import partial

import dask.array as da
import numba as nb
import numpy as np

from raster_tools.dtypes import F32, F64, U8, get_default_null_value
from raster_tools.raster import get_raster

__all__ = [
    "aspect",
    "curvature",
    "easting",
    "hillshade",
    "northing",
    "slope",
    "surface_area_3d",
]

RADIANS_TO_DEGREES = 180 / np.pi
DEGREES_TO_RADIANS = np.pi / 180


def _finalize_rs(rs, data):
    # Invalidate edges
    mask = rs._mask
    mask[:, 0, :] = True
    mask[:, -1, :] = True
    mask[:, :, 0] = True
    mask[:, :, -1] = True
    nv = rs.null_value
    if not rs._masked:
        nv = get_default_null_value(data.dtype)
    data = da.where(mask, nv, data)
    rs._data = data
    rs._mask = mask
    return rs


def _map_surface_func(
    data, func, out_dtype, depth={0: 1, 1: 1}, boundary=np.nan
):
    out_data = da.empty_like(data, dtype=out_dtype)
    for bnd in range(data.shape[0]):
        out_data[bnd] = data[bnd].map_overlap(
            func,
            depth=depth,
            boundary=boundary,
            dtype=out_dtype,
            meta=np.array((), dtype=out_dtype),
        )
    return out_data


@nb.jit(nopython=True, nogil=True)
def _surface_area_3d(xarr, res):
    # TODO: handle non-symmetrical resolutions
    dd = (res**2) * 2
    sd = res**2
    outx = np.empty_like(xarr, dtype=F64)
    rows, cols = xarr.shape
    for rw in range(1, rows - 1):
        for cl in range(1, cols - 1):
            ta = 0
            e = xarr[rw, cl]
            a = xarr[rw + 1, cl - 1]
            b = xarr[rw + 1, cl]
            c = xarr[rw + 1, cl + 1]
            d = xarr[rw, cl - 1]
            f = xarr[rw, cl + 1]
            g = xarr[rw - 1, cl - 1]
            h = xarr[rw - 1, cl]
            i = xarr[rw - 1, cl + 1]
            ea = np.sqrt(dd + (e - a) ** 2) * 0.5
            eb = np.sqrt(sd + (e - b) ** 2) * 0.5
            ab = np.sqrt(sd + (a - b) ** 2) * 0.5
            si = (ea + eb + ab) * 0.5
            ta += np.sqrt(si * (si - ea) * (si - eb) * (si - ab))
            ec = np.sqrt(dd + (e - c) ** 2) * 0.5
            bc = np.sqrt(sd + (b - c) ** 2) * 0.5
            si = (ec + eb + bc) * 0.5
            ta += np.sqrt(si * (si - ec) * (si - eb) * (si - bc))
            ef = np.sqrt(sd + (e - f) ** 2) * 0.5
            cf = np.sqrt(sd + (c - f) ** 2) * 0.5
            si = (ec + ef + cf) * 0.5
            ta += np.sqrt(si * (si - ec) * (si - ef) * (si - cf))
            ei = np.sqrt(dd + (e - i) ** 2) * 0.5
            fi = np.sqrt(sd + (f - i) ** 2) * 0.5
            si = (ei + ef + fi) * 0.5
            ta += np.sqrt(si * (si - ei) * (si - ef) * (si - fi))
            eh = np.sqrt(sd + (e - h) ** 2) * 0.5
            hi = np.sqrt(sd + (h - i) ** 2) * 0.5
            si = (ei + eh + hi) * 0.5
            ta += np.sqrt(si * (si - ei) * (si - eh) * (si - hi))
            eg = np.sqrt(dd + (e - g) ** 2) * 0.5
            gh = np.sqrt(sd + (g - h) ** 2) * 0.5
            si = (eg + eh + gh) * 0.5
            ta += np.sqrt(si * (si - eg) * (si - eh) * (si - gh))
            ed = np.sqrt(sd + (e - d) ** 2) * 0.5
            dg = np.sqrt(sd + (d - g) ** 2) * 0.5
            si = (eg + dg + ed) * 0.5
            ta += np.sqrt(si * (si - eg) * (si - ed) * (si - dg))
            ad = np.sqrt(sd + (a - d) ** 2) * 0.5
            si = (ea + ed + ad) * 0.5
            ta += np.sqrt(si * (si - ea) * (si - ed) * (si - ad))
            outx[rw, cl] = ta
    return outx


def surface_area_3d(raster):
    """Calculates the 3D surface area of each raster cell for each raster band.

    The approach is based on Jense, 2004.

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on (typically an elevation
        surface).

    Returns
    -------
    Raster
        The resulting raster of 3D surface area. The bands will have the same
        shape as the original Raster.

    References
    ----------
    * `Jense, 2004 <https://www.fs.usda.gov/treesearch/pubs/20437>`_

    """
    rs = get_raster(raster, null_to_nan=True).copy()
    data = rs._data
    ffun = partial(_surface_area_3d, res=rs.resolution[0])
    out_data = _map_surface_func(data, ffun, F64)
    return _finalize_rs(rs, out_data)


@nb.jit(nopython=True, nogil=True)
def _slope(xarr, res, degrees):
    # ref: https://pro.arcgis.com/en/pro-app/latest/tool-reference/3d-analyst/how-slope-works.htm  # noqa: E501
    outx = np.empty_like(xarr, dtype=F64)
    rows, cols = xarr.shape
    dx = res[0] * 8
    dy = res[1] * 8
    for ri in range(1, rows - 1):
        for ci in range(1, cols - 1):
            a = xarr[ri + 1, ci - 1]
            b = xarr[ri + 1, ci]
            c = xarr[ri + 1, ci + 1]
            d = xarr[ri, ci - 1]
            f = xarr[ri, ci + 1]
            g = xarr[ri - 1, ci - 1]
            h = xarr[ri - 1, ci]
            i = xarr[ri - 1, ci + 1]
            dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / dx
            dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / dy
            rise_run = np.sqrt((dzdx * dzdx) + (dzdy * dzdy))
            if degrees:
                outx[ri, ci] = np.arctan(rise_run) * RADIANS_TO_DEGREES
            else:
                # Percent slope. See the reference above
                outx[ri, ci] = rise_run
    return outx


def slope(raster, degrees=True):
    """Calculates the slope (degrees) of each raster cell for each raster band.

    The approach is based on ESRI's degree slope calculation.

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on
        (typically an elevation surface).
    degrees : boolean
        Indicates whether to output as degrees or percent slope values.
        Default is True (degrees).

    Returns
    -------
    Raster
        The resulting raster of slope values (degrees or percent). The
        bands will have the same shape as the original Raster.

    References
    ----------
    * `ESRI slope <https://pro.arcgis.com/en/pro-app/latest/tool-reference/3d-analyst/how-slope-works.htm>`_

    """  # noqa: E501
    rs = get_raster(raster, null_to_nan=True).copy()
    data = rs._data

    ffun = partial(_slope, res=rs.resolution, degrees=bool(degrees))
    out_data = _map_surface_func(data, ffun, F64)
    return _finalize_rs(rs, out_data)


@nb.jit(nopython=True, nogil=True)
def _aspect(xarr):
    # ref: https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-aspect-works.htm  # noqa: E501
    # At each grid cell, the neighbors are labeled:
    #   g  h  i
    #   d |e| f
    #   a  b  c
    # Where e is the current cell.  This is reversed along the vertical dim
    # compared to ESRI's notes because this package orients the data arrays
    # such that the north/south coordinate increases with the row dim.
    outx = np.empty_like(xarr, dtype=F64)
    rows, cols = xarr.shape
    for ri in range(1, rows - 1):
        for ci in range(1, cols - 1):
            # See above for notes on ordering
            a = xarr[ri + 1, ci - 1]
            b = xarr[ri + 1, ci]
            c = xarr[ri + 1, ci + 1]
            d = xarr[ri, ci - 1]
            f = xarr[ri, ci + 1]
            g = xarr[ri - 1, ci - 1]
            h = xarr[ri - 1, ci]
            i = xarr[ri - 1, ci + 1]
            dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / 8
            dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / 8

            if dzdx == 0 and dzdy == 0:
                outx[ri, ci] = -1.0
            else:
                aspect = np.arctan2(dzdy, -dzdx) * RADIANS_TO_DEGREES
                if aspect <= 90:
                    outx[ri, ci] = 90.0 - aspect
                else:
                    outx[ri, ci] = 450 - aspect
    return outx


def aspect(raster):
    """Calculates the aspect of each raster cell for each raster band.

    The approach is based on ESRI's aspect calculation.

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on (typically an elevation
        surface).

    Returns
    -------
    Raster
        The resulting raster of aspect (degrees). The bands will have the same
        shape as the original Raster.

    References
    ----------
    * `ESRI aspect <https://pro.arcgis.com/en/pro-app/latest/tool-reference/3d-analyst/how-aspect-works.htm>`_

    """  # noqa: E501
    rs = get_raster(raster, null_to_nan=True).copy()
    data = rs._data

    out_data = _map_surface_func(data, _aspect, F64)
    return _finalize_rs(rs, out_data)


@nb.jit(nopython=True, nogil=True)
def _curv(xarr, res):
    # ref: https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-curvature-works.htm  # noqa: E501
    # Neighbors are labeled like so:
    #   z1  z2  z3
    #   z4 |z5| z6
    #   z7  z8  z9
    outx = np.empty_like(xarr, dtype=F64)
    rows, cols = xarr.shape
    ca = res[0] * res[1]
    for ri in range(1, rows - 1):
        for ci in range(1, cols - 1):
            z2 = xarr[ri - 1, ci]
            z4 = xarr[ri, ci - 1]
            z5 = xarr[ri, ci]
            z6 = xarr[ri, ci + 1]
            z8 = xarr[ri + 1, ci]
            dl2 = ((z4 + z6) / 2) - z5
            el2 = ((z2 + z8) / 2) - z5
            outx[ri, ci] = -2 * (dl2 + el2) * 100 / ca
    return outx


def curvature(raster):
    """Calculates the curvature of each raster cell for each raster band.

    The approach is based on ESRI's curvature calculation.

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on (typically an elevation
        surface).

    Returns
    -------
    Raster
        The resulting raster of curvature. The bands will have the same shape
        as the original Raster.

    References
    ----------
    * `ESRI curvature <https://pro.arcgis.com/en/pro-app/latest/tool-reference/3d-analyst/how-curvature-works.htm>`_

    """  # noqa: E501
    rs = get_raster(raster, null_to_nan=True).copy()
    data = rs._data

    ffun = partial(_curv, res=rs.resolution)
    out_data = _map_surface_func(data, ffun, F64)
    return _finalize_rs(rs, out_data)


def _northing_easting(rs, do_northing):
    trig = np.cos if do_northing else np.sin
    data = rs._data
    # Operate on rs._data rather than rs.xrs to avoid xarray's annoying
    # habit of dropping meta data.
    data = trig(np.radians(data))
    if rs._masked:
        data = da.where(rs._mask, rs.null_value, data)
    rs._data = data
    return rs


def northing(raster, is_aspect=False):
    """Calculates the nothing component of each raster cell for each raster
    band.

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on (typically an aspect or
        elevation surface).
    is_aspect : boolean to determine if a aspect raster is specified.
        Indicates if `raster` is an aspect raster or an elevation raster.
        The default is false and assumes that an elevation raster is used.

    Returns
    -------
    Raster
        The resulting raster of northing (-1 to 1). The bands will have the
        same shape as the original Raster.

    """
    if not is_aspect:
        raster = aspect(raster)

    rs = get_raster(raster, null_to_nan=True).copy()
    return _northing_easting(rs, True)


def easting(raster, is_aspect=False):
    """Calculates the easting component of each raster cell for each raster
    band.

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on (typically an aspect or
        elevation surface).
    is_aspect : boolean to determine if a aspect raster is specified.
        Indicates if `raster` is an aspect raster or an elevation raster.
        The default is false and assumes that an elevation raster is used.

    Returns
    -------
    Raster
        The resulting raster of easting (-1 to 1). The bands will have the same
        shape as the original raster.

    """
    if not is_aspect:
        raster = aspect(raster)

    rs = get_raster(raster, null_to_nan=True).copy()
    return _northing_easting(rs, False)


@nb.jit(nopython=True, nogil=True)
def _hillshade(xarr, res, azimuth, altitude):
    # ref: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-hillshade-works.htm  # noqa: E501
    a_rad = (360.0 - azimuth + 90.0) * DEGREES_TO_RADIANS
    z_rad = (90 - altitude) * DEGREES_TO_RADIANS
    # Use F32 since the result will be cast to U8 anyway
    outx = np.empty_like(xarr, dtype=F32)
    rows, cols = xarr.shape
    dx = res[0] * 8
    dy = res[1] * 8
    for ri in range(1, rows - 1):
        for ci in range(1, cols - 1):
            a = xarr[ri + 1, ci - 1]
            b = xarr[ri + 1, ci]
            c = xarr[ri + 1, ci + 1]
            d = xarr[ri, ci - 1]
            f = xarr[ri, ci + 1]
            g = xarr[ri - 1, ci - 1]
            h = xarr[ri - 1, ci]
            i = xarr[ri - 1, ci + 1]
            dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / dx
            dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / dy
            slpr = np.arctan((dzdx * dzdx + dzdy * dzdy) ** 0.5)
            asr = asr = np.arctan2(dzdy, -dzdx)
            if not dzdx == 0:
                if asr < 0:
                    asr = 2 * np.pi + asr
            else:
                if dzdy > 0:
                    asr = np.pi / 2
                elif dzdy < 0:
                    asr = 2 * np.pi - np.pi / 2
                else:
                    pass
            hs = 255.0 * (
                (np.cos(z_rad) * np.cos(slpr))
                + (np.sin(z_rad) * np.sin(slpr) * np.cos(a_rad - asr))
            )
            if hs < 0:
                hs = 0
            outx[ri, ci] = hs
    return outx


def hillshade(raster, azimuth=315, altitude=45):
    """Calculates the hillshade component of each raster cell for each raster
    band.

    This approach is based on ESRI's hillshade calculation.

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on (typically a elevation
        surface).
    azimuth :  scalar
        The azimuth of the sun (degrees).
    altitude : scalar
        The altitude of the sun (degrees).

    Returns
    -------
    Raster
        The resulting raster of hillshade values (0-255, uint8). The bands will
        have the same shape as the original Raster. The null value is set to
        255.

    References
    ----------
    * `ESRI hillshade <https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-hillshade-works.htm>`_

    """  # noqa: E501
    rs = get_raster(raster, null_to_nan=True).copy()
    data = rs._data
    ffun = partial(
        _hillshade, res=rs.resolution, azimuth=azimuth, altitude=altitude
    )
    out_data = _map_surface_func(data, ffun, F32)

    rs = _finalize_rs(rs, out_data)
    rs = rs.replace_null(255)
    rs = rs.set_null_value(255)
    rs = rs.astype(U8)
    return rs
