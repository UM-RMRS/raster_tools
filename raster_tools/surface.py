"""
Surface module used to perform common surface analyses on Raster objects.

ref: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/an-overview-of-the-surface-tools.htm
ref: https://github.com/makepath/xarray-spatial
"""  # noqa: E501
import dask.array as da
import numba as nb
import numpy as np
import xarray as xr

from raster_tools import focal
from raster_tools.dtypes import F32, F64, I32, U8, is_int
from raster_tools.masking import get_default_null_value
from raster_tools.raster import Raster, get_raster
from raster_tools.utils import make_raster_ds, single_band_mappable

__all__ = [
    "aspect",
    "curvature",
    "easting",
    "hillshade",
    "northing",
    "slope",
    "surface_area_3d",
    "tpi",
]

RADIANS_TO_DEGREES = 180 / np.pi
DEGREES_TO_RADIANS = np.pi / 180


def _finalize_rs(rs, data):
    # Invalidate edges
    mask = rs.mask.copy()
    mask[:, 0, :] = True
    mask[:, -1, :] = True
    mask[:, :, 0] = True
    mask[:, :, -1] = True
    coords = rs.xdata.coords
    dims = rs.xdata.dims
    xdata = xr.DataArray(data, coords=coords, dims=dims)
    xmask = xr.DataArray(mask, coords=coords, dims=dims)
    nv = get_default_null_value(data.dtype)
    xdata = xr.where(xmask, nv, xdata).rio.write_nodata(nv)
    ds = make_raster_ds(xdata, xmask)
    if rs.crs is not None:
        ds = ds.rio.write_crs(rs.crs)
    return Raster(ds, _fast_path=True)


@single_band_mappable
@nb.jit(nopython=True, nogil=True)
def _surface_area_3d(xarr, res):
    # TODO: handle non-symmetrical resolutions
    sd = res**2
    dd = sd * 2
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
    rs = get_raster(raster, null_to_nan=True)
    out_data = rs.data.map_overlap(
        _surface_area_3d,
        depth={0: 0, 1: 1, 1: 1},
        boundary=np.nan,
        dtype=F64,
        meta=np.array((), dtype=F64),
        res=rs.resolution[0],
    )
    return _finalize_rs(rs, out_data)


@single_band_mappable
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
    degrees : bool
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
    rs = get_raster(raster, null_to_nan=True)

    # Leave resolution sign as is
    out_data = rs.data.map_overlap(
        _slope,
        depth={0: 0, 1: 1, 1: 1},
        boundary=np.nan,
        dtype=F64,
        meta=np.array((), dtype=F64),
        res=rs.resolution,
        degrees=bool(degrees),
    )
    return _finalize_rs(rs, out_data)


@single_band_mappable
@nb.jit(nopython=True, nogil=True)
def _aspect(xarr):
    # ref: https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-aspect-works.htm  # noqa: E501
    # At each grid cell, the neighbors are labeled:
    #   a  b  c
    #   d |e| f
    #   g  h  i
    # where e is the current cell.
    outx = np.empty_like(xarr, dtype=F64)
    rows, cols = xarr.shape
    for ri in range(1, rows - 1):
        for ci in range(1, cols - 1):
            # See above for notes on ordering
            a = xarr[ri - 1, ci - 1]
            b = xarr[ri - 1, ci]
            c = xarr[ri - 1, ci + 1]
            d = xarr[ri, ci - 1]
            f = xarr[ri, ci + 1]
            g = xarr[ri + 1, ci - 1]
            h = xarr[ri + 1, ci]
            i = xarr[ri + 1, ci + 1]
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
    rs = get_raster(raster, null_to_nan=True)

    out_data = rs.data.map_overlap(
        _aspect,
        depth={0: 0, 1: 1, 1: 1},
        boundary=np.nan,
        dtype=F64,
        meta=np.array((), dtype=F64),
    )
    return _finalize_rs(rs, out_data)


@single_band_mappable
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
    rs = get_raster(raster, null_to_nan=True)

    out_data = rs.data.map_overlap(
        _curv,
        depth={0: 0, 1: 1, 1: 1},
        boundary=np.nan,
        dtype=F64,
        meta=np.array((), dtype=F64),
        res=np.abs(rs.resolution),
    )
    return _finalize_rs(rs, out_data)


def _northing_easting(rs, do_northing):
    trig = np.cos if do_northing else np.sin
    data = rs.data
    # Operate on rs.data rather than rs.xdata to avoid xarray's annoying
    # habit of dropping meta data.
    data = trig(np.radians(data))
    if rs._masked:
        data = da.where(rs.mask, rs.null_value, data)
    rs.xdata.data = data
    return rs


def northing(raster, is_aspect=False):
    """
    Calculates the nothing component of each raster cell for each band.

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on (typically an aspect or
        elevation surface).
    is_aspect : bool, optional
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
    """
    Calculates the easting component of each raster cell for each band.

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on (typically an aspect or
        elevation surface).
    is_aspect : bool
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


@single_band_mappable
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
    """
    Calculates the hillshade component of each raster cell for each band.

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
    rs = get_raster(raster, null_to_nan=True)
    # Specifically leave resolution sign as is
    out_data = rs.data.map_overlap(
        _hillshade,
        depth={0: 0, 1: 1, 1: 1},
        boundary=np.nan,
        dtype=F32,
        meta=np.array((), dtype=F32),
        res=rs.resolution,
        azimuth=azimuth,
        altitude=altitude,
    )

    rs = (
        _finalize_rs(rs, out_data)
        .replace_null(255)
        .set_null_value(255)
        .astype(U8)
    )
    return rs


def tpi(dem, annulus_inner, annulus_outer):
    """Compute the Topographic Position Index of a DEM.

    This function compares each elevation value to the mean of its neighborhood
    to produce a scale-dependent index that highlights ridges (positive values)
    and valleys (negative valleys). Values close to zero, indicate areas with
    constant slope such as plains (slope of zero). The basic function looks
    like this:
        ``tpi = int(dem - focalmean(dem, annulus_inner, annulus_outer) + 0.5)``
    An annulus (donut) is used to select the neighborhood of each pixel. Larger
    radii values select features at larger scales.

    Parameters
    ----------
    dem : Raster or path str
        The DEM raster to use for TPI analysis.
    annulus_inner : int
        The inner radius of the annulus. If ``0``, a circle of radius
        `annulus_outer` is used to select the neighborhood.
    annulus_outer : int
        The outer radius of the annulus. Must be greater than `annulus_inner`.

    Returns
    -------
    tpi : Raster
        The resulting TPI index for `dem` at the scale determined by the
        annulus radii.

    References
    ----------
    * Weiss AD (2001) Topographic position and landforms analysis. Conference
      poster for ‘21st Annual ESRI International User Conference’, 9–13 July
      2001, San Diego, CA. Available at
      http://www.jennessent.com/arcview/TPI_Weiss_poster.htm

    """
    dem = get_raster(dem)
    if not is_int(annulus_inner) or not is_int(annulus_outer):
        raise TypeError(
            "annulus_inner and annulus_outer must be integer values"
        )
    if annulus_inner < 0:
        raise ValueError("annulus_inner must be greater or equal to zero")
    if annulus_outer < 1:
        raise ValueError("annulus_outer must be greater than zero")
    if annulus_inner >= annulus_outer:
        raise ValueError("annulus_inner must be less than annulus_outer")

    if annulus_inner == 0:
        radii = annulus_outer
    else:
        radii = (annulus_inner, annulus_outer)
    rs_tpi = ((dem - focal.focal(dem, "mean", radii)) + 0.5).astype(I32, False)
    if dem._masked:
        rs_tpi = rs_tpi.set_null_value(get_default_null_value(I32))
    return rs_tpi
