import warnings

import dask.array as da
import numpy as np
import xarray as xr

from raster_tools.raster import Raster, _raster_like, get_default_null_value

from ._utils import is_bool, is_float, is_int, is_scalar


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

    xrs = [r._rs for r in rasters]
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
    return _raster_like(rasters[0], rs, mask=mask, null_value=new_nv)
