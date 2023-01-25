from functools import partial

import dask.array as da
import geopandas as gpd
import numba as nb
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
import xarray as xr

from raster_tools.dtypes import F32, I8, is_bool, is_scalar, is_str
from raster_tools.raster import Raster, get_raster
from raster_tools.vector import get_vector


def _trim(x, slices):
    return x[tuple(slices)]


def _rasterize_geoms(geoms, xc, yc):
    # Convert geoms into a boolean (0/1) array using xc and yc for gridding.
    shape = (yc.size, xc.size)
    transform = xr.DataArray(
        da.zeros(shape), coords=(yc, xc), dims=("y", "x")
    ).rio.transform()
    # Use a MemoryFile to avoid writing to disk
    with rio.io.MemoryFile() as memfile, rio.open(
        memfile,
        mode="w+",
        driver="GTiff",
        width=xc.size,
        height=yc.size,
        count=1,
        crs=geoms.crs,
        transform=transform,
        dtype="uint8",
    ) as ds:
        ds.write(np.ones(shape, dtype="uint8"), 1)
        mask, _ = rio.mask.mask(ds, geoms.values, all_touched=True)
        return mask.squeeze()


@nb.jit(nopython=True, nogil=True)
def _length_sum(out, lengths, idxs):
    n = len(lengths)
    for i in range(n):
        out[idxs[i]] += lengths[i]


def _compute_lengths(geometry, targets, weights=None):
    """
    Overlay the targets circles on the geometries in `geometry`. This produces
    a clip of `geometry` for each target. The output is the computed length of
    each clip result. The output lengths array is the same shape as the input
    targets and the elements correspond to the same location in `targets`.

    See geopandas.overlay for more details about the overlay operation.
    """
    geom_df = geometry.to_frame()
    if weights is not None:
        geom_df["weight"] = weights
    targets_df = targets.to_frame()
    # Add column of index values so that clipped geoms can be mapped back to
    # the target that clipped them after the overlay operation. Column values
    # are preserved in overlay.
    targets_df["target_idx"] = np.arange(len(targets))
    overlay = geom_df.overlay(targets_df)
    overlay["len"] = overlay.length
    if weights is not None:
        overlay["len"] *= overlay.weight
    lengths = np.zeros(len(targets), dtype=F32)
    _length_sum(lengths, overlay.len.values, overlay.target_idx.values)
    return lengths


def _get_valid_slice(xc):
    # Build a slice that spans the valid data within the given coordinate array
    left = None
    i = 0
    while np.isnan(xc[i]):
        left = i + 1
        i += 1
    right = None
    i = -1
    while np.isnan(xc[i]):
        right = i
        i -= 1
    return slice(left, right)


def _len_coords(geom):
    return len(geom.coords)


def _calculate_number_vertices(geoms):
    has_exterior = geoms.geom_type == "Polygon"
    has_interior = ~geoms[has_exterior].interiors.isin([[]])
    nverts = 0
    if has_exterior.any():
        nverts += (
            geoms[has_exterior].exterior.apply(_len_coords).astype(int).sum()
        )
        if has_interior.any():
            nverts += (
                geoms[has_exterior][has_interior]
                .interior.apply(_len_coords)
                .astype(int)
                .sum()
            )
    nverts += (
        geoms[geoms.geom_type.isin(set(["LineString", "LinearRing"]))]
        .apply(_len_coords)
        .astype(int)
        .sum()
    )
    return nverts


# Buffering a point with the following resolutions yields polygons with the
# corresponding area percentage of a circle:
# resolution | % area of circle
# -----------+-----------------
#      16    |      99.8
#       8    |      99.4
#       4    |      97.4
#       2    |      90.0
# -----------+-----------------
# The resolution is the number of segments along a quarter arc of a circle so
# the total number of coordinate pairs for a buffered point is 4x the
# resolution. The default resolution for buffering in geopandas and shapely is
# 16 but this is very expensive, in terms of time and memory, for very large
# geometries or large collections of geometries. 8 is also expensive. 4 seems
# to be a good compromise between performance and accuracy.
BUFFER_RES = 4
# The max desired average vertices per geometry in a geoseries. Values higher
# than this tend to cause large fluctuations in memory usage. When combined
# with dask's scheduling, the system has a high chance of running out of
# memory.
VERTS_PER_GEOM_CUTOFF = 15
# The max number of geometries to work on at a time. Large collections of
# geometries rapidly consume large amounts of memory when buffered and
# rasterized. Keeping the number processed at a time low allows dask to better
# handle the memory load.
GEOM_BATCH_SIZE = 700


def _get_indices_and_lengths_core(geoms, xc, yc, radius, weight_values=None):
    # Rasterize to create mask of cells-of-interest and extract the locations
    indices = _rasterize_geoms(
        geoms.buffer(radius, BUFFER_RES), xc, yc
    ).nonzero()
    # Convert to points and buffer them out by the radius
    buffered_cells = gpd.GeoSeries.from_xy(
        xc[indices[1]], yc[indices[0]], crs=geoms.crs
    ).buffer(radius, BUFFER_RES)
    # Get length of all lines that fall within each buffered area.
    lengths = _compute_lengths(geoms, buffered_cells, weight_values)
    return indices, lengths


def _get_indices_and_lengths(geoms, xc, yc, radius, weight_values=None):
    """
    Finds the indices of cells where geoms fall within the given radius and
    computes the lengths of those geoms for each on of the cells.

    If the number of geoms is too large or the geoms are too complex, the data
    is broken into batches. Batches are used because this process is memory
    intensive and can cause large swings in memory use. Dask can inadvertently
    spawn too many workers so that when enough memory swings occur at once,
    system memory is totally consumed and the whole program is killed. The
    smaller batches smooth out the memory consumption and prevent crashing.
    """
    # Explode to turn multi-geometries into single geometries for vertex
    # counting
    geoms = geoms.explode(ignore_index=True)
    nverts = _calculate_number_vertices(geoms)
    n = len(geoms)
    ratio = nverts / n
    if (ratio > VERTS_PER_GEOM_CUTOFF and n > 1) or n > GEOM_BATCH_SIZE:
        n_batches = max(1, int(ratio / VERTS_PER_GEOM_CUTOFF))
        # Clamp between 1 and GEOM_BATCH_SIZE
        batch_size = min(GEOM_BATCH_SIZE, max(1, n // n_batches))
        dfs = []
        # Process the batches
        for i in range(0, n, batch_size):
            batch = geoms.iloc[i : i + batch_size]
            batch_weights = (
                None
                if weight_values is None
                else weight_values.iloc[i : i + batch_size]
            )
            idxs, lens = _get_indices_and_lengths_core(
                batch, xc, yc, radius, batch_weights
            )
            df = pd.DataFrame(
                {"idx0": idxs[0], "idx1": idxs[1], "length": lens}
            )
            dfs.append(df)
        # Merge results, summing co-located length values
        dfs = pd.concat(dfs, ignore_index=True)
        dfs = dfs.groupby(["idx0", "idx1"], as_index=False).sum()
        indices = tuple(dfs[["idx0", "idx1"]].to_numpy().T)
        lengths = dfs["length"].to_numpy()
    else:
        indices, lengths = _get_indices_and_lengths_core(
            geoms, xc, yc, radius, weight_values
        )
    return indices, lengths


def _length_chunk(xc, yc, gdf, radius, field=None, block_info=None):
    xc = xc.ravel()
    yc = yc.ravel()
    # The overlap operation extends the raster into regions with no coords
    # given. The coords in these regions are set to nan. Find the nan coord
    # spans and trim them off.
    xslice = _get_valid_slice(xc)
    yslice = _get_valid_slice(yc)
    xc_valid = xc[xslice]
    yc_valid = yc[yslice]
    out_shape = block_info[None]["chunk-shape"]
    is_3d = len(out_shape) == 3
    if is_3d:
        out_shape = out_shape[1:]

    # Filter out any geoms that could be problematic
    # TODO: throw error if bad geoms detected?
    gdf = gdf[(~gdf.is_empty) & gdf.is_valid]
    # Clip out any features outside of this chunk's area of interest
    gdf = gdf.clip(
        shapely.geometry.box(
            xc_valid.min(),
            yc_valid.min(),
            xc_valid.max(),
            yc_valid.max(),
        )
        .buffer(radius)
        .bounds
    )

    if len(gdf):
        geometry = gdf.geometry
        # Use field as weights
        weight_values = gdf[field] if field is not None else None
        if field is not None and is_bool(weight_values.dtype):
            weight_values = weight_values.astype(I8)
        indices, lengths = _get_indices_and_lengths(
            geometry, xc_valid, yc_valid, radius, weight_values
        )
        output = np.zeros(out_shape, dtype=F32)
        output[yslice, xslice][indices] = lengths
    else:
        output = np.zeros(out_shape, dtype=F32)

    if is_3d:
        output = np.expand_dims(output, axis=0)
    return output


def length(features, like_rast, radius, weighting_field=None):
    """
    Calculate a raster where each cell is the net length of all features within
    a given radius.

    This function returns a raster where each cell contains the sum of the
    lengths of all features that fall within a radius of the cell's center. The
    features are clipped to the circular neighborhood before the length is
    computed. Optional weighting can be added with `weighting_field`.

    Parameters
    ----------
    features : Vector, str
        The line features to compute lengths from.
    like_rast : Raster, str
        A raster to use as a reference grid and CRS. The output raster will be
        on the same grid.
    radius : scalar
        The radius around each grid cell's center to use. Features that fall
        within the radius are clipped to the resulting circle and their
        resulting lengths are summed. Additional weighting of the sum can be
        done using `weighting_field`. This should be in the same units as
        `like_rast`'s CRS.
    weighting_field : str, optional
        If not ``None``, this selects a field in `features` to use as a
        weighting factor in the sum of lengths.

    Returns
    -------
    Raster
        The resulting raster where each cell contains the sum of all feature
        lengths in the circular neighborhood determined by `radius`.


    """
    features = get_vector(features)
    like_rast = get_raster(like_rast)

    if not is_scalar(radius):
        raise TypeError(f"radius must be a scalar. Got type: {type(radius)}.")
    if radius <= 0:
        raise ValueError(f"radius must be greater than zero: Got: {radius!r}.")
    if like_rast.crs is None:
        raise ValueError("like_rast must have a CRS set.")
    if weighting_field is not None:
        if not is_str(weighting_field):
            raise TypeError("weighting_field must be a string.")
        if weighting_field not in features.data:
            raise ValueError(
                "weighting_field must be a field name in features. "
                f"Got: {weighting_field!r}."
            )
        field_dtype = features.data[weighting_field].dtype
        if not (is_scalar(field_dtype) or is_bool(field_dtype)):
            raise TypeError(
                "The field specified by weighting_field must be a scalar type."
                f" Found: {field_dtype}."
            )

    features = features.to_crs(like_rast.crs)
    gdf = features.data
    out_chunks = ((1,), *like_rast.xrs.chunks[1:])

    xc, yc = like_rast.get_chunked_coords()
    xdepth, ydepth = np.ceil(radius / np.abs(like_rast.resolution)).astype(int)

    oxc = da.overlap.overlap(
        xc, depth={0: 0, 1: 0, 2: xdepth}, boundary=np.nan
    )
    oyc = da.overlap.overlap(
        yc, depth={0: 0, 1: ydepth, 2: 0}, boundary=np.nan
    )
    overlap_chunks = ((1,), oyc.chunks[1], oxc.chunks[2])
    rasters = []
    for part in gdf.partitions:
        data = da.map_blocks(
            _length_chunk,
            oxc,
            oyc,
            gdf=part,
            radius=radius,
            field=weighting_field,
            chunks=overlap_chunks,
            meta=np.array((), dtype=F32),
        )
        rasters.append(data[0])
    data = da.stack(rasters)
    # Trim off overlap regions
    trim_func = partial(
        _trim,
        slices=(slice(None), slice(ydepth, -ydepth), slice(xdepth, -xdepth)),
    )
    data = da.map_blocks(
        trim_func,
        data,
        chunks=((1,) * data.shape[0], *out_chunks[1:]),
        meta=np.array((), dtype=F32),
    )
    data = da.sum(data, axis=0, keepdims=True)
    xrs = xr.DataArray(
        data, dims=("band", "y", "x"), coords=([1], like_rast.y, like_rast.x)
    ).rio.write_crs(like_rast.crs)
    return Raster(xrs)
