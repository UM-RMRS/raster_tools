from functools import partial
from itertools import product

import dask.array as da
import numpy as np
import rasterio as rio
import xarray as xr
from affine import Affine
from dask.base import is_dask_collection, tokenize
from dask.highlevelgraph import HighLevelGraph
from odc.geo.geobox import GeoBox, GeoboxTiles
from odc.geo.math import resolve_fill_value, resolve_nodata
from odc.geo.warp import rio_reproject
from odc.geo.xr import xr_coords
from pyproj import Transformer

from raster_tools._grids import reproject_grid
from raster_tools.dtypes import (
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    get_dtype_min_max,
    is_bool,
    is_float,
)
from raster_tools.masking import get_default_null_value
from raster_tools.raster import (
    Raster,
    dataarray_to_xr_raster_ds,
    get_mask_from_data,
    get_raster,
)

__all__ = [
    "reproject",
]

SUPPORTED_RESAMPLE_METHODS = {m.name for m in rio.warp.SUPPORTED_RESAMPLING}


def reproject(
    raster, crs_or_geobox=None, resample_method="nearest", resolution=None
):
    """Reproject to a new projection or resolution.

    This is a lazy operation.

    Parameters
    ----------
    raster : str, Raster
        The raster to reproject.
    crs_or_geobox : int, str, CRS, GeoBox, optional
        The target grid to reproject the raster to. This can be a projection
        string, EPSG code string or integer, a CRS object, or a GeoBox object.
        When a CRS is given, the output grid is the default one GDAL suggests
        for the reprojection (see
        ``rasterio.warp.calculate_default_transform``), which handles rasters
        that contain a pole or cross the antimeridian. GDAL picks the cell
        size from the distance between the raster's corners, so the cells can
        be coarse for such rasters: a polar stereographic raster containing
        the pole becomes a few rows of cells a degree or more across in a
        geographic CRS. Pass `resolution` to get finer cells. `resolution`
        replaces GDAL's cell size in the new CRS, keeping the grid's top-left
        corner, and the grid is grown to whole cells so that it still covers
        GDAL's extent. A geographic grid that this carries past latitude 90
        or -90 is moved back to end at the pole; if its rows would span more
        than 180 degrees, their height is shortened so that they span -90 to
        90 exactly. If `crs_or_geobox` is not provided, `resolution` must be
        specified.
    resample_method : str, optional
        The data resampling method to use. Null pixels are ignored for all
        methods. Some methods require specific versions of GDAL. These are
        noted below. Valid methods are:

        'nearest'
            Nearest neighbor resampling. This is the default.
        'bilinear'
            Bilinear resampling.
        'cubic'
            Cubic resampling.
        'cubic_spline'
            Cubic spline resampling.
        'lanczos'
            Lanczos windowed sinc resampling.
        'average'
            Average resampling, computes the weighted average of all
            contributing pixels.
        'mode'
            Mode resampling, selects the value which appears most often.
        'max'
            Maximum resampling. (GDAL 2.0+)
        'min'
            Minimum resampling. (GDAL 2.0+)
        'med'
            Median resampling. (GDAL 2.0+)
        'q1'
            Q1, first quartile resampling. (GDAL 2.0+)
        'q3'
            Q3, third quartile resampling. (GDAL 2.0+)
        'sum'
            Sum, compute the weighted sum. (GDAL 3.1+)
        'rms'
            RMS, root mean square/quadratic mean. (GDAL 3.3+)
    resolution : int, float, tuple of int or float, optional
        The desired resolution of the reprojected raster. If `crs_or_geobox` is
        unspecified, this is used to reproject to the new resolution while
        maintaining the same CRS. One of `crs_or_geobox` or `resolution` must
        be provided. Both can also be provided. Changing the resolution keeps
        the grid's origin.

    Returns
    -------
    Raster
        The reprojected raster on the new grid. Cells that the source does not
        cover are null. A raster with no null value is given the default null
        value for its dtype. Its valid cells stay valid, even where they hold
        that value, except in a 64-bit integer raster, where those cells are
        null or, with newer GDAL, have their value nudged off it. 64-bit
        integer values beyond 2**53 lose precision when reprojected, because
        GDAL warps them through double precision.

    """
    raster = get_raster(raster)
    if resample_method not in SUPPORTED_RESAMPLE_METHODS:
        raise ValueError(
            f"Unsupported resampling method provided: {resample_method!r}. "
            "Supported methods: {}".format(
                ", ".join(sorted(SUPPORTED_RESAMPLE_METHODS))
            )
        )
    if resolution is not None and resolution <= 0:
        raise ValueError("Resolution must be a positive value")
    if crs_or_geobox is None:
        if resolution is None:
            raise ValueError("Must supply either crs_or_geobox or resolution")
        dst_gb = raster.geobox.zoom_to(resolution=resolution)
    elif isinstance(crs_or_geobox, GeoBox):
        dst_gb = crs_or_geobox
        if resolution is not None:
            dst_gb = dst_gb.zoom_to(resolution=resolution)
    else:
        dst_gb = reproject_grid(
            raster.geobox, crs_or_geobox, resolution=resolution
        )
    if dst_gb == raster.geobox:
        return raster.copy()

    nv = (
        raster.null_value
        if raster._masked
        else get_default_null_value(raster.dtype)
    )
    working = _get_working_type(raster.dtype, raster._masked)
    if working is not None:
        reprojected, xmask = _reproject_with_mask(
            raster, dst_gb, resample_method, nv, *working
        )
    else:
        reprojected = _reproject_xr(
            raster.xdata, dst_gb, resample_method, dst_nodata=nv
        )
        xmask = None
    reprojected = reprojected.rio.write_nodata(nv)
    # reproject sets a "nodata" attribute that has type int for whole-number
    # null values (e.g. -3.4028235e+38 becomes a VERY long python int). This
    # can cause all kinds of issues for downstream operations
    # since .rio will get confused. We are already handling null values so it
    # can be dropped.
    reprojected.attrs.pop("nodata", None)
    if "longitude" in reprojected.dims:
        # odc-geo will rename x/y to lon/lat for lon/lat based projections, so
        # revert to x/y
        lonlat_to_xy = {"longitude": "x", "latitude": "y"}
        reprojected = reprojected.rename(lonlat_to_xy)
        if xmask is not None:
            xmask = xmask.rename(lonlat_to_xy)
    ds = dataarray_to_xr_raster_ds(reprojected, xmask=xmask)
    return Raster(ds, _fast_path=True)


# Unmasked integer data is warped in the next wider signed type, whose
# minimum no source value can equal
_WIDER_INT_TYPE = {U8: I16, I8: I16, U16: I32, I16: I32, U32: I64, I32: I64}


def _get_working_type(dtype, masked):
    """Get the type and null value to warp data in, if one is needed.

    Warping marks the cells that the source does not cover, and null source
    cells, with a null value. That value must not occur in the source's
    valid data, or valid cells that hold it come out null or altered. The
    default null value given to an unmasked raster can occur in its data, and
    bool data has no spare value at all. For those, this returns a working
    dtype and null value that cannot collide, so that the mask can be taken
    from the warped data. ``None`` is returned when the raster's own null
    value can be used, or when its dtype has no wider type (64-bit
    integers).
    """
    dtype = np.dtype(dtype)
    if is_bool(dtype):
        # A sum resample can count past 255 cells, so use a type wider than
        # uint8
        return I16, I16.type(np.iinfo(I16).min)
    if masked:
        return None
    if is_float(dtype):
        # Source NaN values are null when warping in any case
        return dtype, dtype.type(np.nan)
    if dtype in _WIDER_INT_TYPE:
        wide = _WIDER_INT_TYPE[dtype]
        return wide, wide.type(np.iinfo(wide).min)
    return None


def _reproject_with_mask(
    raster, dst_gb, resample_method, nv, work_dtype, work_nv
):
    """Reproject data in a working dtype and return it with its mask.

    The data is warped as `work_dtype` with `work_nv` marking null and
    uncovered cells, and the mask is taken from that. The result is cast back
    to the raster's dtype and its null cells get `nv`.
    """
    dtype = raster.dtype
    src = raster.xdata.astype(work_dtype)
    if raster._masked:
        src = src.where(~raster.xmask, work_nv)
    # Unmasked integer data has no null cells, so it gets no source null
    # value, which also keeps GDAL on the kernels it uses for the data in its
    # own dtype. Float sources treat NaN as null, as they always have.
    src_nv = work_nv if raster._masked or is_float(dtype) else None
    warped = _reproject_xr(
        src, dst_gb, resample_method, dst_nodata=work_nv, src_nodata=src_nv
    )
    xmask = get_mask_from_data(warped, work_nv)
    if is_bool(dtype):
        data = warped != 0
    elif is_float(dtype):
        data = warped
    else:
        # GDAL clamps to the output type's range, so do the same for the
        # source's type
        lo, hi = get_dtype_min_max(dtype)
        data = warped.clip(lo, hi).astype(dtype)
    data = data.where(~xmask, nv)
    data.attrs = {
        k: v
        for k, v in warped.attrs.items()
        if k not in ("nodata", "_FillValue")
    }
    data.encoding = dict(warped.encoding)
    xmask.attrs = {}
    return data, xmask


# Attributes that do not carry over to a reprojected array. These are the ones
# odc-geo drops.
_REPROJECT_SKIP_ATTRS = frozenset(
    ("crs", "crs_wkt", "grid_mapping", "gcps", "epsg", "nodata", "_FillValue")
)


def _reproject_xr(src, dst_gb, resampling, dst_nodata, src_nodata="auto"):
    """Reproject DataArray `src` onto `dst_gb`.

    A dask-backed `src` is warped by :func:`_warp_dask_array`, which reads
    every source block that each output chunk needs, into what
    ``src.odc.reproject`` would give without its null value attributes. Any
    other `src` is passed to ``src.odc.reproject``.
    """
    if not is_dask_collection(src):
        return src.odc.reproject(
            dst_gb,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
        )
    ydim = src.odc.ydim
    src_nodata = resolve_nodata(src_nodata, src.dtype, src.odc.nodata)
    dst_nodata = resolve_nodata(dst_nodata, src.dtype, src_nodata)
    fill_value = resolve_fill_value(dst_nodata, src_nodata, src.dtype)
    data = _warp_dask_array(
        src.data,
        src.odc.geobox,
        dst_gb,
        ydim,
        resampling,
        src_nodata,
        fill_value,
    )
    spatial_dims = set(src.odc.spatial_dims)
    crs_coord = src.odc.crs_coord
    coords = {
        name: coord
        for name, coord in src.coords.items()
        if spatial_dims.isdisjoint(coord.dims)
        and (crs_coord is None or name != crs_coord.name)
    }
    coords.update(xr_coords(dst_gb))
    dims = (*src.dims[:ydim], *dst_gb.dimensions, *src.dims[ydim + 2 :])
    attrs = {
        k: v for k, v in src.attrs.items() if k not in _REPROJECT_SKIP_ATTRS
    }
    out = xr.DataArray(data, coords=coords, dims=dims, attrs=attrs)
    out.encoding["grid_mapping"] = "spatial_ref"
    return out


def _warp_dask_array(
    data, src_gb, dst_gb, ydim, resampling, src_nodata, fill_value
):
    """Lazily warp dask array `data`, on grid `src_gb`, onto `dst_gb`.

    The y and x axes of `data` are `ydim` and the one after it. The output
    has `data`'s dtype and chunk size. Its chunks that need no source blocks
    are filled with `fill_value`; the others are warped from the window of
    source blocks that :func:`_source_windows` picks for them.
    """

    def with_yx(seq, yx):
        return (*seq[:ydim], *yx, *seq[ydim + 2 :])

    dst_tiles = GeoboxTiles(dst_gb, data.chunksize[ydim : ydim + 2])
    windows = _source_windows(src_gb, data.chunks[ydim : ydim + 2], dst_tiles)
    chunks = with_yx(data.chunks, dst_tiles.chunks)
    name = "reproject-" + tokenize(
        data, src_gb, dst_gb, resampling, src_nodata, fill_value
    )
    warp_chunk = partial(
        _warp_chunk,
        axis=ydim,
        resampling=resampling,
        src_nodata=src_nodata,
        dst_nodata=fill_value,
    )
    dsk = {}
    for idx in np.ndindex(tuple(map(len, chunks))):
        yx = idx[ydim : ydim + 2]
        if yx in windows:
            rows, cols, window_gb, thin = windows[yx]
            blocks = [
                (data.name, *with_yx(idx, (r, c))) for r in rows for c in cols
            ]
            dsk[(name, *idx)] = (
                warp_chunk,
                window_gb,
                dst_tiles[yx],
                thin,
                len(cols),
                *blocks,
            )
        else:
            shape = tuple(c[i] for c, i in zip(chunks, idx, strict=False))
            dsk[(name, *idx)] = (np.full, shape, fill_value, data.dtype)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=(data,))
    return da.Array(graph, name, chunks=chunks, dtype=data.dtype)


# rasterio (1.5.1 at least) replaces a transform within 1e-5 of this
# one, taking it for an ungeoreferenced array, with one whose y axis points
# up, so GDAL warps the wrong area.
_RASTERIO_MISREAD_AFFINE = Affine(1, 0, 0, 0, -1, 0)


def _warp_chunk(
    src_gb,
    dst_gb,
    thin,
    ncols,
    *blocks,
    axis,
    resampling,
    src_nodata,
    dst_nodata,
):
    """Warp the window of source blocks `blocks` onto output chunk `dst_gb`.

    `blocks` are the window's blocks row by row, `ncols` to a row, and
    `src_gb` is the window's grid. `thin` says whether the chunk maps within
    a couple of source rows or columns.
    """
    rows = [
        np.concatenate(blocks[i : i + ncols], axis=axis + 1)
        for i in range(0, len(blocks), ncols)
    ]
    src = np.concatenate(rows, axis=axis)
    # Warp onto one more row above a chunk whose transform rasterio misreads
    extra = int(dst_gb.affine.almost_equals(_RASTERIO_MISREAD_AFFINE))
    if extra:
        dst_gb = GeoBox(
            (dst_gb.shape[0] + 1, dst_gb.shape[1]),
            dst_gb.affine * Affine.translation(0, -1),
            dst_gb.crs,
        )
    shape = (*src.shape[:axis], *dst_gb.shape, *src.shape[axis + 2 :])
    dst = np.full(shape, dst_nodata, dtype=src.dtype)
    options = {}
    if thin:
        # GDAL skips a chunk whose cells all map within one source row or
        # column, such as a chunk of a Mercator grid next to a pole, because
        # the source window it computes for the chunk is empty. Widen that
        # window. This is not done for other chunks, because it also changes
        # a few cells on the edge of the data.
        options["SOURCE_EXTRA"] = 1
    # odc-geo's warp passes GDAL the settings odc-geo uses for all of its
    # warps (kernel scale fixed at 1, int8 and bool data warped as wider
    # types), so values match odc-geo's in-memory warp
    rio_reproject(
        src,
        dst,
        src_gb,
        dst_gb,
        resampling,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
        ydim=axis,
        # From points on a chunk's edges alone, GDAL can miss source cells
        # that a chunk next to a pole needs. Sample a grid over it instead.
        SAMPLE_GRID="YES",
        **options,
    )
    return dst[(slice(None),) * axis + (slice(extra, None),)]


# Points sampled along each axis of an output chunk, per chunk, when finding
# the source blocks it reads
_CHUNK_SAMPLES = 16
# Halvings of the gap between a sample point that maps into the source CRS and
# a neighbor that does not, when finding the edge of the area that maps
_EDGE_HALVINGS = 40
# Source cells read beyond the points that output cells map to: GDAL's
# kernels reach up to 3 cells (lanczos) at the fixed scale of 1 that odc-geo
# warps with, plus one cell for rounding and GDAL's approximate transformer
_KERNEL_PAD = 4


def _source_windows(src_gb, src_chunks, dst_tiles):
    """Pick the window of source blocks that each output chunk reads.

    Returns a dict mapping the (row, col) index of every output chunk that
    needs source data to ``(rows, cols, window, thin)``: the block rows and
    block columns of the source it reads, the window's grid, and whether the
    chunk maps within a couple of source rows or columns. The block columns
    are in the window's order. For a lon/lat source spanning 360 degrees,
    they can run past the last column and on from the first, and the window
    then continues past the source's east edge.

    A grid of points over each output chunk, its edges included, is mapped
    into source pixels. Where only some of them map into the source CRS, as
    at the edge of a world projection's domain, points on that edge are found
    between them and added. The chunk reads the blocks under the points'
    bounding box, widened on each axis by the reach of the resampling kernels
    and by how much the mapping bends between points. For a lon/lat source,
    the box is taken around the circle of longitudes, so that a chunk across
    the source's longitude seam reads blocks on both sides of it. A chunk of
    a projected grid that holds a pole also includes the pole, which in a
    lon/lat source is a whole row, so the chunk reads every block column. A
    chunk none of whose points map, or whose box misses the source, reads
    nothing.

    Mapping only a chunk's outline, or its box through EPSG:4326 as odc-geo
    does, misses blocks or fails for chunks that contain a pole or span a
    wide range of longitudes.
    """
    src_gb_crs = src_gb.crs
    dst_gb = dst_tiles.base
    to_src = _pixel_mapper(dst_gb, src_gb)
    ys, y_slices = _axis_samples(dst_tiles.chunks[0])
    xs, x_slices = _axis_samples(dst_tiles.chunks[1])
    px, py = np.meshgrid(xs, ys)
    col, row = to_src(px, py)
    (h_col, h_row), (v_col, v_row) = _domain_edge_points(
        px, py, col, row, to_src
    )
    pole_chunks = _chunks_containing_poles(dst_tiles, src_gb_crs)
    pole_pixels = _pole_pixels(src_gb)
    period = _longitude_period(src_gb)
    y_edges = np.cumsum((0, *src_chunks[0]))
    x_edges = np.cumsum((0, *src_chunks[1]))
    # A lon/lat source that spans 360 degrees continues past its east edge
    # from its west edge
    spans_globe = period is not None and abs(x_edges[-1] - period) < 1e-3
    windows = {}
    for (iy, y_sl), (ix, x_sl) in product(
        enumerate(y_slices), enumerate(x_slices)
    ):
        c = col[y_sl, x_sl]
        r = row[y_sl, x_sl]
        # Edge points between neighbors in this chunk's grid of points
        h_sl = (y_sl, slice(x_sl.start, x_sl.stop - 1))
        v_sl = (slice(y_sl.start, y_sl.stop - 1), x_sl)
        point_cols = [c.ravel(), h_col[h_sl].ravel(), v_col[v_sl].ravel()]
        point_rows = [r.ravel(), h_row[h_sl].ravel(), v_row[v_sl].ravel()]
        poles = pole_chunks.get((iy, ix), ())
        for lat in poles:
            point_cols.append(pole_pixels[lat][:1])
            point_rows.append(pole_pixels[lat][1:])
        point_cols = np.concatenate(point_cols)
        point_rows = np.concatenate(point_rows)
        mapped = np.isfinite(point_cols)
        if not mapped.any():
            continue
        point_cols = point_cols[mapped]
        point_rows = point_rows[mapped]
        thin = min(np.ptp(point_rows), np.ptp(point_cols)) < 2

        pad = _KERNEL_PAD + _bend(r)
        rows = _block_range(
            point_rows.min() - pad, point_rows.max() + pad, y_edges
        )
        if poles and src_gb_crs.geographic:
            cols = list(range(len(x_edges) - 1))
        else:
            cols = _block_columns(point_cols, c, x_edges, period, spans_globe)
        if not rows or not cols:
            continue
        origin = _window_origin(src_gb, rows, cols, y_edges, x_edges)
        if origin.almost_equals(_RASTERIO_MISREAD_AFFINE):
            # Read one more block row or column, so that the window starts
            # elsewhere. A source that itself starts there cannot avoid it.
            if rows.start > 0:
                rows = range(rows.start - 1, rows.stop)
            elif cols[0] > 0 or spans_globe:
                cols = [(cols[0] - 1) % (len(x_edges) - 1), *cols]
            origin = _window_origin(src_gb, rows, cols, y_edges, x_edges)
        shape = (
            int(sum(src_chunks[0][i] for i in rows)),
            int(sum(src_chunks[1][i] for i in cols)),
        )
        window = GeoBox(shape, origin, src_gb_crs)
        windows[(iy, ix)] = (rows, cols, window, thin)
    return windows


def _window_origin(src_gb, rows, cols, y_edges, x_edges):
    """Transform of the window of source blocks `rows` by `cols`."""
    return src_gb.affine * Affine.translation(
        x_edges[cols[0]], y_edges[rows[0]]
    )


def _block_columns(cols, grid_cols, x_edges, period, spans_globe):
    """Block columns, in window order, that source columns `cols` need.

    `cols` are source pixel columns that a chunk's points map to, and
    `grid_cols` the ones its grid of points map to, NaN where they do not
    map. `period` is the number of source columns in 360 degrees for a
    lon/lat source and None otherwise, and `spans_globe` says whether the
    source spans 360 degrees.
    """
    nblocks = len(x_edges) - 1
    if period is None:
        pad = _KERNEL_PAD + _bend(grid_cols)
        return list(_block_range(cols.min() - pad, cols.max() + pad, x_edges))
    # The shortest arc of the circle of longitudes that holds the points
    wrapped = np.mod(cols, period)
    ordered = np.sort(wrapped)
    gaps = np.diff(ordered, append=ordered[0] + period)
    k = int(np.argmax(gaps))
    if k == len(ordered) - 1:
        start, end = ordered[0], ordered[-1]
    else:
        start, end = ordered[k + 1], ordered[k] + period
    # Measure the bend along the arc, where the columns change smoothly
    grid_wrapped = np.mod(grid_cols, period)
    unwrapped = np.where(
        grid_wrapped < start, grid_wrapped + period, grid_wrapped
    )
    pad = _KERNEL_PAD + _bend(unwrapped)
    lo, hi = start - pad, end + pad
    if hi - lo >= period:
        return list(range(nblocks))
    # The parts of the arc that the source covers, in order along the arc
    width = x_edges[-1]
    pieces = []
    for turn in (-period, 0, period):
        a, b = max(lo, turn), min(hi, turn + width)
        if a < b:
            pieces.append((a - turn, b - turn))
    if not pieces:
        return []
    if len(pieces) == 2 and spans_globe:
        # The arc crosses the seam of a source that spans 360 degrees, so the
        # window runs on from its east edge to its west edge
        first = list(_block_range(pieces[0][0], width, x_edges))
        second = list(_block_range(0, pieces[1][1], x_edges))
        if len(set(first + second)) < nblocks:
            return first + second
        return list(range(nblocks))
    lo = min(a for a, _ in pieces)
    hi = max(b for _, b in pieces)
    return list(_block_range(lo, hi, x_edges))


def _longitude_period(src_gb):
    """Source columns in 360 degrees of longitude, or None.

    None is returned for a source that is not lon/lat, or whose columns do
    not run along lines of latitude.
    """
    aff = src_gb.affine
    if not src_gb.crs.geographic or aff.b != 0 or aff.d != 0:
        return None
    return 360 / abs(aff.a)


def _axis_samples(chunks):
    """Sample positions along one axis of a grid, and each chunk's share.

    The positions are pixel edges, evenly spaced within each chunk, and the
    slice for a chunk selects its positions, both of its edges included.
    """
    positions = [0.0]
    slices = []
    start = 0
    for n in chunks:
        first = len(positions) - 1
        k = min(_CHUNK_SAMPLES, n)
        positions.extend(np.linspace(start, start + n, k + 1)[1:])
        slices.append(slice(first, len(positions)))
        start += n
    return np.array(positions), slices


# Degrees by which a point mapped into the source CRS and back may move
_ROUND_TRIP_TOLERANCE = 1e-3


def _pixel_mapper(dst_gb, src_gb):
    """Make a function that maps pixels on `dst_gb` to pixels on `src_gb`.

    The function takes and returns arrays of pixel columns and rows. Points
    that do not map into the source CRS come back NaN. So do points that a
    projected source CRS does not map back to the same place on the globe:
    some projections, such as transverse Mercator far from its central
    meridian, give finite but meaningless coordinates outside the area they
    are valid for. Where the points are on the globe is compared rather than
    their grid coordinates, which can differ by many cells near a
    singularity, such as a pole in Mercator, or by 360 degrees of longitude.
    """
    to_src = to_lonlat = None
    round_trip = False
    if dst_gb.crs != src_gb.crs:
        dst_crs = dst_gb.crs.proj
        to_src = Transformer.from_crs(dst_crs, src_gb.crs.proj, always_xy=True)
        round_trip = not src_gb.crs.geographic
        if round_trip and not dst_crs.is_geographic:
            to_lonlat = Transformer.from_crs(
                dst_crs, dst_crs.geodetic_crs, always_xy=True
            )

    def mapper(cols, rows):
        x, y = dst_gb.affine * (cols, rows)
        # Unmapped points come back from pyproj as infinite, which makes NaN
        # here
        with np.errstate(invalid="ignore"):
            if to_src is not None:
                sx, sy = to_src.transform(x, y)
                if round_trip:
                    back = to_src.transform(sx, sy, direction="INVERSE")
                    start = (x, y)
                    if to_lonlat is not None:
                        start = to_lonlat.transform(*start)
                        back = to_lonlat.transform(*back)
                    off = _degrees_apart(*start, *back)
                    sx = np.where(off <= _ROUND_TRIP_TOLERANCE, sx, np.nan)
                x, y = sx, sy
            col, row = ~src_gb.affine * (x, y)
        col = np.asarray(col, dtype=float)
        row = np.asarray(row, dtype=float)
        unmapped = ~(np.isfinite(col) & np.isfinite(row))
        col[unmapped] = np.nan
        row[unmapped] = np.nan
        return col, row

    return mapper


def _degrees_apart(lon1, lat1, lon2, lat2):
    """Angle in degrees between points on a sphere, given in degrees."""
    points = []
    for lon, lat in ((lon1, lat1), (lon2, lat2)):
        lon, lat = np.radians(lon), np.radians(lat)
        points.append(
            np.stack(
                (
                    np.cos(lat) * np.cos(lon),
                    np.cos(lat) * np.sin(lon),
                    np.sin(lat),
                )
            )
        )
    chord = np.linalg.norm(points[0] - points[1], axis=0)
    return np.degrees(2 * np.arcsin(np.minimum(chord / 2, 1)))


def _domain_edge_points(px, py, col, row, to_src):
    """Find where the area that maps into the source CRS ends.

    `px` and `py` are a grid of output pixel positions and `col` and `row`
    the source pixels that `to_src` maps them to, NaN where they do not map.
    Between each pair of neighbors in the grid where one maps and the other
    does not, the gap is halved until a point that maps lies on the edge of
    the area that does. Returns the source pixels of those points, as
    ``(col, row)`` arrays for the pairs along each row and then for the
    pairs along each column, NaN for pairs that need no edge point.
    """
    mapped = np.isfinite(col)
    result = []
    for first, second in (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        on_edge = mapped[first] != mapped[second]
        edge_col = np.full(on_edge.shape, np.nan)
        edge_row = np.full(on_edge.shape, np.nan)
        if on_edge.any():
            first_in = mapped[first][on_edge]
            # Each pair's ends as (mapped end, unmapped end)
            ends = [
                (
                    np.where(first_in, a[first][on_edge], a[second][on_edge]),
                    np.where(first_in, a[second][on_edge], a[first][on_edge]),
                )
                for a in (px, py, col, row)
            ]
            (in_x, out_x), (in_y, out_y) = ends[:2]
            in_col, in_row = ends[2][0], ends[3][0]
            for _ in range(_EDGE_HALVINGS):
                mid_x = (in_x + out_x) / 2
                mid_y = (in_y + out_y) / 2
                mid_col, mid_row = to_src(mid_x, mid_y)
                ok = np.isfinite(mid_col)
                in_x = np.where(ok, mid_x, in_x)
                in_y = np.where(ok, mid_y, in_y)
                out_x = np.where(ok, out_x, mid_x)
                out_y = np.where(ok, out_y, mid_y)
                in_col = np.where(ok, mid_col, in_col)
                in_row = np.where(ok, mid_row, in_row)
            edge_col[on_edge] = in_col
            edge_row[on_edge] = in_row
        result.append((edge_col, edge_row))
    return result


def _pole_pixels(src_gb):
    """Map each pole to source pixels: {latitude: array([col, row])}."""
    to_src = Transformer.from_crs("EPSG:4326", src_gb.crs.proj, always_xy=True)
    lats = np.array([90.0, -90.0])
    with np.errstate(invalid="ignore"):
        x, y = to_src.transform(np.zeros(2), lats)
        col, row = ~src_gb.affine * (x, y)
    return {
        float(lat): np.array([c, r])
        if np.isfinite([c, r]).all()
        else np.full(2, np.nan)
        for lat, c, r in zip(lats, col, row, strict=True)
    }


def _chunks_containing_poles(dst_tiles, src_crs):
    """Find the chunks of a projected grid that hold a pole.

    Returns a dict mapping each such chunk to the latitudes of the poles it
    holds. Only chunks of a grid whose CRS differs from `src_crs` are
    returned, and those within a cell of a pole count. A geographic grid is
    not checked: there a pole is a grid edge, which the mapped points
    include.
    """
    dst_gb = dst_tiles.base
    if dst_gb.crs == src_crs or dst_gb.crs.geographic:
        return {}
    to_dst = Transformer.from_crs("EPSG:4326", dst_gb.crs.proj, always_xy=True)
    lats = np.array([90.0, -90.0])
    with np.errstate(invalid="ignore"):
        pole_x, pole_y = to_dst.transform(np.zeros(2), lats)
        pole_cols, pole_rows = ~dst_gb.affine * (pole_x, pole_y)
    y_edges = np.cumsum((0, *dst_tiles.chunks[0]))
    x_edges = np.cumsum((0, *dst_tiles.chunks[1]))
    chunks = {}
    for lat, c, r in zip(lats, pole_cols, pole_rows, strict=True):
        if np.isfinite(c) and np.isfinite(r):
            for chunk in product(
                _block_range(r - 1, r + 1, y_edges),
                _block_range(c - 1, c + 1, x_edges),
            ):
                chunks.setdefault(chunk, []).append(float(lat))
    return chunks


def _bend(a):
    """Largest second difference of sampled values `a` along either axis.

    This measures how far a mapping strays from straight lines between the
    sample points, so padding by it keeps the points' bounding box around
    what lies between them. NaN values are skipped.
    """
    bend = 0.0
    for axis in (0, 1):
        if a.shape[axis] > 2:
            diffs = np.abs(np.diff(a, 2, axis=axis))
            if np.isfinite(diffs).any():
                bend = max(bend, float(np.nanmax(diffs)))
    return bend


def _block_range(lo, hi, edges):
    """Indexes of the blocks, bounded by `edges`, that span [lo, hi] meets."""
    lo = max(lo, edges[0])
    hi = min(hi, edges[-1])
    if lo >= hi:
        return range(0)
    first = np.searchsorted(edges, lo, side="right") - 1
    last = np.searchsorted(edges, hi, side="left") - 1
    return range(int(first), int(last) + 1)
