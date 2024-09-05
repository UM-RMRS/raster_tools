from functools import partial

import dask.array as da
import geopandas as gpd
import numba as nb
import numpy as np
import rasterio as rio
import shapely
from dask.diagnostics import ProgressBar
from packaging import version
from rasterio.enums import MergeAlg
from rasterio.env import GDALVersion
from rasterio.features import rasterize as rio_rasterize

from raster_tools.dtypes import F64, I8, I16, I64, U8, U64, is_float, is_int
from raster_tools.masking import get_default_null_value
from raster_tools.raster import data_to_raster_like, get_raster
from raster_tools.utils import list_reshape_2d
from raster_tools.vector import get_vector

__all__ = ["rasterize"]


_RIO_64BIT_INTS_SUPPORTED = GDALVersion.runtime().at_least("3.5") and (
    version.parse(rio.__version__) >= version.parse("1.3")
)


def _get_rio_dtype(dtype):
    if dtype == I8:
        return I16
    # GDAL >= 3.5 and Rasterio >= 1.3 support 64-bit (u)ints
    if dtype in (I64, U64) and not _RIO_64BIT_INTS_SUPPORTED:
        return F64
    return dtype


def _rio_rasterize_wrapper(
    shape, transform, geometry, values, out_dtype, fill, all_touched
):
    rio_dtype = _get_rio_dtype(out_dtype)
    values_dtype = _get_rio_dtype(values.dtype)
    if values_dtype != values.dtype:
        values = values.astype(values_dtype)

    rast_array = rio_rasterize(
        zip(geometry, values),
        out_shape=shape,
        transform=transform,
        fill=fill,
        all_touched=all_touched,
        merge_alg=MergeAlg.replace,
        dtype=rio_dtype,
    )

    if rio_dtype != out_dtype:
        rast_array = rast_array.astype(out_dtype)
    return rast_array


def _rio_mask(geoms, shape, transform, all_touched, invert):
    fill, geom_value = (1, 0) if invert else (0, 1)
    return rio_rasterize(
        geoms,
        out_shape=shape,
        fill=fill,
        transform=transform,
        all_touched=all_touched,
        default_value=geom_value,
    )


def _rasterize_onto_chunk(
    gdf,
    transform,
    out_dtype,
    fill,
    all_touched,
    overlap_resolve_method,
    block_info=None,
):
    """
    Rasterize a set of features onto a chunk. The cells that touch features
    receive the value of the feature given by the "values" column in gdf. If
    gdf doesn't have a "values" column, the index is used. Cells with
    overlapping features receive a value based on the sorting determined by
    overlap_resolve_method.
    """
    shape_2d = block_info[None]["chunk-shape"]
    valid = ~(gdf.geometry.is_empty | gdf.geometry.isna())
    gdf = gdf[valid]
    if len(gdf) == 0:
        return np.full(shape_2d, fill, dtype=out_dtype)
    use_index = "values" not in gdf
    if use_index:
        gdf = gdf.reset_index(names="values")

    # Trim anything outside the chunk
    chunk_bbox = shapely.geometry.box(
        *rio.transform.array_bounds(*shape_2d, transform)
    )
    geoms_bbox = shapely.geometry.box(*gdf.total_bounds)
    if not shapely.intersects(chunk_bbox, geoms_bbox):
        return np.full(shape_2d, fill, dtype=out_dtype)
    # The clip call shuffles the dataframe. Add column that can be used to
    # return the dataframe to the original order after clipping.
    # ref: https://github.com/geopandas/geopandas/issues/2937
    gdf["idx"] = np.arange(len(gdf))
    gdf = gdf.clip(chunk_bbox)
    gdf = gdf.sort_values(by=["idx"])
    if not len(gdf):
        return np.full(shape_2d, fill, dtype=out_dtype)
    gdf = gdf.drop(labels="idx", axis=1)

    # Sort the dataframe to match the specified overlap resolution method.
    # rasterio's raesterize function replaces cells with overlapping features
    # with the value that came last. To account for this, we have to order the
    # data such that the desired 'winning' values come last. E.g. for 'max',
    # the data must be sorted in ascending order and for 'min' the data must be
    # sorted in descending order.
    #
    # When sorting, stability doesn't matter since the same value will get
    # burned in regardless. Place Nan values at the front though so they get
    # replaced by valid values when possible.
    if overlap_resolve_method == "first":
        # Reverse order so that earlier featues will get rasterized last
        gdf = gdf.iloc[::-1]
    elif overlap_resolve_method == "last":
        # Change nothing
        pass
    elif overlap_resolve_method == "min":
        # Sort so that smaller values get rasterized last
        gdf = gdf.sort_values(
            by=["values"], ascending=False, na_position="first"
        )
    else:
        # "max"
        # Sort so that larger values get rasterized last
        gdf = gdf.sort_values(by=["values"], na_position="first")

    geometry = gdf.geometry.to_numpy()
    values = gdf["values"].to_numpy()
    if use_index:
        values += 1

    return _rio_rasterize_wrapper(
        shape_2d, transform, geometry, values, out_dtype, fill, all_touched
    )


def _mask_onto_chunk(
    gdf, transform, all_touched=True, invert=False, block_info=None
):
    """
    Burn features onto a chunk. Cells that touch a feature get 1 and the rest
    get 0. This is flipped if invert is True.
    """
    fill = 1 if invert else 0
    shape_2d = block_info[None]["chunk-shape"]
    # Trim anything outside the chunk
    chunk_bbox = shapely.geometry.box(
        *rio.transform.array_bounds(*shape_2d, transform)
    )
    geoms_bbox = shapely.geometry.box(*gdf.total_bounds)
    if not shapely.intersects(chunk_bbox, geoms_bbox):
        return np.full(shape_2d, fill, dtype="uint8")
    gdf = gdf.clip(chunk_bbox)
    if not len(gdf):
        return np.full(shape_2d, fill, dtype="uint8")

    return _rio_mask(
        gdf.geometry.to_numpy(), shape_2d, transform, all_touched, invert
    )


@nb.jit(nopython=True, nogil=True)
def _resolve_first(x, fill):
    # Take first valid values along band dimension
    out = np.where(x[-2] == fill, x[-1], x[-2])
    return np.expand_dims(out, 0)


@nb.jit(nopython=True, nogil=True)
def _resolve_last(x, fill):
    # Take last valid values along band dimension
    out = np.where(x[-1] == fill, x[-2], x[-1])
    return np.expand_dims(out, 0)


@nb.jit(nopython=True, nogil=True)
def _all_axis0(x):
    # Compute all along first axis
    out = np.empty((1, x.shape[1], x.shape[2]), dtype=np.bool_)
    nr, nc = x.shape[1:]
    for r in range(nr):
        for c in range(nc):
            out[0, r, c] = np.all(x[:, r, c])
    return out


@nb.jit(nopython=True, nogil=True)
def _min_axis0(x):
    # Compute min along first axis
    out = np.empty((1, x.shape[1], x.shape[2]), dtype=x.dtype)
    nr, nc = x.shape[1:]
    for r in range(nr):
        for c in range(nc):
            out[0, r, c] = np.min(x[:, r, c])
    return out


@nb.jit(nopython=True, nogil=True)
def _max_axis0(x):
    # Compute max along first axis
    out = np.empty((1, x.shape[1], x.shape[2]), dtype=x.dtype)
    nr, nc = x.shape[1:]
    for r in range(nr):
        for c in range(nc):
            out[0, r, c] = np.max(x[:, r, c])
    return out


@nb.jit(nopython=True, nogil=True)
def _resolve_min(x, fill):
    # Take min values along band dimension
    mask = x == fill
    out = np.where(mask, np.nanmax(x), x)
    out = _min_axis0(out)
    out = np.where(_all_axis0(mask), fill, out)
    return out


@nb.jit(nopython=True, nogil=True)
def _resolve_max(x, fill):
    # Take max values along band dimension
    mask = x == fill
    out = np.where(mask, np.nanmin(x), x)
    out = _max_axis0(out)
    out = np.where(_all_axis0(mask), fill, out)
    return out


_RESOLVE_KW_TO_FUNC = {
    "first": _resolve_first,
    "last": _resolve_last,
    "min": _resolve_min,
    "max": _resolve_max,
}


def _reduction_wrapper(
    x, fill, resolve_func, axis=None, keepdims=False, **kwargs
):
    if x.size == 0:
        # Do nothing for dask test calls. Ignore keepdims here because dask
        # gets confused otherwise.
        return x
    # Assuming x has dims (B, Y, X), where B in {1, 2}
    # Do nothing if input only has one band. Nothing to reduce. Copy in order
    # to follow dask best practices.
    out = x.copy() if x.shape[0] == 1 else resolve_func(x, fill=fill)
    if keepdims:
        return out
    return out[0]


def _copy(x, *args, **kwargs):
    return x.copy()


def _reduce_stacked_feature_rasters_custom(
    stack, fill, overlap_resolve_method, keepdims
):
    agg_func = partial(
        _reduction_wrapper,
        fill=fill,
        resolve_func=_RESOLVE_KW_TO_FUNC[overlap_resolve_method],
    )
    reduced = da.reduction(
        stack,
        chunk=_copy,
        combine=agg_func,
        aggregate=agg_func,
        axis=0,
        keepdims=keepdims,
        dtype=stack.dtype,
        split_every=2,
    )
    return reduced


def _reduce_stacked_feature_rasters(
    chunk_stack,
    fill=None,
    overlap_resolve_method=None,
    mask=False,
    mask_invert=False,
    keepdims=False,
):
    if not mask:
        # Apply tailored reduction
        return _reduce_stacked_feature_rasters_custom(
            chunk_stack, fill, overlap_resolve_method, keepdims=keepdims
        )

    if mask_invert:
        # Features marked by 0. Use min to propagate 0s over 1s
        chunk = da.min(chunk_stack, axis=0, keepdims=keepdims)
    else:
        # Features marked by 1. Use max to propagate 1s over 0s
        chunk = da.max(chunk_stack, axis=0, keepdims=keepdims)
    return chunk


def _rasterize_spatial_matches(
    matches,
    partitions,
    like_chunk_rasters,
    all_touched,
    fill=None,
    target_dtype=None,
    overlap_resolve_method=None,
    mask=False,
    mask_invert=False,
):
    # Create a list for holding rasterization results. Each element corresponds
    # to a chunk in the like raster. All elements start as None. Elements will
    # will be replaced by a stack of dask arrays, if that chunk intersects a
    # vector partition. Each array is a vector partition that has been
    # rasterized to the corresponding like-chunk's grid.
    out_chunks = [None] * len(like_chunk_rasters)
    # Group by partition and iterate over the groups
    for ipart, grp in matches.groupby("part_idx"):
        # Get the vector partition
        part = partitions[ipart]
        # Iterate over the chunks that intersected the vector partition and
        # rasterize the partition to each intersecting chunk's grid
        for _, row in grp.iterrows():
            little_like = like_chunk_rasters[row.flat_idx]
            if not mask:
                chunk = da.map_blocks(
                    _rasterize_onto_chunk,
                    dtype=target_dtype,
                    chunks=little_like.shape[1:],
                    meta=np.array((), dtype=target_dtype),
                    # func args
                    gdf=part,
                    transform=little_like.affine,
                    out_dtype=target_dtype,
                    fill=fill,
                    all_touched=all_touched,
                    overlap_resolve_method=overlap_resolve_method,
                )
            else:
                chunk = da.map_blocks(
                    _mask_onto_chunk,
                    dtype=U8,
                    chunks=little_like.shape[1:],
                    meta=np.array((), dtype=U8),
                    # Func args
                    gdf=part,
                    transform=little_like.affine,
                    all_touched=all_touched,
                    invert=mask_invert,
                )
            if out_chunks[row.flat_idx] is None:
                out_chunks[row.flat_idx] = []
            out_chunks[row.flat_idx].append(chunk)
    return out_chunks


def _raw_rasterized_chunks_to_dask_array(
    raw_chunk_list,
    like_chunk_rasters,
    like_blocks_shape,
    fill,
    target_dtype=None,
    overlap_resolve_method=None,
    mask=False,
    mask_invert=False,
):
    processed_chunks = []
    for fi, oc in enumerate(raw_chunk_list):
        if oc is None:
            # Chunk did not intersect any partitions. Fill with the fill value
            processed_chunks.append(
                da.full_like(
                    like_chunk_rasters[fi].data[0], fill, dtype=target_dtype
                )
            )
        else:
            # The chunk intersected 1 or more partitions. Reduce stack of
            # arrays (partitions rasterized to the chunk's grid) to a single
            # array using the specified overlap resolution method or by merging
            # masks together.
            chunk = _reduce_stacked_feature_rasters(
                da.stack(oc, axis=0),
                fill=fill,
                overlap_resolve_method=overlap_resolve_method,
                mask=mask,
                mask_invert=mask_invert,
                keepdims=False,
            )
            processed_chunks.append(chunk)
    # Stack back into 2D grid of chunks
    processed_chunks = list_reshape_2d(processed_chunks, like_blocks_shape)
    # Convert to Dask array made up of the processed chunks and add band dim of
    # size 1.
    return da.block([processed_chunks])


def _rasterize_spatial_aware(
    dgdf,
    like,
    field=None,
    fill=None,
    target_dtype=None,
    overlap_resolve_method=None,
    all_touched=True,
    mask=False,
    mask_invert=False,
):
    if dgdf.spatial_partitions is None:
        raise ValueError("No spatial partitions found on input dataframe.")

    if like.nbands > 1:
        # Only need one band
        like = like.get_bands(1)

    sparts = dgdf.spatial_partitions.to_frame("geometry")
    if mask:
        fill = 1 if mask_invert else 0
        target_dtype = U8
        # Only need geometry for masking
        dgdf = dgdf.geometry.to_frame("geometry")
    else:
        # Transform to minimal dask dataframe.
        if field is not None:
            dgdf = dgdf[[field, "geometry"]].rename(columns={field: "values"})
        else:
            dgdf = dgdf.geometry.to_frame("geometry")
    sparts["part_idx"] = np.arange(dgdf.npartitions)
    chunk_gdf = like.get_chunk_bounding_boxes()
    chunk_gdf["flat_idx"] = chunk_gdf.index
    # Split the like raster up into sub rasters. One for each chunk. Flatten
    # into a 1D list of rasters.
    like_chunk_rasters = list(like.get_chunk_rasters().ravel())
    # Perform a spatial join between the vector partition bounding polygons and
    # the chunk extents from the like raster. This produces a dataframe with a
    # row for each intersection. It is sorted so that partition order is
    # preserved. This sorting allows later partitions to take precedence over
    # earlier partitions.
    matches = sparts.sjoin(chunk_gdf).sort_values("part_idx")
    # Each element is either None or a list of 2D dask arrays
    raw_chunk_list = _rasterize_spatial_matches(
        matches,
        dgdf.partitions,
        like_chunk_rasters,
        all_touched,
        fill=fill,
        target_dtype=target_dtype,
        overlap_resolve_method=overlap_resolve_method,
        mask=mask,
        mask_invert=mask_invert,
    )
    # Replace None elements with array of fill and squash lists of 2D arrays to
    # single chunk.
    out_data = _raw_rasterized_chunks_to_dask_array(
        raw_chunk_list,
        like_chunk_rasters,
        like.data.blocks.shape[1:],
        fill=fill,
        target_dtype=target_dtype,
        overlap_resolve_method=overlap_resolve_method,
        mask=mask,
        mask_invert=mask_invert,
    )
    return data_to_raster_like(out_data, like, nv=fill)


def _rasterize_spatial_naive(
    df,
    like,
    field=None,
    fill=None,
    target_dtype=None,
    overlap_resolve_method="max",
    all_touched=True,
    mask=False,
    mask_invert=False,
):
    # There is no way to neatly align the dataframe partitions with the raster
    # chunks, if spatial bounds are not known. Because of this, we burn in each
    # partition on its own like-sized raster and then merge the results. We do
    # this by creating mock spatial partitions where each partition covers all
    # of like. The spatial aware code can then handle it from here.
    nparts = df.npartitions
    like_bbox = shapely.geometry.box(*like.bounds)
    sparts = gpd.GeoSeries([like_bbox for i in range(nparts)], crs=like.crs)
    df.spatial_partitions = sparts
    return _rasterize_spatial_aware(
        df,
        like,
        field=field,
        fill=fill,
        target_dtype=target_dtype,
        overlap_resolve_method=overlap_resolve_method,
        all_touched=all_touched,
        mask=mask,
        mask_invert=mask_invert,
    )


def rasterize(
    features,
    like,
    field=None,
    overlap_resolve_method="last",
    mask=False,
    mask_invert=False,
    null_value=None,
    all_touched=True,
    use_spatial_aware=False,
    show_progress=False,
):
    """Convert vector feature data to a raster.

    This function can be used to either rasterize features using values from a
    particular data field or to create a raster mask of zeros and ones. Using
    values to rasterize is the default. Use `mask=True` to generate a raster
    mask. If no data field is specified, the underlying dataframe's index plus
    one is used. NOTE: because of limitations in dask, dataframe index values
    are not guaranteed to be unique across the dataframe. Cells that do not
    touch or overlap any features are marked as null.

    To add a column of unique IDs for each feature, see
    :func:`raster_tools.vector.add_objectid_column` or
    :meth:`raster_tools.vector.Vector.add_objectid_column`.

    This operation can be greatly accelerated if the provided `features`
    object has been spatially shuffled or had spatial partitions calculated.
    There are a few ways to do this. For `Vector` or `GeoDataFrame`/`GeoSeries`
    objects, you can use the `spatial_shuffle` or
    `calculate_spatial_partitions` methods. `calculate_spatial_partitions`
    simply computes the spatial bounds of each partition in the data.
    `spatial_shuffle` shuffles the data into partitions of spatially near
    groups and calculates the spatial bounds at the same time. This second
    method is more expensive but provides a potentially greater speed up for
    rasterization. The `use_spatial_aware` flag can also be provided to this
    function. This causes the spatial partitions to be calculated before
    rasterization.

    .. note::
        If the CRS for `features` does not match the CRS for `like`, `features`
        will be transformed to `like`'s CRS. This operation causes spatial
        partition information to be lost. It is recommended that the CRSs for
        both are matched ahead of time.

    Parameters
    ----------
    features : Vector, GeoDataFrame, dask_geopandas.GeoDataFrame
        Vector data to rasterize.
    like : Raster
        A raster to use for grid and CRS information. The resulting raster will
        be on the same grid as `like`.
    field : str, optional
        The name of a field to use for cell values when rasterizing the
        vector features. If None or not specified, the underlying dataframe's
        index plus 1 is used. The default is to use the index plus 1.
    overlap_resolve_method : str, optional
        The method used to resolve overlaping features. Default is `"last"`.
        The available methods are:

        'first'
            Cells with overlapping features will receive the value from the
            feature that appears first in the feature table.
        'last'
            Cells with overlapping features will receive the value from the
            feature that appears last in the feature table.
        'min'
            Cells with overlap will receive the value from the feature with the
            smallest value.
        'max'
            Cells with overlap will receive the value from the feature with the
            largest value.

    mask : bool, optional
        If ``True``, the features are rasterized as a mask. Cells that do not
        touch a feature are masked out. Cells that touch the features are set
        to ``1``. If `mask_invert` is also ``True``, this is inverted. If
        `mask` is ``False``, the features are rasterized using `field` to
        retrieve values from the underlying dataframe. `field` is ignored, if
        this option is a used. Default is ``False``.
    mask_invert : bool, optional
        If ``True`` cells that are inside or touch a feature are masked out. If
        ``False``, cells that do not touch a feature are masked out. Default is
        ``False``.
    null_value : scalar, optional
        The value to use in cells with no feature data, when not masking.
    all_touched : bool, optional
        If ``True``, grid cells that the vector touches will be burned in.
        If False, only cells with a center point inside of the vector
        perimeter will be burned in.
    use_spatial_aware : bool, optional
        Force the use of spatial aware rasterization. If ``True`` and
        `features` is not already spatially indexed, a spatial index will be
        computed. Alternatively, if ``True`` and `features`'s CRS differs from
        `like`, a new spatial index in a common CRS will be computed. If
        `features` already has a spatial index and its CRS matches `like`, this
        argument is ignored. Default is ``False``.
    show_progress : bool, optional
        If `use_spatial_aware` is ``True``, this flag causes a progress bar to
        be displayed for spatial indexing. Default is ``False``.

    Returns
    -------
    Raster
        The resulting single band raster of rasterized features. This raster
        will be on the same grid as `like`.

    """
    gdf = get_vector(features).data

    like = get_raster(like)
    if not mask:
        if isinstance(field, str):
            if field not in gdf:
                raise ValueError(f"Invalid field name: {repr(field)}")
            dtype = gdf[field].dtype
            if not is_int(dtype) and not is_float(dtype):
                raise ValueError(
                    "The specified field must be a scalar data type"
                )
            target_dtype = dtype
        elif field is not None:
            raise ValueError(f"Could not understand 'field' value: {field!r}")
        else:
            target_dtype = I64
        if null_value is not None:
            if not is_int(null_value) and not is_float(null_value):
                raise TypeError("null_value must be a scalar")
        elif field is not None:
            null_value = get_default_null_value(target_dtype)
        else:
            null_value = 0

        if overlap_resolve_method not in {"first", "last", "min", "max"}:
            raise ValueError(
                "Invalid value for overlap_resolve_method: "
                f"{overlap_resolve_method!r}"
            )
    else:
        target_dtype = U8

    if gdf.crs != like.crs and like.crs is not None:
        # This will clear spatial_partitions
        gdf = gdf.to_crs(like.crs)

    if use_spatial_aware and gdf.spatial_partitions is None:
        gdf = gdf.copy()
        if show_progress:
            with ProgressBar():
                gdf.calculate_spatial_partitions()
        else:
            gdf.calculate_spatial_partitions()

    if gdf.spatial_partitions is not None:
        return _rasterize_spatial_aware(
            gdf,
            like,
            field,
            fill=null_value,
            target_dtype=target_dtype,
            all_touched=all_touched,
            overlap_resolve_method=overlap_resolve_method,
            mask=mask,
            mask_invert=mask_invert,
        )
    else:
        return _rasterize_spatial_naive(
            gdf,
            like=like,
            field=field,
            fill=null_value,
            target_dtype=target_dtype,
            overlap_resolve_method=overlap_resolve_method,
            all_touched=all_touched,
            mask=mask,
            mask_invert=mask_invert,
        )
