import os
import sys

import dask
import dask.array as da
import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from dask.delayed import delayed
from packaging import version
from rasterio.enums import MergeAlg
from rasterio.env import GDALVersion
from rasterio.features import rasterize as rio_rasterize

from raster_tools.dtypes import (
    F64,
    I8,
    I16,
    I64,
    U64,
    is_float,
    is_int,
    is_str,
)
from raster_tools.masking import get_default_null_value
from raster_tools.raster import Raster, get_raster

PYOGRIO_SUPPORTED = sys.version_info >= (3, 8)
if PYOGRIO_SUPPORTED:
    import warnings

    import pyogrio as ogr

    warnings.filterwarnings(
        "ignore",
        message=".*Measured \\(M\\) geometry types are not supported.*",
    )
else:
    import fiona

__all__ = [
    "Vector",
    "get_vector",
    "list_layers",
    "count_layer_features",
    "open_vectors",
]


class VectorError(Exception):
    pass


def _is_series(geo):
    return isinstance(geo, (gpd.GeoSeries, dgpd.GeoSeries))


def _is_frame(geo):
    return isinstance(geo, (gpd.GeoDataFrame, dgpd.GeoDataFrame))


def list_layers(path):
    """List the layers in a vector source file."""
    if not os.path.exists(path):
        raise IOError(f"No such file or directory: {path!r}")
    if PYOGRIO_SUPPORTED:
        layers = ogr.list_layers(path)
        return list(layers[:, 0])
    else:
        return fiona.listlayers(path)


def count_layer_features(path, layer):
    """Count the number of features in a layer of a vector source file.

    Parameters
    ----------
    path : str
        The path to a vector source file.
    layer : str, int
        The layer to count features from. Can be the 0-based index or a string.

    Returns
    -------
    int
        The number of features.

    """
    if not os.path.exists(path):
        raise IOError(f"No such file or directory: {path!r}")
    if not is_int(layer) and not is_str(layer):
        raise TypeError("The layer must be an int or string.")
    if is_int(layer) and layer < 0:
        raise ValueError("The layer must be positive.")

    if PYOGRIO_SUPPORTED:
        info = ogr.read_info(path, layer=layer)
        return info["features"]
    else:
        with fiona.open(path, layer=layer) as src:
            return len(src)


def _df_asign_index(df, index):
    df.index = index
    return df


def _read_file_delayed(path, layer, npartitions):
    total_size = count_layer_features(path, layer)
    batch_size = (total_size // npartitions) + 1

    row_offset = 0
    dfs = []
    divs = []
    while row_offset < total_size:
        left = row_offset
        right = min(row_offset + batch_size, total_size)
        rows = slice(left, right)
        df = delayed(gpd.read_file)(path, rows=rows, layer=layer)
        index = pd.RangeIndex(left, right)
        df = delayed(_df_asign_index)(df, index)
        dfs.append(df)
        divs.append(left)
        row_offset += batch_size
    divs.append(right - 1)
    meta = gpd.read_file(path, layer=layer, rows=1)
    divs = tuple(divs)
    # Returns a dask_geopandas GeoDataFrame
    return dd.from_delayed(dfs, meta, divisions=divs), total_size


def _normalize_layers_arg(layers):
    if isinstance(layers, (list, tuple)):
        if len(layers) == 0:
            raise ValueError("Layers argument was empty")
        first = layers[0]
        is_ftype = is_str if is_str(first) else is_int
        if not all(is_ftype(layer) for layer in layers):
            raise TypeError(
                "Layers must be all name strings or all integer indices"
            )
        return layers
    elif is_int(layers) or is_str(layers):
        return [layers]
    else:
        raise TypeError(f"Could not understand layers argument: {layers!r}")


_TARGET_CHUNK_SIZE = 50_000


def open_vectors(path, layers=None):
    """Open a vector source.

    This can be any GDAL/OGR compatible source.

    Parameters
    ----------
    path : str, Path
        The vector source path.
    layers : list of int, int, list of str, str, or None, optional
        The layer(s) to read from the source. This can be a single layer
        int or string, a list of layer ints/strings, or ``None``. If ``None``,
        all layers are read in. If int(s) are given, they are treated as
        0-based layer indices.

    Returns
    -------
    Vector, list of Vector
        If only a single layer was specified or found, a single Vector object
        is returned. Otherwise, a list of Vector objects is returned.

    """
    if not os.path.exists(path):
        raise IOError(f"No such file or directory: {path!r}")

    src_layers = list_layers(path)
    n_layers = len(src_layers)
    if layers is not None:
        layers = _normalize_layers_arg(layers)
        src_layers_set = set(src_layers)
        if is_str(layers[0]) and any(
            layer not in src_layers_set for layer in layers
        ):
            raise ValueError("Invalid layer name")
        elif is_int(layers[0]) and any(
            layer >= n_layers or layer < 0 for layer in layers
        ):
            raise ValueError(
                "Invalid layer index. Index either outside valid range or less"
                " than 0"
            )
    else:
        layers = src_layers

    dfs = []
    for layer in layers:
        n = count_layer_features(path, layer=layer)
        if PYOGRIO_SUPPORTED:
            df = dgpd.read_file(
                path, layer=layer, chunksize=_TARGET_CHUNK_SIZE
            )
            pair = (df, n)
        else:
            n_parts = max(n // _TARGET_CHUNK_SIZE, 1)
            pair = _read_file_delayed(path, layer, n_parts)
        dfs.append(pair)

    if len(dfs) > 1:
        return [Vector(df, n) for df, n in dfs]
    return Vector(*dfs[0])


def get_vector(src):
    if isinstance(src, Vector):
        return src
    elif is_str(src):
        result = open_vectors(src)
        if not isinstance(result, Vector):
            # More than one layer was found
            raise ValueError("Input source must only have 1 layer")
        return result
    elif _is_series(src) or _is_frame(src):
        return Vector(src)
    else:
        raise TypeError("Invalid vector input")


def _get_len_from_divisions(divs):
    # This should never overcount but may undercount by 1 because
    # dask.dataframe special cases the last division when creating a dataframe.
    #
    # NOTE: Dask dataframes have a divisions property. This is a tuple of
    # locations along the index where partitions start and stop. It looks like
    # this: (0, 10, 20, 29) for a dataframe with 3 divisions and 30 elements.
    # The first element is the start of the first partition and is inclusive.
    # The next is the exclusive end of the first partition and the inclusive
    # start of the second partition. This continues until the last partition
    # where a monkey wrench is thrown in. The last division in the tuple is
    # inclusive but dask treats it like every other division, which makes
    # things less clear and potentially makes the size of the dataframe
    # inexact. Because partitions can be merged etc., there is no way to tell
    # if the last partition in a dataframe you have been given is actually the
    # original last partition. Dask also does not make an effort to keep track
    # of the total length. The best way I have found to get the length without
    # computing it, is to go by the divisions. Because of the special case and
    # the inability to differentiate it from other partitions, this func may
    # undercount by 1. It will give the exact value only if the original last
    # partition is no longer part of given dataframe. It is safer to undercount
    # than overcount however.
    n = 0
    for i in range(len(divs) - 1):
        left = divs[i]
        right = divs[i + 1]
        n += right - left
    return n


def _guess_geo_len(geo):
    if not dask.is_dask_collection(geo):
        return len(geo)

    if geo.known_divisions:
        return _get_len_from_divisions(geo.divisions)
    raise ValueError("Divisions must be set.")


def _get_best_effort_geo_len(geo):
    if not dask.is_dask_collection(geo) or geo.npartitions == 1:
        return len(geo)

    n = _guess_geo_len(geo.partitions[:-1])
    # Evaluate the last partition to get its exact size. Hopefully this is the
    # special-cased partition. If it isn't, this will undercount by 1. See
    # the note in _get_len_from_divisions.
    n += len(geo.partitions[-1])
    return n


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
        zip(geometry.values, values),
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


def _rasterize_block(
    xc, yc, geometry, values, out_dtype, fill, all_touched, block_info=None
):
    shape_2d = block_info[None]["chunk-shape"]
    valid = ~(geometry.is_empty | geometry.isna())
    geometry = geometry[valid]
    values = values[valid.values]
    if len(geometry) == 0:
        return np.full(shape_2d, fill, dtype=out_dtype)

    xc = xc.ravel()
    yc = yc.ravel()
    dummy_data = da.zeros(shape_2d)
    dummy_xrs = xr.DataArray(dummy_data, coords=(yc, xc), dims=("y", "x"))
    transform = dummy_xrs.rio.transform()
    return _rio_rasterize_wrapper(
        shape_2d, transform, geometry, values, out_dtype, fill, all_touched
    )


def _rasterize_partition(
    df, xlike, field, fill, target_dtype, all_touched=True
):
    if xlike.ndim > 2:
        # Trim off band dim since its not needed
        xlike = xlike[0]
    geometry = df.geometry
    values = df.index if field is None else df[field]
    values = (
        values.to_dask_array()
        if dask.is_dask_collection(values)
        else values.to_numpy()
    )
    if field is None:
        # Convert to 1-based index
        values = values + 1
        # Convert to minimum dtype
        values = values.astype(target_dtype)
    # Chunk up coords to match data. Dask will provide the correct coord chunks
    # to the rasterizing func in map_blocks
    xc = da.from_array(xlike.x.data, chunks=xlike.chunks[1]).reshape((1, -1))
    yc = da.from_array(xlike.y.data, chunks=xlike.chunks[0]).reshape((-1, 1))

    result_data = da.map_blocks(
        _rasterize_block,
        xc,
        yc,
        chunks=xlike.chunks,
        meta=np.array((), dtype=target_dtype),
        # func args
        geometry=geometry,
        values=values,
        out_dtype=target_dtype,
        fill=fill,
        all_touched=all_touched,
    )
    return result_data


def _vector_to_raster_dask(df, size, xlike, field=None, all_touched=True):
    if not dask.is_dask_collection(xlike):
        raise ValueError("xlike must be a dask collection")

    # There is no way to neatly align the dataframe partitions with the raster
    # chunks. Because of this, we burn in each partition on its own raster and
    # then merge the results.
    fill = 0
    if field is None:
        target_dtype = np.min_scalar_type(size)
    else:
        target_dtype = df[field].dtype
        if target_dtype.kind == "u":
            # Null values are difficult for unsigned ints so cast up to the
            # next largest width signed int. If target_dtype is U64, this will
            # cast to a F64.
            target_dtype = np.promote_types(target_dtype, I16)
        fill = get_default_null_value(target_dtype)
    results = []
    parts = df.partitions if dask.is_dask_collection(df) else [df]
    for part in parts:
        res = _rasterize_partition(
            part,
            xlike,
            field,
            fill,
            target_dtype=target_dtype,
            all_touched=all_touched,
        )
        results.append(res)
    # (n-partitions, yn, xn)
    result_data = da.stack(results, axis=0)
    # (1, yn, xn)
    result_data = result_data.max(axis=0, keepdims=True)
    x = xlike.x.data
    y = xlike.y.data
    result = xr.DataArray(result_data, coords=([1], y, x), dims=xlike.dims)
    if xlike.rio.crs is not None:
        result = result.rio.write_crs(xlike.rio.crs)
    result = result.rio.write_nodata(fill)
    return result


def _geoms_to_raster_mask(xc, yc, geoms, all_touched=True, block_info=None):
    xc = xc.ravel()
    yc = yc.ravel()
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
        mask, _ = rio.mask.mask(ds, geoms.values, all_touched=all_touched)
        return mask.squeeze()


def _vector_to_raster_mask(geoms, like, all_touched=True):
    xc, yc = like.get_chunked_coords()
    xc = xc[0]
    yc = yc[0]
    parts = []
    for part in geoms.partitions:
        data = da.map_blocks(
            _geoms_to_raster_mask,
            xc,
            yc,
            geoms=part,
            all_touched=all_touched,
            chunks=like.data.chunks[1:],
            meta=np.array((), dtype=I8),
        )
        parts.append(data)
    data = da.stack(parts, axis=0).max(axis=0, keepdims=True)
    x = like.x
    y = like.y
    xrs = xr.DataArray(data, coords=([1], y, x), dims=like.xdata.dims)
    if like.crs is not None:
        xrs = xrs.rio.write_crs(like.crs)
    xrs = xrs.rio.write_nodata(0)
    return xrs


def _normalize_geo_data(geo):
    if not _is_series(geo) and not _is_frame(geo):
        raise TypeError(
            "Invalid data type. Must be some type GeoDataFrame or"
            f" GeoSeries. Got {type(geo)}."
        )
    if _is_series(geo):
        geo = geo.to_frame("geometry")
    if dask.is_dask_collection(geo) and not geo.known_divisions:
        raise ValueError(
            "Unknown divisions set on input data. Divisions must be set."
        )
    return geo


_GMTRY = "geometry"


class Vector:
    """A class representing vector data.

    Take care to provide the actual number of features in the data when
    creating a new Vector.
    """

    def __init__(self, geo, size=None):
        """Create a `Vector` object.

        Parameters
        ----------
        geo : GeoDataFrame or GeoSeries
            The vector data to use. This can be a geopandas or dask_geopandas
            object.
        size : int, optional
            The number of features. If not provided, A best effort will be made
            to determine the number of features. Part of the data may be
            temporarily loaded.

        """
        geo = _normalize_geo_data(geo)
        if not dask.is_dask_collection(geo):
            self._size = len(geo)
            geo = dgpd.from_geopandas(geo, npartitions=1)
            self._geo = geo
        else:
            self._geo = geo
            # NOTE: self._size is not data-dependant so when adding features
            # that can affect length, care must be taken to properly update
            # _size.
            if size is None:
                self._size = _get_best_effort_geo_len(geo)
            else:
                if not is_int(size):
                    raise TypeError("Size must be an int.")

                # The guess will underestimate by 1 in most cases
                size_guess = _guess_geo_len(geo)
                if size == size_guess or size == (size_guess + 1):
                    self._size = size
                else:
                    raise ValueError(
                        "Given size does not match divisions in data"
                    )

    def __len__(self):
        return self._size

    def __repr__(self):
        lazy = dask.is_dask_collection(self._geo)
        return "<Vector(shapes: {}, fields: {}, CRS: {}, lazy:{})>".format(
            self.size, self.shape[1], self.crs, lazy
        )

    @property
    def table(self):
        """The vector attribute table."""
        return self._geo.drop(columns=[_GMTRY])

    @property
    def data(self):
        return self._geo

    @property
    def size(self):
        """
        The number of vectors contained. NaN is returned if the vector is lazy.

        Use `len(vector)` to trigger computation of the length.
        """
        return len(self)

    @property
    def shape(self):
        """The shape of the attribute table."""
        if not dask.is_dask_collection(self._geo):
            return self.table.shape
        return (self.size, len(self.field_names))

    @property
    def crs(self):
        """The vector CRS."""
        return rio.crs.CRS.from_wkt(self._geo.crs.to_wkt())

    @property
    def field_schema(self):
        """A mapping of the field names to dtypes."""
        return self.table.dtypes.to_dict()

    @property
    def field_names(self):
        """The field names."""
        return list(self.field_schema.keys())

    @property
    def field_dtypes(self):
        """The field dtypes."""
        return list(self.field_schema.values())

    @property
    def geometry(self):
        """The vector geometry series."""
        return self._geo.geometry

    @property
    def tasks(self):
        """The number of lazy operations left to be computed."""
        if dask.is_dask_collection(self._geo):
            # TODO: take into account partitions?
            return len(self._geo.dask)
        return 0

    @property
    def bounds(self):
        """Return a bounds array or dask array: (minx, miny, maxx, maxy).

        If the vector is lazy, the output is a dask array.
        """
        return self.geometry.total_bounds

    def copy(self):
        """Copies the vector."""
        return Vector(self._geo.copy())

    def eval(self):
        """Computes the built-up chain of operations on the underlying data."""
        if dask.is_dask_collection(self._geo):
            return Vector(self._geo.compute(), self._size)
        return self

    def to_lazy(self, npartitions=None):
        """Converts to a lazy dask-backed Vector.

        Parameters
        ----------
        npartitions : int
            The number of partitions to use.

        Returns
        -------
        Vector
            A lazy copy of the vector.

        """
        if dask.is_dask_collection(self._geo):
            return self
        if npartitions is None:
            # TODO: calculate better value based on df size
            npartitions = 1
        return Vector(
            dgpd.from_geopandas(self._geo, npartitions=npartitions), self.size
        )

    def to_dask(self):
        """
        Returns the underlying data as a :obj:`dask_geopandas.GeoDataFrame`.
        """
        return self.to_lazy()._geo

    def to_dataframe(self):
        """Returns the underlying GeoDataFrame."""
        return self.data

    def to_shapely(self):
        """Returns the vector data as a list of shapely geometries objects."""
        return self.geometry.to_list()

    def to_raster(self, like, field=None, all_touched=True):
        """Convert vector data to a raster.

        Parameters
        ----------
        like : Raster
            A to use for grid and CRS information. The resulting raster will be
            on the same grid as `like`.
        field : str, optional
            The name of a field to use for fill values when rasterizing the
            vector features.
        all_touched : bool, optional
            If ``True``, grid cells that the vector touches will be burned in.
            If False, only cells with a center point inside of the vector
            perimeter will be burned in.

        Returns
        -------
        Raster
            The resulting raster. The result will have a single band. Each
            vector shape is assigned a 1-based integer value equal to its order
            in the original vector collection. This integer value is what is
            burned in at the corresponding grid cells. The dtype of the result
            will be the minimum, unsigned integer type that can fit the ID
            values. The null value is 0.

        """
        like = get_raster(like)
        if field is not None:
            if not is_str(field):
                raise TypeError("Field must be a string")
            if field not in self._geo:
                raise ValueError(f"Invalid field name: {repr(field)}")
            dtype = self.field_schema[field]
            if not is_int(dtype) and not is_float(dtype):
                raise ValueError(
                    "The specified field must be a scalar data type"
                )

        xrs = _vector_to_raster_dask(
            self.to_crs(like.crs)._geo,
            self.size,
            xlike=like.xdata,
            field=field,
            all_touched=all_touched,
        )
        return Raster(xrs)

    def to_raster_mask(self, like, all_touched=True):
        """Convert vector data to a raster mask.

        Parameters
        ----------
        like : Raster
            A to use for grid and CRS information. The resulting raster will be
            on the same grid as `like`.
        all_touched : bool, optional
            If ``True``, grid cells that the vector touches will be burned in.
            If False, only cells with a center point inside of the vector
            perimeter will be burned in.

        Returns
        -------
        Raster
            The resulting raster. The result will have a single band. All cells
            that fall under the vector data are masked with ``1``. The null
            value is 0.

        """
        like = get_raster(like)

        xrs = _vector_to_raster_mask(
            self.to_crs(like.crs).geometry, like, all_touched=all_touched
        )
        return Raster(xrs)

    def to_crs(self, crs):
        """Transform the vector coordinates to the specified `crs`.

        Parameters
        ----------
        crs : str, :obj:`pyproj.CRS`
            The crs to transform the vector coordinates to. The value can be
            anything accepted by
            :meth:`pyproj.CRS.from_user_input()`,
            such as an EPSG string (eg "EPSG:4326") or a WKT string.

        Returns
        -------
        The transformed vector(s).

        """
        return Vector(self._geo.to_crs(crs))

    def cast_field(self, name, dtype):
        """Cast a field to the specified `dtype`.

        Parameters
        ----------
        name : str
            The field name to cast.
        dtype : str, type, or numpy.dtype
            The dtype to cast to.

        Returns
        -------
        Vector
            A new vector with the updated attribute table.

        """
        if not is_str(name):
            raise TypeError("Name must be a string")
        if name not in self._geo:
            raise ValueError(f"Inalid attribute name: {name}")
        geo = self._geo.copy()
        geo[name] = geo[name].astype(dtype)
        return Vector(geo)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise NotImplementedError("Slicing is not supported")
        if not is_int(idx):
            raise TypeError(f"Invalid index type: {type(idx)}")
        if idx >= self.size:
            raise IndexError(
                f"Index {idx} is too large for vector with size {self.size}"
            )
        # Convert negative indices to positive because the index only handles +
        if idx < 0:
            old_idx = idx
            idx = self.size + idx
            if idx < 0:
                raise IndexError(
                    f"Negative index {old_idx} is too small for vector with "
                    f"size {self.size}."
                )
        # [[idx]] forces loc to return a data frame. The new frames index is
        # [idx], however. This breaks future iteration we reset it to [0] to
        # allow iteration over the resulting Vector.
        subgeo = self._geo.loc[[idx]]
        subgeo.index = dask.array.from_array([0]).to_dask_dataframe()
        return Vector(subgeo, 1)

    def buffer(self, *args, **kwargs):
        """Apply the buffer operation to the vector geometries.

        Parameters
        ----------
        *args : positional args
            Arguments to be passed on to GoePandas.
        **kwargs : keyword args
            Keyword arguments to be passed onto GeoPandas.

        Returns
        -------
        Vector
            The buffered vector.

        """
        return Vector(self._geo.buffer(*args, **kwargs))

    def simplify(self, *args, **kwargs):
        """Simplify the vector geometries.

        Parameters
        ----------
        *args : positional args
            Arguments to be passed on to GoePandas.
        **kwargs : keyword args
            Keyword arguments to be passed onto GeoPandas.

        Returns
        -------
        Vector
            The simplify vector.

        """
        return Vector(self._geo.simplify(*args, **kwargs))

    def save(self, path, **fiona_kwargs):
        """Save the vector data to a file.

        By default, an ESRI shapefile is written.

        Parameters
        ----------
        path : str
            Output location to save the vector data to.
        **fiona_kwargs : keyword args
            Keyword arguments to pass on to :func:`fiona.open` such as
            ``layer='name'``, ``driver='GPKG'``, or ``mode='a'``

        Returns
        -------
        Vector
            Returns a new vector pointing to the saved location.

        """
        (geo,) = dask.compute(self._geo)
        geo.to_file(path, **fiona_kwargs)
        return open_vectors(path)

    def model_predict(self, model, fields, n_outputs=1, out_prefix="pred_"):
        """Predict new columns using a model.

        `layer`'s values are used as predictors for the model to produce new
        predictions. The resulting predictions are used to create a new vector
        with the results appended as new columns.

        The `model` argument must provide a `predict` method. If the desired
        model does not provide a `predict` function,
        :class:`ModelPredictAdaptor` can be used to wrap it and make it
        compatible with this function. information.

        Parameters
        ----------
        layer : vector or path str
            Vector with attribute columns.
        model : object
            The model used to estimate new values. It must have a `predict`
            method that takes an array-like object of shape `(N, M)`, where `N`
            is the number of samples and `M` is the number of
            features/predictor variables. The `predict` method should return an
            `(N, [n_outputs])` shape result. If only one variable is resurned,
            then the `n_outputs` dimension is optional.
        fields : list of str
            The names of columns used in the model
        n_outputs : int, optional
            The number of output variables from the model. Each output variable
            produced by the model is converted to a band in the output raster.
            The default is ``1``.
        out_prefix : str, optional
            The prefix to use when naming the resulting column(s). The prefix
            will be combined with the 1-based number of the output variable.
            The Default is `"pred_"`.

        Returns
        -------
        Vector
            The resulting vector with estimated values appended as a columns
            (pred_1, pred_2, pred_3 ...).

        """
        from raster_tools.general import model_predict_vector

        return model_predict_vector(self, model, fields, n_outputs, out_prefix)
