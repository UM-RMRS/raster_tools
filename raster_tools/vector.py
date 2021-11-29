import os

import dask
import dask.dataframe as dd
import dask_geopandas as dgpd
import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from dask.delayed import delayed
from rasterio.enums import MergeAlg
from rasterio.features import rasterize as rio_rasterize

from raster_tools import Raster
from raster_tools.raster import is_raster_class

from ._types import F64, U64
from ._utils import is_int, is_str

__all__ = ["open_vectors", "Vector"]


class VectorError(Exception):
    pass


def _df_asign_index(df, index):
    df.index = index
    return df


def _read_file_delayed(path, layer, npartitions):
    with fiona.open(path, layer=layer) as src:
        total_size = len(src)

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
    return dd.from_delayed(dfs, meta, divisions=divs)


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


_LAZY_THRESHOLD = 100_000


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

    src_layers = fiona.listlayers(path)
    n = len(src_layers)
    if layers is not None:
        layers = _normalize_layers_arg(layers)
        src_layers_set = set(src_layers)
        if is_str(layers[0]) and any(
            layer not in src_layers_set for layer in layers
        ):
            raise ValueError("Invalid layer name")
        elif is_int(layers[0]) and any(
            layer >= n or layer < 0 for layer in layers
        ):
            raise ValueError(
                "Invalid layer index. Index either outside valid range or less"
                " than 0"
            )
    else:
        layers = src_layers

    dfs = []
    for layer in layers:
        with fiona.open(path, layer=layer) as fsrc:
            n = len(fsrc)
        if n >= _LAZY_THRESHOLD:
            # TODO: determine number of partitions
            dfs.append(_read_file_delayed(path, layer, 10))
        else:
            dfs.append(gpd.read_file(path, layer=layer))
    if len(dfs) > 1:
        return [Vector(df) for df in dfs]
    return Vector(dfs[0])


def _get_geo_len(geo, error=False):
    if dask.is_dask_collection(geo):
        n = len(geo)
        if not np.isnan(n):
            return n
        if error:
            raise ValueError(
                "Could not determine size of lazy geo data. Divisions not set."
            )
        return None
    return len(geo)


def _rasterize_block(block, geometry, values, dtype, all_touched):
    shape = block.shape[1:]
    rast_array = rio_rasterize(
        zip(geometry.values, values),
        out_shape=shape,
        transform=block.rio.transform(),
        fill=0,
        all_touched=all_touched,
        merge_alg=MergeAlg.replace,
        dtype=dtype,
    )
    # Add band dimension to array
    block.data = rast_array[None]
    return block


def _rasterize_map_blocks(template, geometry, values, dtype, all_touched):
    # Force computation here to avoid computing in each block-rasterization
    # call.
    geometry, values = dask.compute(geometry, values)
    # Filter invalid geometries
    valid = ~(geometry.is_empty | geometry.isna())
    geometry = geometry[valid]
    values = values[valid.values]

    result = template.map_blocks(
        _rasterize_block,
        args=[
            geometry,
            values,
            dtype,
            all_touched,
        ],
        template=template,
    )
    return result.data


def _rasterize_partition(df, xlike, values, all_touched=True):
    if xlike.shape[0] > 1:
        xlike = xlike[:1]
    geometry = df.geometry
    if values.dtype == U64:
        # rasterio doesn't like uint64
        # This is very unlikely. n would have to be huge. If this triggers,
        # other things will likely break.
        values.astype(F64)
    template = xr.zeros_like(xlike, dtype=values.dtype)

    # geometry is likely lazy. It could also potentially wrap a very large
    # feature collection. It must be computed at some point before mapping
    # the rasterization function onto the template blocks. To avoid doing this
    # here and now, we wrap the rasterization operation in a delayed object.
    lazy_result_data = delayed(_rasterize_map_blocks)(
        template, geometry, values, values.dtype, all_touched
    )
    result = template.copy()
    result.data = dask.array.from_delayed(
        lazy_result_data,
        template.shape,
        dtype=values.dtype,
        meta=template.data,
    )
    return result


def _vector_to_raster_dask(df, xlike, all_touched=True):
    if not dask.is_dask_collection(xlike):
        raise ValueError("xlike must be a dask collection")

    N = _get_geo_len(df, error=True)
    values_dtype = np.min_scalar_type(N)
    if not dask.is_dask_collection(df) or df.npartitions == 1:
        values = np.arange(1, N + 1, dtype=values_dtype)
        return _rasterize_partition(df, xlike, values, all_touched=all_touched)

    # There is no way to neatly align the dataframe partitions with the raster
    # chunks. Because of this, we burn in each partition on its own raster and
    # then merge the results.
    results = []
    offset = 0
    for part in df.partitions:
        n = _get_geo_len(part, error=True)
        values = dask.array.arange(
            offset + 1, offset + n + 1, dtype=values_dtype
        )
        res = _rasterize_partition(
            part,
            xlike,
            values,
            all_touched=all_touched,
        )
        offset += n
        results.append(res)
    # Merge along band dim using max
    result = xr.concat(results, dim="band").max(axis=0, keep_attrs=True)
    result = result.expand_dims("band")
    result["band"] = [1]
    result["spatial_ref"] = xlike.spatial_ref
    return result


def _parse_input_raster(raster):
    if not is_raster_class(raster):
        if is_str(raster):
            raster = Raster(raster)
        else:
            raise TypeError(
                "Raster input must be a Raster type or path string"
            )
    return raster


def _is_series(geo):
    return isinstance(geo, (gpd.GeoSeries, dgpd.GeoSeries))


def _is_frame(geo):
    return isinstance(geo, (gpd.GeoDataFrame, dgpd.GeoDataFrame))


_GMTRY = "geometry"


class Vector:
    """A class representing vector data.

    Avoid explicitly creating `Vector` objects. Use :func:`open_vectors`
    instead.

    """

    def __init__(self, geo):
        """Create a `Vector` object.

        Parameters
        ----------
        geo : GeoDataFrame or GeoSeries
            The vector data to use. This can be as geopandas or dask_geopandas
            object.

        """
        if not _is_series(geo) and not _is_frame(geo):
            raise TypeError(
                "Invalid data type. Must be some type GeoDataFrame or"
                f" GeoSeries. Got {type(geo)}."
            )
        if _is_series(geo):
            geo = geo.to_frame()
        self._geo = geo
        size = _get_geo_len(self._geo)
        if size is None:
            raise VectorError(
                "Could not determine size of dask data. Make sure to set the"
                " divisions."
            )

    def __len__(self):
        return _get_geo_len(self._geo)

    def __repr__(self):
        return "<Vector(shapes: {}, fields: {}, CRS: {}, lazy:{})>".format(
            self.size,
            self.shape[1],
            self.crs,
            dask.is_dask_collection(self._geo),
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
        """The number of vectors contained."""
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
            return Vector(self._geo.compute())
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
        return Vector(dgpd.from_geopandas(self._geo, npartitions=npartitions))

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

    def to_raster(self, like, all_touched=True):
        """Convert vector data to a raster.

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
            The resulting raster. The result will have a single band. Each
            vector shape is assigned a 1-based integer value equal to its order
            in the original vector collection. This integer value is what is
            burned in at the corresponding grid cells. The dtype of the result
            will be the minimum, unsigned integer type that can fit the ID
            values. The null value is 0.

        """
        like = _parse_input_raster(like).to_lazy()

        xrs = _vector_to_raster_dask(
            self.to_crs(like.crs)._geo,
            xlike=like.to_xarray(),
            all_touched=all_touched,
        )
        return Raster(xrs).set_null_value(0)

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
        # [[idx]] forces loc to return a data frame
        return Vector(self._geo.loc[[idx]])

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
            Returns itself.

        """
        self._geo.to_file(path, **fiona_kwargs)
        return self
