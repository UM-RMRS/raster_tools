import os
import sys

import dask
import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from dask.delayed import delayed
from dask.diagnostics import ProgressBar

from raster_tools.dtypes import I64, is_int, is_str

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
    "add_objectid_column",
    "count_layer_features",
    "get_vector",
    "list_layers",
    "open_vectors",
]


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
    parts = []
    divs = []
    while row_offset < total_size:
        left = row_offset
        right = min(row_offset + batch_size, total_size)
        rows = slice(left, right)
        part = delayed(gpd.read_file)(path, rows=rows, layer=layer)
        index = pd.RangeIndex(left, right)
        part = delayed(_df_asign_index)(part, index)
        parts.append(part)
        divs.append(left)
        row_offset += batch_size
    divs.append(right - 1)
    meta = gpd.read_file(path, layer=layer, rows=1)
    divs = tuple(divs)
    # Returns a dask_geopandas GeoDataFrame
    return dd.from_delayed(parts, meta, divisions=divs), total_size


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

    layer_vecs = []
    for layer in layers:
        n = count_layer_features(path, layer=layer)
        if PYOGRIO_SUPPORTED:
            layer_vec = dgpd.read_file(
                path, layer=layer, chunksize=_TARGET_CHUNK_SIZE
            )
            pair = (layer_vec, n)
        else:
            n_parts = max(n // _TARGET_CHUNK_SIZE, 1)
            pair = _read_file_delayed(path, layer, n_parts)
        layer_vecs.append(pair)

    if len(layer_vecs) > 1:
        return [Vector(layer_vec, n) for layer_vec, n in layer_vecs]
    return Vector(*layer_vecs[0])


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


def get_dask_geodataframe(src):
    if isinstance(src, Vector):
        return src.data
    elif is_str(src):
        return get_vector(src).data
    elif isinstance(src, gpd.GeoSeries):
        return dgpd.from_geopandas(src.to_frame("geometry"), npartitions=1)
    elif isinstance(src, dgpd.GeoSeries):
        return src.to_frame("geometry")
    elif isinstance(src, gpd.GeoDataFrame):
        return dgpd.from_geopandas(src, npartitions=1)
    elif isinstance(src, dgpd.GeoDataFrame):
        return src
    else:
        raise TypeError(
            "Could not coerce input to a dask_geopandas.GeoDataFrame"
        )


def _add_objid(feats, name, base, partition_info=None):
    feats = feats.copy()
    feats[name] = np.arange(1, len(feats) + 1, dtype=I64) + (
        partition_info["number"] * base
    )
    return feats


_OBJECTID_BASE = 10**9


def add_objectid_column(features, name=None, _base=_OBJECTID_BASE):
    """Add a column of unique ID values to the set of features.

    Parameters
    ----------
    features : Vector, str, GeoDataFrame, GeoSeries
        The features to add the column to.
    name : str, optional
        The name for the column. If not given, the ESRI OBJECTID convention is
        used. For example, if no name is given, 'OBJECTID' will be used. If
        'OBJECTID' is already present, 'OBJECTID_1' is used.

    Returns
    -------
    result : Vector, dask_geopandas.GeoDataFrame
        The updated vector or dataframe with the new column of unique values.

    """
    return_vector = isinstance(features, (Vector, str))
    feats = get_dask_geodataframe(features)

    if name is not None:
        if name in feats:
            raise ValueError(f"name already exists in dataframe: {name!r}")
    else:
        prefix = "OBJECTID"
        if prefix not in feats:
            name = prefix
        else:
            n = 1
            # Use ESRI convention of OBJECTID_{n} for duplicate OBJECTID
            name = f"{prefix}_{n}"
            while name in feats:
                n += 1
                name = f"{prefix}_{n}"
    meta = feats._meta.copy()
    meta[name] = np.array((), dtype=I64)
    df_new = feats.map_partitions(_add_objid, name, _base, meta=meta)
    if feats.spatial_partitions is not None:
        df_new.spatial_partitions = feats.spatial_partitions.copy()
    if return_vector:
        return Vector(df_new)
    return df_new


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

    def eval(self):  # noqa: A003
        """Computes the built-up chain of operations on the underlying data."""
        # TODO: deprecate in favor of load
        return self.load()

    def load(self):
        """Compute delayed operations and load the result into memory."""
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

    def to_raster(
        self,
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

        This method can be used to either rasterize features using values from
        a particular data field or to create a raster mask of zeros and ones.
        Using values to rasterize is the default. Use `mask=True` to generate a
        raster mask. If no data field is specified, the underlying dataframe's
        index is used. NOTE: because of limitations in dask, dataframe index
        values are not guaranteed to be unique across the dataframe. Cells that
        do not touch or overlap any features are marked as null.

        To add a column of unique IDs for each feature, see
        :func:`raster_tools.vector.add_objectid_column` or
        :meth:`raster_tools.vector.Vector.add_objectid_column`.

        This operation can be greatly accelerated if the provided `features`
        object has been spatially shuffled or had spatial partitions
        calculated. There are a few ways to do this. For `Vector` or
        `GeoDataFrame`/`GeoSeries` objects, you can use the `spatial_shuffle`
        or `calculate_spatial_partitions` methods.
        `calculate_spatial_partitions` simply computes the spatial bounds of
        each partition in the data. `spatial_shuffle` shuffles the data into
        partitions of spatially near groups and calculates the spatial bounds
        at the same time. This second method is more expensive but provides a
        potentially greater speed up for rasterization. The `use_spatial_aware`
        flag can also be provided to this function. This causes the spatial
        partitions to be calculated before rasterization.

        .. note::
            If the CRS for this vector does not match the CRS for `like`,
            this vector will be transformed to `like`'s CRS. This operation
            causes spatial partition information to be lost. It is recommended
            that the CRSs for both are matched ahead of time.

        Parameters
        ----------
        like : Raster
            A raster to use for grid and CRS information. The resulting raster
            will be on the same grid as `like`.
        field : str, optional
            The name of a field to use for cell values when rasterizing the
            vector features. If None or not specified, the underlying
            dataframe's index is used. The default is to use the index.
        overlap_resolve_method : str, optional
            The method used to resolve overlaping features. Default is
            `"last"`. The available methods are:

            'first'
                Cells with overlapping features will receive the value from the
                feature that appears first in the feature table.
            'last'
                Cells with overlapping features will receive the value from the
                feature that appears last in the feature table.
            'min'
                Cells with overlap will receive the value from the feature with
                the smallest value.
            'max'
                Cells with overlap will receive the value from the feature with
                the largest value.

        mask : bool, optional
            If ``True``, the features are rasterized as a mask. Cells that do
            not touch a feature are masked out. Cells that touch the features
            are set to ``1``. If `mask_invert` is also ``True``, this is
            inverted. If `mask` is ``False``, the features are rasterized using
            `field` to retrieve values from the underlying dataframe. `field`
            is ignored, if this option is a used. Default is ``False``.
        mask_invert : bool, optional
            If ``True`` cells that are inside or touch a feature are masked
            out. If ``False``, cells that do not touch a feature are masked
            out. Default is ``False``.
        null_value : scalar, optional
            The value to use in cells with no feature data, when not masking.
        all_touched : bool, optional
            If ``True``, grid cells that the vector touches will be burned in.
            If False, only cells with a center point inside of the vector
            perimeter will be burned in.
        use_spatial_aware : bool, optional
            Force the use of spatial aware rasterization. If ``True`` and
            `features` is not already spatially indexed, a spatial index will
            be computed. Alternatively, if ``True`` and `features`'s CRS
            differs from `like`, a new spatial index in a common CRS will be
            computed. If `features` already has a spatial index and its CRS
            matches `like`, this argument is ignored. Default is ``False``.
        show_progress : bool, optional
            If `use_spatial_aware` is ``True``, this flag causes a progress bar
            to be displayed for spatial indexing. Default is ``False``.

        Returns
        -------
        Raster
            The resulting single band raster of rasterized features. This
            raster will be on the same grid as `like`.

        """
        from raster_tools.rasterize import rasterize

        return rasterize(
            self,
            like,
            field=field,
            overlap_resolve_method=overlap_resolve_method,
            mask=mask,
            mask_invert=mask_invert,
            null_value=null_value,
            all_touched=all_touched,
            use_spatial_aware=use_spatial_aware,
            show_progress=show_progress,
        )

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

    def add_objectid_column(self, name=None):
        """Add a column of unique ID values to the set of features.

        Parameters
        ----------
        name : str, optional
            The name for the column. If not given, the ESRI OBJECTID convention
            is used. For example, if no name is given, 'OBJECTID' will be used.
            If 'OBJECTID' is already present, 'OBJECTID_1' is used, etc.

        Returns
        -------
        result : Vector
            The updated vector with the new column of unique values.

        """
        return add_objectid_column(self, name=name)

    def spatial_shuffle(self, show_progress=False):
        """Shuffle the features into spatially grouped partitions.

        This sorts the features into partitions such that each partition
        contains features that are near to one another. This has the added
        benefit of computing the spatial extents of each partition.

        .. note::
            This causes partial computation to take place.

        """
        if show_progress:
            with ProgressBar():
                data = self._geo.spatial_shuffle()
        else:
            data = self._geo.spatial_shuffle()
        return Vector(data)

    def calculate_spatial_partitions(self, show_progress=False):
        """Calculate the spatial bounds of the underlying data.

        Parameters
        ----------
        show_progress : bool, optional
            Show a progressbar for the calculation. This operation can take a
            significant amount of time, so a progressbar may be helpful.
            Default is ``False``.

        .. note::
            This causes partial computation to take place.

        """
        data = self._geo.copy()
        if show_progress:
            with ProgressBar():
                data.calculate_spatial_partitions()
        else:
            data.calculate_spatial_partitions()
        return Vector(data)
