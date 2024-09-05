"""
Description: general module used to perform common spatial analyses
on Raster objects

* `ESRI Generalization Tools <https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/an-overview-of-the-generalization-tools.htm>`_
* `ESRI Local Tools <https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/an-overview-of-the-local-tools.htm>`_

"""  # noqa: E501

import os
import re
from collections.abc import Iterable, Sequence
from functools import partial

import dask.array as da
import numba as nb
import numpy as np
import pandas as pd
import xarray as xr
from dask_image import ndmeasure as ndm
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    grey_dilation,
    grey_erosion,
)

from raster_tools.creation import empty_like
from raster_tools.dtypes import (
    BOOL,
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    get_common_dtype,
    is_bool,
    is_int,
    is_scalar,
    is_str,
)
from raster_tools.exceptions import RemapFileParseError
from raster_tools.focal import _get_offsets
from raster_tools.masking import (
    get_default_null_value,
    reconcile_nullvalue_with_dtype,
)
from raster_tools.raster import (
    Raster,
    data_to_raster_like,
    data_to_xr_raster_ds_like,
    dataarray_to_xr_raster_ds,
    get_raster,
    xr_where_with_meta,
)
from raster_tools.stat_common import (
    nan_unique_count_jit,
    nanargmax_jit,
    nanargmin_jit,
    nanasm_jit,
    nanentropy_jit,
    nanmode_jit,
)

__all__ = [
    "ModelPredictAdaptor",
    "aggregate",
    "band_concat",
    "dilate",
    "erode",
    "local_stats",
    "model_predict_vector",
    "model_predict_raster",
    "reclassify",
    "regions",
    "remap_range",
    "where",
]

# TODO: mosaic


def _create_labels(xarr, wd, uarr=None):
    if uarr is None:
        uarr = da.unique(xarr).compute()
    # Drop 0 so they are skipped later
    uarr = uarr[uarr != 0]

    cum_num = 0
    result = da.zeros_like(xarr, dtype=U64)
    for v in uarr:
        labeled_array, num_features = ndm.label((xarr == v), structure=wd)
        result += da.where(
            labeled_array > 0, labeled_array + cum_num, 0
        ).astype(result.dtype)
        cum_num += num_features
    return result


def regions(raster, neighbors=4, unique_values=None):
    """Calculates the unique regions (patches) within a raster band.

    The approach is based on ESRI's region group calculation.

    Parameters
    ----------
    raster : Raster or path str
        The raster to perform the calculation on. All unique non zero values
        will be used to define unique regions.
    neighbors : int, optional
        The neighborhood connectivity value. Valid values are 4 and 8. If
        4, a rook pattern is used, e.g. the neighbors along the horizontal and
        vertical directions are used. if 8, then all of the 8 neighbors are
        used. Default is 4.
    unique_values : array or list, optional
        Values that represent zones from which regions will be made. Values not
        included will be grouped together to form one large zone. If `None`,
        each unique value in the raster will be considered a zone and will be
        calculated up front.

    Returns
    -------
    Raster
        The resulting raster of unique regions values. The bands will have the
        same shape as the original Raster. The null value mask from the origial
        raster will be applied to the result.

    References
    ----------
    * `ESRI: Region Group <https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/region-group.htm>`_

    """  # noqa: E501
    raster = get_raster(raster)

    if not is_int(neighbors):
        raise TypeError(
            f"neighbors argument must be an int. Got {type(neighbors)}"
        )
    if neighbors == 4:
        wd = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    elif neighbors == 8:
        wd = np.ones((3, 3), dtype=int)
    else:
        raise ValueError(
            f"Invalid value for neighbors parameter: {repr(neighbors)}"
        )

    if unique_values is not None:
        if isinstance(unique_values, (np.ndarray, da.Array, Sequence)):
            unique_values = np.asarray(unique_values)
        else:
            raise TypeError("Invalid type for unique_values parameter")

    data = raster.data
    # Copy to prevent input and output rasters from having same reference
    mask = raster.mask.copy()
    if raster._masked:
        # Set null values to 0 to skip them in the labeling phase
        data = da.where(raster.mask, 0, data)
    data_out = [_create_labels(band, wd, unique_values) for band in data]
    data_out = da.stack(data_out, axis=0)
    nv = None
    if raster._masked:
        nv = reconcile_nullvalue_with_dtype(raster.null_value, U64)
        data_out = da.where(mask, nv, data_out)
    ds = data_to_xr_raster_ds_like(data_out, raster.xdata, mask=mask, nv=nv)
    return Raster(ds, _fast_path=True)


@nb.jit(nopython=True, nogil=True, parallel=True)
def _coarsen_chunk(x, axis, func, out_dtype, check_nan):
    dims = sorted(set(range(len(x.shape))) - set(axis))
    shape = (x.shape[dims[0]], x.shape[dims[1]], x.shape[dims[2]])
    out = np.empty(shape, out_dtype)
    for i in range(shape[0]):
        for j in nb.prange(shape[1]):
            for k in nb.prange(shape[2]):
                v = func(x[i, j, :, k, :])
                if check_nan and np.isnan(v):
                    # It doesn't matter what value is swapped with nan values
                    # since those cells will be masked out later.
                    v = 0
                out[i, j, k] = v
    return out


def _coarsen_block_map(x, axis, agg_func, out_dtype, check_nan):
    dims = sorted(set(range(len(x.shape))) - set(axis))
    chunks = tuple(x.chunks[d] for d in dims)
    return da.map_blocks(
        partial(
            _coarsen_chunk,
            axis=axis,
            func=agg_func,
            out_dtype=out_dtype,
            check_nan=check_nan,
        ),
        x,
        chunks=chunks,
        drop_axis=axis,
        meta=np.array((), dtype=out_dtype),
    )


def _get_coarsen_dtype(stat, window_size, input_dtype):
    if stat == "unique":
        return np.min_scalar_type(window_size)
    if stat in ("mode", "min", "max"):
        return input_dtype
    if input_dtype == F32:
        return F32
    return F64


_COARSEN_STYPE_TO_FUNC = {
    "max": lambda x: x.max(),
    "mean": lambda x: x.mean(),
    "median": lambda x: x.median(),
    "min": lambda x: x.min(),
    "prod": lambda x: x.prod(),
    "std": lambda x: x.std(),
    "sum": lambda x: x.sum(),
    "var": lambda x: x.var(),
}
_COARSEN_STYPE_TO_CUSTOM_FUNC = {
    "asm": nanasm_jit,
    "entropy": nanentropy_jit,
    "mode": nanmode_jit,
    "unique": nan_unique_count_jit,
}


def _get_unique_dtype(cur_dtype):
    if cur_dtype in (I8, U8):
        return I16
    if cur_dtype == U16:
        return I32
    if cur_dtype == U32:
        return I64
    return cur_dtype


def aggregate(raster, expand_cells, stype):
    """Creates a Raster of aggregated cell values for a new resolution.

    The approach is based on ESRI's aggregate and majority filter functions.

    Parameters
    ----------
    raster : Raster or path str
        Input Raster object or path string
    expand_cells : 2-tuple, list, array-like
        Tuple, array, or list of the number of cells to expand in y and x
        directions. The first element corresponds to the y dimension and the
        second to the x dimension.
    stype : str
        Summarization type. Valid opition are mean, std, var, max, min,
        unique prod, median, mode, sum, unique, entropy, asm.

    Returns
    -------
    Raster
        The resulting raster of aggregated values.

    References
    ----------
    * `ESRI aggregate <https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/aggregate.htm>`_
    * `ESRI majority filter <https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/majority-filter.htm>`_

    """  # noqa: E501
    expand_cells = np.atleast_1d(expand_cells)
    if not is_int(expand_cells.dtype):
        raise TypeError("expand_cells must contain integers")
    if expand_cells.shape != (2,):
        raise ValueError("expand_cells must contain 2 elements")
    if (expand_cells == 1).all():
        raise ValueError("expand_cells values cannont both be one")
    if not (expand_cells >= 1).all():
        raise ValueError("All expand_cells values must be >= 1")
    if not is_str(stype):
        raise TypeError("stype argument must be a string")
    stype = stype.lower()
    if (
        stype not in _COARSEN_STYPE_TO_FUNC
        and stype not in _COARSEN_STYPE_TO_CUSTOM_FUNC
    ):
        raise ValueError(f"Invalid stype argument: {repr(stype)}")
    orig_dtype = get_raster(raster).dtype
    raster = get_raster(raster, null_to_nan=True)

    xdata = raster.xdata
    xmask = raster.xmask
    dim_map = {"y": expand_cells[0], "x": expand_cells[1]}
    xdatac = xdata.coarsen(dim=dim_map, boundary="trim")
    if stype in _COARSEN_STYPE_TO_FUNC:
        xdata = _COARSEN_STYPE_TO_FUNC[stype](xdatac)
    else:
        custom_stat_func = _COARSEN_STYPE_TO_CUSTOM_FUNC[stype]
        check_nan = stype == "mode"
        out_dtype = _get_coarsen_dtype(
            stype, np.prod(expand_cells), orig_dtype
        )
        # Use partial because reduce seems to be bugged and grabs kwargs that
        # it shouldn't.
        xdata = xdatac.reduce(
            partial(
                _coarsen_block_map,
                agg_func=custom_stat_func,
                out_dtype=out_dtype,
                check_nan=check_nan,
            )
        )
    if raster._masked:
        # Coarsen mask as well
        xmask = raster.xmask.coarsen(dim=dim_map, boundary="trim").all()

        if stype == "unique":
            xdata = xdata.astype(_get_unique_dtype(xdata.dtype))
        elif stype == "mode":
            # Cast back to original dtype. Original null value will work
            xdata = xdata.astype(orig_dtype)
        # Replace null cells with null value acording to mask
        nv = get_default_null_value(xdata.dtype)
        xdata = xr_where_with_meta(xmask, nv, xdata, nv=nv)
    else:
        xmask = xr.zeros_like(xdata, dtype=bool)
    ds = dataarray_to_xr_raster_ds(xdata, xmask, crs=raster.crs)
    return Raster(ds, _fast_path=True)


class ModelPredictAdaptor:
    """
    An adaptor class that allows models without a `predict` method to be used
    with :func:`model_predict_raster` and :func:`model_predict_vector`.

    Examples
    --------
    For a model function:

    >>> def my_model_func(x):
    ...     return np.sin(x) + np.cos(x)
    >>> model = ModelPredictAdaptor(my_model_func)
    >>> result_raster = rts.general.model_predict_raster(
    ...         predictors, model, n_outputs=predictors.nbands
    ... )

    For a model function lacking a `predict` method:

    >>> class MyModel:
    ...     # ...
    ...     def transform(x):
    ...         # ...
    ...         return result
    >>> model = ModelPredictAdaptor(MyModel(), "transform")
    >>> result_raster = rts.general.model_predict_raster(predictors, model)

    """

    def __init__(self, model, method=None):
        """
        Constructs an adaptor.

        Parameters
        ----------
        model : callable, object
            The model function or object. The `method` parameter determines
            how this object is handled.
        method : None, str, callable, optional
            Determines how input data is provided to `model`. If ``None``,
            `model` is called with the data as the argument, e.g. `model(x)`.
            If a string, the adaptor passes input data to the method on `model`
            with the same name, e.g. for `method="my_func"`,
            `model.my_func(x)` is called.

        """
        if method is None:

            def func(x):
                return model(x)

            self.method = func
        elif isinstance(method, str):
            if not hasattr(model, method):
                raise AttributeError(
                    f"The given model object has no attribute '{method}'"
                )

            def func(x):
                return getattr(model, method)(x)

            self.method = func
        else:
            raise ValueError(
                "Could not understand the method argument. method must be "
                "None or a string."
            )

    def predict(self, x):
        """Pass the input data to the wrapped model and produce the results"""
        return self.method(x)


def _pred_raster(xarr, model, nbands):
    b, y, x = xarr.shape
    vl_arr = np.moveaxis(xarr, 0, -1).reshape(y * x, b)
    pred = model.predict(np.nan_to_num(vl_arr))
    if pred.ndim == 1:
        pred = np.expand_dims(pred, 1)
    return np.moveaxis(pred, -1, 0).reshape(nbands, y, x)


def model_predict_raster(raster, model, n_outputs=1):
    """
    Generate a new raster using the provided model to predict new values.

    The raster's values are used as the predictor inputs for `model`. Each band
    in the input raster is used as a separate input variable. Outputs are
    raster surfaces where each band corresponds to a variable output by
    `model`.

    The `model` argument must provide a `predict` method. If the desired model
    does not provide a `predict` function, :class:`ModelPredictAdaptor` can be
    used to wrap it and make it compatible with this function.

    Parameters
    ----------
    model : object
        The model used to estimate new values. It must have a `predict` method
        that takes an array-like object of shape `(N, M)`, where `N` is the
        number of samples and `M` is the number of features/predictor
        variables. The `predict` method should return an `(N, [n_outputs])`
        shape result. If only one variable is resurned, then the `n_outputs`
        dimension is optional.
    n_outputs : int, optional
        The number of output variables from the model. Each output variable
        produced by the model is converted to a band in the output raster. The
        default is ``1``.

    Returns
    -------
    Raster
        The resulting raster of estimated values. Each band corresponds to an
        output variable produced by `model`.

    """
    in_raster = get_raster(raster, null_to_nan=True)
    n_outputs = int(n_outputs)
    if n_outputs < 1:
        raise ValueError("n_outputs must be greater than 0")
    in_chunks = in_raster.xdata.chunksizes
    nbands = in_raster.nbands

    data = in_raster.data.rechunk(chunks=(nbands, "auto", "auto"))
    interim_chunks = data.chunks

    data = data.map_blocks(
        _pred_raster,
        model=model,
        nbands=n_outputs,
        chunks=(n_outputs, interim_chunks[1], interim_chunks[2]),
        meta=np.array((), dtype=F64),
        dtype=F64,
    ).rechunk((1, in_chunks["y"], in_chunks["x"]))
    xdata = xr.DataArray(
        data,
        coords={
            "band": np.arange(n_outputs) + 1,
            "y": in_raster.y,
            "x": in_raster.x,
        },
    )
    if in_raster._masked:
        xmask = in_raster.xmask.any(dim="band", keepdims=True)
        if n_outputs > 1:
            xmask = xr.concat([xmask] * n_outputs, "band")
        xdata = xdata.rio.write_nodata(get_default_null_value(xdata.dtype))
    else:
        xmask = xr.zeros_like(xdata, dtype=bool)
    ds = dataarray_to_xr_raster_ds(xdata, xmask)
    if in_raster.crs is not None:
        ds = ds.rio.write_crs(in_raster.crs)
    return Raster(ds, _fast_path=True).burn_mask()


def _pred_df(df, model, columns, n_outputs, prefix):
    x = df[columns].to_numpy()
    valid_mask = ~np.isnan(x).any(axis=1)

    y = model.predict(x[valid_mask])
    if y.ndim == 1:
        y = np.expand_dims(y, 1)
    pred = np.empty((len(df), n_outputs), dtype=F64)
    pred[valid_mask] = y
    pred[~valid_mask] = np.nan

    dic = {}
    if pred.shape[1] > 1:
        for i in range(pred.shape[1]):
            dic[f"{prefix}{i + 1}"] = pred[:, i]
    else:
        dic[f"{prefix}{1}"] = pred.squeeze()
    df2 = pd.DataFrame(dic, index=df.index)
    return pd.concat([df, df2], axis=1)


def model_predict_vector(
    features, model, fields, n_outputs=1, out_prefix="pred_"
):
    """Predict new columns using a model.

    `features`' values are used as predictors for the model to produce new
    predictions. The resulting predictions are used to create a new vector with
    the results appended as new columns.

    The `model` argument must provide a `predict` method. If the desired model
    does not provide a `predict` function, :class:`ModelPredictAdaptor` can be
    used to wrap it and make it compatible with this function. information.

    Parameters
    ----------
    features : vector or path str
        Vector with attribute columns.
    model : object
        The model used to estimate new values. It must have a `predict` method
        that takes an array-like object of shape `(N, M)`, where `N` is the
        number of samples and `M` is the number of features/predictor
        variables. The `predict` method should return an `(N, [n_outputs])`
        shape result. If only one variable is resurned, then the `n_outputs`
        dimension is optional.
    fields : list of str
        The names of columns used in the model
    n_outputs : int, optional
        The number of output variables from the model. Each output variable
        produced by the model is converted to a band in the output raster. The
        default is ``1``.
    out_prefix : str, optional
        The prefix to use when naming the resulting column(s). The prefix will
        be combined with the 1-based number of the output variable. The Default
        is `"pred_"`.

    Returns
    -------
    Vector
        The resulting vector with estimated values appended as a columns
        (pred_1, pred_2, pred_3 ...).

    """
    from raster_tools.vector import get_vector

    vc = get_vector(features).copy()
    feats = vc._geo
    fields = list(fields)
    for field in fields:
        if field not in feats:
            raise KeyError(f"Invalid field name: '{field}'")
    n_outputs = int(n_outputs)
    if n_outputs < 1:
        raise ValueError("n_outputs must be greater than 0")
    if not isinstance(out_prefix, str):
        raise TypeError("out_prefix must be a string")
    for i in range(n_outputs):
        i += 1
        c = f"{out_prefix}{i}"
        if c in feats:
            raise KeyError(
                "out_prefix produced a column name that is already present in "
                "the feature: '{c}'."
            )

    meta = feats._meta.copy()
    new_meta_cols = pd.DataFrame(
        {
            f"{out_prefix}{i + 1}": np.array([], dtype=F64)
            for i in range(n_outputs)
        }
    )
    meta = pd.concat([feats._meta, new_meta_cols], axis=1)
    vc._geo = feats.map_partitions(
        _pred_df,
        model=model,
        columns=fields,
        n_outputs=n_outputs,
        prefix=out_prefix,
        meta=meta,
    )
    return vc


@nb.jit(nopython=True, nogil=True)
def _local_chunk(x, func, out):
    dshape = x.shape
    rws = dshape[1]
    clms = dshape[2]
    for r in range(rws):
        for c in range(clms):
            farr = x[:, r, c]
            out[0, r, c] = func(farr)
    return out


def _get_local_chunk_func(func, out_dtype):
    def wrapped(x):
        return _local_chunk(
            x, func, np.empty((1, *x.shape[1:]), dtype=out_dtype)
        )

    return wrapped


_LOCAL_STYPE_TO_NUMPY_FUNC = {
    "mean": np.nanmean,
    "std": np.nanstd,
    "var": np.nanvar,
    "max": np.nanmax,
    "min": np.nanmin,
    "prod": np.nanprod,
    "sum": np.nansum,
    "median": np.nanmedian,
}
# Use custom min/max band because numpy versions throw errors if all values are
# nan.
_LOCAL_STYPE_TO_CUSTOM_FUNC = {
    "asm": nanasm_jit,
    "entropy": nanentropy_jit,
    "minband": nanargmin_jit,
    "maxband": nanargmax_jit,
    "mode": nanmode_jit,
    "unique": nan_unique_count_jit,
}


def _local(data, stype):
    bnds = data.shape[0]
    orig_chunks = data.chunks
    # Rechunk so band dim is contiguous and chunk sizes are reasonable
    data = da.rechunk(data, chunks=(bnds, "auto", "auto"))

    ffun = _LOCAL_STYPE_TO_CUSTOM_FUNC[stype]
    if stype == "unique":
        out_dtype = np.min_scalar_type(data.shape[0])
    elif stype == "mode":
        out_dtype = data.dtype
    elif stype in ("minband", "maxband"):
        # min/max band inner funcs will return < 0 if all values are nan which
        # will cause overflow for the unsigned type returned by
        # min_scalar_type. This is fine since we can mask them out later.
        out_dtype = np.min_scalar_type(data.shape[0] - 1)
    elif data.dtype == F32:
        out_dtype = F32
    else:
        out_dtype = F64
    ffun = _get_local_chunk_func(ffun, out_dtype)

    out_chunks = ((1,), *data.chunks[1:])
    data_out = da.map_blocks(
        ffun,
        data,
        chunks=out_chunks,
        dtype=out_dtype,
        meta=np.array((), dtype=out_dtype),
    )
    # Rechunk again to expand/contract chunks to a reasonable size and split
    # bands apart
    return da.rechunk(data_out, chunks=(1, *orig_chunks[1:]))


def local_stats(raster, stype):
    """Creates a Raster of summarized values across bands.

    The approach is based on ESRI's local function.

    Parameters
    ----------
    raster : Raster or path str
        Input Raster object or path string
    stype : str
        Summarization type. Valid opitions are mean, std, var, max, min,
        maxband, minband, prod, sum, mode, median, unique, entropy, asm

    Returns
    -------
    Raster
        The resulting raster of values aggregated along the band dimension.

    References
    ----------
    * `ESRI local <https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/an-overview-of-the-local-tools.htm>`_

    """  # noqa: E501
    if not is_str(stype):
        raise TypeError("stype argument must be a string")
    stype = stype.lower()
    if (
        stype not in _LOCAL_STYPE_TO_NUMPY_FUNC
        and stype not in _LOCAL_STYPE_TO_CUSTOM_FUNC
    ):
        raise ValueError(f"Invalid stype aregument: {repr(stype)}")
    orig_dtype = get_raster(raster).dtype
    raster = get_raster(raster, null_to_nan=True)

    xdata = raster.xdata.copy()
    xmask = raster.xmask
    if stype in _LOCAL_STYPE_TO_NUMPY_FUNC:
        xdata = xdata.reduce(
            _LOCAL_STYPE_TO_NUMPY_FUNC[stype], dim="band", keepdims=True
        )
    else:
        data = xdata.data
        xdata = empty_like(raster.get_bands(1)).xdata
        xdata.data = _local(data, stype)
    if raster._masked:
        xmask = xmask.reduce(
            np.all,
            dim="band",
            keepdims=True,
        )
    else:
        xmask = xr.zeros_like(xmask, dtype=bool)

    ds_out = dataarray_to_xr_raster_ds(xdata, xmask)
    if raster._masked:
        if stype in ("unique", "minband", "maxband"):
            xdata = xdata.astype(_get_unique_dtype(xdata.dtype))
        elif stype == "mode":
            # Cast back to original dtype. Original null value will work
            xdata = xdata.astype(orig_dtype)
        nv = get_default_null_value(xdata.dtype)
        xdata = xr_where_with_meta(xmask, nv, xdata, nv=nv)
    ds_out = dataarray_to_xr_raster_ds(xdata, xmask, crs=raster.crs)
    return Raster(ds_out, _fast_path=True)


def _morph_op_chunk(x, footprint, cval, morph_op, binary=False):
    if x.ndim > 2:
        x = x[0]
    if not binary:
        morph_func = grey_dilation if morph_op == "dilation" else grey_erosion
        out = morph_func(x, footprint=footprint, mode="constant", cval=cval)
    else:
        morph_func = (
            binary_dilation if morph_op == "dilation" else binary_erosion
        )
        out = morph_func(x, structure=footprint)
    return out[None]


def _get_footprint(size):
    if isinstance(size, Iterable):
        size = tuple(size)
        if len(size) != 2:
            raise ValueError("size sequence must have lenght 2.")
        if not all(is_int(s) for s in size):
            raise TypeError("size sequence must only contain ints.")
    elif is_int(size):
        size = (size, size)
    else:
        raise TypeError("size input must be an int or sequence of ints.")
    if not all(s > 0 for s in size):
        raise ValueError("size values must be greater than 0.")
    if all(s == 1 for s in size):
        raise ValueError("At least one size value must be greater than 1.")
    footprint = np.ones(size) > 0
    return footprint


def _get_fill(dtype, op):
    type_info = np.iinfo(dtype) if is_int(dtype) else np.finfo(dtype)
    return type_info.max if op == "erosion" else type_info.min


def _erosion_or_dilation_filter(raster, footprint, op):
    data = raster.data
    fill = _get_fill(raster.dtype, op)
    if raster._masked:
        data = da.where(raster.mask, fill, data)
    rpad, cpad = _get_offsets(footprint.shape)
    # Take max because map_overlap does not support asymmetrical overlaps when
    # a boundary value is given
    depth = {0: 0, 1: max(rpad), 2: max(cpad)}
    data = da.map_overlap(
        partial(
            _morph_op_chunk,
            footprint=footprint,
            cval=fill,
            morph_op=op,
        ),
        data,
        depth=depth,
        boundary=fill,
        dtype=raster.dtype,
        meta=np.array((), dtype=raster.dtype),
    )
    mask = raster.mask
    if raster._masked:
        mask = da.map_overlap(
            partial(
                _morph_op_chunk,
                footprint=footprint,
                cval=fill,
                morph_op=op,
                binary=True,
            ),
            ~mask,
            depth=depth,
            boundary=False,
            dtype=BOOL,
            meta=np.array((), dtype=BOOL),
        )
        mask = ~mask
        data = da.where(mask, raster.null_value, data)
    else:
        mask = mask.copy()
    ds_out = data_to_xr_raster_ds_like(
        data, raster.xdata, mask=mask, nv=raster.null_value
    )
    return Raster(ds_out, _fast_path=True)


def dilate(raster, size):
    """Perform dilation on a raster

    Dilation increases the thickness of raster features. Features with higher
    values will cover features with lower values. At each cell, the miximum
    value within a window, defined by `size`, is stored in the output location.
    This is very similar to the max focal filter except that raster features
    are dilated (expanded) into null regions. Dilation is performed on each
    band separately.

    Parameters
    ----------
    raster : Raster or path str
        The raster to dilate
    size : int or 2-tuple of ints
        The shape of the window to use when dilating. A Single int is
        interpreted as the size of a square window. A tuple of 2 ints is used
        as the dimensions of rectangular window. At least one value must be
        greater than 1. Values cannot be less than 1.

    Returns
    -------
    Raster
        The resulting raster with eroded features. This raster will have the
        same shape and meta data as the original

    See also
    --------
    erode, raster_tools.focal.focal

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Dilation_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    """
    raster = get_raster(raster)
    footprint = _get_footprint(size)

    return _erosion_or_dilation_filter(raster, footprint, "dilation")


def erode(raster, size):
    """Perform erosion on a raster

    Erosion increases the thickness of raster features. Features with higher
    values will cover features with lower values. At each cell, the miximum
    value within a window, defined by `size`, is stored in the output location.
    This is very similar to the max focal filter except that raster features
    are eroded (contracted) into null regions. Erosion is performed on each
    band separately.

    Parameters
    ----------
    raster : Raster or path str
        The raster to erode
    size : int or 2-tuple of ints
        The shape of the window to use when eroding. A Single int is
        interpreted as the size of a square window. A tuple of 2 ints is used
        as the dimensions of rectangular window. At least one value must be
        greater than 1. Values cannot be less than 1.

    Returns
    -------
    Raster
        The resulting raster with eroded features. This raster will have the
        same shape and meta data as the original

    See also
    --------
    dilate, raster_tools.focal.focal

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Erosion_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    """
    raster = get_raster(raster)
    footprint = _get_footprint(size)

    return _erosion_or_dilation_filter(raster, footprint, "erosion")


def band_concat(rasters):
    """Join a sequence of rasters along the band dimension.

    Parameters
    ----------
    rasters : sequence of Rasters and/or paths
        The rasters to concatenate. These can be a mix of Rasters and paths.
        All rasters must have the same shape in the x and y dimensions.

    Returns
    -------
    Raster
        The resulting concatenated Raster.

    """
    rasters = [get_raster(raster) for raster in rasters]
    if not rasters:
        raise ValueError("No rasters provided")
    if len(rasters) == 1:
        return rasters[0]

    # TODO: make join a user option or set join="exact"?
    ds = xr.concat([r.to_dataset() for r in rasters], dim="band", join="inner")
    ds["band"] = np.arange(np.sum([r.nbands for r in rasters])) + 1
    if any(r._masked for r in rasters):
        nv = get_default_null_value(ds.raster.dtype)
        ds["raster"] = xr_where_with_meta(ds.mask, nv, ds.raster, nv=nv)
    crs = ([None] + [r.crs for r in rasters if r.crs is not None]).pop()
    if ds.rio.crs is None and crs is not None:
        ds = ds.rio.write_crs(crs)

    return Raster(ds, _fast_path=True)


@nb.jit(nopython=True, nogil=True)
def _remap_values(x, mask, null, start_end_values, new_values, inclusivity):
    outx = np.full_like(x, null)
    bands, rows, columns = x.shape
    rngs = start_end_values.shape[0]
    for bnd in range(bands):
        for rw in range(rows):
            for cl in range(columns):
                if mask[bnd, rw, cl]:
                    continue
                vl = x[bnd, rw, cl]
                remap = False
                for imap in range(rngs):
                    left, right = start_end_values[imap]
                    new = new_values[imap]
                    if inclusivity == 0:
                        remap = left <= vl < right
                    elif inclusivity == 1:
                        remap = left < vl <= right
                    elif inclusivity == 2:
                        remap = left <= vl <= right
                    elif inclusivity == 3:
                        remap = left < vl < right
                    if remap:
                        outx[bnd, rw, cl] = new
                        break
                if not remap:
                    outx[bnd, rw, cl] = x[bnd, rw, cl]
    return outx


def _normalize_mappings(mappings):
    if not isinstance(mappings, (list, tuple)):
        raise TypeError(
            "Mappings must be either single 3-tuple or list of 3-tuples of "
            "scalars"
        )
    if not len(mappings):
        raise ValueError("No mappings provided")
    if is_scalar(mappings[0]):
        mappings = [mappings]
    try:
        mappings = [list(m) for m in mappings]
    except TypeError:
        raise TypeError(
            "Mappings must be either single 3-tuple or list of 3-tuples of "
            "scalars"
        ) from None
    starts = []
    ends = []
    outs = []
    for m in mappings:
        if len(m) != 3:
            raise ValueError(
                "Mappings must be either single 3-tuple or list of 3-tuples of"
                " scalars"
            )
        if not all(is_scalar(mi) for mi in m[:2]):
            raise TypeError("Mapping min and max values must be scalars")
        if any(np.isnan(mi) for mi in m[:2]):
            raise ValueError("Mapping min and max values cannot be NaN")
        if m[0] >= m[1]:
            raise ValueError(
                "Mapping min value must be strictly less than max value:"
                f" {m[0]}, {m[1]}"
            )
        if not is_scalar(m[2]) and m[2] is not None:
            raise ValueError(
                "The new value in a mapping must be a scalar or None. "
                f"Got {m[2]!r}."
            )
        starts.append(m[0])
        ends.append(m[1])
        outs.append(m[2])
    return starts, ends, outs


def remap_range(raster, mapping, inclusivity="left"):
    """Remaps values based on a mapping or list of mappings.

    Mappings are applied all at once with earlier mappings taking
    precedence.

    Parameters
    ----------
    raster : Raster or str
        Path string or Raster to perform remap on.
    mapping : 3-tuple of scalars or list of 3-tuples of scalars
        A tuple or list of tuples containing ``(min, max, new_value)``
        scalars. The mappiing(s) map values between the min and max to the
        ``new_value``. If ``new_value`` is ``None``, the matching pixels will
        be marked as null. If `mapping` is a list and there are mappings that
        conflict or overlap, earlier mappings take precedence. `inclusivity`
        determines which sides of the range are inclusive and exclusive.
    inclusivity : str, optional
        Determines whether to be inclusive or exclusive on either end of the
        range. Default is `'left'`.

        'left' [min, max)
            Left (min) side is inclusive and right (max) side is exclusive.
        'right' (min, max]
            Left (min) side is exclusive and right (max) side is inclusive.
        'both' [min, max]
            Both sides are inclusive.
        'none' (min, max)
            Both sides are exclusive.

    Returns
    -------
    Raster
        The resulting Raster.

    """
    raster = get_raster(raster)
    map_starts, map_ends, map_outs = _normalize_mappings(mapping)
    if not is_str(inclusivity):
        raise TypeError(
            f"inclusivity must be a str. Got type: {type(inclusivity)}"
        )
    inc_map = dict(zip(("left", "right", "both", "none"), range(4)))
    if inclusivity not in inc_map:
        raise ValueError(f"Invalid inclusivity value. Got: {inclusivity!r}")
    if not all(m is None for m in map_outs):
        mappings_common_dtype = get_common_dtype(
            [m for m in map_outs if m is not None]
        )
    else:
        mappings_common_dtype = raster.dtype
    out_dtype = np.promote_types(raster.dtype, mappings_common_dtype)
    # numba doesn't understand f16 so use f32 and then downcast
    f16_workaround = out_dtype == F16
    if raster._masked:
        nv = raster.null_value
    elif None in map_outs:
        nv = get_default_null_value(out_dtype)
    else:
        nv = None
    start_end_values = np.array([[s, e] for s, e in zip(map_starts, map_ends)])
    new_values = np.array([v if v is not None else nv for v in map_outs])

    data = raster.data
    mask = raster.mask
    if out_dtype != data.dtype:
        if not f16_workaround:
            data = data.astype(out_dtype)
        else:
            data = data.astype(F32)
    elif f16_workaround:
        data = data.astype(F32)
    data = da.map_blocks(
        _remap_values,
        data,
        mask,
        # Can't pass None object to numba code
        null=0 if nv is None else nv,
        start_end_values=start_end_values,
        new_values=new_values,
        inclusivity=inc_map[inclusivity],
        dtype=data.dtype,
        meta=np.array((), dtype=data.dtype),
    )
    if f16_workaround:
        data = data.astype(F16)
    return data_to_raster_like(data, raster, nv=nv)


def _get_where_nv(x, y):
    assert not ((x is None) and (y is None))
    if (x is None or is_scalar(x)) and (y is None or is_scalar(y)):
        # Both are either None or a scalar so get default
        return get_default_null_value(
            get_common_dtype([a for a in (x, y) if a is not None])
        )
    if isinstance(x, Raster) and isinstance(y, Raster):
        # No easy way to pick so just get a default
        return get_default_null_value(np.promote_types(x.dtype, y.dtype))
    if x is None or is_scalar(x):
        # y cannot be None or a scalar here. y must be a raster
        return y.null_value if y._masked else get_default_null_value(y.dtype)
    else:
        # x must be a raster but not y
        return x.null_value if x._masked else get_default_null_value(x.dtype)


def where(condition, true_rast, false_rast):
    """
    Return elements chosen from `true_rast` or `false_rast` depending on
    `condition`.

    The mask for the result is a combination of all three raster inputs. Any
    cells that are masked in the condition raster will be masked in the output.
    The rest of the cells are masked based on which raster they were taken
    from, as determined by the condition raster, and if the cell in that raster
    was null. Effectively, the resulting mask is determined as follows:
    `where(condition.mask, True, where(condition, true_rast.mask,
    false_rast.mask)`.

    `true_rast` and `false_rast` can both be ``None`` to indicate a null value
    but they cannot both be ``None`` at the same time. An error is raised in
    that case.

    Parameters
    ----------
    condition : str or Raster
        A boolean or int raster that indicates where elements in the result
        should be selected from. If the condition is an int raster, it is
        coerced to bool using `condition > 0`.  ``True`` cells pull values from
        `true_rast` and ``False`` cells pull from `y`. *str* is treated as a
        path to a raster.
    true_rast : scalar, Raster, str, None
        Raster or scalar to pull from if the corresponding location in
        `condition` is ``True``. If None, this is treated as a null value.
    false_rast : scalar, Raster, str, None
        Raster or scalar to pull from if the corresponding location in
        `condition` is ``False``. If None, this is treated as a null value.

    Returns
    -------
    Raster
        The resulting Raster.

    """
    condition = get_raster(condition)
    if not is_bool(condition.dtype) and not is_int(condition.dtype):
        raise TypeError(
            "Condition argument must be a boolean or integer raster"
        )
    args = []
    for r, name in [(true_rast, "true_rast"), (false_rast, "false_rast")]:
        if not is_scalar(r) and r is not None:
            try:
                r = get_raster(r)
            except TypeError as err:
                raise TypeError(
                    f"Could not understand {name} argument. Got: {r!r}"
                ) from err
        args.append(r)
    if all(a is None for a in args):
        raise ValueError("'true_rast' and 'false_rast' cannot both be None")
    out_crs = None
    for r in [condition, true_rast, false_rast]:
        crs = getattr(r, "crs", None)
        if crs is not None:
            out_crs = crs
            break
    masked = condition._masked
    masked |= any(r._masked if isinstance(r, Raster) else False for r in args)
    masked |= any(a is None for a in args)

    x, y = args
    cd = condition.xdata
    if is_int(condition.dtype):
        # if condition.dtype is not bool then must be an int raster so
        # assume that condition is raster of 0 and 1 values.
        # condition > 0 will grab all 1/True values.
        cd = cd > 0
    if masked:
        nv = _get_where_nv(x, y)
        if x is None:
            x = nv
        if y is None:
            y = nv
        if all(is_scalar(a) for a in (x, y)):
            dtype = get_common_dtype((x, y))
            x = dtype.type(x)
            y = dtype.type(y)
        cd = xr.where(condition.xmask, False, cd)
        xd = getattr(x, "xdata", x)
        yd = getattr(y, "xdata", y)
        cm = condition.xmask
        xm = getattr(x, "xmask", False)
        ym = getattr(y, "xmask", False)
        data = xr.where(cd, xd, yd)
        mask = xr.where(cd, xm, ym)
        mask = xr.where(cm, True, mask)
        data = xr.where(mask, nv, data)
        # Pick up any new null values caused by x/y being None or equal to nv
        mask = data == nv if not np.isnan(nv) else np.isnan(data)
        data = data.rio.write_nodata(nv)
    else:
        xd = getattr(x, "xdata", x)
        yd = getattr(y, "xdata", y)
        data = xr.where(cd, xd, yd)
        mask = xr.zeros_like(data, dtype=bool)
    ds = dataarray_to_xr_raster_ds(data, mask)
    if out_crs is not None:
        ds = ds.rio.write_crs(out_crs)
    return Raster(ds, _fast_path=True)


@nb.jit(nopython=True, nogil=True)
def _reclassify_chunk(
    x, mask, mapping_from, mapping_to, unmapped_to_null, null, out_dtype
):
    mapping = dict(zip(mapping_from, mapping_to))
    out = np.empty_like(x, dtype=out_dtype)
    nb, ny, nx = x.shape
    for b in range(nb):
        for i in range(ny):
            for j in range(nx):
                if mask[b, i, j]:
                    out[b, i, j] = null
                    continue
                v = x[b, i, j]
                reclass = v in mapping
                if reclass:
                    out[b, i, j] = mapping[v]
                elif unmapped_to_null:
                    out[b, i, j] = null
                else:
                    out[b, i, j] = v
    return out


# Matches: 0, 0.0, .0, 12, 1.2, .2, 1.2e34, 1.2e-34, .2e+34, 0e+12, 1.2E23
_INT_OR_FLOAT_RE = (
    r"(?:[-+]?(?:(?:0|[1-9]\d*)(?:\.\d*)?|\.\d+)?)(?:[eE][-+]?\d+)?"
)
_REMAPPING_LINE_PATTERN = re.compile(
    rf"^\s*(?P<from>{_INT_OR_FLOAT_RE})\s*:"
    rf"\s*(?P<to>{_INT_OR_FLOAT_RE}|NoData)\s*$"
)


def _str_to_float_or_int(string):
    # String matched regex for a number value. Check if float specific
    # characters are in the string.
    if any(c in string for c in ".eE"):
        return float(string)
    return int(string)


def _parse_ascii_remap_file(fd):
    mapping = {}
    for line_no, line in enumerate(fd):
        line = line.strip()
        if not line:
            continue
        m = _REMAPPING_LINE_PATTERN.match(line)
        if m is None or not m.group("from") or not m.group("to"):
            raise RemapFileParseError(
                f"Invalid remapping on line {line_no + 1}: {line!r}"
            )
        from_str = m.group("from")
        to_str = m.group("to")
        k = _str_to_float_or_int(from_str)
        v = _str_to_float_or_int(to_str) if to_str != "NoData" else None
        if k in mapping:
            raise ValueError(f"Found duplicate mapping: '{k}:{v}'.")
        mapping[k] = v
    return mapping


def _get_remapping(mapping):
    if is_str(mapping):
        if os.path.exists(mapping):
            with open(mapping) as fd:
                mapping = _parse_ascii_remap_file(fd)
            if not mapping:
                raise ValueError("No mapping found in the provided file")
        else:
            raise IOError(f"No such file: {mapping!r}")
    elif not isinstance(mapping, dict):
        raise TypeError(
            f"Remapping must be a str or dict. Got: {type(mapping)!r}"
        )
    return mapping


def reclassify(raster, remapping, unmapped_to_null=False):
    """Reclassify the input raster values based on a mapping.

    Parameters
    ----------
    raster : str, Raster
        The input raster to reclassify. Can be a path string or Raster object.
    remapping : str, dict
        Can be either a ``dict`` or a path string. If a ``dict`` is provided,
        the keys will be reclassified to the corresponding values. It is
        possible to map values to the null value by providing ``None`` in the
        mapping. If a path string, it is treated as an ASCII remap file where
        each line looks like ``a:b`` and ``a`` and ``b`` are scalars. ``b`` can
        also be "NoData". This indicates that ``a`` will be mapped to the null
        value. The output values of the mapping can cause type promotion. If
        the input raster has integer data and one of the outputs in the mapping
        is a float, the result will be a float raster.
    unmapped_to_null : bool, optional
        If ``True``, values not included in the mapping are instead mapped to
        the null value. Default is ``False``.

    Returns
    -------
    Raster
        The remapped raster.

    """
    raster = get_raster(raster)
    remapping = _get_remapping(remapping)

    out_dtype = raster.dtype
    mapping_out_values = [v for v in remapping.values() if v is not None]
    if len(mapping_out_values):
        out_dtype = np.promote_types(
            get_common_dtype(mapping_out_values), out_dtype
        )
    if raster._masked:
        nv = raster.null_value
    else:
        nv = get_default_null_value(out_dtype)
    masked = raster._masked or unmapped_to_null
    for k in remapping:
        if remapping[k] is None:
            remapping[k] = nv
            masked = True

    mapping_from = np.array(list(remapping))
    mapping_to = np.array(list(remapping.values())).astype(out_dtype)
    data = da.map_blocks(
        _reclassify_chunk,
        raster.data,
        raster.mask,
        mapping_from=mapping_from,
        mapping_to=mapping_to,
        unmapped_to_null=unmapped_to_null,
        null=nv,
        out_dtype=out_dtype,
        dtype=out_dtype,
        meta=np.array((), dtype=out_dtype),
    )
    if not masked:
        nv = None
    return data_to_raster_like(data, raster, nv=nv)
