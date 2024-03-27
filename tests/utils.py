import dask
import numpy as np
import rasterio as rio
import xarray as xr

from raster_tools.dtypes import is_bool, is_scalar
from raster_tools.raster import Raster
from raster_tools.utils import is_strictly_decreasing, is_strictly_increasing


def assert_coords_equal(ac, bc):
    assert isinstance(
        ac,
        (
            xr.core.coordinates.DataArrayCoordinates,
            xr.core.coordinates.DatasetCoordinates,
        ),
    ) and isinstance(
        bc,
        (
            xr.core.coordinates.DataArrayCoordinates,
            xr.core.coordinates.DatasetCoordinates,
        ),
    )
    assert list(ac.keys()) == list(bc.keys())
    for k in ac:
        assert np.allclose(ac[k], bc[k])


def assert_valid_raster(raster):
    assert isinstance(raster, Raster)
    assert len(raster.shape) == 3
    assert "raster" in raster._ds
    assert "mask" in raster._ds
    assert raster._ds.raster.shape == raster._ds.mask.shape
    # Coords and dims
    # xr.Dataset seems to sort dims alphabetically (sometimes)
    assert set(raster._ds.dims) == {"band", "y", "x"}
    assert raster._ds.raster.dims == ("band", "y", "x")
    assert raster._ds.mask.dims == ("band", "y", "x")
    # Make sure that all dims have coords
    for d in raster._ds.raster.dims:
        assert d in raster._ds.coords
        assert d in raster._ds.raster.coords
        assert d in raster._ds.mask.coords
    # Make sure that coords are consistent
    assert_coords_equal(raster._ds.coords, raster._ds.raster.coords)
    assert_coords_equal(raster._ds.coords, raster._ds.mask.coords)
    if raster.shape[2] > 1:
        assert is_strictly_increasing(raster._ds.x)
    if raster.shape[1] > 1:
        assert is_strictly_decreasing(raster._ds.y)
    assert np.allclose(
        raster._ds.band.to_numpy(), np.arange(raster.shape[0]) + 1
    )
    assert raster._ds.band.to_numpy()[0] == 1
    # is lazy
    assert dask.is_dask_collection(raster._ds)
    assert dask.is_dask_collection(raster._ds.raster)
    assert dask.is_dask_collection(raster._ds.raster.data)
    assert dask.is_dask_collection(raster._ds.mask)
    assert dask.is_dask_collection(raster._ds.mask.data)
    assert raster._ds.raster.chunks == raster._ds.mask.chunks
    # size 1 chunks along band dim
    assert raster._ds.raster.data.chunksize[0] == 1
    assert raster._ds.mask.data.chunksize[0] == 1
    # Null value stuff
    assert is_bool(raster._ds.mask.dtype)
    nv = raster.null_value
    assert nv is None or is_scalar(nv) or is_bool(nv)
    if raster.null_value is not None:
        assert "_FillValue" in raster._ds.raster.attrs
        assert np.allclose(
            raster.null_value, raster._ds.raster.rio.nodata, equal_nan=True
        )
        assert np.allclose(
            raster._ds.raster.attrs["_FillValue"],
            raster._ds.raster.rio.nodata,
            equal_nan=True,
        )
        mask = raster._ds.mask.data.compute()
        data = raster._ds.raster.data.compute()
        data_copy = data.copy()
        data_copy[mask] = raster.null_value
        assert np.allclose(data, data_copy, equal_nan=True)
    # Check for expected chunksize mismatch with actual size
    for expected_chunksize, actual_chunksize in zip(
        [c.shape for c in raster.data.blocks.ravel()],
        [c.compute().shape for c in raster.data.blocks.ravel()],
    ):
        assert expected_chunksize == actual_chunksize
    assert raster.shape == raster.data.compute().shape
    assert raster.shape == raster.mask.compute().shape
    # CRS
    assert isinstance(raster.crs, rio.crs.CRS) or raster.crs is None
    assert raster.crs == raster._ds.rio.crs
    assert raster.crs == raster._ds.raster.rio.crs
    assert raster.crs == raster._ds.mask.rio.crs


def assert_rasters_similar(left, right, check_nbands=True, check_chunks=True):
    assert isinstance(left, Raster) and isinstance(right, Raster)
    if left is right:
        return

    assert set(left._ds.dims) == set(right._ds.dims)
    assert np.allclose(left.x, right.x)
    assert np.allclose(left.y, right.y)
    assert left.crs == right.crs
    assert left.affine == right.affine
    if check_nbands:
        assert left.nbands == right.nbands
    if check_chunks:
        assert left.data.chunks[1:] == right.data.chunks[1:]


def arange_nd(shape, dtype=None, mod=np):
    return mod.arange(np.prod(shape), dtype=dtype).reshape(shape)


def arange_raster(shape, dtype=None):
    assert len(shape) in (2, 3)
    return Raster(arange_nd(shape, dtype=dtype))
