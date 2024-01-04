import numpy as np
import pytest
import xarray as xr

from raster_tools.io import DimensionsError, open_dataset
from raster_tools.masking import get_default_null_value
from tests.utils import assert_valid_raster


@pytest.mark.parametrize(
    "path,crs",
    [
        ("tests/data/raster/grib_example_1.grib", None),
        ("tests/data/raster/grib_example_2.grib2", None),
        ("tests/data/raster/nc_example_1.nc", None),
        ("tests/data/raster/nc_example_2.nc", None),
        ("tests/data/raster/nc_example_3.nc", "EPSG:4326"),
    ],
)
def test_open_dataset(path, crs):
    # mask_and_scale is on by default for netcdf files now. This replaces all
    # _FillValue occurances with nan and drops the _FIllValue attribute. Set it
    # to false so that we can make sure the fill value is being detected.
    meta = xr.open_dataset(path, decode_coords="all", mask_and_scale=False)
    vars_ = list(meta.data_vars)
    vars_ = [v for v in vars_ if meta[v].squeeze().ndim <= 3]

    result = open_dataset(
        path,
        ignore_extra_dim_errors=True,
        # See comment above about turning off mask_and_scale
        xarray_kwargs={"mask_and_scale": False},
    )

    assert isinstance(result, dict)
    assert len(result) == len(vars_)
    assert sorted(vars_) == sorted(result)
    for k, v in result.items():
        assert_valid_raster(v)
        assert v.crs == crs
        assert v.affine is not None
        fill = meta[k].rio.nodata
        if fill is None:
            fill = np.nan
        if not np.isnan(fill):
            assert v.null_value == fill
        else:
            assert v.null_value == get_default_null_value(v.dtype)
        assert v.xdata.dims == ("band", "y", "x")
        truth_data = meta[k].squeeze().compute().to_numpy()
        ycoord = meta[meta.rio.y_dim].to_numpy()
        if ycoord[0] < ycoord[-1]:
            truth_data = truth_data[..., ::-1, :]
        data = v.data.compute().squeeze()
        assert np.allclose(truth_data, data)


def test_open_dataset_extra_dims_error():
    with pytest.raises(DimensionsError):
        open_dataset("tests/data/raster/nc_example_2.nc")
    assert (
        open_dataset(
            "tests/data/raster/nc_example_2.nc", ignore_extra_dim_errors=True
        )
        is not None
    )


def test_open_dataset_crs():
    assert xr.open_dataset("tests/data/raster/nc_example_1.nc").rio.crs is None
    dataset = open_dataset(
        "tests/data/raster/nc_example_1.nc", crs="EPSG:4326"
    )
    assert dataset["t2m"].crs == "EPSG:4326"
