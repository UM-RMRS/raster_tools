import operator
import unittest

import affine
import dask
import numpy as np
import pytest
import rasterio as rio
import xarray as xr

import raster_tools.focal as focal
from raster_tools import Raster, band_concat
from raster_tools.dtypes import (
    DTYPE_INPUT_TO_DTYPE,
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
    is_int,
    is_scalar,
)

TEST_ARRAY = np.array(
    [
        [1, 3, 4, 4, 3, 2],
        [7, 3, 2, 6, 4, 6],
        [5, 8, 7, 5, 6, 6],
        [1, 4, 5, -1, 5, 1],
        [4, 7, 5, -1, 2, 6],
        [1, 2, 2, 1, 3, 4],
    ]
)


def rs_eq_array(rs, ar):
    return (rs.xrs.values == ar).all()


class TestRasterCreation(unittest.TestCase):
    def test_ctor_errors(self):
        with self.assertRaises(ValueError):
            Raster(np.ones(4))
        with self.assertRaises(ValueError):
            Raster(np.ones((1, 3, 4, 4)))

    def test_increasing_coords(self):
        # This raster has an inverted y axis
        rs = Raster("tests/data/elevation_small.tif")
        x, y = rs.xrs.x.values, rs.xrs.y.values
        self.assertTrue((np.diff(x) > 0).all())
        self.assertTrue((np.diff(y) > 0).all())

        rs = Raster(TEST_ARRAY)
        x, y = rs.xrs.x.values, rs.xrs.y.values
        self.assertTrue((np.diff(x) > 0).all())
        self.assertTrue((np.diff(y) > 0).all())

    def test_creation_from_numpy(self):
        for nprs in [np.ones((6, 6)), np.ones((1, 6, 6)), np.ones((4, 5, 5))]:
            rs = Raster(nprs)
            shape = nprs.shape if len(nprs.shape) == 3 else (1, *nprs.shape)
            self.assertEqual(rs.shape, shape)
            self.assertTrue(rs_eq_array(rs, nprs))

        rs = Raster(TEST_ARRAY)
        # Band dim has been added
        self.assertTrue(rs.shape == (1, 6, 6))
        # Band dim starts at 1
        self.assertTrue((rs.xrs.band == [1]).all())
        # x/y dims start at 0 and increase
        self.assertTrue((rs.xrs.x == np.arange(0, 6)).all())
        self.assertTrue((rs.xrs.y == np.arange(0, 6)).all())
        # No null value determined for int type
        self.assertIsNone(rs.null_value)

        rs = Raster(TEST_ARRAY.astype(float))
        self.assertTrue(np.isnan(rs.null_value))


class TestProperties(unittest.TestCase):
    def test__attrs(self):
        rs = Raster("tests/data/elevation_small.tif")
        self.assertDictEqual(rs._attrs, rs.xrs.attrs)
        rs._attrs = {}
        self.assertDictEqual(rs._attrs, {})

    def test__masked(self):
        rs = Raster("tests/data/elevation_small.tif")
        self.assertTrue(rs._masked)
        rs = Raster("tests/data/null_values.tiff")
        self.assertTrue(rs._masked)
        x = np.ones((1, 3, 3))
        rs = Raster(x)
        self.assertTrue(rs._masked)
        rs = Raster(x.astype(int))
        self.assertFalse(rs._masked)

    def test__values(self):
        rs = Raster("tests/data/elevation_small.tif")
        self.assertTrue((rs._values == rs.xrs.values).all())

    def test__null_value(self):
        rs = Raster("tests/data/elevation_small.tif")
        self.assertEqual(rs._null_value, rs.xrs.attrs["_FillValue"])
        rs._null_value = 1
        self.assertEqual(rs._null_value, 1)
        self.assertEqual(rs.xrs.attrs["_FillValue"], 1)

    def test_null_value(self):
        rs = Raster("tests/data/elevation_small.tif")
        self.assertEqual(rs.null_value, rs.xrs.attrs["_FillValue"])

    def test_dtype(self):
        rs = Raster("tests/data/elevation_small.tif")
        self.assertTrue(rs.dtype == rs.xrs.dtype)

    def test_shape(self):
        rs = Raster("tests/data/elevation_small.tif")
        self.assertTrue(rs.shape == rs.xrs.shape)
        self.assertIsInstance(rs.shape, tuple)

    def test_crs(self):
        rs = Raster("tests/data/elevation_small.tif")
        self.assertIsInstance(rs.crs, rio.crs.CRS)
        self.assertTrue(rs.crs == rs.xrs.rio.crs)

        x = np.arange(25).reshape((5, 5))
        rs = Raster(x)
        self.assertIsNone(rs.crs)

    def test_affine(self):
        rs = Raster("tests/data/elevation_small.tif")
        self.assertIsInstance(rs.affine, affine.Affine)
        self.assertTrue(rs.affine == rs.xrs.rio.transform())

    def test_resolution(self):
        rs = Raster("tests/data/elevation_small.tif")
        self.assertTupleEqual(rs.resolution, rs.xrs.rio.resolution(True))

        r = np.arange(25).reshape((5, 5))
        rs = Raster(r)
        self.assertTupleEqual(rs.resolution, rs.xrs.rio.resolution(True))


def test_property_xrs():
    rs = Raster("tests/data/elevation_small.tif")
    assert hasattr(rs, "xrs")
    assert isinstance(rs.xrs, xr.DataArray)
    assert rs.xrs is rs._rs


def test_property_pxrs():
    data = np.arange(25).reshape((1, 5, 5))
    rs = Raster(data).remap_range((0, 10, -999)).set_null_value(-999)
    assert not np.isnan(rs).any().compute()
    assert hasattr(rs, "pxrs")
    assert isinstance(rs.pxrs, xr.DataArray)
    assert np.isnan(rs.pxrs).sum().compute() == 10
    assert rs.pxrs is not rs._rs
    tdata = np.where(data < 10, np.nan, data)
    assert np.allclose(rs.pxrs, tdata, equal_nan=True)

    rs = Raster(data)
    assert not np.isnan(rs).any().compute()
    assert rs.pxrs is rs._rs
    assert not np.isnan(rs.pxrs).any().compute()


def test_property__data():
    rs = Raster("tests/data/elevation_small.tif")
    assert hasattr(rs, "_data")
    assert isinstance(rs._data, dask.array.Array)
    assert rs._data is rs._rs.data


_BINARY_ARITHMETIC_OPS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.pow,
    operator.truediv,
    operator.floordiv,
    operator.mod,
]
_BINARY_COMPARISON_OPS = [
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
    operator.eq,
    operator.ne,
]


@pytest.mark.parametrize("op", _BINARY_ARITHMETIC_OPS + _BINARY_COMPARISON_OPS)
@pytest.mark.parametrize("operand", [-2.0, -1, 0, 2, 3.0, True])
@pytest.mark.parametrize(
    "rs_type", [F16, F32, F64, I16, I32, I64, I8, U16, U32, U64, U8]
)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:overflow")
def test_binary_ops_arithmetic_against_scalar(op, operand, rs_type):
    x = np.arange(4 * 5 * 5).reshape((4, 5, 5)).astype(rs_type)
    rs = Raster(x).remap_range((0, 10, 0)).set_null_value(0)
    x = rs._values

    result = op(operand, rs)
    truth = op(operand, x)
    assert np.allclose(result, truth, equal_nan=True)
    assert result.dtype == truth.dtype
    assert result._attrs == rs._attrs
    assert np.allclose(result._mask, rs._mask)
    assert np.all(result.xrs.spatial_ref == rs.xrs.spatial_ref).values
    # Make sure raising ints to negative ints raises TypeError
    if (
        op == operator.pow
        and is_int(rs.dtype)
        and is_int(operand)
        and operand < 0
        # Numpy promotes to float64 in this case
        and rs.dtype != U64
    ):
        with pytest.raises(TypeError):
            op(rs, operand)
    else:
        truth = op(x, operand)
        result = op(rs, operand)
        assert np.allclose(result, truth, equal_nan=True)
        assert result.dtype == truth.dtype
        assert result._attrs == rs._attrs
        assert np.allclose(result._mask, rs._mask)
        assert np.all(result.xrs.spatial_ref == rs.xrs.spatial_ref).values


unknown_chunk_array = dask.array.ones((5, 5))
unknown_chunk_array = unknown_chunk_array[unknown_chunk_array > 0]


@pytest.mark.parametrize("op", _BINARY_ARITHMETIC_OPS + _BINARY_COMPARISON_OPS)
@pytest.mark.parametrize(
    "operand,error",
    [
        ([1], False),
        (np.array([1]), False),
        (np.array([[1]]), False),
        (np.ones((5, 5)), False),
        (np.ones((1, 5, 5)), False),
        (np.ones((4, 5, 5)), False),
        ([1, 2, 3, 4], False),
        (range(1, 5), False),
        (np.array([1, 2, 3, 4]), False),
        (dask.array.ones((5, 5)), False),
        (np.array([1, 1]), True),
        (unknown_chunk_array, True),
    ],
)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:overflow")
def test_binary_ops_arithmetic_against_array(op, operand, error):
    data = np.arange(4 * 5 * 5).reshape((4, 5, 5)).astype(float)
    rs = Raster(data).remap_range((0, 10, 0)).set_null_value(0)
    data = rs._values

    if error:
        with pytest.raises(ValueError):
            op(rs, operand)
        with pytest.raises(ValueError):
            op(operand, rs)
    else:
        if len(operand) == len(data) and (
            isinstance(operand, (list, range)) or operand.size == 4
        ):
            truth1 = op(data, np.array(operand).reshape((-1, 1, 1)))
            truth2 = op(np.array(operand).reshape((-1, 1, 1)), data)
        else:
            truth1 = op(data, operand)
            truth2 = op(operand, data)
        result = op(rs, operand)
        (truth1,) = dask.compute(truth1)
        assert np.allclose(result, truth1, equal_nan=True)
        assert result._attrs == rs._attrs
        assert np.allclose(result._mask, rs._mask)
        assert np.all(result.xrs.spatial_ref == rs.xrs.spatial_ref).values
        result = op(operand, rs)
        (truth2,) = dask.compute(truth2)
        assert np.allclose(result, truth2, equal_nan=True)
        assert result._attrs == rs._attrs
        assert np.allclose(result._mask, rs._mask)
        assert np.all(result.xrs.spatial_ref == rs.xrs.spatial_ref).values


@pytest.mark.parametrize("op", _BINARY_ARITHMETIC_OPS + _BINARY_COMPARISON_OPS)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:overflow")
def test_binary_ops_arithmetic_against_raster(op):
    data = np.arange(4 * 5 * 5).reshape((4, 5, 5)).astype(float)
    rs = Raster(data).remap_range((0, 10, 0)).set_null_value(0)
    data = rs._values
    data2 = np.ones_like(data, dtype=int) * 2
    rs2 = Raster(data2)

    truth1 = op(data, data2)
    truth2 = op(data2, data)
    result = op(rs, rs2)
    assert np.allclose(result, truth1, equal_nan=True)
    assert result._attrs == rs._attrs
    assert np.allclose(result._mask, rs._mask)
    assert np.all(result.xrs.spatial_ref == rs.xrs.spatial_ref).values
    result = op(rs2, rs)
    assert np.allclose(result, truth2, equal_nan=True)
    assert result._attrs == rs._attrs
    assert np.allclose(result._mask, rs._mask)
    assert np.all(result.xrs.spatial_ref == rs.xrs.spatial_ref).values


def test_binary_ops_arithmetic_inplace():
    data = np.arange(4 * 5 * 5).reshape((4, 5, 5))
    rs = Raster(data)

    r = rs.copy()
    rr = r
    r += 3
    t = data + 3
    assert np.allclose(r, t)
    assert rr is r

    r = rs.copy()
    rr = r
    r -= 3
    t = data - 3
    assert np.allclose(r, t)
    assert rr is r

    r = rs.copy()
    rr = r
    r *= 3
    t = data * 3
    assert np.allclose(r, t)
    assert rr is r

    r = rs.copy()
    rr = r
    r **= 3
    t = data**3
    assert np.allclose(r, t)
    assert rr is r

    r = rs.copy()
    rr = r
    r /= 3
    t = data / 3
    assert np.allclose(r, t)
    assert rr is r

    r = rs.copy()
    rr = r
    r //= 3
    t = data // 3
    assert np.allclose(r, t)
    assert rr is r

    r = rs.copy()
    rr = r
    r %= 3
    t = data % 3
    assert np.allclose(r, t)
    assert rr is r


_NP_UFUNCS = list(
    filter(
        lambda x: isinstance(x, np.ufunc)
        # not valid for rasters
        and x not in (np.isnat, np.matmul, np.binary_repr),
        map(lambda x: getattr(np, x), dir(np)),
    )
)
_NP_UFUNCS_NIN_SINGLE = list(filter(lambda x: x.nin == 1, _NP_UFUNCS))
_NP_UFUNCS_NIN_MULT = list(filter(lambda x: x.nin > 1, _NP_UFUNCS))


@pytest.mark.parametrize("ufunc", _NP_UFUNCS_NIN_SINGLE)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_ufuncs_single_input(ufunc):
    data = np.arange(4 * 5 * 5).reshape((4, 5, 5))
    rs = Raster(data).remap_range((0, 10, 0)).set_null_value(0)
    data = rs._values

    if ufunc.__name__.startswith("logical"):
        truth = ufunc(data > 0)
    else:
        truth = ufunc(data)
    result = ufunc(rs)
    if ufunc.nout == 1:
        assert np.allclose(result, truth, equal_nan=True)
        assert np.allclose(result._mask, rs._mask, equal_nan=True)
        assert result._attrs == rs._attrs
    else:
        for (r, t) in zip(result, truth):
            assert np.allclose(r, t, equal_nan=True)
            assert np.allclose(r._mask, rs._mask, equal_nan=True)
            assert r._attrs == rs._attrs


@pytest.mark.parametrize("ufunc", _NP_UFUNCS_NIN_MULT)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:overflow")
def test_ufuncs_multiple_input_against_scalar(ufunc):
    data = np.arange(4 * 5 * 5).reshape((4, 5, 5))
    rs = Raster(data).remap_range((0, 10, 0)).set_null_value(0)
    data = rs._values
    extra = ufunc.nin - 1

    args = [rs] + [2 for i in range(extra)]
    args_np = [getattr(a, "_values", a) for a in args]
    if ufunc.__name__.startswith("bitwise") or ufunc.__name__.startswith(
        "logical"
    ):
        truth = ufunc(*[a > 0 for a in args_np])
    else:
        truth = ufunc(*args_np)
    result = ufunc(*args)
    if ufunc.nout == 1:
        assert np.allclose(result, truth, equal_nan=True)
        assert np.allclose(result._mask, rs._mask, equal_nan=True)
        assert result._attrs == rs._attrs
    else:
        for (r, t) in zip(result, truth):
            assert np.allclose(r, t, equal_nan=True)
            assert np.allclose(r._mask, rs._mask, equal_nan=True)
            assert r._attrs == rs._attrs
    # Reflected
    args = args[::-1]
    args_np = args_np[::-1]
    if ufunc.__name__.startswith("bitwise") or ufunc.__name__.startswith(
        "logical"
    ):
        truth = ufunc(*[a > 0 for a in args_np])
    else:
        truth = ufunc(*args_np)
    result = ufunc(*args)
    if ufunc.nout == 1:
        assert np.allclose(result, truth, equal_nan=True)
        assert np.allclose(result._mask, rs._mask, equal_nan=True)
        assert result._attrs == rs._attrs
    else:
        for (r, t) in zip(result, truth):
            assert np.allclose(r, t, equal_nan=True)
            assert np.allclose(r._mask, rs._mask, equal_nan=True)
            assert r._attrs == rs._attrs


@pytest.mark.parametrize("ufunc", _NP_UFUNCS_NIN_MULT)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:overflow")
def test_ufuncs_multiple_input_against_raster(ufunc):
    data = np.arange(4 * 5 * 5).reshape((4, 5, 5))
    rs = Raster(data).remap_range((0, 10, 0)).set_null_value(0)
    data = rs._values
    extra = ufunc.nin - 1
    other = Raster(np.ones_like(data) * 2)
    other.xrs.data[0, 0, 0] = 25
    other = other.set_null_value(25)
    assert other._mask.compute().sum() == 1
    mask = rs._mask | other._mask

    args = [rs] + [other.copy() for i in range(extra)]
    args_np = [a._values for a in args]
    if ufunc.__name__.startswith("bitwise") or ufunc.__name__.startswith(
        "logical"
    ):
        truth = ufunc(*[a > 0 for a in args_np])
    else:
        truth = ufunc(*args_np)
    result = ufunc(*args)
    if ufunc.nout == 1:
        assert np.allclose(result, truth, equal_nan=True)
        assert np.allclose(result._mask, mask, equal_nan=True)
        assert result._attrs == rs._attrs
    else:
        for (r, t) in zip(result, truth):
            assert np.allclose(r, t, equal_nan=True)
            assert np.allclose(r._mask, mask, equal_nan=True)
            assert r._attrs == rs._attrs
    # Test reflected
    args = args[::-1]
    args_np = args_np[::-1]
    if ufunc.__name__.startswith("bitwise") or ufunc.__name__.startswith(
        "logical"
    ):
        truth = ufunc(*[a > 0 for a in args_np])
    else:
        truth = ufunc(*args_np)
    result = ufunc(*args)
    if ufunc.nout == 1:
        assert np.allclose(result, truth, equal_nan=True)
        assert np.allclose(result._mask, mask, equal_nan=True)
        assert result._attrs == other._attrs
    else:
        for (r, t) in zip(result, truth):
            assert np.allclose(r, t, equal_nan=True)
            assert np.allclose(r._mask, mask, equal_nan=True)
            assert r._attrs == other._attrs


def test_invert():
    data = np.arange(4 * 5 * 5).reshape((4, 5, 5))
    rs = Raster(data).remap_range((0, 10, 0)).set_null_value(0)
    data = rs._values

    assert np.allclose(np.invert(rs), np.invert(data))
    assert np.allclose(~rs, np.invert(data))
    assert np.allclose(
        np.invert(rs.astype(bool)), np.invert(data.astype(bool))
    )
    with pytest.raises(TypeError):
        ~rs.astype(float)


@pytest.mark.parametrize(
    "func",
    [
        np.all,
        np.any,
        np.max,
        np.mean,
        np.min,
        np.prod,
        np.std,
        np.sum,
        np.var,
    ],
)
def test_reductions(func):
    data = np.arange(3 * 5 * 5).reshape((3, 5, 5)) - 25
    rs = Raster(data).set_null_value(4)
    valid = ~rs._mask.compute()

    fname = func.__name__
    if fname in ("amin", "amax"):
        # Drop leading 'a'
        fname = fname[1:]
    assert hasattr(rs, fname)
    truth = func(data[valid])
    for result in (func(rs), getattr(rs, fname)()):
        assert isinstance(result, dask.array.Array)
        assert result.size == 1
        assert result.shape == ()
        if fname in ("all", "any"):
            assert isinstance(result.compute(), (bool, np.bool_))
        else:
            assert is_scalar(result.compute())
        assert np.allclose(result, truth, equal_nan=True)
    if fname in ("all", "any"):
        rs = rs > 0
        data = data > 0
        truth = func(data[valid])
        assert np.allclose(func(rs), truth, equal_nan=True)
        assert np.allclose(getattr(rs, fname)(), truth, equal_nan=True)


class TestAstype(unittest.TestCase):
    def test_astype(self):
        rs = Raster("tests/data/elevation_small.tif")
        for type_code, dtype in DTYPE_INPUT_TO_DTYPE.items():
            self.assertEqual(rs.astype(type_code).dtype, dtype)
            self.assertEqual(rs.astype(type_code).eval().dtype, dtype)
            self.assertEqual(rs.astype(dtype).dtype, dtype)
            self.assertEqual(rs.astype(dtype).eval().dtype, dtype)

    def test_wrong_type_codes(self):
        rs = Raster("tests/data/elevation_small.tif")
        with self.assertRaises(ValueError):
            rs.astype("not float32")
        with self.assertRaises(ValueError):
            rs.astype("other")

    def test_dtype_property(self):
        rs = Raster("tests/data/elevation_small.tif")
        self.assertEqual(rs.dtype, rs.xrs.dtype)

    def test_astype_str_uppercase(self):
        rs = Raster("tests/data/elevation_small.tif")
        for type_code, dtype in DTYPE_INPUT_TO_DTYPE.items():
            if isinstance(type_code, str):
                type_code = type_code.upper()
                self.assertEqual(rs.astype(type_code).eval().dtype, dtype)


class TestRasterAttrsPropagation(unittest.TestCase):
    def test_ctor_attrs(self):
        r1 = Raster("tests/data/elevation_small.tif")
        true_attrs = r1._attrs.copy()
        r2 = Raster(Raster("tests/data/elevation_small.tif"))
        test_attrs = {"test": 0}
        r3 = Raster("tests/data/elevation_small.tif")
        r3._attrs = test_attrs
        self.assertEqual(r2._attrs, true_attrs)
        self.assertEqual(r3._attrs, test_attrs)

    def test_astype_attrs(self):
        rs = Raster("tests/data/elevation_small.tif")
        attrs = rs._attrs
        self.assertEqual(rs.astype(int)._attrs, attrs)

    def test_convolve_attrs(self):
        rs = Raster("tests/data/elevation_small.tif")
        attrs = rs._attrs
        self.assertEqual(focal.convolve(rs, np.ones((3, 3)))._attrs, attrs)

    def test_focal_attrs(self):
        rs = Raster("tests/data/elevation_small.tif")
        attrs = rs._attrs
        self.assertEqual(focal.focal(rs, "max", 3)._attrs, attrs)

    def test_band_concat_attrs(self):
        rs = Raster("tests/data/elevation_small.tif")
        attrs = rs._attrs
        rs2 = Raster("tests/data/elevation2_small.tif")
        self.assertEqual(band_concat([rs, rs2])._attrs, attrs)


class TestCopy(unittest.TestCase):
    def test_copy(self):
        rs = Raster("tests/data/elevation_small.tif")
        copy = rs.copy()
        self.assertIsNot(rs, copy)
        self.assertIsNot(rs.xrs, copy.xrs)
        self.assertIsNot(rs._attrs, copy._attrs)
        self.assertTrue((rs.xrs == copy.xrs).all())
        self.assertEqual(rs._attrs, copy._attrs)


class TestSetCrs(unittest.TestCase):
    def test_set_crs(self):
        rs = Raster("tests/data/elevation_small.tif")
        self.assertTrue(rs.crs != 4326)

        rs4326 = rs.set_crs(4326)
        self.assertTrue(rs4326.crs != rs.crs)
        self.assertTrue(rs4326.crs == 4326)
        self.assertTrue(np.allclose(rs._values, rs4326._values))


class TestSetNullValue(unittest.TestCase):
    def test_set_null_value(self):
        rs = Raster("tests/data/null_values.tiff")
        ndv = rs.null_value
        rs2 = rs.set_null_value(0)
        self.assertEqual(rs.null_value, ndv)
        self.assertEqual(rs._attrs["_FillValue"], ndv)
        self.assertEqual(rs2._attrs["_FillValue"], 0)

        rs = Raster("tests/data/elevation_small.tif")
        nv = rs.null_value
        rs2 = rs.set_null_value(None)
        self.assertEqual(rs.null_value, nv)
        self.assertEqual(rs._attrs["_FillValue"], nv)
        self.assertIsNone(rs2.null_value)
        self.assertIsNone(rs2._attrs["_FillValue"])


class TestReplaceNull(unittest.TestCase):
    def test_replace_null(self):
        fill_value = 0
        rs = Raster("tests/data/null_values.tiff")
        nv = rs.null_value
        rsnp = rs._values
        rsnp_replaced = rsnp.copy()
        rsnp_replaced[rsnp == rs.null_value] = fill_value
        rs = rs.replace_null(fill_value)
        self.assertTrue(np.allclose(rs._values, rsnp_replaced, equal_nan=True))
        self.assertEqual(rs.null_value, nv)
        self.assertTrue(rs._mask.sum().compute() == 0)


class TestWhere(unittest.TestCase):
    def test_where(self):
        rs = Raster("tests/data/elevation_small.tif")
        c = rs > 1100

        r = rs.where(c, 0)
        rsnp = np.asarray(rs)
        truth = np.where(rsnp > 1100, rsnp, 0)
        self.assertTrue(np.allclose(r, truth, equal_nan=True))
        self.assertTrue(
            np.allclose(
                rs.where(c, "tests/data/elevation_small.tif"),
                rs,
                equal_nan=True,
            )
        )

        c = c.astype(int)
        r = rs.where(c, 0)
        self.assertTrue(np.allclose(r, truth, equal_nan=True))

        self.assertTrue(rs._masked)
        self.assertTrue(r._masked)
        self.assertTrue(rs.crs is not None)
        self.assertTrue(r.crs == rs.crs)
        self.assertDictEqual(r._attrs, rs._attrs)

        with self.assertRaises(TypeError):
            cf = c.astype(float)
            rs.where(cf, 0)
        with self.assertRaises(TypeError):
            rs.where(c, None)


class TestToNullMask(unittest.TestCase):
    def test_to_null_mask(self):
        rs = Raster("tests/data/null_values.tiff")
        nv = rs.null_value
        rsnp = rs._values
        truth = rsnp == nv
        self.assertTrue(rs_eq_array(rs.to_null_mask(), truth))
        # Test case where no null values
        rs = Raster("tests/data/elevation_small.tif")
        truth = np.full(rs.shape, False, dtype=bool)
        self.assertTrue(rs_eq_array(rs.to_null_mask(), truth))


class TestEval(unittest.TestCase):
    def test_eval(self):
        rs = Raster("tests/data/elevation_small.tif")
        rsnp = rs.xrs.values
        rs += 2
        rsnp += 2
        rs -= rs
        rsnp -= rsnp
        rs *= -1
        rsnp *= -1
        result = rs.eval()
        # Make sure new raster returned
        self.assertIsNot(rs, result)
        self.assertIsNot(rs.xrs, result.xrs)
        # Make sure that original raster is still lazy
        self.assertTrue(dask.is_dask_collection(rs.xrs))
        self.assertTrue(rs_eq_array(result, rsnp))
        self.assertTrue(dask.is_dask_collection(result.xrs))
        # 2 operations: 1 copy and 1 chunk operation
        self.assertEqual(len(result._data.dask), 2)
        self.assertTrue(dask.is_dask_collection(result._mask))
        # 1 operation: dask.array.from_array()
        self.assertTrue(len(result._mask.dask), 1)


class TestToDask(unittest.TestCase):
    def test_to_dask(self):
        rs = Raster("tests/data/elevation2_small.tif")
        self.assertTrue(isinstance(rs.to_dask(), dask.array.Array))
        self.assertIs(rs.to_dask(), rs._data)
        self.assertTrue(isinstance(rs.eval().to_dask(), dask.array.Array))


class TestGetBands(unittest.TestCase):
    def test_get_bands(self):
        rs = Raster("tests/data/multiband_small.tif")
        rsnp = rs.xrs.values
        self.assertTrue(rs_eq_array(rs.get_bands(1), rsnp[:1]))
        self.assertTrue(rs_eq_array(rs.get_bands(2), rsnp[1:2]))
        self.assertTrue(rs_eq_array(rs.get_bands(3), rsnp[2:3]))
        self.assertTrue(rs_eq_array(rs.get_bands(4), rsnp[3:4]))
        for bands in [[1], [1, 2], [1, 1], [3, 1, 2], [4, 3, 2, 1]]:
            np_bands = [i - 1 for i in bands]
            result = rs.get_bands(bands)
            self.assertTrue(np.allclose(result, rsnp[np_bands]))
            bnd_dim = list(range(1, len(bands) + 1))
            self.assertTrue(np.allclose(result.xrs.band, bnd_dim))

        self.assertTrue(len(rs.get_bands(1).shape) == 3)

        for bands in [0, 5, [1, 5], [0]]:
            with self.assertRaises(IndexError):
                rs.get_bands(bands)
        with self.assertRaises(ValueError):
            rs.get_bands([])


def test_burn_mask():
    data = np.arange(25).reshape((1, 5, 5))
    rs = Raster(data).remap_range((0, 10, -999)).set_null_value(-999)
    true_mask = data < 10
    true_state = data.copy()
    true_state[true_mask] = -999
    assert np.allclose(true_mask, rs._mask)
    assert np.allclose(true_state, rs._values)

    rs.xrs.data = data
    assert np.allclose(rs, data)
    assert np.allclose(rs.burn_mask(), true_state)

    data = np.arange(25).reshape((1, 5, 5))
    rs = Raster(data).remap_range((20, 26, 999)).set_null_value(999)
    rs = rs > 15
    assert rs.dtype == np.dtype(bool)
    assert rs.null_value == 999
    true_state = data > 15
    true_state = np.where(data >= 20, False, true_state)
    print(rs.burn_mask()._values)
    print(true_state)
    assert np.allclose(rs.burn_mask(), true_state)


if __name__ == "__main__":
    unittest.main()
