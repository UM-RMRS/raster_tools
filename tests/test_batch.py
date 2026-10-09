import operator

import numpy as np
import pytest

from raster_tools.batch import parse_batch_script
from raster_tools.exceptions import BatchScriptParseError
from raster_tools.masking import get_default_null_value
from raster_tools.raster import Raster

OPS = [
    ("+", "esriRasterPlus", operator.add),
    ("-", "esriRasterMinus", operator.sub),
    ("*", "esriRasterMultiply", operator.mul),
    ("/", "esriRasterDivide", operator.truediv),
    ("%", "esriRasterMode", operator.mod),
    ("**", "esriRasterPower", operator.pow),
]
SYMBOL_OPS = [(sym, func) for sym, _, func in OPS]
ALL_OP_NAMES = [(sym, func) for sym, _, func in OPS] + [
    (esri, func) for _, esri, func in OPS
]


@pytest.fixture
def batch_dir(tmp_path):
    left = np.arange(1, 17, dtype="float64").reshape(1, 4, 4) / 4
    right = np.arange(16, 0, -1, dtype="float64").reshape(1, 4, 4) / 8
    Raster(left).save(str(tmp_path / "left.tif"))
    Raster(right).save(str(tmp_path / "right.tif"))
    return tmp_path, left, right


def _run_script(directory, text):
    path = directory / "script.bch"
    path.write_text(text)
    return parse_batch_script(str(path)).final_raster


@pytest.mark.parametrize("op,func", ALL_OP_NAMES)
def test_arithmetic_raster_raster(batch_dir, op, func):
    directory, left, right = batch_dir
    result = _run_script(
        directory, f"out = ARITHMETIC(left.tif;right.tif;{op})\n"
    )
    assert isinstance(result, Raster)
    np.testing.assert_allclose(result.to_numpy(), func(left, right))


@pytest.mark.parametrize("op,func", SYMBOL_OPS)
def test_arithmetic_raster_scalar(batch_dir, op, func):
    directory, left, _ = batch_dir
    result = _run_script(directory, f"out = ARITHMETIC(left.tif;2.5;{op})\n")
    assert isinstance(result, Raster)
    np.testing.assert_allclose(result.to_numpy(), func(left, 2.5))


@pytest.mark.parametrize("op,func", SYMBOL_OPS)
def test_arithmetic_scalar_raster(batch_dir, op, func):
    directory, left, _ = batch_dir
    result = _run_script(directory, f"out = ARITHMETIC(2.5;left.tif;{op})\n")
    assert isinstance(result, Raster)
    np.testing.assert_allclose(result.to_numpy(), func(2.5, left))


def test_arithmetic_uses_earlier_script_results(batch_dir):
    directory, left, right = batch_dir
    script = (
        "# Comments and blank lines are skipped\n"
        "\n"
        "sum = ARITHMETIC(left.tif;right.tif;+)\n"
        "out = ARITHMETIC(sum;3;*)\n"
    )
    result = _run_script(directory, script)
    np.testing.assert_allclose(result.to_numpy(), (left + right) * 3)


def test_arithmetic_through_raster_constructor(batch_dir):
    directory, left, right = batch_dir
    path = directory / "script.bch"
    path.write_text("out = ARITHMETIC(left.tif;right.tif;-)\n")
    result = Raster(str(path))
    np.testing.assert_allclose(result.to_numpy(), left - right)


def test_arithmetic_propagates_nulls(tmp_path):
    data = np.arange(1, 17, dtype="float64").reshape(1, 4, 4)
    data[0, 0, 0] = -1
    Raster(data).set_null_value(-1).save(str(tmp_path / "nulls.tif"))
    result = _run_script(tmp_path, "out = ARITHMETIC(nulls.tif;2;*)\n")
    mask = result.mask.compute()
    assert mask[0, 0, 0]
    assert mask.sum() == 1
    np.testing.assert_allclose(result.to_numpy()[~mask], (data * 2)[~mask])


@pytest.mark.parametrize(
    "line",
    [
        "out = ARITHMETIC(left.tif;right.tif;^)\n",
        "out = ARITHMETIC(left.tif;right.tif)\n",
        "out = ARITHMETIC(1;2;+)\n",
    ],
)
def test_arithmetic_invalid_lines(batch_dir, line):
    directory, _, _ = batch_dir
    with pytest.raises(BatchScriptParseError, match="Script Line 1"):
        _run_script(directory, line)


def test_remap(batch_dir):
    directory, left, _ = batch_dir
    result = _run_script(directory, "out = REMAP(left.tif;0:2:10,2:5:20)\n")
    expected = np.where(left < 2, 10, 20)
    np.testing.assert_allclose(result.to_numpy(), expected)


@pytest.fixture
def multiband_dir(tmp_path):
    data = np.arange(48, dtype="float64").reshape(3, 4, 4)
    Raster(data).save(str(tmp_path / "multi.tif"))
    return tmp_path, data


def test_extract_band_single(multiband_dir):
    directory, data = multiband_dir
    result = _run_script(directory, "out = EXTRACTBAND(multi.tif;2)\n")
    assert result.nbands == 1
    np.testing.assert_allclose(result.to_numpy(), data[1:2])


def test_extract_band_multiple_in_given_order(multiband_dir):
    directory, data = multiband_dir
    result = _run_script(directory, "out = EXTRACTBAND(multi.tif;3;1)\n")
    assert result.nbands == 2
    np.testing.assert_allclose(result.to_numpy(), data[[2, 0]])


@pytest.mark.parametrize(
    "line",
    [
        "out = EXTRACTBAND(multi.tif;4)\n",
        "out = EXTRACTBAND(multi.tif;0)\n",
        "out = EXTRACTBAND(multi.tif;1.5)\n",
        "out = EXTRACTBAND(multi.tif;one)\n",
        "out = EXTRACTBAND(multi.tif)\n",
    ],
)
def test_extract_band_invalid_lines(multiband_dir, line):
    directory, _ = multiband_dir
    with pytest.raises(
        BatchScriptParseError, match="Script Line 1: EXTRACTBAND Error"
    ):
        _run_script(directory, line)


@pytest.fixture
def signed_dir(tmp_path):
    # Saved without a null value so SETNULL must pick a default one
    data = np.arange(-8, 8, dtype="int16").reshape(1, 4, 4)
    Raster(data).save(str(tmp_path / "signed.tif"))
    return tmp_path, data


def test_set_null_negative_and_multiple_ranges(signed_dir):
    directory, data = signed_dir
    result = _run_script(directory, "out = SETNULL(signed.tif;-6:-3;2.5:4)\n")
    expected_mask = ((data >= -6) & (data < -3)) | ((data >= 2.5) & (data < 4))
    assert result.dtype == np.dtype("int16")
    assert result.null_value == get_default_null_value(np.dtype("int16"))
    mask = result.mask.compute()
    np.testing.assert_array_equal(mask, expected_mask)
    values = result.to_numpy()
    np.testing.assert_array_equal(values[~mask], data[~mask])
    assert np.all(values[mask] == result.null_value)


def test_set_null_keeps_existing_null_value(tmp_path):
    data = np.arange(1, 17, dtype="float64").reshape(1, 4, 4)
    data[0, 0, 0] = -1
    Raster(data).set_null_value(-1).save(str(tmp_path / "nulls.tif"))
    result = _run_script(tmp_path, "out = SETNULL(nulls.tif;10:12)\n")
    assert result.null_value == -1
    mask = result.mask.compute()
    expected_mask = (data == -1) | ((data >= 10) & (data < 12))
    np.testing.assert_array_equal(mask, expected_mask)


@pytest.mark.parametrize(
    "line",
    [
        "out = SETNULL(signed.tif)\n",
        "out = SETNULL(signed.tif;-3)\n",
        "out = SETNULL(signed.tif;-3:-1:0)\n",
        "out = SETNULL(signed.tif;a:b)\n",
        "out = SETNULL(signed.tif;-1:-3)\n",
        "out = SETNULL(signed.tif;nan:1)\n",
    ],
)
def test_set_null_invalid_lines(signed_dir, line):
    directory, _ = signed_dir
    with pytest.raises(
        BatchScriptParseError, match="Script Line 1: SETNULL Error"
    ):
        _run_script(directory, line)


def test_composite(batch_dir):
    directory, left, right = batch_dir
    result = _run_script(
        directory, "out = COMPOSITE(left.tif;right.tif;left.tif)\n"
    )
    assert result.nbands == 3
    np.testing.assert_allclose(
        result.to_numpy(), np.concatenate([left, right, left])
    )


def test_composite_requires_two_rasters(batch_dir):
    directory, _, _ = batch_dir
    script = "# one raster is not enough\nout = COMPOSITE(left.tif)\n"
    with pytest.raises(BatchScriptParseError) as excinfo:
        _run_script(directory, script)
    assert str(excinfo.value) == (
        "Script Line 2: COMPOSITE Error: at least 2 rasters are required"
    )
