import operator

import numpy as np
import pytest

from raster_tools.batch import parse_batch_script
from raster_tools.exceptions import BatchScriptParseError
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
