"""Parser for raster batch scripts (``.bch`` files).

A batch script holds one assignment per line::

    name = FUNCTION(arg;arg;...)

Text after ``#`` is a comment and blank lines are skipped. Function names
are case-insensitive. A raster argument is either a name assigned on an
earlier line or a path to a raster file; relative paths are resolved
against the directory holding the script. The script's result is the
raster assigned on its last line.

Functions:

``OPENRASTER(path)``
    Open the raster at `path`.
``ARITHMETIC(left;right;op)``
    Apply a binary operation. `left` and `right` are rasters or numbers,
    and at least one must be a raster. `op` is one of ``+``, ``-``,
    ``*``, ``/``, ``%`` (modulo), ``**`` or the equivalent ArcObjects name
    ``esriRasterPlus``, ``esriRasterMinus``, ``esriRasterMultiply``,
    ``esriRasterDivide``, ``esriRasterMode`` or ``esriRasterPower``.
``EXTRACTBAND(raster;band[;band...])``
    Extract one or more bands, in the order given. Bands are integers
    starting at 1.
``NULLTOVALUE(raster;value)``
    Replace null cells with `value`.
``REMAP(raster;min:max:new[,min:max:new...])``
    Set cells in ``[min, max)`` to `new`. Earlier groups take precedence
    where ranges overlap.
``SETNULL(raster;min:max[;min:max...])``
    Mark cells in ``[min, max)`` as null. The raster's null value is kept
    if it has one; otherwise the default null value for its data type is
    used. Bounds may be negative, e.g. ``SETNULL(dem;-9999:-1)``.
``COMPOSITE(raster;raster[;raster...])``
    Stack the bands of two or more rasters into one raster.
``SAVEFUNCTIONRASTER(raster;name;directory;type[;nodata[;bw[;bh]]])``
    Save `raster` to ``directory/name`` with the extension for `type`
    (only ``TIFF`` is supported). `nodata` defaults to 0. `bw` and `bh`
    set the tile width and height; `bw` alone gives square tiles.
"""

import operator
import os
import re

from raster_tools._stack import stack_bands
from raster_tools.exceptions import BatchScriptParseError
from raster_tools.raster import Raster
from raster_tools.utils import validate_file


def _split_strip(s, delimeter):
    return [si.strip() for si in s.split(delimeter)]


def _batch_error(msg, line_no):
    raise BatchScriptParseError(f"Script Line {line_no}: {msg}")


FTYPE_TO_EXT = {
    "TIFF": "tif",
}


_ESRI_OP_TO_OP = {
    "esriRasterPlus": "+",
    "+": "+",
    "esriRasterMinus": "-",
    "-": "-",
    "esriRasterMultiply": "*",
    "*": "*",
    "esriRasterDivide": "/",
    "/": "/",
    # ArcObjects documents esriRasterMode as "The mode operation (%)", so
    # it is treated as the modulo operator.
    "esriRasterMode": "%",
    "%": "%",
    "esriRasterPower": "**",
    "**": "**",
}
_OP_TO_FUNC = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "%": operator.mod,
    "**": operator.pow,
}
_FUNC_PATTERN = re.compile(r"^(?P<func>[A-Za-z]+)\((?P<args>[^\(\)]+)\)$")


class _BatchScripParserState:
    def __init__(self, path):
        validate_file(path)
        self.path = os.path.abspath(path)
        self.location = os.path.dirname(self.path)
        self.rasters = {}
        self.final_raster = None

    def get_raster(self, name_or_path):
        if name_or_path in self.rasters:
            return self.rasters[name_or_path]
        else:
            # Handle relative paths. Assume they are relative to the batch file
            if not os.path.isabs(name_or_path):
                name_or_path = os.path.join(self.location, name_or_path)
            validate_file(name_or_path)
            return Raster(name_or_path)


def _batch_parse_arithmetic(state, args_str, line_no):
    try:
        left_arg, right_arg, op = _split_strip(args_str, ";")
    except ValueError:
        _batch_error(
            "ARITHMETIC Error: requires 3 arguments (left;right;operation)",
            line_no,
        )
    if op not in _ESRI_OP_TO_OP:
        _batch_error(f"Unknown arithmetic operation {repr(op)}", line_no)
    func = _OP_TO_FUNC[_ESRI_OP_TO_OP[op]]
    try:
        left = float(left_arg)
    except ValueError:
        left = state.get_raster(left_arg)
    try:
        right = float(right_arg)
    except ValueError:
        right = state.get_raster(right_arg)
    if not (isinstance(left, Raster) or isinstance(right, Raster)):
        _batch_error(
            "ARITHMETIC Error: at least one argument must be a raster",
            line_no,
        )
    return func(left, right)


def _batch_parse_extract_band(state, args_str, line_no):
    raster, *band_args = _split_strip(args_str, ";")
    if not band_args:
        _batch_error(
            "EXTRACTBAND Error: requires a raster and at least one band",
            line_no,
        )
    try:
        bands = [int(sb) for sb in band_args]
    except ValueError:
        _batch_error(
            "EXTRACTBAND Error: band values must be integers", line_no
        )
    rs = state.get_raster(raster)
    try:
        return rs.get_bands(bands)
    except IndexError as e:
        _batch_error(f"EXTRACTBAND Error: {e}", line_no)


def _batch_parse_null_to_value(state, args_str, line_no):
    left, *right = _split_strip(args_str, ";")
    if len(right) > 1:
        _batch_error("NULLTOVALUE Error: Too many arguments", line_no)
    value = float(right[0])
    return state.get_raster(left).replace_null(value)


def _batch_parse_remap(state, args_str, line_no):
    raster, *args = _split_strip(args_str, ";")
    if len(args) > 1:
        _batch_error("REMAP Error: Too many argument dividers", line_no)
    args = args[0]
    remaps = []
    for group in _split_strip(args, ","):
        try:
            values = [float(v) for v in _split_strip(group, ":")]
        except ValueError:
            _batch_error("REMAP Error: values must be numbers", line_no)
        if len(values) != 3:
            _batch_error(
                "REMAP Error: requires 3 values separated by ':'", line_no
            )
        left, right, new = values
        if right <= left:
            _batch_error(
                "REMAP Error: the min value must be less than the max value",
                line_no,
            )
        remaps.append((left, right, new))
    if len(remaps) == 0:
        _batch_error("REMAP Error: No remap values found", line_no)
    return state.get_raster(raster).remap_range(remaps)


def _batch_parse_composite(state, args_str, line_no):
    rasters = [state.get_raster(path) for path in _split_strip(args_str, ";")]
    if len(rasters) < 2:
        _batch_error(
            "COMPOSITE Error: at least 2 rasters are required", line_no
        )
    return stack_bands(rasters)


def _batch_parse_open(state, args_str, line_no):
    try:
        return state.get_raster(args_str)
    except Exception as e:
        _batch_error(f"Error while opening raster: {repr(e)}", line_no)


def _batch_parse_save(state, args_str, line_no):
    # From c# files:
    #  (inRaster;outName;outWorkspace;rasterType;nodata;blockwidth;blockheight)
    # nodata;blockwidth;blockheight are optional
    try:
        in_rs, out_name, out_dir, type_, *extra = _split_strip(args_str, ";")
    except ValueError:
        _batch_error(
            "SAVEFUNCTIONRASTER Error: Incorrect number of arguments", line_no
        )
    n = len(extra)
    bwidth = None
    bheight = None
    nodata = 0
    if n >= 1:
        nodata = float(extra[0])
    if n >= 2:
        bwidth = int(extra[1])
    if n == 3:
        bheight = int(extra[2])
    if n > 3:
        _batch_error("SAVEFUNCTIONRASTER Error: Too many arguments", line_no)
    if type_ not in FTYPE_TO_EXT:
        _batch_error("SAVEFUNCTIONRASTER Error: Unknown file type", line_no)
    raster = state.get_raster(in_rs)
    out_name = os.path.join(out_dir, out_name)
    ext = FTYPE_TO_EXT[type_]
    out_name += f".{ext}"
    blocksize = None
    if bwidth is not None and bheight is not None:
        blocksize = (bheight, bwidth)
    elif bwidth is not None:
        blocksize = bwidth
    return raster.save(out_name, null_value=nodata, blocksize=blocksize)


def _batch_parse_set_null(state, args_str, line_no):
    raster, *range_args = _split_strip(args_str, ";")
    if not range_args:
        _batch_error(
            "SETNULL Error: requires a raster and at least one min:max range",
            line_no,
        )
    ranges = []
    for sr in range_args:
        bounds = _split_strip(sr, ":")
        if len(bounds) != 2:
            _batch_error(
                f"SETNULL Error: ranges must be min:max pairs: {sr!r}",
                line_no,
            )
        try:
            left, right = (float(b) for b in bounds)
        except ValueError:
            _batch_error(
                f"SETNULL Error: range bounds must be numbers: {sr!r}",
                line_no,
            )
        if not left < right:
            _batch_error(
                "SETNULL Error: the min value must be less than the max value",
                line_no,
            )
        ranges.append((left, right, None))
    # A None replacement value marks the matching cells as null
    return state.get_raster(raster).remap_range(ranges)


_FUNC_TO_PARSER = {
    "ARITHMETIC": _batch_parse_arithmetic,
    "COMPOSITE": _batch_parse_composite,
    "EXTRACTBAND": _batch_parse_extract_band,
    "NULLTOVALUE": _batch_parse_null_to_value,
    "OPENRASTER": _batch_parse_open,
    "REMAP": _batch_parse_remap,
    "SAVEFUNCTIONRASTER": _batch_parse_save,
    "SETNULL": _batch_parse_set_null,
}


def parse_batch_script(path):
    state = _BatchScripParserState(path)
    with open(state.path) as fd:
        lines = fd.readlines()
    last_raster = None
    for i, line in enumerate(lines):
        # Ignore comments
        line, *_ = _split_strip(line, "#")
        if not line:
            continue
        lh, rh = _split_strip(line, "=")
        state.rasters[lh] = _parse_raster(state, lh, rh, i + 1)
        last_raster = lh
    state.final_raster = state.rasters[last_raster]
    return state


def _parse_raster(state, dst, expr, line_no):
    mat = _FUNC_PATTERN.match(expr)
    if mat is None:
        _batch_error("Could not parse function on line", line_no)
    func = mat["func"].upper()
    args = mat["args"]
    if func not in _FUNC_TO_PARSER:
        _batch_error(f"Unknown function {repr(func)}", line_no)
    return _FUNC_TO_PARSER[func](state, args, line_no)
