import os
import re

from raster_tools.dtypes import is_int, is_scalar
from raster_tools.general import band_concat
from raster_tools.masking import get_default_null_value
from raster_tools.raster import Raster
from raster_tools.utils import validate_file


class BatchScriptParseError(BaseException):
    pass


def _split_strip(s, delimeter):
    return [si.strip() for si in s.split(delimeter)]


def _batch_error(msg, line_no):
    raise BatchScriptParseError(f"Script Line {line_no}: {msg}")


def _parse_user_number(str_val):
    from ast import literal_eval

    # This may raise a ValueError if the string is not a valid literal
    val = literal_eval(str_val)
    if not is_scalar(val):
        raise TypeError("Must be a scalar")
    return val


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
    "esriRasterMode": "%",
    "%": "%",
    "esriRasterPower": "**",
    "**": "**",
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
    left_arg, right_arg, op = _split_strip(args_str, ";")
    op = _ESRI_OP_TO_OP[op]
    if op not in _ESRI_OP_TO_OP:
        _batch_error(f"Unknown arithmetic operation {repr(op)}", line_no)
    op = _ESRI_OP_TO_OP[op]
    try:
        left = float(left_arg)
    except ValueError:
        left = state.get_raster(left_arg)
    try:
        right = float(right_arg)
    except ValueError:
        right = state.get_raster(right_arg)
    return left._binary_arithmetic(right, op)


def _batch_parse_extract_band(state, args_str, line_no):
    rs = state.get_raster(args_str.pop(0))
    bands = []
    for sb in args_str:
        try:
            b = _parse_user_number(sb)
            if not is_int(b):
                raise ValueError()
        except ValueError:
            _batch_error("Error parsing band value", line_no)
        except TypeError:
            _batch_error("Band values must be integers", line_no)
        bands.append(b)
    return rs.get_bands(bands)


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
    args = []
    for group in remaps:
        args.extend(group)
    return state.get_raster(raster).remap_range(*args)


def _batch_parse_composite(state, args_str, line_no):
    on_line = f" on line {line_no}"
    rasters = [state.get_raster(path) for path in _split_strip(args_str, ";")]
    if len(rasters) < 2:
        _batch_error(
            "COMPOSITE Error: at least 2 rasters are required", on_line
        )
    return band_concat(rasters)


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
    return raster.save(out_name, nodata, bwidth, bheight)


def _batch_parse_set_null(state, args_str, line_no):
    rs = state.get_raster(args_str.pop(0))
    if not rs._masked:
        rs.set_null_value(get_default_null_value(rs.dtype))
    sranges = [_split_strip(r, "-") for r in args_str]
    ranges = []
    for sr in sranges:
        try:
            lh = _parse_user_number(sr[0])
            rh = _parse_user_number(sr[1])
            ranges.extend((lh, rh, rs.null_value))
        except ValueError:
            _batch_error("Error parsing range value", line_no)
        except TypeError:
            _batch_error("Range bounds must be numbers", line_no)
    return rs.remap_range(*ranges).set_null_value(rs.null_value)


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
