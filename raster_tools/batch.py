import os
import re

from .io import FTYPE_TO_EXT
from .raster import Raster, _BINARY_ARITHMETIC_OPS
from ._utils import validate_file


class BatchScriptParseError(BaseException):
    pass


def _split_strip(s, delimeter):
    return [si.strip() for si in s.split(delimeter)]


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
_ARITHMETIC_OPS_MAP = {}
_FUNC_PATTERN = re.compile(r"^(?P<func>[A-Za-z]+)\((?P<args>[^\(\)]+)\)$")


class BatchScript:
    def __init__(self, path):
        validate_file(path)
        self.path = path
        self.rasters = {}
        self.final_raster = None

    def parse(self):
        with open(self.path) as fd:
            lines = fd.readlines()
        last_raster = None
        for i, line in enumerate(lines):
            # Ignore comments
            line, *_ = _split_strip(line, "#")
            if not line:
                continue
            lh, rh = _split_strip(line, "=")
            self.rasters[lh] = self._parse_raster(lh, rh, i + 1)
            last_raster = lh
        self.final_raster = self.rasters[last_raster]
        return self

    def _parse_raster(self, dst, expr, line_no):
        mat = _FUNC_PATTERN.match(expr)
        if mat is None:
            raise BatchScriptParseError(
                f"Could not parse function on line {line_no}"
            )
        func = mat["func"].upper()
        args = mat["args"]
        raster = None
        if func == "ARITHMETIC":
            raster = self._arithmetic_args_to_raster(args, line_no)
        elif func == "NULLTOVALUE":
            return self._null_to_value_args_to_raster(args, line_no)
        elif func == "REMAP":
            raster = self._remap_args_to_raster(args, line_no)
        elif func == "COMPOSITE":
            raise NotImplementedError()
        elif func == "OPENRASTER":
            raster = Raster(args)
        elif func == "SAVEFUNCTIONRASTER":
            raster = self._save_args_to_raster(args, line_no)
        else:
            raise BatchScriptParseError(
                f"Unknown function on line {line_no}: '{func}'"
            )
        return raster

    def _arithmetic_args_to_raster(self, args_str, line_no):
        left_raster, right_raster, op = _split_strip(args_str, ";")
        op = _ESRI_OP_TO_OP[op]
        if op not in _BINARY_ARITHMETIC_OPS:
            raise BatchScriptParseError(
                f"Uknown arithmetic operation on line {line_no}: '{op}'"
            )
        else:
            op = _BINARY_ARITHMETIC_OPS[op]
        left = self._get_raster(left_raster)
        right = self._get_raster(right_raster)
        return left._binary_arithmetic(right, op)

    def _null_to_value_args_to_raster(self, args_str, line_no):
        on_line = f" on line {line_no}"
        left, *right = _split_strip(args_str, ";")
        if len(right) > 1:
            raise BatchScriptParseError(
                "NULLTOVALUE Error: Too many arguments" + on_line
            )
        value = float(right[0])
        return self._get_raster(left).replace_null(value)

    def _remap_args_to_raster(self, args_str, line_no):
        on_line = f" on line {line_no}"
        raster, *args = _split_strip(args_str, ";")
        if len(args) > 1:
            raise BatchScriptParseError(
                "REMAP Error: Too many arguments" + on_line
            )
        args = args[0]
        try:
            values = [float(v) for v in _split_strip(args, ":")]
        except ValueError:
            raise BatchScriptParseError(
                "REMAP Error: values must be numbers" + on_line
            )
        if len(values) != 3:
            raise BatchScriptParseError(
                "REMAP Error: requires 3 values separated by ':'" + on_line
            )
        left, right, new = values
        if right <= left:
            raise BatchScriptParseError(
                "REMAP Error: the min value must be less than the max value"
                + on_line
            )
        return self._get_raster(raster).remap_range(left, right, new)

    def _save_args_to_raster(self, args_str, line_no):
        on_line = f" on line {line_no}"
        # From c# files:
        #  (inRaster;outName;outWorkspace;rasterType;nodata;blockwidth;blockheight)
        # nodata;blockwidth;blockheight are optional
        try:
            in_rs, out_name, out_dir, type_, *extra = _split_strip(
                args_str, ";"
            )
        except ValueError:
            raise BatchScriptParseError(
                "SAVEFUNCTIONRASTER Error: Incorrect number of arguments"
                + on_line
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
            raise BatchScriptParseError(
                "SAVEFUNCTIONRASTER Error: Too many arguments" + on_line
            )
        if type_ not in FTYPE_TO_EXT:
            raise BatchScriptParseError(
                "SAVEFUNCTIONRASTER Error: Unknown file type" + on_line
            )
        raster = self._get_raster(in_rs)
        out_name = os.path.join(out_dir, out_name)
        ext = FTYPE_TO_EXT[type_]
        out_name += f".{ext}"
        return raster.save(out_name, nodata, bwidth, bheight)

    def _get_raster(self, name_or_path):
        if name_or_path in self.rasters:
            return self.rasters[name_or_path]
        else:
            validate_file(name_or_path)
            return Raster(name_or_path)
