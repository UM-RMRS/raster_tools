import importlib
from typing import TYPE_CHECKING

import raster_tools._compat  # noqa: F401
from raster_tools import clipping, focal, line_stats, surface, zonal
from raster_tools._mosaic import mosaic
from raster_tools._padding import pad
from raster_tools._stack import split_bands, stack_bands
from raster_tools._version import __version__  # noqa
from raster_tools.blocks import (
    geo_map_blocks,
    geo_map_overlap,
    map_blocks,
    map_overlap,
)
from raster_tools.creation import (
    constant_raster,
    empty_like,
    full_like,
    ones_like,
    random_raster,
    zeros_like,
)
from raster_tools.general import band_concat, reclassify, remap_range
from raster_tools.io import open_dataset
from raster_tools.raster import (
    Raster,
    data_to_raster,
    data_to_raster_like,
    data_to_xr_raster,
    data_to_xr_raster_ds,
    data_to_xr_raster_ds_like,
    data_to_xr_raster_like,
    dataarray_to_raster,
    dataarray_to_xr_raster,
    dataarray_to_xr_raster_ds,
    get_raster,
)
from raster_tools.vector import (
    Vector,
    count_layer_features,
    list_layers,
    open_vectors,
)
from raster_tools.warp import reproject

if TYPE_CHECKING:
    from raster_tools import distance

# Importing distance eagerly compiles its explicitly-signed numba
# functions, adding seconds to import. Load it lazily (PEP 562) so only
# code that uses it pays that cost.
_LAZY_SUBMODULES = frozenset(("distance",))


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"raster_tools.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY_SUBMODULES)


__all__ = [
    "Raster",
    "Vector",
    "band_concat",
    "clipping",
    "constant_raster",
    "count_layer_features",
    "data_to_raster",
    "data_to_raster_like",
    "data_to_xr_raster",
    "data_to_xr_raster_ds",
    "data_to_xr_raster_ds_like",
    "data_to_xr_raster_like",
    "dataarray_to_raster",
    "dataarray_to_xr_raster",
    "dataarray_to_xr_raster_ds",
    "distance",
    "empty_like",
    "focal",
    "full_like",
    "geo_map_blocks",
    "geo_map_overlap",
    "get_raster",
    "line_stats",
    "list_layers",
    "map_blocks",
    "map_overlap",
    "mosaic",
    "ones_like",
    "open_dataset",
    "open_vectors",
    "pad",
    "random_raster",
    "reclassify",
    "remap_range",
    "reproject",
    "split_bands",
    "stack_bands",
    "surface",
    "zeros_like",
    "zonal",
]
