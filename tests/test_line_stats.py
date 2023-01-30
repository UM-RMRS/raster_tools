# isort: off
# TODO(pygeos): remove this once shapely is the default backend for geopandas.
# Force raster_tools._compat to be loaded before geopandas when running tests
import raster_tools  # noqa: F401

# isort: on

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import LinearRing, LineString, Point, Polygon, box

from raster_tools import line_stats
from raster_tools.raster import Raster
from tests.utils import assert_rasters_similar, assert_valid_raster


@pytest.mark.parametrize(
    "geoms,n",
    [
        (gpd.GeoSeries(), 0),
        (gpd.GeoSeries([Point(0, 0)]), 1),
        (gpd.GeoSeries([LineString([(0, 0), (1, 1), (2, 2)])]), 3),
        (gpd.GeoSeries([box(0, 0, 1, 1)]), 5),
        (gpd.GeoSeries([LinearRing([(0, 0), (1, 1), (1, 0)])]), 4),
        (
            gpd.GeoSeries(
                [
                    Polygon([(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]),
                    # Polygon with interior ring
                    Polygon(
                        [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)],
                        [[(1, 0), (1.5, 0.5), (1, 1), (0.5, 0.5), (1, 0)]],
                    ),
                ]
            ),
            15,
        ),
        (
            gpd.GeoSeries(
                [
                    Point(0, 0),
                    LineString([(0, 0), (1, 1), (2, 2)]),
                    Polygon([(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]),
                    # Polygon with interior ring
                    Polygon(
                        [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)],
                        [[(1, 0), (1.5, 0.5), (1, 1), (0.5, 0.5), (1, 0)]],
                    ),
                ]
            ),
            19,
        ),
    ],
)
def test__calculate_number_vertices(geoms, n):
    count = line_stats._calculate_number_vertices(geoms)
    assert count == n


template_22 = xr.DataArray(
    np.ones((2, 2)), dims=("y", "x"), coords=([1, 0], [0, 1])
).rio.write_crs("5070")
template_44 = xr.DataArray(
    np.ones((4, 4)),
    dims=("y", "x"),
    coords=(np.arange(0, 4)[::-1], np.arange(0, 4)),
).rio.write_crs("5070")


def like_template(template, data):
    out = template.copy()
    out.data = data
    return out


@pytest.mark.parametrize(
    "geoms,like,radius,truth,weight",
    [
        (
            gpd.GeoSeries([LineString([(0, 1), (1, 1)])], crs="EPSG:5070"),
            Raster(template_22),
            1,
            Raster(like_template(template_22, np.array([[1, 1], [0, 0]]))),
            False,
        ),
        (
            gpd.GeoSeries([LineString([(0, 1), (1, 1)])], crs="EPSG:5070"),
            Raster(template_22).chunk((1, 1, 1)),
            1,
            Raster(like_template(template_22, np.array([[1, 1], [0, 0]]))),
            False,
        ),
        (
            gpd.GeoSeries([LineString([(0, 1), (1, 1)])], crs="EPSG:5070"),
            Raster(template_22),
            2,
            Raster(like_template(template_22, np.array([[1, 1], [1, 1]]))),
            False,
        ),
        (
            gpd.GeoSeries(
                [LineString([(-0.5, 1), (1.5, 1)])], crs="EPSG:5070"
            ),
            Raster(template_22),
            1,
            Raster(like_template(template_22, np.array([[1.5, 1.5], [0, 0]]))),
            False,
        ),
        (
            gpd.GeoSeries(
                [LineString([(0, 3), (2, 3), (2, 0)])], crs="EPSG:5070"
            ),
            Raster(template_44),
            1,
            Raster(
                like_template(
                    template_44,
                    np.array(
                        [
                            [1, 2, 2, 0],
                            [0, 0, 2, 0],
                            [0, 0, 2, 0],
                            [0, 0, 1, 0],
                        ]
                    ),
                )
            ),
            False,
        ),
        (
            gpd.GeoSeries(
                [LineString([(0, 3), (2, 3), (2, 0)])], crs="EPSG:5070"
            ),
            Raster(template_44).chunk((1, 2, 2)),
            1,
            Raster(
                like_template(
                    template_44,
                    np.array(
                        [
                            [1, 2, 2, 0],
                            [0, 0, 2, 0],
                            [0, 0, 2, 0],
                            [0, 0, 1, 0],
                        ]
                    ),
                )
            ),
            False,
        ),
        (
            gpd.GeoDataFrame(
                {
                    "weights": [3, 4],
                    "geometry": [
                        LineString([(0, 3), (2, 3)]),
                        LineString([(2, 3), (2, 0)]),
                    ],
                },
                crs="EPSG:5070",
            ),
            Raster(template_44),
            1,
            Raster(
                like_template(
                    template_44,
                    np.array(
                        [
                            [1, 2, 2, 0],
                            [0, 0, 2, 0],
                            [0, 0, 2, 0],
                            [0, 0, 1, 0],
                        ]
                    ),
                )
            ),
            False,
        ),
        (
            gpd.GeoDataFrame(
                {
                    "weights": [3, 4],
                    "geometry": [
                        LineString([(0, 3), (2, 3)]),
                        LineString([(2, 3), (2, 0)]),
                    ],
                },
                crs="EPSG:5070",
            ),
            Raster(template_44),
            1,
            Raster(
                like_template(
                    template_44,
                    np.array(
                        [
                            [3, 6, 7, 0],
                            [0, 0, 8, 0],
                            [0, 0, 8, 0],
                            [0, 0, 4, 0],
                        ]
                    ),
                )
            ),
            True,
        ),
        (
            gpd.GeoDataFrame(
                {
                    "weights": [3, 4],
                    "geometry": [
                        LineString([(0, 3), (2, 3)]),
                        LineString([(2, 3), (2, 0)]),
                    ],
                },
                crs="EPSG:5070",
            ),
            Raster(template_44).chunk((1, 2, 2)),
            1,
            Raster(
                like_template(
                    template_44,
                    np.array(
                        [
                            [3, 6, 7, 0],
                            [0, 0, 8, 0],
                            [0, 0, 8, 0],
                            [0, 0, 4, 0],
                        ]
                    ),
                )
            ),
            True,
        ),
    ],
)
def test_length(geoms, like, radius, truth, weight):
    result = line_stats.length(
        geoms, like, radius, weighting_field="weights" if weight else None
    )

    assert_valid_raster(result)
    assert np.allclose(result, truth)
    assert_rasters_similar(result, like)
    assert result.mask.compute().sum() == 0
