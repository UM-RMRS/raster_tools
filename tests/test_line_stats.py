# isort: off
# TODO(pygeos): remove this once shapely is the default backend for geopandas.
# Force raster_tools._compat to be loaded before geopandas when running tests
import raster_tools  # noqa: F401

# isort: on

import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import LinearRing, LineString, Point, Polygon, box

from raster_tools import line_stats
from raster_tools.raster import Raster
from raster_tools.vector import Vector
from tests.utils import (
    assert_rasters_similar,
    assert_valid_raster,
    make_raster,
)


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
        (
            dgpd.from_geopandas(
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
                npartitions=2,
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


def test_length_with_radius_larger_than_chunk():
    # Rechunk so that right most chunk's width is less than the radius.
    # The rechunk-to-fit-depth dance happens inside geo_map_overlap on
    # a copy now, so the result lands back on the user's declared
    # chunking (here (90, 10)) -- the user shouldn't have to know that
    # internal rechunking happened.
    like = make_raster(
        "arange", shape=(1, 100, 100), chunksize=(1, 100, 90), crs="5070"
    )
    geoms = gpd.GeoDataFrame(
        # zero length
        {"geometry": [LineString([(0, 3), (0, 3)])]},
        crs="EPSG:5070",
    )
    expected = Raster(np.zeros((100, 100))).set_crs("5070")

    result = line_stats.length(Vector(geoms), like, 20)
    assert_valid_raster(result)
    assert np.allclose(result, expected)
    assert result.data.chunks == ((1,), (100,), (90, 10))


_simple_line = gpd.GeoSeries([LineString([(0, 1), (1, 1)])], crs="EPSG:5070")


@pytest.mark.parametrize(
    "features,like,radius,weighting_field,exc,match",
    [
        (
            _simple_line,
            Raster(template_22),
            [1, 2],
            None,
            TypeError,
            "radius must be a scalar",
        ),
        (
            _simple_line,
            Raster(template_22),
            0,
            None,
            ValueError,
            "radius must be greater than zero",
        ),
        (
            _simple_line,
            Raster(template_22),
            -1,
            None,
            ValueError,
            "radius must be greater than zero",
        ),
        (
            _simple_line,
            make_raster("ones", shape=(1, 2, 2), crs=None),
            1,
            None,
            ValueError,
            "like_rast must have a CRS",
        ),
        (
            gpd.GeoDataFrame(
                {"weights": [1.0], "geometry": [LineString([(0, 1), (1, 1)])]},
                crs="EPSG:5070",
            ),
            Raster(template_22),
            1,
            123,
            TypeError,
            "weighting_field must be a string",
        ),
        (
            gpd.GeoDataFrame(
                {"weights": [1.0], "geometry": [LineString([(0, 1), (1, 1)])]},
                crs="EPSG:5070",
            ),
            Raster(template_22),
            1,
            "missing",
            ValueError,
            "weighting_field must be a field name",
        ),
        (
            gpd.GeoDataFrame(
                {
                    "label": ["a"],
                    "geometry": [LineString([(0, 1), (1, 1)])],
                },
                crs="EPSG:5070",
            ),
            Raster(template_22),
            1,
            "label",
            TypeError,
            "must be a scalar type",
        ),
    ],
)
def test_length_invalid_inputs(
    features, like, radius, weighting_field, exc, match
):
    with pytest.raises(exc, match=match):
        line_stats.length(
            features, like, radius, weighting_field=weighting_field
        )


def test_length_reprojects_features():
    # Same line as the (0, 1)->(1, 1) case in 5070, but expressed in 4326.
    # line_stats.length must reproject features to like_rast's CRS.
    like = Raster(template_22)
    line_5070 = gpd.GeoSeries([LineString([(0, 1), (1, 1)])], crs="EPSG:5070")
    line_4326 = line_5070.to_crs("EPSG:4326")

    result_native = line_stats.length(line_5070, like, 1)
    result_reproj = line_stats.length(line_4326, like, 1)

    assert_valid_raster(result_reproj)
    assert_rasters_similar(result_reproj, like)
    assert np.allclose(result_reproj, result_native)


def test_length_bool_weighting_field():
    # Two parallel lines; only the True-weighted one should contribute. The
    # expected output equals running with just the True line and no weighting.
    gdf_weighted = gpd.GeoDataFrame(
        {
            "weights": [True, False],
            "geometry": [
                LineString([(0, 3), (2, 3)]),
                LineString([(0, 0), (2, 0)]),
            ],
        },
        crs="EPSG:5070",
    )
    gdf_true_only = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 3), (2, 3)])]},
        crs="EPSG:5070",
    )
    like = Raster(template_44)
    expected = line_stats.length(gdf_true_only, like, 1).values
    result = line_stats.length(
        gdf_weighted, like, 1, weighting_field="weights"
    )
    assert_valid_raster(result)
    assert np.allclose(result.values, expected)


def _high_vertex_line(n_verts, y, crs="EPSG:5070"):
    xs = np.linspace(0, 3, n_verts)
    return LineString([(float(x), y) for x in xs])


def test_length_triggers_batched_by_vertex_ratio(monkeypatch):
    # Two lines with many vertices each push nverts/n above the cutoff.
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                _high_vertex_line(60, 3),
                _high_vertex_line(60, 2),
            ],
        },
        crs="EPSG:5070",
    )
    like = Raster(template_44)

    # Run unbatched by raising the thresholds above any reachable value.
    monkeypatch.setattr(line_stats, "VERTS_PER_GEOM_CUTOFF", 10**9)
    monkeypatch.setattr(line_stats, "GEOM_BATCH_SIZE", 10**9)
    expected = line_stats.length(gdf, like, 1).values

    # Now allow the batched path to fire.
    monkeypatch.setattr(line_stats, "VERTS_PER_GEOM_CUTOFF", 15)
    monkeypatch.setattr(line_stats, "GEOM_BATCH_SIZE", 700)
    result = line_stats.length(gdf, like, 1)

    assert_valid_raster(result)
    assert np.allclose(result.values, expected)


def test_length_triggers_batched_by_count(monkeypatch):
    # Many simple lines; force batching by lowering GEOM_BATCH_SIZE.
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, y), (3, y)]) for y in (0.0, 1.0, 2.0, 3.0)
            ],
        },
        crs="EPSG:5070",
    )
    like = Raster(template_44)

    monkeypatch.setattr(line_stats, "GEOM_BATCH_SIZE", 10**9)
    monkeypatch.setattr(line_stats, "VERTS_PER_GEOM_CUTOFF", 10**9)
    expected = line_stats.length(gdf, like, 1).values

    monkeypatch.setattr(line_stats, "GEOM_BATCH_SIZE", 2)
    result = line_stats.length(gdf, like, 1)

    assert_valid_raster(result)
    assert np.allclose(result.values, expected)


def test_length_filters_empty_geoms():
    # Empty LineStrings must be filtered out; result must match running with
    # only the non-empty line.
    valid_line = LineString([(0, 3), (2, 3), (2, 0)])
    gdf_mixed = gpd.GeoDataFrame(
        {"geometry": [valid_line, LineString()]},
        crs="EPSG:5070",
    )
    gdf_clean = gpd.GeoDataFrame({"geometry": [valid_line]}, crs="EPSG:5070")
    like = Raster(template_44)

    result = line_stats.length(gdf_mixed, like, 1)
    expected = line_stats.length(gdf_clean, like, 1)

    assert_valid_raster(result)
    assert np.allclose(result.values, expected.values)


@pytest.mark.parametrize(
    "geoms",
    [
        gpd.GeoSeries([Point(0, 1)], crs="EPSG:5070"),
        gpd.GeoSeries(
            [Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])],
            crs="EPSG:5070",
        ),
        gpd.GeoSeries(
            [
                LineString([(0, 1), (1, 1)]),
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            ],
            crs="EPSG:5070",
        ),
    ],
)
def test_length_rejects_non_line_geoms(geoms):
    with pytest.raises(TypeError, match="only supports line geometries"):
        line_stats.length(geoms, Raster(template_22), 1)


def test_length_with_linearring_input():
    # A LinearRing's length equals its perimeter. Compare with the same
    # vertex sequence as a closed LineString.
    ring = LinearRing([(2, 2), (4, 2), (4, 4), (2, 4)])
    line = LineString([(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)])
    like = make_raster("ones", shape=(1, 10, 10), crs="5070")

    result_ring = line_stats.length(
        gpd.GeoSeries([ring], crs="EPSG:5070"), like, 1
    )
    result_line = line_stats.length(
        gpd.GeoSeries([line], crs="EPSG:5070"), like, 1
    )

    assert_valid_raster(result_ring)
    assert np.any(result_ring.values > 0)
    assert np.allclose(result_ring.values, result_line.values)


def test_length_multiband_like_uses_first_band():
    geoms = gpd.GeoSeries([LineString([(0, 1), (1, 1)])], crs="EPSG:5070")
    single = Raster(template_22)
    # Build a 3-band template that shares template_22's coords/CRS.
    multi_xr = xr.DataArray(
        np.ones((3, 2, 2)),
        dims=("band", "y", "x"),
        coords={"band": [1, 2, 3], "y": [1, 0], "x": [0, 1]},
    ).rio.write_crs("5070")
    multi = Raster(multi_xr)

    result_multi = line_stats.length(geoms, multi, 1)
    result_single = line_stats.length(geoms, single, 1)

    assert_valid_raster(result_multi)
    assert result_multi.shape[0] == 1
    assert np.allclose(result_multi.values, result_single.values)
