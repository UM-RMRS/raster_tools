# isort: off
# TODO(pygeos): remove this once shapely is the default backend for geopandas.
# Force raster_tools._compat to be loaded before geopandas when running tests
import raster_tools as rts  # noqa: F401

# isort: on

import os

import geopandas as gpd
import numpy as np
import pytest
import shapely
from affine import Affine
from rasterio.enums import MergeAlg
from rasterio.features import rasterize as rio_rasterize

from raster_tools import _rasterize_numba, rasterize
from raster_tools._rasterize_numba import (
    _NumbaUnsupported,
    numba_mask,
    numba_rasterize_wrapper,
)
from tests.utils import make_raster

# Exercise this module under both rasterization backends via the shared
# conftest fixture, matching the other rasterization test modules.
pytestmark = pytest.mark.usefixtures("_rasterize_backend")

_EXACT_DTYPES = [
    "uint8",
    "uint16",
    "uint32",
    "int16",
    "int32",
    "float32",
    "float64",
]


def test_rasterize_numba_warmup():
    # Selectable with `pytest -p no:xdist -k rasterize_numba_warmup` to
    # populate numba's on-disk cache serially before a parallel run.
    _rasterize_numba._warmup()


def _rio(geoms, values, shape, transform, all_touched, dtype):
    return rio_rasterize(
        zip(geoms, values, strict=True),
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        merge_alg=MergeAlg.replace,
        dtype=dtype,
    )


def _numba(geoms, values, shape, transform, all_touched, dtype):
    return numba_rasterize_wrapper(
        shape,
        transform,
        np.array(geoms, dtype=object),
        np.asarray(values),
        np.dtype(dtype),
        0,
        all_touched,
    )


def _rio_mask_oracle(geoms, shape, transform, all_touched, invert):
    fill, geom_value = (1, 0) if invert else (0, 1)
    out = np.full(shape, fill, dtype="uint8")
    rio_rasterize(
        np.array(geoms, dtype=object),
        out=out,
        transform=transform,
        all_touched=all_touched,
        default_value=geom_value,
    )
    return out


def _rand_affine(rng):
    a = rng.uniform(0.3, 3.0)
    e = -rng.uniform(0.3, 3.0)
    c = rng.uniform(-30, 30)
    f = rng.uniform(-30, 30)
    return Affine(a, 0.0, c, 0.0, e, f)


def _world_bounds(transform, nx, ny):
    a, b, c, d, e, f = transform[:6]
    xs = []
    ys = []
    for col, row in ((0, 0), (nx, 0), (nx, ny), (0, ny)):
        xs.append(a * col + b * row + c)
        ys.append(d * col + e * row + f)
    return min(xs), max(xs), min(ys), max(ys)


def _make_polygons(rng, nx, ny, transform, snapped):
    geoms = []
    for _ in range(int(rng.integers(1, 4))):
        if snapped:
            k = int(rng.integers(3, 7))
            pts = rng.integers(0, min(nx, ny), size=(k, 2)).astype(float)
            if rng.random() < 0.5:
                pts += 0.5
        else:
            xmin, xmax, ymin, ymax = _world_bounds(transform, nx, ny)
            k = int(rng.integers(3, 7))
            pts = np.column_stack(
                [rng.uniform(xmin, xmax, k), rng.uniform(ymin, ymax, k)]
            )
        poly = shapely.Polygon(pts).buffer(0)
        if poly.is_empty or poly.geom_type not in (
            "Polygon",
            "MultiPolygon",
        ):
            continue
        if not snapped and rng.random() < 0.3:
            # Punch a hole to exercise interior rings.
            cx, cy = poly.centroid.x, poly.centroid.y
            hole = shapely.Point(cx, cy).buffer(0.3)
            holed = poly.difference(hole)
            if not holed.is_empty and holed.geom_type in (
                "Polygon",
                "MultiPolygon",
            ):
                poly = holed
        geoms.append(poly)
    return geoms


def _make_lines(rng, nx, ny, transform, snapped):
    geoms = []
    xmin, xmax, ymin, ymax = _world_bounds(transform, nx, ny)
    for _ in range(int(rng.integers(1, 4))):
        choice = rng.random()
        if choice < 0.2:
            # Axis-aligned segment.
            x0 = rng.uniform(xmin, xmax)
            y0 = rng.uniform(ymin, ymax)
            if rng.random() < 0.5:
                geoms.append(shapely.LineString([(x0, y0), (xmax, y0)]))
            else:
                geoms.append(shapely.LineString([(x0, y0), (x0, ymax)]))
        elif choice < 0.3:
            # Single-pixel / zero-length degenerate segment.
            x0 = rng.uniform(xmin, xmax)
            y0 = rng.uniform(ymin, ymax)
            geoms.append(shapely.LineString([(x0, y0), (x0, y0)]))
        elif choice < 0.45:
            # MultiLineString with a couple of parts.
            parts = []
            for _ in range(int(rng.integers(2, 4))):
                k = int(rng.integers(2, 5))
                parts.append(
                    np.column_stack(
                        [
                            rng.uniform(xmin, xmax, k),
                            rng.uniform(ymin, ymax, k),
                        ]
                    )
                )
            geoms.append(shapely.MultiLineString(parts))
        else:
            k = int(rng.integers(2, 6))
            geoms.append(
                shapely.LineString(
                    np.column_stack(
                        [
                            rng.uniform(xmin, xmax, k),
                            rng.uniform(ymin, ymax, k),
                        ]
                    )
                )
            )
    return geoms


def _make_points(rng, nx, ny, transform, snapped):
    geoms = []
    xmin, xmax, ymin, ymax = _world_bounds(transform, nx, ny)
    for _ in range(int(rng.integers(1, 4))):
        if not snapped and rng.random() < 0.25:
            # MultiPoint expands to its constituent points.
            k = int(rng.integers(2, 5))
            pts = np.column_stack(
                [rng.uniform(xmin, xmax, k), rng.uniform(ymin, ymax, k)]
            )
            geoms.append(shapely.MultiPoint(pts))
            continue
        if snapped or rng.random() < 0.3:
            # Land on an integer world coordinate (pixel corner on a unit
            # grid) to hit the corner-rounding path.
            x = float(int(rng.uniform(xmin, xmax)))
            y = float(int(rng.uniform(ymin, ymax)))
        else:
            x = rng.uniform(xmin, xmax)
            y = rng.uniform(ymin, ymax)
        geoms.append(shapely.Point(x, y))
    return geoms


_MAKERS = {
    "poly": _make_polygons,
    "line": _make_lines,
    "point": _make_points,
}


def _draw_case(rng):
    family = str(rng.choice(["poly", "line", "point"]))
    all_touched = bool(rng.integers(0, 2))
    dtype = str(rng.choice(_EXACT_DTYPES))
    snapped = bool(rng.random() < 0.25)
    if snapped:
        nx = int(rng.integers(6, 20))
        ny = int(rng.integers(6, 20))
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, ny)
    else:
        nx = int(rng.integers(5, 25))
        ny = int(rng.integers(5, 25))
        transform = _rand_affine(rng)
    geoms = _MAKERS[family](rng, nx, ny, transform, snapped)
    return family, all_touched, dtype, nx, ny, transform, geoms


def _fuzz_cases():
    return int(os.environ.get("RASTERIZE_FUZZ_CASES", "300"))


def test_numba_rasterize_matches_rasterio_fuzz():
    rng = np.random.default_rng(0)
    for case in range(_fuzz_cases()):
        family, all_touched, dtype, nx, ny, transform, geoms = _draw_case(rng)
        if not geoms:
            continue
        values = rng.integers(1, 51, len(geoms))
        shape = (ny, nx)
        expected = _rio(geoms, values, shape, transform, all_touched, dtype)
        got = _numba(geoms, values, shape, transform, all_touched, dtype)
        if not np.array_equal(expected, got):
            diff = np.argwhere(expected != got)
            raise AssertionError(
                f"fuzz mismatch: seed=0 case={case} family={family} "
                f"all_touched={all_touched} dtype={dtype} "
                f"transform={transform!r} ndiff={len(diff)} "
                f"first_diff={diff[:5].tolist()} "
                f"wkt={[g.wkt for g in geoms]}"
            )


def test_numba_mask_matches_rasterio_fuzz():
    rng = np.random.default_rng(1)
    for case in range(_fuzz_cases()):
        family, all_touched, _, nx, ny, transform, geoms = _draw_case(rng)
        if not geoms:
            continue
        invert = bool(rng.integers(0, 2))
        shape = (ny, nx)
        expected = _rio_mask_oracle(
            geoms, shape, transform, all_touched, invert
        )
        got = numba_mask(
            np.array(geoms, dtype=object),
            shape,
            transform,
            all_touched,
            invert,
        )
        if not np.array_equal(expected, got):
            diff = np.argwhere(expected != got)
            raise AssertionError(
                f"mask fuzz mismatch: seed=1 case={case} family={family} "
                f"all_touched={all_touched} invert={invert} "
                f"transform={transform!r} ndiff={len(diff)} "
                f"wkt={[g.wkt for g in geoms]}"
            )


# --- Explicit regression cases (section 1.8 and the added edge cases) ---

_TR = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)


def _one(geom, all_touched, dtype="int32", value=1, shape=(8, 8), tr=_TR):
    expected = _rio([geom], [value], shape, tr, all_touched, dtype)
    got = _numba([geom], [value], shape, tr, all_touched, dtype)
    np.testing.assert_array_equal(got, expected)
    return got


@pytest.mark.parametrize("all_touched", [False, True])
def test_box_edges_on_pixel_centers(all_touched):
    _one(shapely.box(2.5, 2.5, 5.5, 5.5), all_touched)


@pytest.mark.parametrize("all_touched", [False, True])
def test_zero_area_polygon_burns_nothing(all_touched):
    out = _one(shapely.Polygon([(1, 1), (5, 1), (3, 1), (1, 1)]), all_touched)
    assert int((out != 0).sum()) == 0


@pytest.mark.parametrize("all_touched", [False, True])
def test_horizontal_line_on_row_boundary(all_touched):
    _one(shapely.LineString([(1, 6), (7, 6)]), all_touched)


def test_vertical_line_on_col_boundary_all_touched():
    _one(shapely.LineString([(3, 1), (3, 7)]), True)


def test_point_on_pixel_corner():
    out = _one(shapely.Point(3, 7), False)
    assert out[3, 3] != 0
    assert int((out != 0).sum()) == 1


def test_point_on_right_edge_burns_nothing():
    out = _one(shapely.Point(8, 2), False)
    assert int((out != 0).sum()) == 0


def test_linestring_reversal_matches_rasterio():
    # A multi-segment line whose Bresenham walk differs by direction; the
    # kernel must reverse coordinates the way GDAL does.
    line = shapely.LineString([(1.2, 1.7), (5.8, 3.1), (2.4, 6.9)])
    _one(line, False)
    _one(line, True)


@pytest.mark.parametrize("all_touched", [False, True])
def test_multipolygon_overlapping_solid_parts(all_touched):
    mp = shapely.MultiPolygon(
        [shapely.box(1, 1, 5, 5), shapely.box(3, 3, 7, 7)]
    )
    _one(mp, all_touched)


@pytest.mark.parametrize("all_touched", [False, True])
def test_polygon_with_hole(all_touched):
    ph = shapely.Polygon(
        [(1, 1), (7, 1), (7, 7), (1, 7)],
        holes=[[(3, 3), (5, 3), (5, 5), (3, 5)]],
    )
    _one(ph, all_touched)


def test_far_off_line_endpoint_int_guard():
    tr = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 20.0)
    far = shapely.LineString([(5, 5), (5e10, 5e10)])
    for all_touched in (False, True):
        expected = _rio([far], [1], (20, 20), tr, all_touched, "int32")
        got = _numba([far], [1], (20, 20), tr, all_touched, "int32")
        np.testing.assert_array_equal(got, expected)


def test_far_off_and_nan_point_burn_nothing():
    tr = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 20.0)
    geoms = [shapely.Point(5e10, 5e10), shapely.Point()]
    got = numba_rasterize_wrapper(
        (20, 20),
        tr,
        np.array(geoms, dtype=object),
        np.array([1, 2]),
        np.dtype("int32"),
        0,
        False,
    )
    assert int((got != 0).sum()) == 0


_TR20 = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 20.0)


@pytest.mark.parametrize("all_touched", [False, True])
def test_nan_vertex_polygon_does_not_crash_and_matches_gdal(all_touched):
    # A NaN vertex is filtered out upstream and GDAL's behaviour on it is
    # undefined, but the fill kernel must stay memory-safe: it skips edges
    # touching a NaN vertex rather than indexing with ceil(NaN). Here GDAL
    # burns nothing, so the kernel must agree and burn only 0s.
    poly = shapely.Polygon([(2, 2), (8, 2), (8, 8), (np.nan, 8), (2, 2)])
    got = _numba([poly], [7], (20, 20), _TR20, all_touched, "int32")
    expected = _rio([poly], [7], (20, 20), _TR20, all_touched, "int32")
    np.testing.assert_array_equal(got, expected)
    assert set(np.unique(got).tolist()) <= {0, 7}


@pytest.mark.parametrize("all_touched", [False, True])
def test_huge_finite_vertex_polygon_matches_gdal(all_touched):
    # A vertex far outside the C int range is clamped exactly as GDAL clamps
    # it, so the bucketed fill stays pixel-identical to rasterio.
    poly = shapely.Polygon([(2, 2), (8, 2), (8, 8), (1e300, 8), (2, 2)])
    got = _numba([poly], [7], (20, 20), _TR20, all_touched, "int32")
    expected = _rio([poly], [7], (20, 20), _TR20, all_touched, "int32")
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("all_touched", [False, True])
def test_inf_vertex_polygon_does_not_crash(all_touched):
    # An infinite vertex is clamped by the row-window checks before any
    # integer conversion, so the kernel stays memory-safe and bounded. Its
    # exact output is not required to match GDAL (the committed kernel does
    # not either), only to burn valid, in-range values.
    poly = shapely.Polygon([(2, 2), (8, 2), (8, 8), (np.inf, 8), (2, 2)])
    got = _numba([poly], [7], (20, 20), _TR20, all_touched, "int32")
    assert set(np.unique(got).tolist()) <= {0, 7}


@pytest.mark.parametrize(
    "value,expected_burn", [(2.7, 3), (2.5, 3), (-2.5, -3)]
)
def test_fractional_value_rounds_half_away_from_zero(value, expected_burn):
    box = shapely.box(2, 2, 6, 6)
    got = _numba([box], np.array([value]), (8, 8), _TR, False, "int32")
    expected = _rio([box], [value], (8, 8), _TR, False, "int32")
    np.testing.assert_array_equal(got, expected)
    assert set(np.unique(got).tolist()) == {0, expected_burn}


def test_none_and_empty_in_batch_stay_aligned():
    geoms = [
        shapely.box(1, 1, 3, 3),
        None,
        shapely.Polygon(),
        shapely.box(4, 4, 6, 6),
    ]
    values = np.array([10, 20, 30, 40])
    got = _numba(geoms, values, (8, 8), _TR, False, "int32")
    expected = _rio(
        [g for g in geoms if g is not None and not g.is_empty],
        [10, 40],
        (8, 8),
        _TR,
        False,
        "int32",
    )
    np.testing.assert_array_equal(got, expected)
    burned = set(np.unique(got).tolist())
    assert burned == {0, 10, 40}


@pytest.mark.parametrize(
    "geom2d,geom3d",
    [
        (
            shapely.box(2, 2, 6, 6),
            shapely.Polygon([(2, 2, 9), (6, 2, 9), (6, 6, 9), (2, 6, 9)]),
        ),
        (
            shapely.LineString([(1, 1), (6, 6)]),
            shapely.LineString([(1, 1, 5), (6, 6, 5)]),
        ),
        (shapely.Point(3.5, 3.5), shapely.Point(3.5, 3.5, 9)),
    ],
)
def test_z_coordinate_ignored(geom2d, geom3d):
    got2d = _numba([geom2d], [1], (8, 8), _TR, False, "int32")
    got3d = _numba([geom3d], [1], (8, 8), _TR, False, "int32")
    np.testing.assert_array_equal(got3d, got2d)
    expected = _rio([geom3d], [1], (8, 8), _TR, False, "int32")
    np.testing.assert_array_equal(got3d, expected)


@pytest.mark.parametrize("all_touched", [False, True])
def test_bowtie_ring_even_odd_parity(all_touched):
    # Raw self-intersecting ring, no buffer(0) repair.
    bow = shapely.Polygon([(1, 1), (6, 6), (6, 1), (1, 6), (1, 1)])
    _one(bow, all_touched)


# Self-intersecting rings whose winding GDAL reads from the local turn at the
# lowest-rightmost vertex, which disagrees with the global signed area, and
# whose horizontal edges land on pixel-center scanlines. Each burns a
# different set depending on ring orientation, so parity depends on
# reproducing GDAL's clockwise normalization rather than a signed-area sign.
_SELF_INTERSECTING_WKTS = [
    "POLYGON ((19.5 19.5, 10.5 11.5, 3.5 2.5, 9.5 11.5, 17.5 11.5, "
    "15.5 6.5, 19.5 19.5))",
    "POLYGON ((19.5 19.5, 10.5 18.5, 0.5 15.5, 16.5 15.5, 4.5 13.5, "
    "19.5 19.5))",
    "POLYGON ((7.5 17.5, 17.5 5.5, 19.5 8.5, 2.5 6.5, 16.5 5.5, 1.5 5.5, "
    "7.5 17.5))",
    "POLYGON ((17.5 9.5, 17.5 19.5, 4.5 0.5, 15.5 19.5, 7.5 3.5, 16.5 9.5, "
    "17.5 9.5))",
    "POLYGON ((15.5 1.5, 16.5 2.5, 7.5 10.5, 16.5 15.5, 12.5 1.5, 15.5 1.5))",
    "POLYGON ((4.5 9.5, 17.5 9.5, 13.5 6.5, 8.5 16.5, 4.5 9.5))",
    "POLYGON ((9.5 7.5, 8.5 7.5, 7.5 3.5, 13.5 8.5, 7.5 14.5, 16.5 15.5, "
    "9.5 7.5))",
]


@pytest.mark.parametrize("wkt", _SELF_INTERSECTING_WKTS)
@pytest.mark.parametrize("all_touched", [False, True])
def test_self_intersecting_ring_center_scanline_parity(wkt, all_touched):
    geom = shapely.from_wkt(wkt)
    tr = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 20.0)
    _one(geom, all_touched, shape=(20, 20), tr=tr)
    # Winding-invariant: the reversed ring must burn the same set, since GDAL
    # normalizes both to clockwise before filling.
    reversed_geom = shapely.Polygon(list(geom.exterior.coords)[::-1])
    _one(reversed_geom, all_touched, shape=(20, 20), tr=tr)


@pytest.mark.parametrize("all_touched", [False, True])
def test_multilinestring_matches_rasterio(all_touched):
    mls = shapely.MultiLineString(
        [[(1.2, 1.7), (6.4, 6.1)], [(1.5, 6.5), (6.5, 1.5)]]
    )
    out = _one(mls, all_touched)
    assert int((out != 0).sum()) > 0


@pytest.mark.parametrize("all_touched", [False, True])
def test_multipoint_matches_rasterio(all_touched):
    mp = shapely.MultiPoint([(2.5, 7.5), (5.5, 4.5), (3.5, 5.5)])
    out = _one(mp, all_touched)
    assert int((out != 0).sum()) == 3


def _readonly_object_array(geoms):
    arr = np.array(geoms, dtype=object)
    arr.setflags(write=False)
    return arr


@pytest.mark.parametrize(
    "make_geoms",
    [_readonly_object_array, lambda geoms: gpd.GeoSeries(geoms, index=[10])],
    ids=["readonly_array", "geoseries"],
)
def test_numba_rasterize_accepts_readonly_and_geoseries(make_geoms):
    box = shapely.box(2, 2, 6, 6)
    got = numba_rasterize_wrapper(
        (8, 8),
        _TR,
        make_geoms([box]),
        np.array([5]),
        np.dtype("int32"),
        0,
        False,
    )
    expected = _rio([box], [5], (8, 8), _TR, False, "int32")
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize(
    "make_geoms",
    [_readonly_object_array, lambda geoms: gpd.GeoSeries(geoms, index=[10])],
    ids=["readonly_array", "geoseries"],
)
def test_numba_mask_accepts_readonly_and_geoseries(make_geoms):
    box = shapely.box(2, 2, 6, 6)
    got = numba_mask(make_geoms([box]), (8, 8), _TR, False, False)
    expected = _rio_mask_oracle([box], (8, 8), _TR, False, False)
    np.testing.assert_array_equal(got, expected)


# --- Fallback contract ---


def _dispatch_raises(geoms, transform=_TR):
    out = np.zeros((8, 8), dtype="int32")
    values = np.arange(1, len(geoms) + 1)
    with pytest.raises(_NumbaUnsupported):
        _rasterize_numba._dispatch_burn(
            out, transform, np.array(geoms, dtype=object), values, False
        )


def test_dispatch_falls_back_on_mixed_family():
    _dispatch_raises(
        [shapely.box(0, 0, 2, 2), shapely.LineString([(0, 0), (3, 3)])]
    )


def test_dispatch_falls_back_on_geometrycollection():
    gc = shapely.GeometryCollection(
        [shapely.Point(1, 1), shapely.box(0, 0, 2, 2)]
    )
    _dispatch_raises([gc])


def test_dispatch_falls_back_on_rotation():
    _dispatch_raises(
        [shapely.box(2, 2, 6, 6)], Affine(1.0, 0.2, 0.0, 0.1, -1.0, 8.0)
    )


@pytest.mark.parametrize(
    "geoms",
    [
        [
            shapely.box(2, 2, 8, 8),
            shapely.LineString([(1, 1), (9, 9)]),
            shapely.Point(5, 5),
        ],
        [
            shapely.GeometryCollection(
                [shapely.Point(3, 3), shapely.box(4, 4, 8, 8)]
            ),
            shapely.box(1, 1, 5, 5),
        ],
    ],
)
def test_public_rasterize_numba_backend_falls_back(monkeypatch, geoms):
    like = make_raster(
        "zeros",
        shape=(1, 12, 12),
        affine=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 12.0),
        chunksize=(1, 6, 6),
    )
    gdf = gpd.GeoDataFrame(
        {"values": np.arange(1, len(geoms) + 1)},
        geometry=geoms,
        crs="EPSG:3857",
    )
    monkeypatch.setattr(rasterize, "RASTERIZE_BACKEND", "rasterio")
    ref = rasterize.rasterize(
        gdf, like, field="values", all_touched=True, null_value=0
    ).to_numpy()
    monkeypatch.setattr(rasterize, "RASTERIZE_BACKEND", "numba")
    got = rasterize.rasterize(
        gdf, like, field="values", all_touched=True, null_value=0
    ).to_numpy()
    np.testing.assert_array_equal(got, ref)


# --- Scratch-cap fallback ---


def _poly_kernel_arrays(poly, transform):
    geometry = np.array([poly], dtype=object)
    ids = shapely.get_type_id(geometry)
    _, coords, offsets = _rasterize_numba._polygonal_ragged(geometry, ids)
    px, py = _rasterize_numba._to_pixel(coords, transform)
    ring_offsets = offsets[0].astype(np.int64)
    ring_start = np.ascontiguousarray(ring_offsets[:-1])
    ring_stop = np.ascontiguousarray(ring_offsets[1:])
    x = np.ascontiguousarray(coords[:, 0])
    y = np.ascontiguousarray(coords[:, 1])
    rev = _rasterize_numba._ring_reverse_mask(x, y, ring_start, ring_stop)
    return px, py, ring_start, ring_stop, rev


def _burn_poly_capped(poly, value, shape, transform, all_touched, dtype, cap):
    geometry = np.array([poly], dtype=object)
    ids = shapely.get_type_id(geometry)
    name, coords, offsets = _rasterize_numba._polygonal_ragged(geometry, ids)
    px, py = _rasterize_numba._to_pixel(coords, transform)
    values = _rasterize_numba._cast_burn_values(
        np.array([value]), np.dtype(dtype)
    )
    out = np.zeros(shape, dtype=dtype)
    _rasterize_numba._burn_polygonal(
        out,
        px,
        py,
        coords,
        offsets,
        values,
        name,
        all_touched,
        scratch_cap=cap,
    )
    return out


@pytest.mark.parametrize("all_touched", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_scratch_cap_fallback_matches_rasterio(all_touched, reverse):
    # A one-entry scratch makes the bucketed fill give up on any real
    # polygon, so the per-row fallback fills it. The box's horizontal edges
    # sit on pixel-center scanlines, the only branch in the fallback that
    # depends on ring orientation, so both windings are checked.
    tr = _TR20
    shape = (20, 20)
    ring = [(2.5, 2.5), (2.5, 12.5), (12.5, 12.5), (12.5, 2.5), (2.5, 2.5)]
    if reverse:
        ring = ring[::-1]
    poly = shapely.Polygon(ring)
    got = _burn_poly_capped(poly, 7, shape, tr, all_touched, "int32", 1)
    expected = _rio([poly], [7], shape, tr, all_touched, "int32")
    np.testing.assert_array_equal(got, expected)
    assert int((got != 0).sum()) > 0


def test_scratch_cap_forces_per_row_fallback():
    # A one-entry scratch cannot hold the crossings, so _fill_bucketed reports
    # it did not complete and the caller routes to the per-row kernel; a
    # scratch large enough lets the bucketed kernel finish. Both kernels
    # reproduce rasterio, including the horizontal edges on pixel centers.
    tr = _TR20
    shape = (20, 20)
    poly = shapely.box(2.5, 2.5, 12.5, 12.5)
    px, py, ring_start, ring_stop, rev = _poly_kernel_arrays(poly, tr)
    nrings = len(ring_start)
    ny = shape[0]
    row_count = np.empty(ny + 1, dtype=np.int64)
    row_off = np.empty(ny + 1, dtype=np.int64)

    tiny = np.empty(1, dtype=np.int64)
    done = _rasterize_numba._fill_bucketed(
        np.zeros(shape, dtype="int32"),
        px,
        py,
        ring_start,
        ring_stop,
        rev,
        0,
        nrings,
        7,
        row_count,
        row_off,
        tiny,
    )
    assert not done

    big = np.empty(10_000, dtype=np.int64)
    out_bucketed = np.zeros(shape, dtype="int32")
    done = _rasterize_numba._fill_bucketed(
        out_bucketed,
        px,
        py,
        ring_start,
        ring_stop,
        rev,
        0,
        nrings,
        7,
        row_count,
        row_off,
        big,
    )
    assert done

    out_fallback = np.zeros(shape, dtype="int32")
    _rasterize_numba._fill_one_polygon(
        out_fallback, px, py, ring_start, ring_stop, rev, 0, nrings, 7
    )
    expected = _rio([poly], [7], shape, tr, False, "int32")
    np.testing.assert_array_equal(out_bucketed, expected)
    np.testing.assert_array_equal(out_fallback, expected)


# --- Missing geometry inside a numba run ---


def test_numba_run_drops_none_between_polygons(monkeypatch):
    # A None between two polygons falls into one numba run; it must be
    # dropped with its value rather than reaching to_ragged_array.
    p1 = shapely.box(1, 1, 3, 3)
    p2 = shapely.box(4, 4, 6, 6)
    geoms = np.array([p1, None, p2], dtype=object)
    values = np.array([10, 20, 30])
    shape = (8, 8)
    monkeypatch.setattr(rasterize, "RASTERIZE_BACKEND", "numba")
    got = rasterize._rio_rasterize_wrapper(
        shape, _TR, geoms, values, np.dtype("int32"), 0, False
    )
    monkeypatch.setattr(rasterize, "RASTERIZE_BACKEND", "rasterio")
    expected = rasterize._rio_rasterize_wrapper(
        shape,
        _TR,
        np.array([p1, p2], dtype=object),
        np.array([10, 30]),
        np.dtype("int32"),
        0,
        False,
    )
    np.testing.assert_array_equal(got, expected)
    assert set(np.unique(got).tolist()) == {0, 10, 30}


def test_numba_mask_run_drops_none_between_polygons(monkeypatch):
    p1 = shapely.box(1, 1, 3, 3)
    p2 = shapely.box(4, 4, 6, 6)
    geoms = np.array([p1, None, p2], dtype=object)
    shape = (8, 8)
    monkeypatch.setattr(rasterize, "RASTERIZE_BACKEND", "numba")
    got = rasterize._rio_mask(geoms, shape, _TR, False, False)
    monkeypatch.setattr(rasterize, "RASTERIZE_BACKEND", "rasterio")
    expected = rasterize._rio_mask(
        np.array([p1, p2], dtype=object), shape, _TR, False, False
    )
    np.testing.assert_array_equal(got, expected)
    assert int((got != 0).sum()) > 0


def test_numba_run_drops_none_adjacent_to_geometrycollection(monkeypatch):
    # A None next to a GeometryCollection joins the numba run of the
    # neighbouring polygon; the collection stays on the rasterio run.
    p1 = shapely.box(1, 1, 3, 3)
    gc = shapely.GeometryCollection(
        [shapely.Point(5, 5), shapely.box(4, 4, 7, 7)]
    )
    geoms = np.array([p1, None, gc], dtype=object)
    values = np.array([10, 20, 30])
    shape = (8, 8)
    monkeypatch.setattr(rasterize, "RASTERIZE_BACKEND", "numba")
    got = rasterize._rio_rasterize_wrapper(
        shape, _TR, geoms, values, np.dtype("int32"), 0, False
    )
    monkeypatch.setattr(rasterize, "RASTERIZE_BACKEND", "rasterio")
    expected = rasterize._rio_rasterize_wrapper(
        shape,
        _TR,
        np.array([p1, gc], dtype=object),
        np.array([10, 30]),
        np.dtype("int32"),
        0,
        False,
    )
    np.testing.assert_array_equal(got, expected)
