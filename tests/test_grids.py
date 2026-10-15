import numpy as np
import pytest
from affine import Affine
from odc.geo.geobox import GeoBox
from pyproj import Transformer

from raster_tools import _grids
from tests.utils import gdal_default_grid


def test_grids_close_different_crs():
    from odc.geo.geobox import GeoBox

    a = GeoBox.from_bbox((0, 0, 100, 100), crs=3857, resolution=1, tight=True)
    b = GeoBox.from_bbox((0, 0, 100, 100), crs=4326, resolution=1, tight=True)
    assert not _grids.grids_close(a, b)


def test_grids_close_different_shape():
    from odc.geo.geobox import GeoBox

    a = GeoBox.from_bbox((0, 0, 100, 100), crs=3857, resolution=1, tight=True)
    b = GeoBox.from_bbox((0, 0, 200, 100), crs=3857, resolution=1, tight=True)
    assert a.shape != b.shape
    assert not _grids.grids_close(a, b)


def test_grids_close_identical():
    from odc.geo.geobox import GeoBox

    a = GeoBox.from_bbox((0, 0, 100, 100), crs=3857, resolution=1, tight=True)
    assert _grids.grids_close(a, a)


@pytest.mark.parametrize(
    "cell_size, expected",
    [
        pytest.param(30.0, True, id="identical"),
        pytest.param(30.0 + 1e-9, True, id="float_noise"),
        # 0.02 m per cell over 1000 cells moves the last cell by 0.67 cells
        pytest.param(30.02, False, id="cell_size_drifts"),
    ],
)
def test_grids_close_judges_cell_size_drift_across_grid(cell_size, expected):
    from affine import Affine

    a = GeoBox((1000, 1000), Affine(30.0, 0, 0, 0, -30.0, 0), 5070)
    b = GeoBox((1000, 1000), Affine(cell_size, 0, 0, 0, -cell_size, 0), 5070)
    assert _grids.grids_close(a, b) is expected
    assert _grids.are_all_grids_same([a, b]) is expected


def test_grids_close_origin_offset():
    from affine import Affine

    a = GeoBox((10, 10), Affine(30.0, 0, 0, 0, -30.0, 0), 5070)
    b = GeoBox((10, 10), Affine(30.0, 0, 0.01, 0, -30.0, 0), 5070)
    c = GeoBox((10, 10), Affine(30.0, 0, 0, 0, -30.0, -1.0), 5070)
    assert _grids.grids_close(a, b)
    assert not _grids.grids_close(a, c)


def test_grids_close_rotated_grid_tolerates_float_noise():
    from affine import Affine

    # Rotated 90 degrees, so the a and e terms are zero
    a = GeoBox((10, 10), Affine(0, 30.0, 0, 30.0, 0, 0), 5070)
    b = GeoBox((10, 10), Affine(0, 30.0, 1e-9, 30.0, 0, 0), 5070)
    c = GeoBox((10, 10), Affine(0, 30.0, 15.0, 30.0, 0, 0), 5070)
    assert _grids.grids_close(a, b)
    assert not _grids.grids_close(a, c)


def test_are_all_grids_same_detects_mismatch():
    # Regression: a past bug made 2-grid comparisons always return True.
    from odc.geo.geobox import GeoBox

    a = GeoBox.from_bbox((0, 0, 100, 100), crs=3857, resolution=1, tight=True)
    b = GeoBox.from_bbox((0, 0, 200, 200), crs=3857, resolution=1, tight=True)
    assert _grids.are_all_grids_same([a, a])
    assert not _grids.are_all_grids_same([a, b])
    assert not _grids.are_all_grids_same([a, a, b])


def test_are_all_grids_same_tolerates_subpixel_noise():
    # Some published products carry sub-pixel FP noise in otherwise-shared
    # grids (observed up to ~1e-4 in CRS units). Strict GeoBox equality
    # would reject those and trigger needless reprojection passes, so
    # are_all_grids_same applies a sub-pixel tolerance.
    from affine import Affine
    from odc.geo.geobox import GeoBox

    # Realistic 30m grid (EPSG:5070 / NAD83 Albers).
    a = GeoBox.from_bbox(
        (500_000, 4_000_000, 500_000 + 30 * 100, 4_000_000 + 30 * 100),
        crs=5070,
        resolution=30,
        tight=True,
    )
    at = a.affine

    # 1e-4 m drift in origin -- realistic FP noise -- must still count
    # as "same".
    drifted = GeoBox(
        a.shape,
        Affine(at.a, at.b, at.c + 1e-4, at.d, at.e, at.f + 1e-4),
        a.crs,
    )
    assert _grids.are_all_grids_same([a, drifted])

    # Half-a-pixel shift must not be tolerated (that's a real offset).
    half_pixel = GeoBox(
        a.shape,
        Affine(at.a, at.b, at.c + 15, at.d, at.e, at.f),
        a.crs,
    )
    assert not _grids.are_all_grids_same([a, half_pixel])


def _bbox_grid(xmin, ymin, xmax, ymax, crs=5070, resolution=30):
    return GeoBox.from_bbox(
        (xmin, ymin, xmax, ymax),
        crs=crs,
        resolution=resolution,
        tight=True,
    )


def test_get_grid_bounds_matches_input_bbox():
    a = _bbox_grid(0, 0, 3000, 3000)
    assert _grids.get_grid_bounds(a) == pytest.approx((0, 0, 3000, 3000))


def test_get_grid_bbox_returns_polygon_of_expected_bounds():
    a = _bbox_grid(0, 0, 3000, 3000)
    bbox = _grids.get_grid_bbox(a)
    assert bbox.bounds == pytest.approx((0, 0, 3000, 3000))


def test_reproject_grid_changes_crs():
    a = _bbox_grid(500_000, 4_000_000, 500_000 + 3000, 4_000_000 + 3000)
    reprojected = _grids.reproject_grid(a, 4326)
    assert reprojected.crs == 4326


def test_combine_grids_invalid_how_raises():
    a = _bbox_grid(0, 0, 3000, 3000)
    with pytest.raises(ValueError, match="how must be"):
        _grids.combine_grids([a], how="bogus")


def test_combine_grids_identical_grids_returns_input():
    a = _bbox_grid(0, 0, 3000, 3000)
    result = _grids.combine_grids([a, a])
    assert _grids.grids_close(result, a)


def test_combine_grids_identical_grids_with_dst_crs_reprojects():
    a = _bbox_grid(500_000, 4_000_000, 500_000 + 3000, 4_000_000 + 3000)
    result = _grids.combine_grids([a, a], dst_crs=4326)
    assert result.crs == 4326


def test_combine_grids_union_covers_both_inputs():
    a = _bbox_grid(0, 0, 3000, 3000)
    b = _bbox_grid(3000, 3000, 6000, 6000)
    result = _grids.combine_grids([a, b], how="union")
    xmin, ymin, xmax, ymax = _grids.get_grid_bounds(result)
    assert xmin == pytest.approx(0)
    assert ymin == pytest.approx(0)
    assert xmax == pytest.approx(6000)
    assert ymax == pytest.approx(6000)


def test_combine_grids_intersection_of_overlap():
    a = _bbox_grid(0, 0, 3000, 3000)
    b = _bbox_grid(1500, 1500, 4500, 4500)
    result = _grids.combine_grids([a, b], how="intersection")
    xmin, ymin, xmax, ymax = _grids.get_grid_bounds(result)
    assert xmin == pytest.approx(1500)
    assert ymin == pytest.approx(1500)
    assert xmax == pytest.approx(3000)
    assert ymax == pytest.approx(3000)


def test_combine_grids_intersection_empty_raises():
    a = _bbox_grid(0, 0, 3000, 3000)
    b = _bbox_grid(10_000, 10_000, 13_000, 13_000)
    with pytest.raises(ValueError, match="intersection.*empty"):
        _grids.combine_grids([a, b], how="intersection")


def test_combine_grids_intersection_of_touching_grids_raises():
    # Touching grids intersect in a line, which holds no cells
    a = _bbox_grid(0, 0, 3000, 3000)
    b = _bbox_grid(3000, 0, 6000, 3000)
    with pytest.raises(ValueError, match="intersection.*empty"):
        _grids.combine_grids([a, b], how="intersection")
    corner = _bbox_grid(3000, 3000, 6000, 6000)
    with pytest.raises(ValueError, match="intersection.*empty"):
        _grids.combine_grids([a, corner], how="intersection")


def _unaligned_grid():
    # 30 m cells whose origin is not a multiple of the cell size
    return GeoBox((10, 10), Affine(30, 0, 1007, 0, -30, 2013), "EPSG:5070")


def _unaligned_grid_in_utm():
    # GDAL's default grid for _unaligned_grid in EPSG:32613
    return gdal_default_grid(_unaligned_grid(), "EPSG:32613")


def test_reproject_grid_gives_gdal_default_grid():
    grid = _unaligned_grid()
    result = _grids.reproject_grid(grid, "EPSG:32613")
    assert result == _unaligned_grid_in_utm()


def test_reproject_grid_with_resolution_covers_gdal_default_extent():
    grid = _unaligned_grid()
    default = _unaligned_grid_in_utm()
    result = _grids.reproject_grid(grid, "EPSG:32613", resolution=20)
    assert result == gdal_default_grid(grid, "EPSG:32613", resolution=20)
    assert result.crs == "EPSG:32613"
    assert result.resolution.xy == (20, -20)
    # The top-left corner is kept and the size rounded up
    xmin, ymin, xmax, ymax = result.boundingbox
    dxmin, dymin, dxmax, dymax = default.boundingbox
    assert (xmin, ymax) == pytest.approx((dxmin, dymax))
    assert xmax >= dxmax
    assert ymin <= dymin


def test_reproject_grid_same_crs_returns_grid():
    grid = _unaligned_grid()
    assert _grids.reproject_grid(grid, grid.crs) == grid


def test_reproject_grid_same_crs_with_resolution_keeps_origin():
    grid = _unaligned_grid()
    result = _grids.reproject_grid(grid, grid.crs, resolution=20)
    assert result.crs == grid.crs
    assert result.resolution.xy == (20, -20)
    assert (result.affine.c, result.affine.f) == (1007, 2013)


@pytest.mark.parametrize("resolution", [0, -30])
def test_reproject_grid_nonpositive_resolution_raises(resolution):
    with pytest.raises(ValueError, match="positive"):
        _grids.reproject_grid(
            _unaligned_grid(), "EPSG:32613", resolution=resolution
        )


def test_combine_grids_union_across_crs_takes_gdal_default_extent():
    grid = _unaligned_grid()
    expected = _unaligned_grid_in_utm()
    result = _grids.combine_grids([expected, grid], how="union")
    assert _grids.grids_close(result, expected)


def test_combine_grids_intersection_across_crs_takes_gdal_default_extent():
    grid = _unaligned_grid()
    expected = _unaligned_grid_in_utm()
    xmin, ymin, xmax, ymax = expected.extent.boundingbox
    # A larger grid with the same cell size that contains the other grid
    cover = GeoBox.from_bbox(
        (xmin - 1000, ymin - 1000, xmax + 1000, ymax + 1000),
        crs=32613,
        resolution=expected.resolution,
        tight=True,
    )
    result = _grids.combine_grids([cover, grid], how="intersection")
    assert _grids.grids_close(result, expected)


def _polar_grid(half_width, cell_size, crs="EPSG:3413"):
    # A square polar stereographic grid centered on the pole
    n = round(2 * half_width / cell_size)
    return GeoBox(
        (n, n),
        Affine(cell_size, 0, -half_width, 0, -cell_size, half_width),
        crs,
    )


def _antimeridian_grid():
    # 200 x 100 km in UTM zone 60N, crossing longitude 180 near 45N
    return GeoBox(
        (100, 200), Affine(1000, 0, 600_000, 0, -1000, 5_000_000), 32660
    )


def _edge_lonlat(grid, npoints=400):
    # Longitudes and latitudes of points along the edges of `grid`
    height, width = grid.shape
    t = np.linspace(0, 1, npoints)
    cols = np.concatenate([t * width, t * width, 0 * t, 0 * t + width])
    rows = np.concatenate([0 * t, 0 * t + height, t * height, t * height])
    x, y = grid.affine * (cols, rows)
    to_lonlat = Transformer.from_crs(grid.crs.proj, 4326, always_xy=True)
    return to_lonlat.transform(x, y)


def _assert_usable_lonlat_grid(result):
    xmin, ymin, xmax, ymax = result.boundingbox
    assert result.crs == "EPSG:4326"
    assert np.all(np.isfinite([xmin, ymin, xmax, ymax]))
    assert -90 - 1e-9 <= ymin < ymax <= 90 + 1e-9
    ny, nx = result.shape
    assert nx > 1
    assert ny >= 1


@pytest.mark.parametrize(
    "half_width,cell_size", [(3_000_000, 25_000), (100_000, 2000)]
)
def test_reproject_grid_containing_pole_to_lonlat(half_width, cell_size):
    # https://github.com/UM-RMRS/raster_tools/issues/78
    grid = _polar_grid(half_width, cell_size)
    result = _grids.reproject_grid(grid, "EPSG:4326")
    assert result == gdal_default_grid(grid, "EPSG:4326")
    _assert_usable_lonlat_grid(result)
    xmin, _, xmax, ymax = result.boundingbox
    assert ymax == pytest.approx(90)
    assert xmin == pytest.approx(-180)
    assert xmax == pytest.approx(180)


@pytest.mark.parametrize("resolution", [None, 0.7, 7])
@pytest.mark.parametrize(
    "half_width,cell_size", [(3_000_000, 25_000), (100_000, 2000)]
)
def test_reproject_grid_containing_south_pole_to_lonlat(
    half_width, cell_size, resolution
):
    # For the small grid, GDAL's default grid is one row of cells with zero
    # height
    grid = _polar_grid(half_width, cell_size, crs="EPSG:3031")
    _, lat = _edge_lonlat(grid)
    result = _grids.reproject_grid(grid, "EPSG:4326", resolution=resolution)
    _assert_usable_lonlat_grid(result)
    assert -result.resolution.y > 0
    if resolution is not None:
        assert result.resolution.xy == pytest.approx((resolution, -resolution))
    xmin, ymin, xmax, ymax = result.boundingbox
    assert xmin == pytest.approx(-180)
    assert xmax >= 180 - 1e-9
    assert ymax >= lat.max() - 1e-9
    # GDAL's default grid for the large grid stops short of the pole
    assert ymin < -89


def test_reproject_grid_across_antimeridian_covers_source_latitudes():
    grid = _antimeridian_grid()
    _, lat = _edge_lonlat(grid)
    result = _grids.reproject_grid(grid, "EPSG:4326")
    assert result == gdal_default_grid(grid, "EPSG:4326")
    _assert_usable_lonlat_grid(result)
    _, ymin, _, ymax = result.boundingbox
    assert ymin <= lat.min()
    assert ymax >= lat.max() - 1e-9


def test_reproject_grid_wide_source_keeps_comparable_cell_count():
    # A thin strip from longitude -178.8 to 178.8 along the equator, which
    # does not cross the antimeridian
    grid = GeoBox(
        (20, 2000),
        Affine(19_900, 0, -19_900_000, 0, -19_900, 100_000),
        "EPSG:3857",
    )
    result = _grids.reproject_grid(grid, "EPSG:4326")
    _assert_usable_lonlat_grid(result)
    ny, nx = result.shape
    src_cells = grid.shape[0] * grid.shape[1]
    assert src_cells / 4 < nx * ny < src_cells * 4


def test_reproject_grid_with_edges_off_globe_to_lonlat():
    # The whole northern hemisphere, with every corner off the globe
    grid = GeoBox(
        (100, 100),
        Affine(130_000, 0, -6.5e6, 0, -130_000, 6.5e6),
        "+proj=ortho +lat_0=90 +lon_0=0 +datum=WGS84",
    )
    result = _grids.reproject_grid(grid, "EPSG:4326")
    _assert_usable_lonlat_grid(result)
    assert result.shape[0] > 1


def _polar_grids_of_two_sizes():
    big = GeoBox.from_bbox((-3e6, -3e6, 3e6, 3e6), "EPSG:3413", resolution=1e4)
    small = GeoBox.from_bbox(
        (-1e5, -1e5, 1e5, 1e5), "EPSG:3413", resolution=1000
    )
    return big, small


@pytest.mark.parametrize("how", ["union", "intersection"])
def test_combine_grids_to_other_crs_sizes_cells_in_its_units(how):
    # Without a resolution, the cells take the size of the first grid's
    # cells in GDAL's default grid for it in dst_crs, not the size of its
    # cells in meters
    big, small = _polar_grids_of_two_sizes()
    result = _grids.combine_grids([big, small], how=how, dst_crs="EPSG:4326")
    _assert_usable_lonlat_grid(result)
    cell = abs(gdal_default_grid(big, "EPSG:4326").resolution.x)
    assert result.resolution.xy == pytest.approx((cell, -cell))
    xmin, ymin, xmax, ymax = result.boundingbox
    assert ymax == pytest.approx(90)
    assert xmin == pytest.approx(-180)
    assert xmax >= 180
    # The extent is the union or intersection of the GDAL default extents
    extents = [
        gdal_default_grid(g, "EPSG:4326").boundingbox for g in (big, small)
    ]
    expected_ymin = min(e[1] for e in extents)
    if how == "intersection":
        expected_ymin = max(e[1] for e in extents)
    assert expected_ymin - cell < ymin <= expected_ymin


@pytest.mark.parametrize("resolution", [None, 0.7, 7])
def test_combine_grids_containing_south_pole_to_lonlat(resolution):
    big = _polar_grid(3_000_000, 25_000, crs="EPSG:3031")
    small = _polar_grid(100_000, 2000, crs="EPSG:3031")
    result = _grids.combine_grids(
        [big, small], dst_crs="EPSG:4326", resolution=resolution
    )
    _assert_usable_lonlat_grid(result)
    if resolution is not None:
        assert result.resolution.xy == pytest.approx((resolution, -resolution))
    _, lat = _edge_lonlat(big)
    _, ymin, _, ymax = result.boundingbox
    assert ymax >= lat.max() - 1e-9
    assert ymin < -89


@pytest.mark.parametrize("resolution", [0.7, 7, 13, 50])
def test_reproject_grid_with_resolution_covers_gdal_extent(resolution):
    # A global Mollweide grid, whose GDAL default extent nearly reaches both
    # poles. Whole cells of a resolution that does not divide 180 degrees
    # cannot fit between the poles, so the rows are shortened rather than
    # cut off.
    grid = GeoBox.from_bbox(
        (-18_040_095, -9_020_047, 18_040_095, 9_020_047),
        "ESRI:54009",
        resolution=50_000,
    )
    gdal_bounds = gdal_default_grid(grid, "EPSG:4326").boundingbox
    result = _grids.reproject_grid(grid, "EPSG:4326", resolution=resolution)
    xmin, ymin, xmax, ymax = result.boundingbox
    _assert_usable_lonlat_grid(result)
    assert result.resolution.x == pytest.approx(resolution)
    assert 0 < -result.resolution.y <= resolution + 1e-9
    assert xmin == pytest.approx(gdal_bounds[0])
    assert xmax >= gdal_bounds[2] - 1e-9
    assert ymin <= gdal_bounds[1] + 1e-9
    assert ymax >= gdal_bounds[3] - 1e-9


@pytest.mark.parametrize("resolution", [None, 0.1])
@pytest.mark.parametrize("crs", ["EPSG:3413", "EPSG:3031"])
@pytest.mark.parametrize("half_width,cell_size", [(50_000, 1000), (5000, 100)])
def test_reproject_grid_small_grid_containing_pole_has_rows(
    half_width, cell_size, crs, resolution
):
    # GDAL suggests a grid with no rows for these
    grid = _polar_grid(half_width, cell_size, crs=crs)
    _, lat = _edge_lonlat(grid)
    result = _grids.reproject_grid(grid, "EPSG:4326", resolution=resolution)
    _assert_usable_lonlat_grid(result)
    assert result.shape[0] >= 1
    _, ymin, _, ymax = result.boundingbox
    assert ymin <= lat.min() + 1e-9
    assert ymax >= lat.max() - 1e-9


def test_combine_grids_same_lonlat_crs_keeps_grid_past_poles():
    # A global grid with cell centers on the poles is not moved or cut when
    # nothing is reprojected
    world = GeoBox((181, 360), Affine(1, 0, -180.5, 0, -1, 90.5), 4326)
    part = GeoBox((10, 10), Affine(1, 0, 0.5, 0, -1, 0.5), 4326)
    result = _grids.combine_grids([world, part], how="union")
    assert result == world


def test_combine_grids_same_crs_sizes_grids_to_first_cell_size():
    # The second grid is resized to 30 m cells, keeping its origin, before
    # the union is taken: its extent becomes 195..345 by -25..125.
    first = GeoBox((7, 7), Affine(30, 0, 0, 0, -30, 210), 5070)
    second = GeoBox((5, 5), Affine(25, 0, 195, 0, -25, 125), 5070)
    result = _grids.combine_grids([first, second], how="union")
    assert result == GeoBox((8, 12), Affine(30, 0, 0, 0, -30, 210), 5070)
