import pytest
from affine import Affine
from odc.geo.geobox import GeoBox

from raster_tools import _grids


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
    # Smallest 30 m grid covering _unaligned_grid's footprint in EPSG:32613.
    # The footprint spans about 10.5 x 10.9 cells.
    return GeoBox(
        (11, 11),
        Affine(30, 0, 1425861.259191, 0, -30, 2574214.656564),
        "EPSG:32613",
    )


def test_reproject_grid_covers_footprint_without_padding():
    grid = _unaligned_grid()
    result = _grids.reproject_grid(grid, "EPSG:32613")
    # Snapping the origin to a multiple of the cell size would give 14 x 14,
    # and padding the footprint as GeoBox.to_crs does would give 13 x 13.
    assert _grids.grids_close(result, _unaligned_grid_in_utm())


def test_reproject_grid_with_resolution_keeps_tight_origin():
    grid = _unaligned_grid()
    expected = _unaligned_grid_in_utm()
    result = _grids.reproject_grid(grid, "EPSG:32613", resolution=20)
    assert result.crs == "EPSG:32613"
    assert result.resolution.xy == (20, -20)
    assert result.shape == (17, 16)
    assert result.affine.c == pytest.approx(expected.affine.c)
    assert result.affine.f == pytest.approx(expected.affine.f)


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


def test_combine_grids_union_across_crs_is_tight():
    grid = _unaligned_grid()
    expected = _unaligned_grid_in_utm()
    result = _grids.combine_grids([expected, grid], how="union")
    assert _grids.grids_close(result, expected)


def test_combine_grids_intersection_across_crs_is_tight():
    grid = _unaligned_grid()
    expected = _unaligned_grid_in_utm()
    xmin, ymin, xmax, ymax = expected.extent.boundingbox
    # A larger grid on its own lattice that contains the other grid
    cover = _bbox_grid(
        xmin - 1000, ymin - 1000, xmax + 1000, ymax + 1000, crs=32613
    )
    result = _grids.combine_grids([cover, grid], how="intersection")
    assert _grids.grids_close(result, expected)
