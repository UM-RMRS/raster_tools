import os

import numpy as np
import pytest
from affine import Affine
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from raster_tools import Raster, data_to_raster, distance
from raster_tools.distance.cost_distance import (
    _get_strides,
    _normalize_raster_data,
    cost_distance_analysis_numpy,
)
from tests import testdata
from tests.utils import assert_rasters_equal, assert_valid_raster

# Example taken from ESRI docs
SOURCES = np.array(
    [
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0],
    ]
)
SOURCES_IDXS = np.argwhere(SOURCES != 0)
COST_SURF = np.array(
    [
        [1, 3, 4, 4, 3, 2],
        [7, 3, 2, 6, 4, 6],
        [5, 8, 7, 5, 6, 6],
        [1, 4, 5, -1, 5, 1],
        [4, 7, 5, -1, 2, 6],
        [1, 2, 2, 1, 3, 4],
    ]
)
# Cost dist truths
CD_TRUTH_SCALE_1 = np.array(
    [
        [
            [2.0, 0.0, 0.0, 4.0, 7.5, 10.0],
            [6.0, 2.5, 0.0, 4.0, 9.0, 13.86396103],
            [8.0, 7.07106781, 4.5, 4.94974747, 10.44974747, 12.74264069],
            [5.0, 7.5, 10.5, -1.0, 10.62132034, 9.24264069],
            [2.5, 5.65685425, 6.44974747, -1.0, 7.12132034, 11.12132034],
            [0.0, 1.5, 3.5, 5.0, 7.0, 10.5],
        ]
    ]
)
CD_TRUTH_SCALE_5 = np.array(
    [
        [
            [10.0, 0.0, 0.0, 20.0, 37.5, 50.0],
            [30.0, 12.5, 0.0, 20.0, 45.0, 69.31980515],
            [40.0, 35.35533906, 22.5, 24.74873734, 52.24873734, 63.71320344],
            [25.0, 37.5, 52.5, -1.0, 53.10660172, 46.21320344],
            [12.5, 28.28427125, 32.24873734, -1.0, 35.60660172, 55.60660172],
            [0.0, 7.5, 17.5, 25.0, 35.0, 52.5],
        ]
    ]
)
# Traceback Truths
TR_TRUTH_SCALE_1 = np.array(
    [
        [
            [1, 0, 0, 5, 5, 5],
            [7, 1, 0, 5, 5, 6],
            [3, 8, 7, 6, 5, 3],
            [3, 5, 7, -1, 3, 4],
            [3, 4, 4, -1, 4, 5],
            [0, 5, 5, 5, 5, 5],
        ]
    ],
)
# Allocation truths
AL_TRUTH_SCALE_1 = np.array(
    [
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [2, 1, 1, 1, 1, 2],
            [2, 2, 1, 0, 2, 2],
            [2, 2, 2, 0, 2, 2],
            [2, 2, 2, 2, 2, 2],
        ]
    ]
)
AL_TRUTH_IDXS_SCALE_1 = np.array(
    [
        [
            [0, 0, 1, 1, 1, 1],
            [0, 2, 2, 2, 2, 1],
            [3, 2, 2, 2, 2, 3],
            [3, 3, 2, -1, 3, 3],
            [3, 3, 3, -1, 3, 3],
            [3, 3, 3, 3, 3, 3],
        ]
    ]
)


# ---------------------------------------------------------------------------
# Public Raster-level API: cost_distance_analysis
# ---------------------------------------------------------------------------


@pytest.fixture
def costs():
    # Fresh per test -- the underlying data can be mutated by callers.
    return Raster(COST_SURF).set_null_value(-1)


def _make_sources(kind):
    if kind == "raster":
        return Raster(SOURCES).set_null_value(0)
    return SOURCES_IDXS


@pytest.mark.parametrize("kind", ["raster", "index"])
def test_cost_distance_analysis_cost_distance(costs, kind):
    cd, _, _ = distance.cost_distance_analysis(costs, _make_sources(kind))
    assert np.allclose(cd.to_numpy(), CD_TRUTH_SCALE_1, equal_nan=True)
    assert cd.dtype == np.dtype(np.float64)
    assert cd.null_value == -1
    assert cd._masked


@pytest.mark.parametrize("kind", ["raster", "index"])
def test_cost_distance_analysis_traceback(costs, kind):
    # The traceback direction depends only on source *locations*, which are
    # identical for both source kinds, so the truth is shared.
    _, tr, _ = distance.cost_distance_analysis(costs, _make_sources(kind))
    assert np.allclose(tr.to_numpy(), TR_TRUTH_SCALE_1, equal_nan=True)
    assert tr.dtype == np.dtype(np.int8)
    assert tr.null_value == -1


@pytest.mark.parametrize(
    "kind,truth,null",
    [
        ("raster", AL_TRUTH_SCALE_1, 0),
        ("index", AL_TRUTH_IDXS_SCALE_1, -1),
    ],
)
def test_cost_distance_analysis_allocation(costs, kind, truth, null):
    _, _, al = distance.cost_distance_analysis(costs, _make_sources(kind))
    assert np.allclose(al.to_numpy(), truth, equal_nan=True)
    assert al.dtype == np.dtype(np.int64)
    assert al.null_value == null


@pytest.mark.parametrize("kind", ["raster", "index"])
def test_cost_distance_analysis_outputs_valid(costs, kind):
    cd, tr, al = distance.cost_distance_analysis(costs, _make_sources(kind))
    assert_valid_raster(cd)
    assert_valid_raster(tr)
    assert_valid_raster(al)


def _error_case(name):
    dem_path = os.path.join(testdata.RASTER_DIR, "dem_small.tif")
    mb_path = os.path.join(testdata.RASTER_DIR, "multiband_small.tif")
    cs = Raster(COST_SURF).set_null_value(-1)
    srcs = Raster(SOURCES).set_null_value(0)
    dem_int_no_null = testdata.raster.dem_small.astype(
        int, False
    ).set_null_value(None)
    cases = {
        # (costs, sources, elevation, exception, match)
        "multiband_costs_path": (mb_path, srcs, None, ValueError, "multiband"),
        "sources_shape_mismatch": (
            cs,
            dem_path,
            None,
            ValueError,
            "shapes must match",
        ),
        "sources_not_int": (
            dem_path,
            dem_path,
            None,
            TypeError,
            "integer type",
        ),
        "sources_no_null": (
            dem_path,
            dem_int_no_null,
            None,
            ValueError,
            "null value",
        ),
        "sources_bad_shape": (
            cs,
            np.zeros((5, 3), dtype=int),
            None,
            ValueError,
            r"\(M, 2\)",
        ),
        "sources_duplicates": (
            cs,
            np.zeros((5, 2), dtype=int),
            None,
            ValueError,
            "duplicates",
        ),
        # The multiband checks must also apply to already-built Raster
        # objects, not just paths; the remaining cases cover previously
        # untested error branches.
        "multiband_costs_object": (
            Raster(mb_path),
            srcs,
            None,
            ValueError,
            "multiband",
        ),
        "multiband_elevation_object": (
            cs,
            srcs,
            Raster(mb_path),
            ValueError,
            "multiband",
        ),
        "elevation_shape_mismatch": (
            cs,
            srcs,
            dem_path,
            ValueError,
            "same shape",
        ),
        "sources_unintelligible": (
            cs,
            None,
            None,
            ValueError,
            "Could not understand",
        ),
    }
    return cases[name]


@pytest.mark.parametrize(
    "name",
    [
        "multiband_costs_path",
        "sources_shape_mismatch",
        "sources_not_int",
        "sources_no_null",
        "sources_bad_shape",
        "sources_duplicates",
        "multiband_costs_object",
        "multiband_elevation_object",
        "elevation_shape_mismatch",
        "sources_unintelligible",
    ],
)
def test_cost_distance_analysis_errors(name):
    cs, srcs, elev, exc, match = _error_case(name)
    with pytest.raises(exc, match=match):
        distance.cost_distance_analysis(cs, srcs, elev)


def test_cost_distance_analysis_scale_isotropic():
    # 5m isotropic resolution via a real affine (no private _ds surgery).
    cs = data_to_raster(
        COST_SURF[None],
        affine=Affine(5, 0, 0, 0, -5, 0),
        crs="EPSG:3857",
    ).set_null_value(-1)
    srcs = Raster(SOURCES).set_null_value(0)
    cd, _, _ = distance.cost_distance_analysis(cs, srcs)
    assert np.allclose(cd.to_numpy(), CD_TRUTH_SCALE_5, equal_nan=True)


def test_cost_distance_sources_float_error_message():
    dem_path = "tests/data/raster/dem_small.tif"
    with pytest.raises(TypeError) as exc:
        distance.cost_distance_analysis(dem_path, dem_path)
    msg = str(exc.value)
    assert "integer type" in msg
    assert str(Raster(dem_path).dtype) in msg
    assert "GeoTIFF" in msg
    assert "astype" in msg


def test_cost_distance_analysis_crs():
    rs = testdata.raster.dem_small
    srcs = np.array([[1, 1], [20, 30]])
    cd, tr, al = distance.cost_distance_analysis(rs, srcs)
    assert_valid_raster(cd)
    assert_valid_raster(tr)
    assert_valid_raster(al)
    assert cd.crs == rs.crs
    assert tr.crs == rs.crs
    assert al.crs == rs.crs


def test_cost_distance_crs_none():
    # Complements the crs-preserved test: no CRS in, no CRS out.
    cs = Raster(COST_SURF).set_null_value(-1)
    assert cs.crs is None
    cd, tr, al = distance.cost_distance_analysis(cs, _make_sources("raster"))
    assert cd.crs is None
    assert tr.crs is None
    assert al.crs is None


# ---------------------------------------------------------------------------
# Regression tests: source index bounds and x/y scaling order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src",
    [[[-1, 0]], [[0, -1]], [[6, 0]], [[0, 6]]],
    ids=["neg_row", "neg_col", "row_too_large", "col_too_large"],
)
def test_cost_distance_source_index_out_of_bounds(costs, src):
    # Negative indices used to wrap silently and too-large indices raised
    # a bare IndexError. Both must raise a clear ValueError.
    with pytest.raises(ValueError, match="out of bounds"):
        distance.cost_distance_analysis(costs, np.array(src))


def test_cost_distance_nonsquare_resolution_axis_mapping():
    # Raster.resolution is (xres, yres) but the core scales [row, col]
    # moves, so the pair must be reversed before it is passed down. With
    # xres=3, yres=7 a horizontal move must cost 3 and a vertical move 7.
    cs = data_to_raster(
        np.ones((1, 3, 3)),
        affine=Affine(3, 0, 0, 0, -7, 0),
        crs="EPSG:3857",
    )
    cd, _, _ = distance.cost_distance_analysis(cs, np.array([[0, 0]]))
    data = cd.to_numpy()[0]
    assert data[0, 1] == 3.0
    assert data[1, 0] == 7.0


def test_cost_distance_nonsquare_matches_numpy():
    # A non-square-resolution raster must match the oracle-verified numpy
    # function called with scaling in (yres, xres) order.
    rng = np.random.default_rng(0)
    arr = rng.uniform(0.5, 10.0, (8, 8))
    xres, yres = 3.0, 7.0
    cs = data_to_raster(
        arr[None],
        affine=Affine(xres, 0, 0, 0, -yres, 0),
        crs="EPSG:3857",
    )
    src_idxs = np.array([[0, 0], [7, 7]])
    cd, _, _ = distance.cost_distance_analysis(cs, src_idxs)
    sarr = np.full((8, 8), -1, dtype=np.int64)
    sarr[0, 0] = 0
    sarr[7, 7] = 1
    cd_ref, _, _ = cost_distance_analysis_numpy(
        arr, sarr, -1, scaling=(yres, xres)
    )
    reached = np.isfinite(cd_ref)
    data = cd.to_numpy()[0]
    assert np.allclose(data[reached], cd_ref[reached], rtol=1e-12)


# ---------------------------------------------------------------------------
# Wrappers, elevation glue, path/chunk/dtype inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["raster", "index"])
def test_cda_wrappers_match_full(costs, kind):
    cd, tr, al = distance.cost_distance_analysis(costs, _make_sources(kind))
    assert_rasters_equal(
        distance.cda_cost_distance(costs, _make_sources(kind)), cd
    )
    assert_rasters_equal(
        distance.cda_traceback(costs, _make_sources(kind)), tr
    )
    assert_rasters_equal(
        distance.cda_allocation(costs, _make_sources(kind)), al
    )


def _numpy_reference(costs_rs, sources_arr, snull, elevation_rs=None):
    # Glue reference: reuses the same private normalization the public API
    # uses, so these are band-handling / null-fill / mask / scaling-order glue
    # checks. Elevation *correctness* is carried by the oracle path-walk.
    data = _normalize_raster_data(costs_rs)
    scaling = np.abs(costs_rs.resolution)[::-1]
    if elevation_rs is not None:
        edata = _normalize_raster_data(elevation_rs, -9999)
        return cost_distance_analysis_numpy(
            data,
            sources_arr,
            snull,
            elevation=edata,
            elevation_null_value=-9999,
            scaling=scaling,
        )
    return cost_distance_analysis_numpy(
        data, sources_arr, snull, scaling=scaling
    )


def _raster_source_array(srcs_rs):
    return srcs_rs.to_numpy()[0].astype(np.int64), srcs_rs.null_value


def test_cost_distance_elevation_happy_path():
    # 6x6 with a no-null float64 elevation. float64 avoids compiling a fresh
    # numba kernel signature.
    cs = Raster(COST_SURF).set_null_value(-1)
    elev = (np.arange(36, dtype=np.float64).reshape(6, 6)) * 0.5
    elevation = Raster(elev[None])
    srcs = Raster(SOURCES).set_null_value(0)
    cd, tr, al = distance.cost_distance_analysis(cs, srcs, elevation)
    sarr, snull = _raster_source_array(srcs)
    ref_cd, ref_tr, ref_al = _numpy_reference(cs, sarr, snull, elevation)
    reached = np.isfinite(ref_cd)
    data = cd.to_numpy()[0]
    assert np.allclose(data[reached], ref_cd[reached], rtol=1e-9)
    # traceback is the numpy result shifted by +1 (ESRI 0-8 scale) everywhere.
    assert np.array_equal(tr.to_numpy()[0], ref_tr + 1)
    assert np.array_equal(al.to_numpy()[0][reached], ref_al[reached])


def test_cost_distance_elevation_masked_cells():
    # A nan-null elevation cell becomes a barrier: unreached-filled but NOT
    # masked at the Raster level (the output mask tracks costs, not elevation).
    cs = Raster(COST_SURF).set_null_value(-1)
    elev = np.arange(36, dtype=np.float64).reshape(6, 6)
    elev[2, 4] = np.nan
    elevation = Raster(elev[None]).set_null_value(np.nan)
    srcs = Raster(SOURCES).set_null_value(0)
    cd, _, _ = distance.cost_distance_analysis(cs, srcs, elevation)
    sarr, snull = _raster_source_array(srcs)
    ref_cd, _, _ = _numpy_reference(cs, sarr, snull, elevation)
    reached = np.isfinite(ref_cd)
    data = cd.to_numpy()[0]
    mask = cd.mask.compute()[0]
    assert np.allclose(data[reached], ref_cd[reached], rtol=1e-9)
    assert not reached[2, 4]
    assert data[2, 4] == cd.null_value
    assert not mask[2, 4]


def test_cost_distance_accepts_file_paths(tmp_path):
    # Save as float64 costs/elevation and int32 sources so the GeoTIFF
    # round-trip preserves dtypes (int64 would reload as float64 and fail the
    # integer check).
    cs = data_to_raster(
        COST_SURF.astype(np.float64)[None],
        affine=Affine(1, 0, 0, 0, -1, 6),
        crs="EPSG:3857",
    ).set_null_value(-1)
    srcs = data_to_raster(
        SOURCES.astype(np.int32)[None],
        affine=Affine(1, 0, 0, 0, -1, 6),
        crs="EPSG:3857",
    ).set_null_value(0)
    elev = np.arange(36, dtype=np.float64).reshape(6, 6)
    elevation = data_to_raster(
        elev[None], affine=Affine(1, 0, 0, 0, -1, 6), crs="EPSG:3857"
    )
    cpath = str(tmp_path / "costs.tif")
    spath = str(tmp_path / "sources.tif")
    epath = str(tmp_path / "elev.tif")
    cs.save(cpath)
    srcs.save(spath)
    elevation.save(epath)
    cd_o, tr_o, al_o = distance.cost_distance_analysis(cs, srcs, elevation)
    cd_p, tr_p, al_p = distance.cost_distance_analysis(cpath, spath, epath)
    assert_rasters_equal(cd_p, cd_o, check_chunks=False)
    assert_rasters_equal(tr_p, tr_o, check_chunks=False)
    assert_rasters_equal(al_p, al_o, check_chunks=False)


def test_cost_distance_nan_null_costs():
    # nan-null costs vs the -1-barrier version on a fully connected grid (every
    # valid COST_SURF cell is reached). Reached cells must match; the output
    # null value must be nan.
    m1 = Raster(COST_SURF.astype(np.float64)).set_null_value(-1)
    nan_arr = COST_SURF.astype(np.float64)
    nan_arr[nan_arr == -1] = np.nan
    nan_costs = Raster(nan_arr[None]).set_null_value(np.nan)
    cd_m1, _, _ = distance.cost_distance_analysis(
        m1, Raster(SOURCES).set_null_value(0)
    )
    cd_nan, _, _ = distance.cost_distance_analysis(
        nan_costs, Raster(SOURCES).set_null_value(0)
    )
    mask = cd_m1.mask.compute()[0]
    reached = ~mask
    assert np.allclose(
        cd_m1.to_numpy()[0][reached], cd_nan.to_numpy()[0][reached]
    )
    assert np.isnan(cd_nan.null_value)


def test_cost_distance_chunked_costs_input():
    cs = Raster(COST_SURF).set_null_value(-1)
    cd_s, tr_s, al_s = distance.cost_distance_analysis(
        cs, Raster(SOURCES).set_null_value(0)
    )
    cs_c = Raster(COST_SURF).set_null_value(-1).chunk((1, 2, 2))
    cd_c, tr_c, al_c = distance.cost_distance_analysis(
        cs_c, Raster(SOURCES).set_null_value(0)
    )
    # The core collapses to numpy internally, so a chunked input must give the
    # same values. (The output data is single-chunk while its mask inherits
    # the input chunking, so both sides are rechunked to a single block before
    # comparing.)
    single = (1, -1, -1)
    assert_rasters_equal(cd_c.chunk(single), cd_s.chunk(single))
    assert_rasters_equal(tr_c.chunk(single), tr_s.chunk(single))
    assert_rasters_equal(al_c.chunk(single), al_s.chunk(single))


def test_cost_distance_all_null_sources():
    cs = Raster(COST_SURF).set_null_value(-1)
    srcs = Raster(np.zeros((6, 6), dtype=np.int64)).set_null_value(0)
    cd, tr, al = distance.cost_distance_analysis(cs, srcs)
    # ESRI unreached traceback is -1; allocation is the source null everywhere.
    assert np.all(tr.to_numpy()[0] == -1)
    assert np.all(al.to_numpy()[0] == 0)
    mask = cd.mask.compute()[0]
    data = cd.to_numpy()[0]
    assert np.all(data[~mask] == cd.null_value)


def test_cost_distance_unreached_valid_cell_filled_not_masked():
    # A full-height barrier column isolates the right side of the grid.
    arr = np.ones((3, 5), dtype=np.float64)
    arr[:, 2] = -1.0
    cs = Raster(arr[None]).set_null_value(-1)
    cd, _, _ = distance.cost_distance_analysis(cs, np.array([[0, 0]]))
    data = cd.to_numpy()[0]
    mask = cd.mask.compute()[0]
    # The far-side valid cell is unreachable: filled with the null value but
    # NOT masked. Masking unreached-but-valid cells is a deliberate
    # out-of-scope follow-up.
    assert data[0, 4] == cd.null_value
    assert not mask[0, 4]


def test_cost_distance_unsigned_sources():
    # A uint8 sources raster exercises the dtype.kind == "u" branch.
    cs = Raster(COST_SURF).set_null_value(-1)
    srcs = Raster(SOURCES.astype(np.uint8)).set_null_value(0)
    _, _, al = distance.cost_distance_analysis(cs, srcs)
    assert al.dtype == np.dtype(np.int64)
    assert al.null_value == 0
    assert np.allclose(al.to_numpy(), AL_TRUTH_SCALE_1, equal_nan=True)


def test_cost_distance_1x1_public_api():
    # 1x1 grid through the public API with a sources *Raster* (index sources
    # would instead allocate 0). costs has no nulls, so nothing is masked.
    cs = Raster(np.array([[3.0]])[None])
    srcs = Raster(np.array([[7]], dtype=np.int64)[None]).set_null_value(0)
    cd, tr, al = distance.cost_distance_analysis(cs, srcs)
    assert np.array_equal(cd.to_numpy()[0], np.array([[0.0]]))
    assert np.array_equal(tr.to_numpy()[0], np.array([[0]]))
    assert np.array_equal(al.to_numpy()[0], np.array([[7]]))
    assert not cd.mask.compute().any()
    assert tr.null_value == -1


# ---------------------------------------------------------------------------
# Regression tests for findings F1, F3, F5, and F5b (numpy level)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(6, 6), (3, 5), (7, 13), (2, 3, 4), (5, 6, 7)]
)
def test_get_strides_matches_numpy(shape):
    # F1: _mult_accumulate must build a true cumulative product so that
    # _get_strides returns numpy's own C-contiguous element strides. Covers
    # both the 2D and 3D declared signatures.
    itemsize = np.dtype(np.int64).itemsize
    expected = tuple(
        s // itemsize for s in np.empty(shape, dtype=np.int64).strides
    )
    assert tuple(_get_strides(shape)) == expected


def test_cost_distance_1x1_with_source():
    # F3: a 1x1 grid whose single cell is the source. The heap capacity clamp
    # (max(size - 1, 1)) must let this run and return a trivial solution.
    costs = np.array([[3.0]])
    sources = np.array([[7]], dtype=np.int64)
    cd, tb, al = cost_distance_analysis_numpy(costs, sources, -1)
    assert np.array_equal(cd, np.array([[0.0]]))
    assert np.array_equal(tb, np.array([[-1]]))
    assert np.array_equal(al, np.array([[7]]))


def test_cost_distance_1x1_no_source():
    # F3: a 1x1 grid with no source (all-null sources) must not raise and
    # should report the cell as unreached.
    costs = np.array([[3.0]])
    sources = np.array([[-1]], dtype=np.int64)
    cd, tb, al = cost_distance_analysis_numpy(costs, sources, -1)
    assert np.isinf(cd[0, 0])
    assert tb[0, 0] == -2
    assert al[0, 0] == -1


def test_source_on_barrier_cost_does_not_propagate():
    # F5b: a source placed on a barrier-cost (-1) cell must not propagate. The
    # buggy version used the -1 fill as the popped cell's own cost, producing
    # edge weights L * 0.5 * (-1 + new_cost) that were zero or negative. Here
    # the neighbors have cost 0.5, so the buggy weight was -0.25 * L < 0.
    costs = np.full((3, 3), 0.5)
    costs[1, 1] = -1.0
    sources = np.full((3, 3), -1, dtype=np.int64)
    sources[1, 1] = 0
    cd, tb, al = cost_distance_analysis_numpy(costs, sources, -1)
    # Only the source cell is reached; neighbors stay unreached.
    assert cd[1, 1] == 0.0
    neighbors = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]
    for r, c in neighbors:
        assert np.isinf(cd[r, c])
    # No negative cost distances anywhere.
    assert not (cd[np.isfinite(cd)] < 0).any()


def test_barrier_source_alongside_valid_source_no_negatives():
    # F5b: when a barrier-cost source coexists with a valid source, the valid
    # source drives propagation and no negative cost distances appear.
    costs = np.full((3, 3), 1.0)
    costs[0, 0] = -1.0
    sources = np.full((3, 3), -1, dtype=np.int64)
    sources[0, 0] = 0  # source sitting on the barrier cell
    sources[2, 2] = 1  # valid source
    cd, tb, al = cost_distance_analysis_numpy(costs, sources, -1)
    finite = cd[np.isfinite(cd)]
    assert not (finite < 0).any()
    # The barrier source keeps its own zero cumcost but does not seed
    # neighbors; those are reached from the valid source instead.
    assert cd[0, 0] == 0.0
    assert al[0, 1] == 1
    assert al[1, 0] == 1


def test_source_on_null_elevation_does_not_propagate():
    # F5: with elevation, a source on a null-elevation cell must not
    # propagate. The buggy version used the elevation null fill in dz,
    # inflating outgoing edge lengths instead of skipping the cell.
    costs = np.full((3, 3), 1.0)
    elevation = np.full((3, 3), 10.0)
    elev_null = -9999.0
    elevation[1, 1] = elev_null
    sources = np.full((3, 3), -1, dtype=np.int64)
    sources[1, 1] = 0
    cd, tb, al = cost_distance_analysis_numpy(
        costs,
        sources,
        -1,
        elevation=elevation,
        elevation_null_value=elev_null,
    )
    assert cd[1, 1] == 0.0
    neighbors = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]
    for r, c in neighbors:
        assert np.isinf(cd[r, c])
    assert not (cd[np.isfinite(cd)] < 0).any()


def test_cost_distance_source_on_masked_cost_no_negatives():
    # F5b at the Raster level: a source located on a masked (null) cost cell
    # must not yield negative cost distances in the unmasked output.
    costs_arr = np.full((5, 5), 1.0)
    costs_arr[2, 2] = -1.0  # becomes a masked / barrier cell
    src_arr = np.zeros((5, 5), dtype=np.int64)
    src_arr[2, 2] = 5  # source on the masked cost cell
    src_arr[0, 0] = 1  # a valid source
    costs = Raster(costs_arr).set_null_value(-1)
    srcs = Raster(src_arr).set_null_value(0)
    cd, tb, al = distance.cost_distance_analysis(costs, srcs)
    # The masked cost cell is masked out of the output (it holds the source's
    # own zero cumcost), so only the unmasked cells are inspected here.
    data = cd.to_numpy()
    mask = cd.mask.compute()
    valid = data[~mask]
    assert not (valid < 0).any()


# ---------------------------------------------------------------------------
# scipy-oracle property test for cost_distance_analysis_numpy
# ---------------------------------------------------------------------------

# Move order matching _MOVES in cost_distance.py: (drow, dcol).
_ORACLE_MOVES = [
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
]


def _edge_length(dr, dc, sr, sc, elevation, r, c, nr, nc):
    """Single source of truth for the (optionally 3D) move length L."""
    planar_sq = (sr * dr) ** 2 + (sc * dc) ** 2
    if elevation is not None:
        dz = elevation[nr, nc] - elevation[r, c]
        return np.sqrt(planar_sq + dz * dz)
    return np.sqrt(planar_sq)


def _scaling_pair(scaling):
    if np.isscalar(scaling):
        return float(scaling), float(scaling)
    return float(scaling[0]), float(scaling[1])


def _build_oracle_graph(costs, scaling, elevation, elev_null):
    """Build the scipy grid graph and the shared 'valid cell' predicate.

    A cell is a node iff it is non-barrier (cost >= 0 and finite) and, with
    elevation, non-null. The undirected edge weight between neighbors i and j
    is ``L * 0.5 * (cost_i + cost_j)``.
    """
    rows, cols = costs.shape
    sr, sc = _scaling_pair(scaling)
    n = rows * cols

    def valid(r, c):
        if costs[r, c] < 0 or not np.isfinite(costs[r, c]):
            return False
        return not (elevation is not None and elevation[r, c] == elev_null)

    edge_rows, edge_cols, data = [], [], []
    for r in range(rows):
        for c in range(cols):
            if not valid(r, c):
                continue
            i = r * cols + c
            for dr, dc in _ORACLE_MOVES:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if not valid(nr, nc):
                    continue
                length = _edge_length(dr, dc, sr, sc, elevation, r, c, nr, nc)
                edge_rows.append(i)
                edge_cols.append(nr * cols + nc)
                data.append(length * 0.5 * (costs[r, c] + costs[nr, nc]))
    graph = coo_matrix((data, (edge_rows, edge_cols)), shape=(n, n)).tocsr()
    return graph, valid


def _oracle_cost_distance(costs, sources, scaling, elevation, elev_null):
    """Ground-truth cost distance via scipy Dijkstra on the grid graph."""
    rows, cols = costs.shape
    graph, _ = _build_oracle_graph(costs, scaling, elevation, elev_null)
    src_cells = np.flatnonzero(sources.ravel() != -1)
    dist = dijkstra(graph, directed=False, indices=src_cells, min_only=True)
    return dist.reshape(rows, cols)


def _assert_output_structure(cd, tr, al, sources, snull):
    """Vectorized structural invariants shared by every oracle case."""
    assert cd.dtype == np.float64
    assert tr.dtype == np.int8
    assert al.dtype == np.int64
    src_mask = sources != snull
    # Source cells.
    assert np.all(cd[src_mask] == 0.0)
    assert np.all(tr[src_mask] == -1)
    assert np.array_equal(al[src_mask], sources[src_mask])
    # Unreached cells (traceback -2).
    unreached = tr == -2
    assert np.all(np.isinf(cd[unreached]))
    assert np.all(al[unreached] == snull)
    # Reached non-source cells.
    reached_ns = (~src_mask) & (tr != -2)
    assert np.all(tr[reached_ns] >= 0)
    assert np.all(tr[reached_ns] <= 7)
    assert np.all(al[reached_ns] != snull)
    assert np.all(np.isfinite(cd[reached_ns]))


def _assert_traceback_consistent(
    cd, tr, al, costs, scaling, elevation, elev_null, snull
):
    """Implementation-independent traceback/allocation path-walk.

    For each reached non-source cell, its parent edge must be tight, walking
    predecessors must terminate at a source within ``size`` steps without
    repeats, and the allocation must be constant along that path.
    """
    rows, cols = costs.shape
    sr, sc = _scaling_pair(scaling)
    size = rows * cols
    for r in range(rows):
        for c in range(cols):
            if tr[r, c] < 0:  # source (-1) or unreached (-2)
                continue
            dr, dc = _ORACLE_MOVES[tr[r, c]]
            pr, pc = r - dr, c - dc
            # Predecessor is in-bounds, valid, and reached.
            assert 0 <= pr < rows and 0 <= pc < cols
            assert costs[pr, pc] >= 0 and np.isfinite(costs[pr, pc])
            if elevation is not None:
                assert elevation[pr, pc] != elev_null
            assert tr[pr, pc] != -2
            # Parent-edge tightness (the edge is effectively bit-exact).
            length = _edge_length(dr, dc, sr, sc, elevation, pr, pc, r, c)
            expected = cd[pr, pc] + length * 0.5 * (
                costs[pr, pc] + costs[r, c]
            )
            assert np.isclose(cd[r, c], expected, rtol=1e-9, atol=0.0)
            # Walk predecessors back to a source.
            wr, wc = r, c
            seen = set()
            steps = 0
            while tr[wr, wc] != -1:
                assert (wr, wc) not in seen
                seen.add((wr, wc))
                assert al[wr, wc] == al[r, c]
                mdr, mdc = _ORACLE_MOVES[tr[wr, wc]]
                wr, wc = wr - mdr, wc - mdc
                steps += 1
                assert steps <= size
            # Terminal source carries the same allocation label.
            assert al[wr, wc] == al[r, c]


def _random_case(
    seed,
    barrier_frac,
    use_elevation,
    n_sources,
    shape=None,
    special=None,
):
    rng = np.random.default_rng(seed)
    if shape is None:
        rows = int(rng.integers(12, 17))
        cols = int(rng.integers(12, 17))
    else:
        rows, cols = shape
    costs = rng.uniform(0.5, 10.0, (rows, cols))
    if barrier_frac > 0:
        costs[rng.random((rows, cols)) < barrier_frac] = -1.0
    if special is not None:
        for kind, frac in special.items():
            sel = rng.random((rows, cols)) < frac
            if kind == "zero":
                costs[sel] = 0.0
            elif kind == "nan":
                costs[sel] = np.nan
            elif kind == "inf":
                costs[sel] = np.inf
    elevation = None
    elev_null = -9999.0
    if use_elevation:
        elevation = rng.uniform(0.0, 50.0, (rows, cols))
        elevation[rng.random((rows, cols)) < 0.1] = elev_null
    # Source placement must exclude barriers of every flavor. inf >= 0 is True,
    # so isfinite is required alongside the >= 0 test.
    valid_mask = np.isfinite(costs) & (costs >= 0)
    if use_elevation:
        valid_mask &= elevation != elev_null
    valid_cells = np.argwhere(valid_mask)
    rng.shuffle(valid_cells)
    sources = np.full((rows, cols), -1, dtype=np.int64)
    for k in range(min(n_sources, len(valid_cells))):
        r, c = valid_cells[k]
        sources[r, c] = k
    return costs, sources, elevation, elev_null


# category -> kwargs for _random_case plus a "scaling" for the run/oracle.
_ORACLE_CATEGORIES = {
    "no_barriers_isotropic": {
        "barrier_frac": 0.0,
        "scaling": 1.0,
        "use_elevation": False,
        "n_sources": 1,
    },
    "barriers_isotropic": {
        "barrier_frac": 0.15,
        "scaling": 1.0,
        "use_elevation": False,
        "n_sources": 1,
    },
    "barriers_anisotropic": {
        "barrier_frac": 0.15,
        "scaling": (2.0, 0.5),
        "use_elevation": False,
        "n_sources": 1,
    },
    "multisource_barriers": {
        "barrier_frac": 0.15,
        "scaling": 1.0,
        "use_elevation": False,
        "n_sources": 4,
    },
    "elevation_isotropic": {
        "barrier_frac": 0.15,
        "scaling": 1.0,
        "use_elevation": True,
        "n_sources": 1,
    },
    "elevation_anisotropic": {
        "barrier_frac": 0.15,
        "scaling": (2.0, 0.5),
        "use_elevation": True,
        "n_sources": 2,
    },
    "zero_cost": {
        "barrier_frac": 0.1,
        "scaling": 1.0,
        "use_elevation": False,
        "n_sources": 2,
        "special": {"zero": 0.1},
    },
    "nan_costs": {
        "barrier_frac": 0.15,
        "scaling": 1.0,
        "use_elevation": False,
        "n_sources": 1,
        "special": {"nan": 0.1},
    },
    "inf_costs": {
        "barrier_frac": 0.15,
        "scaling": 1.0,
        "use_elevation": False,
        "n_sources": 1,
        "special": {"inf": 0.1},
    },
    "single_row": {
        "barrier_frac": 0.1,
        "scaling": 1.0,
        "use_elevation": False,
        "n_sources": 2,
        "shape": (1, 18),
    },
    "single_col": {
        "barrier_frac": 0.1,
        "scaling": 1.0,
        "use_elevation": False,
        "n_sources": 2,
        "shape": (18, 1),
    },
    "high_barrier": {
        "barrier_frac": 0.45,
        "scaling": 1.0,
        "use_elevation": False,
        "n_sources": 3,
    },
}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("category", list(_ORACLE_CATEGORIES))
def test_cost_distance_analysis_matches_oracle(category, seed):
    params = dict(_ORACLE_CATEGORIES[category])
    scaling = params.pop("scaling")
    costs, sources, elevation, elev_null = _random_case(seed, **params)
    cd, tr, al = cost_distance_analysis_numpy(
        costs,
        sources,
        -1,
        elevation=elevation,
        elevation_null_value=(elev_null if elevation is not None else 0),
        scaling=scaling,
    )
    ref = _oracle_cost_distance(costs, sources, scaling, elevation, elev_null)
    reached = np.isfinite(cd)
    assert np.array_equal(reached, np.isfinite(ref))
    # rtol=1e-12, atol=0.0 is safe even for zero_cost: path costs are either
    # exactly 0.0 in both implementations or bounded below by ~0.125 (the
    # smallest nonzero edge at these parameters). If the cost range is ever
    # widened toward 0, atol must become nonzero.
    assert np.allclose(cd[reached], ref[reached], rtol=1e-12, atol=0.0)
    _assert_output_structure(cd, tr, al, sources, -1)
    _assert_traceback_consistent(
        cd, tr, al, costs, scaling, elevation, elev_null, -1
    )


# ---------------------------------------------------------------------------
# Directed numpy-level tests
# ---------------------------------------------------------------------------


def _single_source(shape):
    sources = np.full(shape, -1, dtype=np.int64)
    sources[0, 0] = 0
    return sources


@pytest.mark.parametrize("scaling", [0.0, -1.0, (1.0, 0.0), (-2.0, 3.0)])
def test_scaling_nonpositive_raises(scaling):
    costs = np.ones((4, 4))
    with pytest.raises(ValueError, match="greater than 0"):
        cost_distance_analysis_numpy(
            costs, _single_source((4, 4)), -1, scaling=scaling
        )


def test_scaling_wrong_size_raises():
    costs = np.ones((4, 4))
    with pytest.raises(ValueError, match="Invalid scaling shape"):
        cost_distance_analysis_numpy(
            costs, _single_source((4, 4)), -1, scaling=[1.0, 2.0, 3.0]
        )


def test_scaling_2d_raises():
    costs = np.ones((4, 4))
    with pytest.raises(ValueError, match="Invalid scaling shape"):
        cost_distance_analysis_numpy(
            costs, _single_source((4, 4)), -1, scaling=[[1.0, 2.0]]
        )


def test_sources_shape_mismatch_raises():
    costs = np.ones((4, 4))
    sources = np.full((3, 3), -1, dtype=np.int64)
    with pytest.raises(ValueError, match="shapes must match"):
        cost_distance_analysis_numpy(costs, sources, -1)


def test_elevation_shape_mismatch_raises():
    costs = np.ones((4, 4))
    elevation = np.ones((3, 3))
    with pytest.raises(ValueError, match="same shape as costs"):
        cost_distance_analysis_numpy(
            costs, _single_source((4, 4)), -1, elevation=elevation
        )


def test_float16_costs_promoted():
    # float16 costs are promoted to float32 internally; an explicit float32
    # cast must give bit-identical outputs.
    rng = np.random.default_rng(0)
    costs16 = rng.uniform(0.5, 5.0, (5, 5)).astype(np.float16)
    sources = np.full((5, 5), -1, dtype=np.int64)
    sources[0, 0] = 0
    sources[4, 4] = 1
    out16 = cost_distance_analysis_numpy(costs16, sources, -1)
    out32 = cost_distance_analysis_numpy(
        costs16.astype(np.float32), sources, -1
    )
    for a, b in zip(out16, out32, strict=True):
        assert np.array_equal(a, b)


def test_float16_elevation_promoted():
    # float16 elevation is promoted to float32 internally. (This compiles one
    # new kernel signature the first time it runs on a worker.)
    rng = np.random.default_rng(1)
    costs = rng.uniform(0.5, 5.0, (5, 5))
    elev16 = rng.uniform(0.0, 20.0, (5, 5)).astype(np.float16)
    sources = np.full((5, 5), -1, dtype=np.int64)
    sources[0, 0] = 0
    out16 = cost_distance_analysis_numpy(
        costs, sources, -1, elevation=elev16, elevation_null_value=-9999.0
    )
    out32 = cost_distance_analysis_numpy(
        costs,
        sources,
        -1,
        elevation=elev16.astype(np.float32),
        elevation_null_value=-9999.0,
    )
    for a, b in zip(out16, out32, strict=True):
        assert np.array_equal(a, b)


def test_constant_elevation_matches_planar():
    # A flat elevation surface must match a run with no elevation at all.
    rng = np.random.default_rng(2)
    costs = rng.uniform(0.5, 5.0, (8, 8))
    sources = np.full((8, 8), -1, dtype=np.int64)
    sources[0, 0] = 0
    sources[7, 7] = 1
    flat = np.full((8, 8), 3.0)
    cd_e, tr_e, al_e = cost_distance_analysis_numpy(
        costs, sources, -1, elevation=flat, elevation_null_value=-9999.0
    )
    cd_p, tr_p, al_p = cost_distance_analysis_numpy(costs, sources, -1)
    reached = np.isfinite(cd_p)
    assert np.array_equal(reached, np.isfinite(cd_e))
    assert np.allclose(cd_e[reached], cd_p[reached], rtol=1e-12)
    assert np.array_equal(tr_e, tr_p)
    assert np.array_equal(al_e, al_p)


def test_deterministic_repeated_runs():
    # The kernel has no randomness: identical inputs give identical outputs.
    rng = np.random.default_rng(3)
    costs = rng.uniform(0.5, 5.0, (10, 10))
    costs[rng.random((10, 10)) < 0.15] = -1.0
    sources = np.full((10, 10), -1, dtype=np.int64)
    sources[0, 0] = 0
    sources[9, 9] = 1
    sources[0, 9] = 2
    out1 = cost_distance_analysis_numpy(costs, sources, -1)
    out2 = cost_distance_analysis_numpy(costs, sources, -1)
    for a, b in zip(out1, out2, strict=True):
        assert np.array_equal(a, b)


def test_all_null_sources_unreached():
    # No sources: the main loop breaks immediately on an empty heap.
    costs = np.ones((4, 4))
    sources = np.full((4, 4), -1, dtype=np.int64)
    cd, tr, al = cost_distance_analysis_numpy(costs, sources, -1)
    assert np.all(np.isinf(cd))
    assert np.all(tr == -2)
    assert np.all(al == -1)
