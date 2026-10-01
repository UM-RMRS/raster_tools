"""Check raster_tools results on a multi-process dask.distributed cluster.

Each operation is computed twice, once on a LocalCluster whose workers are
separate processes and once with the threaded scheduler, and the results must
match. Process-based workers force every graph to be pickled and executed
away from the client, and they split any per-process state (module globals,
caches, locks) that the threaded scheduler shares.
"""

import logging
import os
import warnings

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest

distributed = pytest.importorskip("distributed")

import raster_tools as rts  # noqa: E402
from raster_tools import distance, rasterize  # noqa: E402
from tests import testdata  # noqa: E402

pytestmark = [
    # One xdist worker runs the whole module so the cluster starts once.
    pytest.mark.xdist_group("distributed"),
    pytest.mark.timeout(300),
]

N_WORKERS = 2


@pytest.fixture(scope="module")
def _cluster_client():
    with warnings.catch_warnings():
        # Startup can warn about port selection or memory limits; neither
        # matters for a throwaway local cluster.
        warnings.simplefilter("ignore")
        cluster = distributed.LocalCluster(
            n_workers=N_WORKERS,
            threads_per_worker=1,
            processes=True,
            dashboard_address=None,
            silence_logs=logging.ERROR,
        )
        # Not the default client: nothing outside the fixture below should
        # see the cluster, including later tests on the same xdist worker.
        client = distributed.Client(cluster, set_as_default=False)
    try:
        client.wait_for_workers(N_WORKERS, timeout=120)
        yield client
    finally:
        client.close()
        cluster.close()


@pytest.fixture
def dist_client(_cluster_client):
    # Routing through the config also catches dask.compute calls made inside
    # the library while graphs are being built.
    with dask.config.set(scheduler=_cluster_client):
        yield _cluster_client


def _set_worker_rasterize_backend(backend):
    rasterize.RASTERIZE_BACKEND = backend


def _get_worker_rasterize_backend():
    return rasterize.RASTERIZE_BACKEND


@pytest.fixture(params=["rasterio", "numba"])
def rasterize_backend(request, dist_client, monkeypatch):
    # The backend switch is a module global read when chunks are burned, so
    # it has to be set in every worker process, not just in this one.
    monkeypatch.setattr(rasterize, "RASTERIZE_BACKEND", request.param)
    previous = dist_client.run(_get_worker_rasterize_backend)
    dist_client.run(_set_worker_rasterize_backend, request.param)
    try:
        yield request.param
    finally:
        for addr, value in previous.items():
            dist_client.run(
                _set_worker_rasterize_backend, value, workers=[addr]
            )


def _dem_small():
    return testdata.raster.dem_small.chunk((1, 25, 25))


def _dem_with_nulls():
    return testdata.raster.dem_clipped_small.chunk((1, 125, 125))


def _focal_mean():
    return rts.focal.focal(_dem_with_nulls(), "mean", 5, ignore_null=True)


def _correlate():
    kernel = np.arange(9, dtype="float64").reshape(3, 3)
    return rts.focal.correlate(_dem_small(), kernel)


def _slope():
    return rts.surface.slope(_dem_with_nulls())


def _aspect():
    return rts.surface.aspect(_dem_small())


def _arithmetic():
    dem = _dem_with_nulls()
    return rts.general.where(dem > 1500, dem * 2 - 100, dem / 3)


def _reclassify():
    dem = _dem_small()
    lo, hi = 1400, 1700
    return rts.reclassify(
        dem.astype("int32", new_null_value=-1), {lo: 1, hi: 2}
    )


def _regions():
    # regions computes the unique values of its input while building the
    # graph, so that compute must also go through the cluster. Labeling
    # cost grows quickly with the chunk count, so this uses a 2x2 grid.
    dem = testdata.raster.dem_small.chunk((1, 50, 50))
    classes = rts.remap_range(
        dem, [(0, 1500, 1), (1500, 1700, 2), (1700, 5000, 3)]
    ).astype("uint8", new_null_value=0)
    return rts.general.regions(classes)


def _rasterize_field():
    return rasterize.rasterize(testdata.vector.pods_small, _dem_small())


def _rasterize_mask():
    return rasterize.rasterize(
        testdata.vector.pods_small, _dem_small(), mask=True
    )


def _clip():
    return rts.clipping.clip(testdata.vector.pods_small, _dem_small())


def _reproject():
    return rts.reproject(_dem_small(), "EPSG:4326", "bilinear")


def _cost_distance():
    dem = _dem_small()
    costs = rts.general.where(dem > 0, dem / 1000, -1)
    return distance.cda_cost_distance(costs, [[10, 10], [80, 60]])


def _proximity():
    src = testdata.raster.prox_src.chunk((1, 50, 53))
    return distance.pa_proximity(src, max_distance=15)


RASTER_OPS = {
    "focal_mean": _focal_mean,
    "focal_correlate": _correlate,
    "surface_slope": _slope,
    "surface_aspect": _aspect,
    "general_arithmetic_where": _arithmetic,
    "general_reclassify": _reclassify,
    "general_regions": _regions,
    "clip": _clip,
    "reproject": _reproject,
    "cost_distance": _cost_distance,
    "proximity": _proximity,
}

RASTERIZE_OPS = {
    "rasterize_field": _rasterize_field,
    "rasterize_mask": _rasterize_mask,
}


def _raster_parts(raster, scheduler):
    data, mask = dask.compute(raster.data, raster.mask, scheduler=scheduler)
    return {
        "data": data,
        "mask": mask,
        "null_value": raster.null_value,
        "dtype": raster.dtype,
        "crs": raster.crs,
        "affine": raster.affine,
    }


def _assert_parts_equal(actual, expected):
    np.testing.assert_array_equal(actual["mask"], expected["mask"])
    np.testing.assert_array_equal(actual["data"], expected["data"])
    for key in ("null_value", "dtype", "crs", "affine"):
        a, e = actual[key], expected[key]
        if key == "null_value" and a is not None and e is not None:
            np.testing.assert_array_equal(a, e)
        else:
            assert a == e, key


def _check_matches_threads(build, client):
    # The threaded result is built and computed entirely outside the
    # cluster; the distributed one is built under the client fixture.
    with dask.config.set(scheduler="threads"):
        expected = _raster_parts(build(), "threads")
    actual = _raster_parts(build(), client)
    assert actual["mask"].shape == actual["data"].shape
    _assert_parts_equal(actual, expected)
    return actual


def _worker_pid(block):
    return np.full(block.shape, os.getpid(), dtype="int64")


def test_client_routes_computes_to_worker_processes(dist_client):
    assert dask.base.get_scheduler() == dist_client.get
    x = da.zeros((8, 8), chunks=4, dtype="int64")
    pids = set(np.unique(x.map_blocks(_worker_pid).compute()).tolist())
    worker_pids = set(dist_client.run(os.getpid).values())
    assert len(worker_pids) == N_WORKERS
    assert os.getpid() not in worker_pids
    assert pids <= worker_pids


def test_cluster_is_not_the_default_scheduler(_cluster_client):
    assert dask.config.get("scheduler", None) is not _cluster_client
    assert dask.base.get_scheduler() != _cluster_client.get
    with pytest.raises(ValueError):
        distributed.get_client()


@pytest.mark.parametrize("name", list(RASTER_OPS))
def test_raster_op_matches_threads(name, dist_client):
    _check_matches_threads(RASTER_OPS[name], dist_client)


@pytest.mark.parametrize("name", list(RASTERIZE_OPS))
def test_rasterize_matches_threads(name, rasterize_backend, dist_client):
    workers = set(dist_client.run(_get_worker_rasterize_backend).values())
    assert workers == {rasterize_backend}
    result = _check_matches_threads(RASTERIZE_OPS[name], dist_client)
    assert (~result["mask"]).any()


def test_zonal_stats_matches_threads(dist_client):
    def build():
        return rts.zonal.zonal_stats(
            testdata.vector.pods_small,
            _dem_small(),
            ["mean", "min", "max", "count", "std", "sum"],
        )

    with dask.config.set(scheduler="threads"):
        expected = build().compute(scheduler="threads")
    actual = build().compute(scheduler=dist_client)
    assert len(actual) > 0
    pd.testing.assert_frame_equal(actual.sort_index(), expected.sort_index())


@pytest.fixture
def default_client(_cluster_client):
    # Registered as the global default, as a plain ``Client(cluster)`` call
    # in user code would be. Closing it restores the previous default.
    with distributed.Client(_cluster_client.scheduler.address) as client:
        yield client
    with pytest.raises(ValueError):
        distributed.get_client()


def _resolved_store_lock():
    # The lock dask.array.store substitutes for lock=True, which is what
    # Raster.save hands to rioxarray.
    return dask.utils.get_scheduler_lock(collection=da.Array)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "dask resolves lock=True to a process-local SerializableLock unless "
        "a distributed client is the global default, so Raster.save on a "
        "non-default client lets worker processes write the same GeoTIFF "
        "concurrently and corrupt it"
    ),
)
def test_save_lock_is_process_safe_on_non_default_client(dist_client):
    assert isinstance(_resolved_store_lock(), distributed.Lock)


def test_save_round_trip_matches_threads(default_client, tmp_path):
    assert isinstance(_resolved_store_lock(), distributed.Lock)

    # Small dask chunks against the default 256x256 GeoTIFF tiles put many
    # chunk writes into each tile, so unsynchronized writers would collide.
    def build():
        return testdata.raster.dem_clipped_small.chunk((1, 25, 25))

    expected_path = tmp_path / "threads.tif"
    actual_path = tmp_path / "distributed.tif"
    with dask.config.set(scheduler="threads"):
        build().save(expected_path)
        expected = _raster_parts(rts.Raster(expected_path), "threads")
    build().save(actual_path)
    actual = _raster_parts(rts.Raster(actual_path), "threads")
    assert expected["mask"].any()
    _assert_parts_equal(actual, expected)
