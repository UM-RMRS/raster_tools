import pytest

from raster_tools._compat import (
    MIN_DASK_VERSION,
    _release_tuple,
    check_dask_version,
)


@pytest.mark.parametrize(
    "version_str,expected",
    [
        ("2025.1.0", (2025, 1, 0)),
        ("2026.8.0", (2026, 8, 0)),
        ("2025.1.0+3.gabcdef", (2025, 1, 0)),
        ("2025.1.0rc1", (2025, 1, 0)),
        ("2025.1.0.dev0", (2025, 1, 0)),
        ("2025.1.0.post1", (2025, 1, 0)),
        ("2023.3.0", (2023, 3, 0)),
    ],
)
def test_release_tuple(version_str, expected):
    assert _release_tuple(version_str) == expected


@pytest.mark.parametrize("version_str", ["2023.3.0", "2024.4.1", "2024.12.1"])
def test_check_dask_version_rejects_old_dask(version_str):
    with pytest.raises(ImportError) as excinfo:
        check_dask_version(version_str)
    message = str(excinfo.value)
    assert "requires dask>=2025.1.0" in message
    assert f"found dask {version_str}" in message


@pytest.mark.parametrize(
    "version_str", [".".join(map(str, MIN_DASK_VERSION)), "2026.8.0"]
)
def test_check_dask_version_accepts_supported_dask(version_str):
    check_dask_version(version_str)
