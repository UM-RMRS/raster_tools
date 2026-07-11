import subprocess
import sys

import raster_tools


def test_distance_not_imported_eagerly():
    # Must run in a subprocess: other test modules import distance, so
    # sys.modules in the pytest process is already polluted.
    code = (
        "import sys; import raster_tools; "
        "assert 'raster_tools.distance' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_public_api_resolves():
    for name in raster_tools.__all__:
        assert getattr(raster_tools, name) is not None
    assert set(raster_tools.__all__) <= set(dir(raster_tools))
    assert not hasattr(raster_tools, "not_a_real_attribute")
