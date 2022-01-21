from raster_tools import Raster

def create_raster(file):
    raster = Raster(file)
    return raster
def add_to_raster(raster, file):
    raster.add(file)

def test(file, file2):
    raster = Raster(file)
    raster.add(file2)
    return raster.__array__(float)

print(test("web_app/data/elevation.tif", "web_app/data/elevation2.tif"))