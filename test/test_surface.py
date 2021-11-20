from raster_tools import surface, Raster
import xarray as xr
import dask.array as da
import numpy as np

#Create Rasters sufaces to compare against ESRI created surfaces.
rsPath = r'test\data\elevation.tif'
outDir = r'E:\temp\surfaceTest'
rs=Raster(rsPath)
slopeRs = surface.slope(rs)
slopeRs.save(outDir+"\\slope2.tif")
aspectRs = surface.aspect(rs)
aspectRs.save(outDir+"\\aspect2.tif")
curvRs = surface.curvature(rs)
curvRs.save(outDir+"\\curv2.tif")
surfRs = surface.surfaceArea3d(rs)
norRs = surface.northing(rs,False)
norRs.save(outDir+"\\northing2.tif")
estRs = surface.easting(rs,False)
estRs.save(outDir+"\\easting2.tif")
hsRs = surface.hillshade(rs)
hsRs.save(outDir+"\\hillshade2.tif")


    












