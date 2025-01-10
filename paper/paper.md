---
title: 'Raster Tools: An Open Source Toolbox for Raster Processing'
tags:
  - Python
  - raster
  - rasters
  - raster processing
  - parallel
  - geospatial
  - GIS
  - remote sensing
authors:
  - name: Fredrick Bunt
    orcid: 0000-0003-4139-5355
    affiliation: 1

  - name: Jesse Johnson
    orcid: 0000-0002-7387-6500
    affiliation: 1

  - name: John Hogland
    orcid: 0000-0003-2676-6277
    affiliation: 2
affiliations:
  - index: 1
    name: University of Montana, USA
  - index: 2
    name: Rocky Mountain Research Station, USA
    ror: 04347cr60
date: 10 January 2025
bibliography: paper.bib
---

# Summary

The Raster Tools Python package provides a homogeneous application programmers
interface (API) for building scalable raster data processing pipelines.
Pipelines built with Raster Tools can efficiently process extremely large
raster datasets, including those larger than the system’s available memory.
Raster Tools can be deployed on systems ranging from small laptops to high
performance computing environments. To aid in the construction of raster
processing pipelines, Raster Tools provides a consistent API to a suite of
scalable raster processing functions including focal, zonal, clipping,
convolution, and distance analysis operations.

# Statement of Need

In recent years, geospatial analysis has experienced a dramatic shift driven by
the proliferation of global monitoring programs, new platforms for collecting
geographically based observations, and advancements in remote sensing
technology [@see2024]. This surge in data acquisition has led to the generation
of massive raster datasets characterized by high spatial resolutions, dense
temporal measurements, increased coverage of the electromagnetic spectrum, and
greater spatial extents. Collectively, these mark a significant departure from
the data volumes of the past [@qcrunch]. For example, MODIS satellites [@modis]
are now acquiring 100 m resolution data and SENTINEL-2 [@sentinel2] is
collecting imagery at 10-60 meter spatial resolution, globally at daily and
five day intervals, respectively. The volume of such datasets has posed
unprecedented challenges in terms of processing and analysis. In addition,
there are numerous important products derived from primary data sources
characterized by similar data volumes. Examples include interferometric
synthetic radar estimates of velocities, soil moisture estimates, and
vegetation indices. A specific example is the widely used ERA5 global
reanalysis dataset, which now has a data catalog of roughly 5 petabytes
[@era5].

To meet modern data challenges, Python provides a rich, general purpose, data
processing stack built on tools like Numpy [@numpy], SciPy [@scipy], Xarray
[@xarray], and Dask [@dask]. With the use of GDAL [@gdal] bindings like
Rasterio [@rasterio] and rioxarray [@rioxarray], this stack can be leveraged to
carry out raster processing and geographic information systems (GIS) work. Yet,
it is cumbersome and time consuming to develop pipelines that scale from small
to extremely large raster datasets. This is for several primary reasons. First,
many of the packages in Python’s data stack do not natively scale or only scale
in a way that loads all data into memory, limiting the datasets sizes that can
be processed on smaller hardware. Second, many common GIS operations are not
implemented in the available packages, requiring implementations to be written
from scratch. Third, the package interfaces are not always compatible without a
substantial amount of boilerplate code. Finally, there is no standard way of
handling null or missing data. In the GIS world, this is typically handled with
sentinel values that are outside of the data domain (e.g. -9999), while in much
of Python’s data stack, this is handled with NaN values. These two paradigms
are difficult to reconcile.

Raster Tools overcomes these issues by providing a consistent API for creating
scalable processing pipelines in Python. It wraps the appropriate Python
packages for each operation and takes care of any boilerplate code that is
needed to make them work together. It also provides implementations for common
GIS operations that are not already provided by Python’s data stack. Raster
Tools pipelines can easily integrate with or replace existing pipelines due to
its compatibility with common data objects from Numpy, Xarray, and Dask. This
compatibility with common data objects has the added benefit of allowing Raster
Tools pipelines to integrate with machine learning packages such as
scikit-learn [@sklearn], xgboost [@xgboost], and pytorch [@pytorch].
