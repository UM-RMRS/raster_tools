.. currentmodule:: raster_tools

*******************
Top Level Functions
*******************

Opening Rasters
===============

.. autosummary::
   :toctree: generated/

   Raster
   open_dataset

Combining Rasters
=================

.. autosummary::
   :toctree: generated/

   band_concat
   mosaic
   split_bands
   stack_bands

Creating Rasters from a Template
================================

.. autosummary::
   :toctree: generated/

   constant_raster
   empty_like
   full_like
   ones_like
   random_raster
   zeros_like

Constructing Rasters from Data
==============================

.. autosummary::
   :toctree: generated/

   data_to_raster
   data_to_raster_like
   data_to_xr_raster
   data_to_xr_raster_ds
   data_to_xr_raster_ds_like
   data_to_xr_raster_like
   dataarray_to_raster
   dataarray_to_xr_raster
   dataarray_to_xr_raster_ds
   get_raster

Remapping Rasters
=================

.. autosummary::
   :toctree: generated/

   reclassify
   remap_range

Mapping Functions Over Blocks
=============================

.. autosummary::
   :toctree: generated/

   geo_map_blocks
   geo_map_overlap
   map_blocks
   map_overlap

Padding Rasters
===============

.. autosummary::
   :toctree: generated/

   pad

Vectors
=======

.. autosummary::
   :toctree: generated/

   Vector
   count_layer_features
   list_layers
   open_vectors
