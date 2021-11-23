.. currentmodule:: raster_tools

**************
Raster Objects
**************

Constructing Rasters
====================

.. autosummary::
   :toctree: generated/

   Raster

Attributes
==========

.. autosummary::
   :toctree: generated/

   Raster.affine
   Raster.crs
   Raster.device
   Raster.dtype
   Raster.null_value
   Raster.resolution
   Raster.shape

Operations
==========

Arithmetic
-----------

.. autosummary::
   :toctree: generated/

   Raster.add
   Raster.subtract
   Raster.divide
   Raster.multiply
   Raster.negate
   Raster.mod
   Raster.pow
   Raster.log
   Raster.log10
   Raster.round
   Raster.sqrt

Comparisons
-----------

.. autosummary::
   :toctree: generated/

   Raster.eq
   Raster.ge
   Raster.gt
   Raster.le
   Raster.lt
   Raster.ne

Logical
-------

.. autosummary::
   :toctree: generated/

   Raster.and_
   Raster.or_

Hardware Switching
------------------
.. autosummary::
   :toctree: generated/

   Raster.cpu
   Raster.gpu

Reshaping and Reorganizing
--------------------------

.. autosummary::
   :toctree: generated/

   Raster.clip_box
   Raster.get_bands

Null Data and Remapping
-----------------------

.. autosummary::
   :toctree: generated/

   Raster.replace_null
   Raster.remap_range
   Raster.set_null_value
   Raster.to_null_mask
   Raster.where

Contents / Conversion
---------------------

.. autosummary::
   :toctree: generated/

   Raster.astype
   Raster.copy
   Raster.to_xarray
   Raster.to_dask
   Raster.to_lazy

Georeferencing
--------------

.. autosummary::
   :toctree: generated/

   Raster.set_crs

Raster IO
---------

.. autosummary::
   :toctree: generated/

   Raster.close
   Raster.eval
   Raster.save
