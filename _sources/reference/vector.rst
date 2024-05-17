.. currentmodule:: raster_tools

**************
Vector Objects
**************

Opening Vectors
====================

.. autosummary::
   :toctree: generated/

   open_vectors

Attributes
==========

.. autosummary::
   :toctree: generated/

   Vector.bounds
   Vector.crs
   Vector.data
   Vector.field_dtypes
   Vector.field_names
   Vector.field_schema
   Vector.geometry
   Vector.shape
   Vector.size
   Vector.table
   Vector.tasks

Operations
==========

IO
-------

.. autosummary::
   :toctree: generated/

   Vector.eval
   Vector.save

Contents / Conversions
----------------------

.. autosummary::
   :toctree: generated/

   Vector.to_crs
   Vector.to_dask
   Vector.to_dataframe
   Vector.to_lazy
   Vector.to_raster
   Vector.to_shapely
   Vector.cast_field

Vector Ops
----------

.. autosummary::
   :toctree: generated/

   Vector.buffer
   Vector.simplify
