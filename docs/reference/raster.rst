.. currentmodule:: raster_tools

**************
Raster Objects
**************

Constructing Rasters
====================

.. autosummary::
   :toctree: generated/

   Raster

Properties
==========

.. autosummary::
   :toctree: generated/

   Raster.dtype
   Raster.shape
   Raster.nbands
   Raster.null_value
   Raster.values
   Raster.data
   Raster.xdata
   Raster.mask
   Raster.xmask
   Raster.x
   Raster.y
   Raster.crs
   Raster.affine
   Raster.resolution
   Raster.bounds
   Raster.bandwise

Deprecated Properties
---------------------

.. autosummary::
   :toctree: generated/

   Raster.pxrs
   Raster.xrs

Operations
==========

Arithmetic and Comparison Operations
----------------------------------------------

Raster objects support all arithmetic (``+``, ``-``, ``*``,
``/``, ``//``, ``%``, ``divmod()``, ``**``, ``<<``, ``>>``, ``&``, ``^``,
``|``, ``~``), bitwise/logical operators (``&``, ``|``, ``^``, ``~``), and
comparison operations (``==``, ``<``, ``>``, ``<=``, ``>=``, ``!=``). Because
of this, they can be used like regular scalar variables. Some examples:

* Addition and subtraction: ``rs4 = rs1 + rs2 - rs3``
* Floor division: ``rs3 = rs1 // rs2``
* With scalars: ``rs3 = 1.5 * (rs1 ** (-rs2 - 2))``
* divmod: ``rs_div, rs_mod = divmod(rs1, rs2)``
* Bit shifting: ``rs2 = rs1 << 1``
* Bitwise and: ``rs3 = rs1 & rs2``
* Bitwise complement: ``rs2 = ~rs1``
* Comparison: ``rs_bool = rs1 >= rs2``

General
-------

.. autosummary::
   :toctree: generated/

   Raster.round

Reductions
----------

.. autosummary::
   :toctree: generated/

   Raster.all
   Raster.any
   Raster.max
   Raster.mean
   Raster.min
   Raster.prod
   Raster.std
   Raster.sum
   Raster.var


Methods
=======

Reshaping and Reorganizing
--------------------------

.. autosummary::
   :toctree: generated/

   Raster.chunk
   Raster.get_bands

Null Data and Remapping
-----------------------

.. autosummary::
   :toctree: generated/

   Raster.burn_mask
   Raster.reclassify
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
   Raster.to_dataset
   Raster.to_vector

Georeferencing
--------------

.. autosummary::
   :toctree: generated/

   Raster.index
   Raster.set_crs
   Raster.xy

Raster IO
---------

.. autosummary::
   :toctree: generated/

   Raster.close
   Raster.eval
   Raster.save

Plotting
========

.. autosummary::
   :toctree: generated/

   Raster.plot

Miscellaneous
=============

.. autosummary::
   :toctree: generated/

   Raster.get_chunked_coords
