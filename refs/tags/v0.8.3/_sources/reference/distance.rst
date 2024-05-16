.. currentmodule:: raster_tools.distance

*****************
Distance Analysis
*****************

The distance module contains functions for carrying out distance analysis.
Currently this is limited to 2D and 3D `cost distance analysis`_ and 2D
`proximity analysis`_.

.. _cost distance analysis: https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/understanding-cost-distance-analysis.htm
.. _proximity analysis: https://desktop.arcgis.com/en/arcmap/latest/analyze/commonly-used-tools/proximity-analysis.htm

Cost Distance Operations
========================

.. autosummary::
   :toctree: generated/

   cost_distance_analysis
   cda_allocation
   cda_cost_distance
   cda_traceback

Proximity Operations
====================

.. autosummary::
   :toctree: generated/

   proximity_analysis
   pa_allocation
   pa_direction
   pa_proximity
