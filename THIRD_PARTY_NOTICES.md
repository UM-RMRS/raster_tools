# Third-party notices

Raster Tools is distributed under the GNU General Public License v3 (see
LICENSE). The components below are derived from software under other
licenses whose terms require the notices reproduced here.

## GDAL

The rasterization kernels in `raster_tools/_rasterize_numba.py` are a Python
port of GDAL's rasterizer (`alg/llrasterize.cpp`, `alg/gdalrasterize.cpp`,
and `ogr/ogrcurve.cpp` from https://github.com/OSGeo/gdal).

Copyright (c) 1999-2005, Frank Warmerdam
Copyright (c) 2008-2013, Even Rouault

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
