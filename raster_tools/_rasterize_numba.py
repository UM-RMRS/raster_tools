"""Numba scanline/line rasterization kernels.

These kernels reproduce GDAL's rasterizer (replace mode) pixel for pixel for
axis-aligned, non-rotated affines. They are a private backend for
``raster_tools.rasterize``; the two entry points ``numba_rasterize_wrapper``
and ``numba_mask`` mirror the rasterio wrappers they can stand in for.

Inputs that GDAL handles but these kernels do not (add mode, rotated or sheared
affines, mixed geometry families, GeometryCollections) raise
``_NumbaUnsupported`` so the caller can fall back to the rasterio path.
"""

# The kernels below are a Python port of GDAL's rasterizer. They transcribe
# the algorithms in these GDAL source files:
#
#   alg/llrasterize.cpp    Copyright (c) 2000, Frank Warmerdam
#                          Copyright (c) 2011, Even Rouault
#   alg/gdalrasterize.cpp  Copyright (c) 2005, Frank Warmerdam
#                          Copyright (c) 2008-2013, Even Rouault
#   ogr/ogrcurve.cpp       Copyright (c) 1999, Frank Warmerdam
#
# GDAL is distributed under the MIT license, whose terms follow.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# The kernels transcribe GDAL's rasterizer variable-for-variable (dfX,
# nDeltaX, iX, ...) so they can be checked line by line against the GDAL
# source. Keep those names rather than renaming them to snake_case.
# ruff: noqa: N806

import math

import numba as nb
import numpy as np
import shapely

from raster_tools.dtypes import U8

# C int (int32) range. GDAL skips a line segment whose endpoint pixel
# coordinates fall outside this range and clamps polygon edge crossings to it.
_INT_MAX = 2147483647.0
_INT_MIN = -2147483648.0

# Epsilon GDAL uses to detect geometries aligned with pixel coordinates.
_EPSILON = 1e-4

# shapely geometry type ids grouped by the family that shares a kernel.
_POLYGONAL = frozenset({3, 6})
_LINEAL = frozenset({1, 5})
_PUNCTAL = frozenset({0, 4})


class _NumbaUnsupported(Exception):  # noqa: N818
    """Raised when the numba backend cannot handle an input."""


@nb.jit(nopython=True, nogil=True, cache=True)
def _burn_point(out, iy, ix, value):
    if 0 <= ix < out.shape[1] and 0 <= iy < out.shape[0]:
        out[iy, ix] = value


@nb.jit(nopython=True, nogil=True, cache=True)
def _burn_scanline(out, y, x0, x1, value):
    if x0 > x1:
        return
    if x0 < 0:
        x0 = 0
    nx = out.shape[1]
    if x1 >= nx:
        x1 = nx - 1
    out[y, x0 : x1 + 1] = value


# Upper bound, in int64 entries, on the crossing scratch buffer. A polygon
# whose clamped edge-row spans sum above this is filled with the per-row
# kernel instead, keeping scratch memory bounded for pathological inputs.
_SCRATCH_CAP = 8_000_000


@nb.jit(nopython=True, nogil=True, cache=True)
def _sort_small(a, lo, hi):
    # Insertion sort of a[lo:hi]; crossings per row are usually 2 to 8.
    for i in range(lo + 1, hi):
        v = a[i]
        j = i - 1
        while j >= lo and a[j] > v:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = v


@nb.jit(nopython=True, nogil=True, cache=True)
def _max_span(px, py, ring_start, ring_stop, poly_ring_lo, poly_ring_hi, ny):
    # Largest per-polygon crossing count, used to size the shared scratch.
    # Each edge can cross at most int(clamped dy span) + 2 scanlines. A
    # non-finite y makes the span check fall through and add nothing, which
    # is consistent with such an edge being skipped in the fill.
    best = 0
    for p in range(poly_ring_lo.shape[0]):
        total = 0
        for r in range(poly_ring_lo[p], poly_ring_hi[p]):
            s = ring_start[r]
            e = ring_stop[r]
            for i in range(s, e):
                ind1 = e - 1 if i == s else i - 1
                y1 = py[ind1]
                y2 = py[i]
                if y1 > y2:
                    y1, y2 = y2, y1
                if y1 < 0.0:
                    y1 = 0.0
                if y2 > ny:
                    y2 = float(ny)
                if y2 >= y1:
                    total += int(y2 - y1) + 2
        if total > best:
            best = total
    return best


@nb.jit(nopython=True, nogil=True, cache=True)
def _fill_one_polygon(out, px, py, ring_start, ring_stop, rev, r0, r1, value):
    ny = out.shape[0]
    nx = out.shape[1]
    total = 0
    dminy = 0.0
    dmaxy = 0.0
    seen = False
    for r in range(r0, r1):
        s = ring_start[r]
        e = ring_stop[r]
        total += e - s
        for i in range(s, e):
            v = py[i]
            if not seen:
                dminy = v
                dmaxy = v
                seen = True
            else:
                if v < dminy:
                    dminy = v
                if v > dmaxy:
                    dmaxy = v
    if not seen:
        return
    miny = int(max(0.0, dminy))
    upper = dmaxy
    if upper > ny - 1:
        upper = float(ny - 1)
    maxy = int(upper)
    maxx = nx - 1
    crossings = np.empty(total, dtype=np.int64)
    for y in range(miny, maxy + 1):
        dy = y + 0.5
        count = 0
        for r in range(r0, r1):
            s = ring_start[r]
            e = ring_stop[r]
            flip = rev[r]
            for i in range(s, e):
                if i == s:
                    ind1 = e - 1
                    ind2 = s
                else:
                    ind1 = i - 1
                    ind2 = i
                if flip:
                    tmp = ind1
                    ind1 = ind2
                    ind2 = tmp
                dy1 = py[ind1]
                dy2 = py[ind2]
                if (dy1 < dy and dy2 < dy) or (dy1 > dy and dy2 > dy):
                    continue
                if dy1 < dy2:
                    dx1 = px[ind1]
                    dx2 = px[ind2]
                elif dy1 > dy2:
                    tmp = dy1
                    dy1 = dy2
                    dy2 = tmp
                    dx2 = px[ind1]
                    dx1 = px[ind2]
                else:
                    # Horizontal edge: only the bottom edge of a pair burns.
                    if px[ind1] > px[ind2]:
                        dh1 = math.floor(px[ind2] + 0.5)
                        dh2 = math.floor(px[ind1] + 0.5)
                        if dh1 > maxx or dh2 <= 0:
                            continue
                        hx1 = int(max(dh1, 0.0))
                        hx2 = int(min(dh2, float(nx)))
                        _burn_scanline(out, y, hx1, hx2 - 1, value)
                    continue
                if dy < dy2 and dy >= dy1:
                    t = (dy - dy1) * (dx2 - dx1) / (dy2 - dy1) + dx1
                    if t < _INT_MIN:
                        t = _INT_MIN
                    elif t > _INT_MAX:
                        t = _INT_MAX
                    crossings[count] = int(math.floor(t + 0.5))
                    count += 1
        crossings[:count].sort()
        k = 0
        while k + 1 < count:
            xa = crossings[k]
            xb = crossings[k + 1]
            if xa <= maxx and xb > 0:
                _burn_scanline(out, y, xa, xb - 1, value)
            k += 2


@nb.jit(nopython=True, nogil=True, cache=True)
def _fill_bucketed(
    out,
    px,
    py,
    ring_start,
    ring_stop,
    rev,
    r0,
    r1,
    value,
    row_count,
    row_off,
    xs,
):
    """Fill one polygon by bucketing edge crossings into per-row slots.

    The set of (row, crossing) pairs is identical to the per-row kernel; only
    the visitation order changes, and the per-row sort makes order
    irrelevant. row_count and row_off are int64 scratch of length ny + 1 and
    xs is int64 scratch for the crossings; they are passed in so one
    allocation serves every polygon of a call. A flagged ring is walked in
    reverse by swapping its endpoint indices, which only the horizontal-edge
    direction test observes. An edge with a NaN endpoint is skipped in both
    the counting and the crossing pass so the two stay in step; inf endpoints
    are clamped by the row-window checks and behave as in the per-row kernel.
    Returns True when the polygon was filled and False, without completing,
    when the crossing total exceeds the scratch capacity so the caller can
    fall back to the per-row kernel.
    """
    ny = out.shape[0]
    nx = out.shape[1]
    maxx = nx - 1
    dminy = math.inf
    dmaxy = -math.inf
    for r in range(r0, r1):
        for i in range(ring_start[r], ring_stop[r]):
            v = py[i]
            if v < dminy:
                dminy = v
            if v > dmaxy:
                dmaxy = v
    if dminy == math.inf or dmaxy == -math.inf:
        return True
    # Rows the per-row kernel would visit; nothing outside can burn.
    if dminy < 0.0:
        dminy = 0.0
    if dmaxy > ny - 1:
        dmaxy = float(ny - 1)
    miny = int(dminy)
    maxy = int(dmaxy)
    if miny > maxy:
        return True
    nrows = maxy - miny + 1
    for y in range(nrows + 1):
        row_count[y] = 0
    # Pass 1: count crossings per row and burn horizontal edges. An edge
    # oriented so dy1 < dy2 crosses row y exactly when dy1 <= y + 0.5 < dy2.
    for r in range(r0, r1):
        s = ring_start[r]
        e = ring_stop[r]
        flip = rev[r]
        for i in range(s, e):
            if i == s:
                ind1 = e - 1
                ind2 = s
            else:
                ind1 = i - 1
                ind2 = i
            if flip:
                tmp = ind1
                ind1 = ind2
                ind2 = tmp
            dy1 = py[ind1]
            dy2 = py[ind2]
            if (
                math.isnan(dy1)
                or math.isnan(dy2)
                or math.isnan(px[ind1])
                or math.isnan(px[ind2])
            ):
                continue
            if dy1 == dy2:
                # Horizontal edge burns only when it sits on a scanline
                # center inside the row range, and only right to left.
                yf = dy1 - 0.5
                if yf != math.floor(yf) or yf < miny or yf > maxy:
                    continue
                if px[ind1] > px[ind2]:
                    dh1 = math.floor(px[ind2] + 0.5)
                    dh2 = math.floor(px[ind1] + 0.5)
                    if dh1 > maxx or dh2 <= 0:
                        continue
                    hx1 = int(max(dh1, 0.0))
                    hx2 = int(min(dh2, float(nx)))
                    _burn_scanline(out, int(yf), hx1, hx2 - 1, value)
                continue
            if dy1 > dy2:
                tmp = dy1
                dy1 = dy2
                dy2 = tmp
            ylo = math.ceil(dy1 - 0.5)
            yhi = math.ceil(dy2 - 0.5) - 1.0
            if ylo < miny:
                ylo = float(miny)
            if yhi > maxy:
                yhi = float(maxy)
            if ylo > yhi:
                continue
            a = int(ylo) - miny
            b = int(yhi) - miny
            for y in range(a, b + 1):
                row_count[y] += 1
    total = 0
    for y in range(nrows):
        c = row_count[y]
        row_off[y] = total
        total += c
        row_count[y] = 0
    row_off[nrows] = total
    if total > xs.shape[0]:
        return False
    # Pass 2: place each crossing into its row slot.
    for r in range(r0, r1):
        s = ring_start[r]
        e = ring_stop[r]
        for i in range(s, e):
            if i == s:
                ind1 = e - 1
                ind2 = s
            else:
                ind1 = i - 1
                ind2 = i
            dy1 = py[ind1]
            dy2 = py[ind2]
            if (
                math.isnan(dy1)
                or math.isnan(dy2)
                or math.isnan(px[ind1])
                or math.isnan(px[ind2])
            ):
                continue
            if dy1 == dy2:
                continue
            if dy1 < dy2:
                dx1 = px[ind1]
                dx2 = px[ind2]
            else:
                tmp = dy1
                dy1 = dy2
                dy2 = tmp
                dx2 = px[ind1]
                dx1 = px[ind2]
            ylo = math.ceil(dy1 - 0.5)
            yhi = math.ceil(dy2 - 0.5) - 1.0
            if ylo < miny:
                ylo = float(miny)
            if yhi > maxy:
                yhi = float(maxy)
            if ylo > yhi:
                continue
            a = int(ylo)
            b = int(yhi)
            for y in range(a, b + 1):
                dy = y + 0.5
                t = (dy - dy1) * (dx2 - dx1) / (dy2 - dy1) + dx1
                if t < _INT_MIN:
                    t = _INT_MIN
                elif t > _INT_MAX:
                    t = _INT_MAX
                k = y - miny
                xs[row_off[k] + row_count[k]] = int(math.floor(t + 0.5))
                row_count[k] += 1
    # Pass 3: sort each row's crossings and burn the spans between pairs.
    for k in range(nrows):
        lo = row_off[k]
        hi = row_off[k + 1]
        count = hi - lo
        if count < 2:
            continue
        if count <= 32:
            _sort_small(xs, lo, hi)
        else:
            xs[lo:hi].sort()
        y = k + miny
        j = lo
        while j + 1 < hi:
            xa = xs[j]
            xb = xs[j + 1]
            if xa <= maxx and xb > 0:
                _burn_scanline(out, y, xa, xb - 1, value)
            j += 2
    return True


@nb.jit(nopython=True, nogil=True, cache=True)
def _line_bresenham_part(out, px, py, s, e, value):
    ny = out.shape[0]
    nx = out.shape[1]
    for i in range(s + 1, e):
        dfX = px[i - 1]
        dfY = py[i - 1]
        dfXEnd = px[i]
        dfYEnd = py[i]
        if (
            (dfY < 0.0 and dfYEnd < 0.0)
            or (dfY > ny and dfYEnd > ny)
            or (dfX < 0.0 and dfXEnd < 0.0)
            or (dfX > nx and dfXEnd > nx)
        ):
            continue
        if not (
            _INT_MIN <= dfX <= _INT_MAX
            and _INT_MIN <= dfY <= _INT_MAX
            and _INT_MIN <= dfXEnd <= _INT_MAX
            and _INT_MIN <= dfYEnd <= _INT_MAX
        ):
            continue
        iX = int(math.floor(dfX))
        iY = int(math.floor(dfY))
        iX1 = int(math.floor(dfXEnd))
        iY1 = int(math.floor(dfYEnd))
        nDeltaX = abs(iX1 - iX)
        nDeltaY = abs(iY1 - iY)
        nXStep = -1 if iX > iX1 else 1
        nYStep = -1 if iY > iY1 else 1
        if nDeltaX >= nDeltaY:
            nXError = nDeltaY << 1
            nYError = nXError - (nDeltaX << 1)
            nError = nXError - nDeltaX
            if i != e - 1:
                nDeltaX -= 1
            while nDeltaX >= 0:
                nDeltaX -= 1
                _burn_point(out, iY, iX, value)
                iX += nXStep
                if nError > 0:
                    iY += nYStep
                    nError += nYError
                else:
                    nError += nXError
        else:
            nXError = nDeltaX << 1
            nYError = nXError - (nDeltaY << 1)
            nError = nXError - nDeltaY
            if i != e - 1:
                nDeltaY -= 1
            while nDeltaY >= 0:
                nDeltaY -= 1
                _burn_point(out, iY, iX, value)
                iY += nYStep
                if nError > 0:
                    iX += nXStep
                    nError += nYError
                else:
                    nError += nXError


@nb.jit(nopython=True, nogil=True, cache=True)
def _line_all_touched_part(out, px, py, s, e, value, intersect_only):
    ny = out.shape[0]
    nx = out.shape[1]
    nxf = float(nx)
    nyf = float(ny)
    for i in range(s + 1, e):
        dfX = px[i - 1]
        dfY = py[i - 1]
        dfXEnd = px[i]
        dfYEnd = py[i]
        if (
            (dfY < 0.0 and dfYEnd < 0.0)
            or (dfY > nyf and dfYEnd > nyf)
            or (dfX < 0.0 and dfXEnd < 0.0)
            or (dfX > nxf and dfXEnd > nxf)
        ):
            continue
        if not (
            _INT_MIN <= dfX <= _INT_MAX
            and _INT_MIN <= dfY <= _INT_MAX
            and _INT_MIN <= dfXEnd <= _INT_MAX
            and _INT_MIN <= dfYEnd <= _INT_MAX
        ):
            continue
        if dfX > dfXEnd:
            tmp = dfX
            dfX = dfXEnd
            dfXEnd = tmp
            tmp = dfY
            dfY = dfYEnd
            dfYEnd = tmp
        if abs(dfX - dfXEnd) < 0.01:
            # Vertical segment.
            if intersect_only and (
                abs(dfX - round(dfX)) < _EPSILON
                and abs(dfXEnd - round(dfXEnd)) < _EPSILON
            ):
                continue
            if dfYEnd < dfY:
                tmp = dfY
                dfY = dfYEnd
                dfYEnd = tmp
            iX = int(math.floor(dfXEnd))
            iY = int(math.floor(dfY))
            iYEnd = int(math.floor(dfYEnd - _EPSILON))
            if iX < 0 or iX >= nx:
                continue
            if iY < 0:
                iY = 0
            if iYEnd >= ny:
                iYEnd = ny - 1
            yy = iY
            while yy <= iYEnd:
                _burn_point(out, yy, iX, value)
                yy += 1
            continue
        if abs(dfY - dfYEnd) < 0.01:
            # Horizontal segment.
            if intersect_only and (
                abs(dfY - round(dfY)) < _EPSILON
                and abs(dfYEnd - round(dfYEnd)) < _EPSILON
            ):
                continue
            if dfXEnd < dfX:
                tmp = dfX
                dfX = dfXEnd
                dfXEnd = tmp
            iX = int(math.floor(dfX))
            iY = int(math.floor(dfY))
            iXEnd = int(math.floor(dfXEnd - _EPSILON))
            if iY < 0 or iY >= ny:
                continue
            if iX < 0:
                iX = 0
            if iXEnd >= nx:
                iXEnd = nx - 1
            xx = iX
            while xx <= iXEnd:
                _burn_point(out, iY, xx, value)
                xx += 1
            continue
        dfSlope = (dfYEnd - dfY) / (dfXEnd - dfX)
        if dfXEnd > nxf:
            dfYEnd -= (dfXEnd - nxf) * dfSlope
            dfXEnd = nxf
        if dfX < 0.0:
            dfY += (0.0 - dfX) * dfSlope
            dfX = 0.0
        if dfYEnd > dfY:
            if dfY < 0.0:
                dfX += (0.0 - dfY) / dfSlope
                dfY = 0.0
            if dfYEnd >= nyf:
                dfXEnd += (dfYEnd - nyf) / dfSlope
                if dfXEnd > nxf:
                    dfXEnd = nxf
        else:
            if dfY >= nyf:
                dfX += (nyf - dfY) / dfSlope
                dfY = nyf
            if dfYEnd < 0.0:
                dfXEnd -= dfYEnd / dfSlope
        while dfX >= 0.0 and dfX < dfXEnd:
            iX = int(math.floor(dfX))
            iY = int(math.floor(dfY))
            if 0 <= iY < ny:
                _burn_point(out, iY, iX, value)
            dfStepX = math.floor(dfX + 1.0) - dfX
            dfStepY = dfStepX * dfSlope
            if math.floor(dfY + dfStepY) == iY:
                dfX += dfStepX
                dfY += dfStepY
            elif dfSlope < 0:
                dfStepY = iY - dfY
                if dfStepY > -1e-9:
                    dfStepY = -1e-9
                dfStepX = dfStepY / dfSlope
                dfX += dfStepX
                dfY += dfStepY
            else:
                dfStepY = (iY + 1) - dfY
                if dfStepY < 1e-9:
                    dfStepY = 1e-9
                dfStepX = dfStepY / dfSlope
                dfX += dfStepX
                dfY += dfStepY


@nb.jit(nopython=True, nogil=True, cache=True)
def _burn_points(out, px, py, point_value):
    ny = out.shape[0]
    nx = out.shape[1]
    for i in range(point_value.shape[0]):
        dfX = px[i]
        dfY = py[i]
        if not (math.isfinite(dfX) and math.isfinite(dfY)):
            continue
        if (
            dfX < _INT_MIN
            or dfX > _INT_MAX
            or dfY < _INT_MIN
            or dfY > _INT_MAX
        ):
            continue
        if 0.0 <= dfX < nx and 0.0 <= dfY < ny:
            out[int(dfY), int(dfX)] = point_value[i]


@nb.jit(nopython=True, nogil=True, cache=True)
def _burn_polygons(
    out,
    px,
    py,
    ring_start,
    ring_stop,
    rev,
    poly_ring_lo,
    poly_ring_hi,
    poly_value,
    all_touched,
    scratch_cap,
):
    ny = out.shape[0]
    need = _max_span(
        px, py, ring_start, ring_stop, poly_ring_lo, poly_ring_hi, ny
    )
    if need > scratch_cap:
        need = scratch_cap
    xs = np.empty(need, dtype=np.int64)
    row_count = np.empty(ny + 1, dtype=np.int64)
    row_off = np.empty(ny + 1, dtype=np.int64)
    for p in range(poly_value.shape[0]):
        v = poly_value[p]
        r0 = poly_ring_lo[p]
        r1 = poly_ring_hi[p]
        if all_touched:
            for r in range(r0, r1):
                _line_all_touched_part(
                    out, px, py, ring_start[r], ring_stop[r], v, True
                )
        done = _fill_bucketed(
            out,
            px,
            py,
            ring_start,
            ring_stop,
            rev,
            r0,
            r1,
            v,
            row_count,
            row_off,
            xs,
        )
        if not done:
            _fill_one_polygon(
                out, px, py, ring_start, ring_stop, rev, r0, r1, v
            )


@nb.jit(nopython=True, nogil=True, cache=True)
def _burn_lines(out, px, py, part_start, part_stop, part_value, all_touched):
    for k in range(part_value.shape[0]):
        v = part_value[k]
        s = part_start[k]
        e = part_stop[k]
        if all_touched:
            _line_all_touched_part(out, px, py, s, e, v, False)
        else:
            _line_bresenham_part(out, px, py, s, e, v)


def _to_pixel(coords, transform):
    inv = ~transform
    x = coords[:, 0]
    y = coords[:, 1]
    px = inv.a * x + inv.b * y + inv.c
    py = inv.d * x + inv.e * y + inv.f
    return np.ascontiguousarray(px), np.ascontiguousarray(py)


def _cast_burn_values(values, out_dtype):
    values = np.asarray(values)
    if np.issubdtype(out_dtype, np.integer) and np.issubdtype(
        values.dtype, np.floating
    ):
        # GDAL converts a float burn value into an integer raster by rounding
        # half away from zero, not by truncating. Integer inputs are burned
        # directly so wide integer values keep full precision.
        rounded = np.trunc(values + np.copysign(0.5, values))
        return np.ascontiguousarray(rounded.astype(out_dtype))
    return np.ascontiguousarray(values.astype(out_dtype))


def _reverse_parts(px, py, part_start, part_stop):
    n = px.shape[0]
    if n == 0:
        return px, py
    lengths = part_stop - part_start
    per_start = np.repeat(part_start, lengths)
    per_stop = np.repeat(part_stop, lengths)
    positions = np.arange(n)
    rev = per_start + per_stop - 1 - positions
    return np.ascontiguousarray(px[rev]), np.ascontiguousarray(py[rev])


@nb.jit(nopython=True, nogil=True, cache=True)
def _ring_reverse_mask(x, y, ring_start, ring_stop):
    """Flag rings GDAL would reverse to normalize them to clockwise.

    Transcribes OGRCurve::isClockwise: the winding is read from the local
    turn at the lowest-rightmost vertex, falling back to a signed-area sum
    only for degenerate pivots. For a self-intersecting ring the local turn
    and the global signed area can disagree, so the two must not be
    interchanged. A ring GDAL finds counterclockwise is reversed.
    """
    eps = 1.0e-5
    nrings = ring_start.shape[0]
    reverse = np.zeros(nrings, dtype=np.bool_)
    for r in range(nrings):
        s = ring_start[r]
        e = ring_stop[r]
        n = e - s
        if n < 3:
            continue
        use_fallback = False
        v = 0
        selx = x[s]
        sely = y[s]
        before_sel = -1
        next_sel = -1
        next_is_next_sel = True
        before = 0
        for i in range(1, n - 1):
            curx = x[s + i]
            cury = y[s + i]
            if next_is_next_sel:
                next_sel = i
                next_is_next_sel = False
            if cury < sely or (cury == sely and curx > selx):
                v = i
                before_sel = before
                selx = curx
                sely = cury
                use_fallback = False
                next_is_next_sel = True
            elif cury == sely and curx == selx:
                use_fallback = True
            before = i
        if next_is_next_sel:
            next_sel = n - 2
        if v == 0:
            before_sel = n - 2
        bsx = x[s + before_sel]
        bsy = y[s + before_sel]
        if abs(bsx - selx) < eps and abs(bsy - sely) < eps:
            use_fallback = True
        dx0 = bsx - selx
        dy0 = bsy - sely
        if v + 1 >= n - 1:
            next_sel = 0
        nsx = x[s + next_sel]
        nsy = y[s + next_sel]
        if abs(nsx - selx) < eps and abs(nsy - sely) < eps:
            use_fallback = True
        dx1 = nsx - selx
        dy1 = nsy - sely
        crossproduct = dx1 * dy0 - dx0 * dy1
        clockwise = True
        decided = False
        if not use_fallback:
            if crossproduct > 0.0:
                clockwise = False
                decided = True
            elif crossproduct < 0.0:
                clockwise = True
                decided = True
        if not decided:
            dfSum = x[s] * (y[s + 1] - y[s])
            for i in range(1, n - 1):
                dfSum += x[s + i] * (y[s + i + 1] - y[s + i - 1])
            dfSum += x[s + n - 1] * (y[s] - y[s + n - 2])
            clockwise = dfSum < 0.0
        if not clockwise:
            reverse[r] = True
    return reverse


def _polygonal_ragged(geometry, ids):
    """Return (name, coords, offsets) for a polygon-family array.

    An array that is all POLYGON with no interior rings gets its ring and
    geometry offsets straight from the per-geometry coordinate counts, which
    avoids to_ragged_array's interior-ring materialization. MultiPolygon
    arrays and polygons with holes go through to_ragged_array. The layout is
    the one to_ragged_array would produce for the same input.
    """
    if (ids == 3).all():
        nint = shapely.get_num_interior_rings(geometry)
        if not nint.any():
            coords = shapely.get_coordinates(geometry)
            counts = shapely.get_num_coordinates(geometry).astype(np.int64)
            ring_offsets = np.empty(len(geometry) + 1, dtype=np.int64)
            ring_offsets[0] = 0
            np.cumsum(counts, out=ring_offsets[1:])
            geom_offsets = np.arange(len(geometry) + 1, dtype=np.int64)
            return "POLYGON", coords, (ring_offsets, geom_offsets)
    geom_type, coords, offsets = shapely.to_ragged_array(geometry)
    return geom_type.name, coords, offsets


def _burn_polygonal(
    out,
    px,
    py,
    coords,
    offsets,
    values,
    name,
    all_touched,
    scratch_cap=_SCRATCH_CAP,
):
    if name == "POLYGON":
        ring_offsets = offsets[0].astype(np.int64)
        geom_offsets = offsets[1].astype(np.int64)
        poly_ring_lo = geom_offsets[:-1]
        poly_ring_hi = geom_offsets[1:]
        poly_value = values
    else:
        ring_offsets = offsets[0].astype(np.int64)
        poly_offsets = offsets[1].astype(np.int64)
        geom_offsets = offsets[2].astype(np.int64)
        poly_ring_lo = poly_offsets[:-1]
        poly_ring_hi = poly_offsets[1:]
        poly_value = np.repeat(values, np.diff(geom_offsets))
    ring_start = np.ascontiguousarray(ring_offsets[:-1])
    ring_stop = np.ascontiguousarray(ring_offsets[1:])
    # GDAL normalizes every ring to clockwise before scan-filling. Rather
    # than physically reordering vertices, flag the rings GDAL would reverse
    # and let the kernel swap each flagged ring's endpoints; only the
    # horizontal-edge direction test depends on that order. Orientation is
    # measured on the untransformed world coordinates, matching GDAL.
    x = np.ascontiguousarray(coords[:, 0])
    y = np.ascontiguousarray(coords[:, 1])
    rev = _ring_reverse_mask(x, y, ring_start, ring_stop)
    _burn_polygons(
        out,
        px,
        py,
        ring_start,
        ring_stop,
        rev,
        np.ascontiguousarray(poly_ring_lo),
        np.ascontiguousarray(poly_ring_hi),
        np.ascontiguousarray(poly_value),
        all_touched,
        scratch_cap,
    )


def _burn_lineal(out, px, py, offsets, values, name, all_touched):
    if name == "LINESTRING":
        geom_offsets = offsets[0].astype(np.int64)
        part_start = geom_offsets[:-1]
        part_stop = geom_offsets[1:]
        part_value = values
    else:
        part_offsets = offsets[0].astype(np.int64)
        geom_offsets = offsets[1].astype(np.int64)
        part_start = part_offsets[:-1]
        part_stop = part_offsets[1:]
        part_value = np.repeat(values, np.diff(geom_offsets))
    part_start = np.ascontiguousarray(part_start)
    part_stop = np.ascontiguousarray(part_stop)
    # GDAL pushes each LineString's coordinates in reverse order, which
    # changes the Bresenham walk, so reverse each part before burning.
    px, py = _reverse_parts(px, py, part_start, part_stop)
    _burn_lines(
        out,
        px,
        py,
        part_start,
        part_stop,
        np.ascontiguousarray(part_value),
        all_touched,
    )


def _burn_punctal(out, px, py, offsets, values, name):
    if name == "POINT":
        point_value = values
    else:
        geom_offsets = offsets[0].astype(np.int64)
        point_value = np.repeat(values, np.diff(geom_offsets))
    _burn_points(out, px, py, np.ascontiguousarray(point_value))


def _dispatch_burn(out, transform, geometry, values, all_touched):
    if transform.b != 0.0 or transform.d != 0.0:
        raise _NumbaUnsupported("rotated or sheared affine")
    geometry = np.asarray(geometry, dtype=object)
    if not geometry.flags.writeable:
        # to_ragged_array writes into the geometry object buffer (through
        # get_rings for polygons), so a read-only array, such as the view a
        # GeoSeries exposes, must be copied to a writable buffer first. The
        # copy only moves object pointers and only runs when needed, so the
        # common writable-input path stays cheap.
        geometry = np.array(geometry, dtype=object)
    ids = shapely.get_type_id(geometry)
    valid = ids >= 0
    if not valid.any():
        return
    present = {int(v) for v in np.unique(ids[valid])}
    if present <= _POLYGONAL:
        name, coords, offsets = _polygonal_ragged(geometry, ids)
        px, py = _to_pixel(coords, transform)
        _burn_polygonal(
            out, px, py, coords, offsets, values, name, all_touched
        )
    elif present <= _LINEAL:
        geom_type, coords, offsets = shapely.to_ragged_array(geometry)
        px, py = _to_pixel(coords, transform)
        _burn_lineal(out, px, py, offsets, values, geom_type.name, all_touched)
    elif present <= _PUNCTAL:
        geom_type, coords, offsets = shapely.to_ragged_array(geometry)
        px, py = _to_pixel(coords, transform)
        _burn_punctal(out, px, py, offsets, values, geom_type.name)
    else:
        raise _NumbaUnsupported("mixed families or unsupported geometry type")


def numba_rasterize_wrapper(
    shape, transform, geometry, values, out_dtype, fill, all_touched
):
    # np.zeros gets calloc's lazily faulted pages, so a fill of zero (the
    # common no-data value) skips the memset of a fresh output chunk.
    if fill == 0:
        out = np.zeros(shape, dtype=out_dtype)
    else:
        out = np.full(shape, fill, dtype=out_dtype)
    geometry = np.asarray(geometry)
    values = _cast_burn_values(values, out_dtype)
    _dispatch_burn(out, transform, geometry, values, all_touched)
    return out


def numba_mask(geoms, shape, transform, all_touched, invert):
    fill, geom_value = (1, 0) if invert else (0, 1)
    geoms = np.asarray(geoms)
    if fill == 0:
        out = np.zeros(shape, dtype=U8)
    else:
        out = np.full(shape, fill, dtype=U8)
    values = np.full(len(geoms), geom_value, dtype=U8)
    _dispatch_burn(out, transform, geoms, values, all_touched)
    return out


def _warmup():
    """Compile every kernel once per output dtype, serially.

    Running this before a parallel test session populates numba's on-disk
    cache without the workers racing to write it.
    """
    from affine import Affine

    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0)
    box = shapely.box(0.2, 0.2, 1.8, 1.8)
    line = shapely.LineString([(0.2, 0.2), (1.8, 1.8)])
    point = shapely.Point(0.5, 0.5)
    dtypes = (
        np.dtype("uint8"),
        np.dtype("uint16"),
        np.dtype("uint32"),
        np.dtype("int16"),
        np.dtype("int32"),
        np.dtype("float32"),
        np.dtype("float64"),
    )
    for dtype in dtypes:
        for geom, all_touched in (
            (box, False),
            (box, True),
            (line, False),
            (line, True),
            (point, False),
        ):
            numba_rasterize_wrapper(
                (2, 2),
                transform,
                np.array([geom], dtype=object),
                np.ones(1, dtype=dtype),
                dtype,
                0,
                all_touched,
            )
    numba_mask(np.array([box], dtype=object), (2, 2), transform, True, False)
