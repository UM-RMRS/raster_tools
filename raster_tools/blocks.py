"""Block-mapping primitives for Rasters.

Provides :func:`map_blocks` and :func:`map_overlap` as thin wrappers over
the corresponding ``dask.array`` functions, plus :func:`geo_map_blocks`
and :func:`geo_map_overlap` which hand each user callback georeferenced
:class:`xarray.DataArray` blocks (with band/y/x coords, CRS, nodata)
instead of raw NumPy blocks. All four are also available as
:class:`~raster_tools.Raster` methods (``r.map_blocks(func, ...)``); the
calling raster is the first input and the method docstrings are shared
verbatim with the functions.

These are **shape-preserving** by default: ``func`` must return an
array-like of the same shape as a single input data block (the
overlap-included shape, for the ``_overlap`` variants). The output
Raster lands on the first input's grid; ``chunks=`` and other
graph-construction options are not exposed. The one exception is the
``out_bands`` argument of :func:`map_blocks` / :func:`geo_map_blocks`,
which changes the **band** count of the output (the y/x grid is still
preserved); the ``_overlap`` variants remain strictly shape-preserving.

Also defines :class:`GeoBlockInfo`, the geo-aware analog of dask's
``block_info`` dict: a per-block metadata object carrying the block's
:class:`~odc.geo.geobox.GeoBox`, its band/row/col slices into the parent
array, and helpers for padding/shifting the block window and reconstructing
an :class:`xarray.DataArray` for the block's data. The geo wrappers
build on this primitive to hand each user callback a georeferenced
``DataArray`` (with band/y/x coords, CRS, nodata) instead of a raw
NumPy block.
"""

import dask
import dask.array as da
import numpy as np
import xarray as xr
from affine import Affine
from dask.utils import has_keyword, parse_bytes
from odc.geo.geobox import GeoBox

from raster_tools._grids import are_all_grids_same
from raster_tools.dask_utils import chunks_to_array_locations
from raster_tools.dtypes import is_int
from raster_tools.masking import get_default_null_value
from raster_tools.raster import Raster, data_to_raster_like, get_raster
from raster_tools.utils import nan_equal

__all__ = [
    "GeoBlockInfo",
    "geo_block_infos",
    "geo_block_infos_as_dask",
    "geo_map_blocks",
    "geo_map_overlap",
    "map_blocks",
    "map_overlap",
]


class GeoBlockInfo:
    """Geo-aware metadata for a single dask block.

    Mirrors dask's per-block ``block_info`` dict, but carries the block's
    geographic grid (a 2D :class:`~odc.geo.geobox.GeoBox`) plus enough
    parent-array context for chunk-local code to locate itself within the
    full raster. Holds no pixel data.

    Parameters
    ----------
    geobox : odc.geo.geobox.GeoBox
        The 2D grid (``(ny, nx)``) for this block. Source of truth for
        ``affine``, ``crs``, and ``bbox``.
    band_slice, row_slice, col_slice : slice
        Half-open ``[start, stop)`` slices identifying where this block
        sits inside the parent array's band, row, and column axes.
    chunk_location : tuple of int
        The ``(bi, yi, xi)`` block index in the parent array's ``numblocks``
        grid. Preserved across mutators (still names the originating block
        even after padding/shifting).
    parent_affine : affine.Affine
        Affine of the full raster this block came from. Needed by callers
        like ``Raster.to_points`` to map block-local cells back to parent
        row/col.
    parent_shape : tuple of int
        ``(nbands, ny, nx)`` of the full raster. Used for parent-array
        ravel-index computations.
    """

    def __init__(
        self,
        geobox,
        band_slice,
        row_slice,
        col_slice,
        chunk_location,
        parent_affine,
        parent_shape,
    ):
        self.geobox = geobox
        self.band_slice = band_slice
        self.row_slice = row_slice
        self.col_slice = col_slice
        self.chunk_location = chunk_location
        self.parent_affine = parent_affine
        self.parent_shape = parent_shape

    def __repr__(self):
        return (
            f"<{type(self).__module__}.{type(self).__name__}"
            f" shape={self.shape}, chunk_location={self.chunk_location}>"
        )

    @property
    def affine(self):
        return self.geobox.affine

    @property
    def crs(self):
        return self.geobox.crs

    @property
    def bbox(self):
        return self.geobox.extent.geom

    @property
    def shape(self):
        """3D ``(nbands, ny, nx)`` block shape."""
        return (
            self.band_slice.stop - self.band_slice.start,
            self.geobox.shape.y,
            self.geobox.shape.x,
        )

    @property
    def x(self):
        return self.geobox.coordinates["x"].values

    @property
    def y(self):
        return self.geobox.coordinates["y"].values

    @property
    def band(self):
        return np.arange(self.band_slice.start, self.band_slice.stop)

    def to_dataarray(self, data, *, mask=None, nodata=None):
        """
        Wrap a block's pixel data as a georeferenced :class:`xarray.DataArray`.

        Builds an ``xr.DataArray`` with ``band``, ``y``, ``x`` coords drawn
        from this block's geobox and band slice, attaches the CRS via
        ``rio.write_crs``, and (optionally) writes ``nodata``.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel data with shape matching :attr:`shape`.
        mask : numpy.ndarray, optional
            Boolean mask with the same shape as ``data``. If provided,
            this method returns a 2-tuple of ``(xdata, xmask)`` with
            matching coords.
        nodata : scalar, optional
            Nodata value to write into the data DataArray's metadata.

        Returns
        -------
        xarray.DataArray or tuple of xarray.DataArray
            ``xdata`` if ``mask`` is None; otherwise ``(xdata, xmask)``.
        """
        if data.shape != self.shape:
            raise ValueError(
                f"data shape {data.shape} does not match block shape "
                f"{self.shape}"
            )
        coords = {"band": self.band, "y": self.y, "x": self.x}
        xdata = xr.DataArray(
            data, dims=("band", "y", "x"), coords=coords
        ).rio.set_spatial_dims(y_dim="y", x_dim="x")
        if self.crs is not None:
            xdata = xdata.rio.write_crs(self.crs)
        if nodata is not None:
            xdata = xdata.rio.write_nodata(nodata)
        if mask is None:
            return xdata
        if mask.shape != self.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match block shape "
                f"{self.shape}"
            )
        xmask = xr.DataArray(
            mask, dims=("band", "y", "x"), coords=coords
        ).rio.set_spatial_dims(y_dim="y", x_dim="x")
        if self.crs is not None:
            xmask = xmask.rio.write_crs(self.crs)
        return xdata, xmask

    # -- mutators ----------------------------------------------------------
    #
    # All mutators return a NEW GeoBlockInfo (immutable semantics). They
    # adjust the geobox (translating the affine when growing on the top/left)
    # and update row_slice / col_slice to track the window within the parent
    # array. parent_affine, parent_shape, band_slice, and chunk_location are
    # preserved.
    #
    # Negative pad values trim that side. Positive shift values move the
    # window down (pad_y) or right (pad_x).

    def _resize(self, dy_top, dy_bottom, dx_left, dx_right):
        if not all(is_int(v) for v in (dy_top, dy_bottom, dx_left, dx_right)):
            raise TypeError("resize amounts must be integers")
        new_ny = self.geobox.shape.y + dy_top + dy_bottom
        new_nx = self.geobox.shape.x + dx_left + dx_right
        if new_ny < 0 or new_nx < 0:
            raise ValueError(
                f"resize would produce negative shape: ({new_ny}, {new_nx})"
            )
        new_affine = self.geobox.affine
        if dy_top != 0 or dx_left != 0:
            # Affine.translation(xoff, yoff): xoff in cols, yoff in rows.
            # Growing on top/left moves the upper-left pixel center
            # outward, which means a negative col/row translation in the
            # affine's input space.
            new_affine = new_affine * Affine.translation(-dx_left, -dy_top)
        new_geobox = GeoBox((new_ny, new_nx), new_affine, self.crs)
        return GeoBlockInfo(
            new_geobox,
            self.band_slice,
            slice(
                self.row_slice.start - dy_top,
                self.row_slice.stop + dy_bottom,
            ),
            slice(
                self.col_slice.start - dx_left,
                self.col_slice.stop + dx_right,
            ),
            self.chunk_location,
            self.parent_affine,
            self.parent_shape,
        )

    def pad_y(self, top, bottom):
        """Grow (or trim, with negative values) the block on top/bottom."""
        return self._resize(top, bottom, 0, 0)

    def pad_x(self, left, right):
        """Grow (or trim, with negative values) the block on left/right."""
        return self._resize(0, 0, left, right)

    def pad(self, y, x=None):
        """Symmetrically pad the block in both axes by ``y`` and ``x``.

        ``x`` defaults to ``y``. Each axis is grown by twice the given
        value because both sides are padded.
        """
        if x is None:
            x = y
        return self._resize(y, y, x, x)

    def shift_y(self, n):
        """Slide the block window vertically. Positive ``n`` moves down."""
        return self._resize(-n, n, 0, 0)

    def shift_x(self, n):
        """Slide the block window horizontally. Positive ``n`` moves right."""
        return self._resize(0, 0, -n, n)

    def shift(self, y, x=None):
        """Slide the window in both axes."""
        if x is None:
            x = y
        return self._resize(-y, y, -x, x)


def _slices_from_locations(locations):
    return [slice(start, stop) for start, stop in locations]


def geo_block_infos(raster):
    """Build a NumPy object-array of :class:`GeoBlockInfo` for a Raster.

    The returned array has shape ``raster.data.numblocks`` and each entry
    describes the corresponding dask block.
    """
    parent_geobox = raster.geobox
    parent_affine = raster.affine
    parent_shape = raster.shape
    chunks = raster.data.chunks
    numblocks = raster.data.numblocks
    band_slices = _slices_from_locations(chunks_to_array_locations(chunks[0]))
    row_slices = _slices_from_locations(chunks_to_array_locations(chunks[1]))
    col_slices = _slices_from_locations(chunks_to_array_locations(chunks[2]))
    out = np.empty(numblocks, dtype=object)
    for bi, yi, xi in np.ndindex(numblocks):
        rs = row_slices[yi]
        cs = col_slices[xi]
        block_geobox = parent_geobox[rs, cs]
        out[bi, yi, xi] = GeoBlockInfo(
            block_geobox,
            band_slices[bi],
            rs,
            cs,
            (bi, yi, xi),
            parent_affine,
            parent_shape,
        )
    return out


def geo_block_infos_as_dask(raster):
    """Return a chunksize-1 dask object-array of :class:`GeoBlockInfo`.

    Suitable as a side-input to :func:`dask.array.map_blocks` aligned 1:1
    with the data blocks.
    """
    return da.from_array(geo_block_infos(raster), chunks=1)


# Kwarg names reserved for map_blocks's per-block injection. The user's
# func opts in to any of these by including them as named parameters.
# The caller of map_blocks may not pass them via **kwargs.
_MAP_BLOCKS_RESERVED_KWARGS = frozenset(
    {
        "input_masks",
        "input_null_values",
        "block_info",
        "block_id",
        "out_null_value",
    }
)

# Kwarg names that dask's map_blocks / map_overlap accept as graph-
# construction options. If a caller passes one via our **kwargs path it
# would silently alter dask's graph (or, for names dask doesn't
# recognize on the path taken, be forwarded to their per-block func) --
# either way bypassing our shape-preserving / mask-rebuilding contract.
# Reject upfront with a message pointing them at dask directly.
# ``enforce_ndim`` is map_blocks-only and ``allow_rechunk`` is
# map_overlap-only, but both flavors share this set; rejecting a name
# the other flavor wouldn't consume is harmless and keeps the contract
# uniform.
_DASK_RESERVED_KWARGS = frozenset(
    {
        "chunks",
        "name",
        "token",
        "drop_axis",
        "new_axis",
        "enforce_ndim",
        "concatenate",
        "align_arrays",
        "trim",
        "allow_rechunk",
    }
)


def _meta_dtype(meta):
    """Read ``meta``'s dtype without forcing a host-side materialization.

    Prefer the ``.dtype`` attribute (numpy / cupy / sparse / dask
    arrays all have one) and fall back to ``np.asarray`` only for
    array-likes that don't expose it directly.
    """
    dtype = getattr(meta, "dtype", None)
    if dtype is not None:
        return np.dtype(dtype)
    return np.asarray(meta).dtype


def _out_dtype_hint(dtype, meta):
    """Resolve the user-supplied output dtype, if any.

    Wrapper builders use this to pre-resolve ``out_null_value``
    independently of dask's per-chunk ``block_info[None]["dtype"]``,
    which is ``None`` when ``meta=`` is set (dask doesn't carry the
    dtype through block_info in that case, even though the meta's
    dtype is the actual output dtype).
    """
    if meta is not None:
        return _meta_dtype(meta)
    if dtype is not None:
        return np.dtype(dtype)
    return None


def _check_no_dask_kwargs(kwargs):
    collision = _DASK_RESERVED_KWARGS & kwargs.keys()
    if collision:
        raise ValueError(
            f"these kwargs are dask graph-construction options and are "
            f"not forwarded to your func; call dask directly if you need "
            f"them: {sorted(collision)}"
        )


def _check_reserved_kwargs(kwargs, reserved_set, fname):
    """Raise if any reserved-injection kwarg name leaked through ``**kwargs``.

    The four public functions (and the inference helpers) reserve a
    set of kwarg names for per-block injection; passing any of them
    via ``**kwargs`` would silently clobber the wrapper's value.
    """
    collision = reserved_set & kwargs.keys()
    if collision:
        raise ValueError(
            f"these kwargs are reserved by {fname} for per-block "
            f"injection and cannot be passed by the caller: "
            f"{sorted(collision)}"
        )


def _check_shape_aligned(rasters):
    """Raise on any raster that doesn't match ``rasters[0]``'s 3D shape.

    Used by the four public functions and the inference / resolve
    helpers. Permissive: doesn't check CRS / affine -- callers that
    require strict grid alignment add :func:`are_all_grids_same`.
    """
    ref = rasters[0]
    for i, r in enumerate(rasters[1:], 1):
        if r.shape != ref.shape:
            raise ValueError(
                f"raster {i} shape {r.shape} does not match raster 0 "
                f"shape {ref.shape}. Use ``Raster.reproject`` to align "
                f"inputs to the same grid first, e.g. "
                f"``r{i}.reproject(r0.geobox)``."
            )


def _check_dtype_meta_agree(dtype, meta):
    """Raise if both dtype and meta are set and their dtypes disagree.

    dask doesn't validate this combination -- it lets ``.dtype``
    report the kwarg while ``_meta`` carries the meta's dtype and
    the actual computed dtype is whatever ``func`` returns. Catch
    the mismatch upfront.
    """
    if dtype is None or meta is None:
        return
    meta_dtype = _meta_dtype(meta)
    if np.dtype(dtype) != meta_dtype:
        raise ValueError(
            f"dtype={dtype!r} conflicts with meta.dtype={meta_dtype!r}; "
            "pass one or the other, or matching values."
        )


def map_blocks(
    func,
    *rasters,
    dtype=None,
    null_value=None,
    meta=None,
    out_bands=None,
    return_mask=False,
    **kwargs,
):
    """Apply ``func`` block-wise across one or more aligned rasters.

    Thin wrapper over :func:`dask.array.map_blocks`. Each call to ``func``
    receives one block from each input raster, in input order. The output
    :class:`~raster_tools.Raster` adopts its CRS, affine, and x/y coords
    from the first input; its mask is derived from the output data by
    default, or set explicitly by ``func`` via ``return_mask`` (see Notes).

    Available both as a module function and as a
    :class:`~raster_tools.Raster` method: ``r1.map_blocks(func, r2, ...)``
    is equivalent to ``map_blocks(func, r1, r2, ...)`` -- in the method
    form the calling raster is the first input. References to the "first
    input" below mean ``r1`` in either spelling.

    **Per-block contract**

    Per-block kwargs (opt-in): ``input_masks``, ``input_null_values``,
    ``block_info``, ``block_id``, ``out_null_value``. Name any of these in
    ``func``'s signature to receive them per chunk; see below.

    By default the output Raster's mask is rebuilt from the output data and the
    resolved output null value (``out_data == null_value``, or
    ``np.isnan(out_data)`` for NaN nulls) -- write the sentinel only at cells
    you want masked. Cells your func happens to leave equal to the sentinel
    will appear masked even if you didn't intend them to; cells you wanted
    masked but didn't write the sentinel to will not. This is true regardless
    of the input rasters' masks: input masks do **not** carry through
    unchanged. To set the output mask explicitly instead -- decoupling nullness
    from the data values and avoiding both failure modes above -- pass
    ``return_mask=True`` and have ``func`` return a ``(data, mask)`` pair; see
    ``return_mask`` below.

    Always passed positionally::

        func(*input_data, **kwargs)

    where ``input_data`` is a tuple of N NumPy blocks, one per input raster, in
    input order.

    The user can also opt in to receive per-block extras by including named
    parameters in ``func``'s signature. Detection mirrors dask's own
    ``block_info=`` / ``block_id=`` mechanism and uses
    :func:`inspect.signature` (via :func:`dask.utils.has_keyword`).
    Recognized names:

    - ``input_masks`` -- tuple of N ``np.ndarray`` (bool) per-block mask
      arrays, parallel to ``input_data``.
    - ``input_null_values`` -- tuple of N scalars, each input raster's
      ``null_value`` (``None`` if unset).
    - ``block_info`` -- dask's standard per-block info dict (see
      :func:`dask.array.map_blocks`).
    - ``block_id`` -- dask's standard per-block index tuple: the block's
      ``chunk-location`` (equivalently ``block_info[None]["chunk-location"]``).
      ``None`` during the meta inference call.
    - ``out_null_value`` -- scalar; the resolved output null value the wrapper
      will use to derive the output mask. Write this sentinel at cells you want
      masked. See "Output null value resolution" in Notes below.

    A function whose only kwargs absorber is ``**kwargs`` (e.g. ``def f(*args,
    **kwargs):``) does NOT trigger any of these injections --
    ``inspect.signature`` only sees explicit parameter names. Name the kwargs
    you want.

    **Reserved kwargs**

    ``input_masks``, ``input_null_values``, ``block_info``, ``block_id``, and
    ``out_null_value`` are reserved. Passing any of them via ``map_blocks``'s
    own ``**kwargs`` raises ``ValueError`` -- otherwise the wrapper's injection
    would silently clobber the caller's value.

    Parameters
    ----------
    func : callable
        Per-block function. See "Per-block contract" above for the full
        signature rules. Must return an array-like that dask can ingest
        (NumPy ndarray, cupy ndarray, sparse array, etc.) with the same
        shape as a single input data block. For non-numpy backends, also
        pass ``meta=`` so dask's output meta is correct.
    *rasters : Raster or str
        The input rasters. In the method form the calling raster is the
        first input and ``*rasters`` holds any additional ones; in the
        function form at least one is required. Path strings are accepted.
        Only the 3D shape is validated; CRS and affine are *not* checked.
        The caller is responsible for aligning inputs -- typically via
        ``r2.reproject(r1.geobox)``. Inputs are auto-rechunked to the first
        input's chunk structure, so same-shape inputs with differing
        chunking are handled. For a geo-aware variant that strictly
        requires matching grids, see :func:`~raster_tools.geo_map_blocks`.
    dtype : dtype-like, optional
        Output dtype. When ``None`` (default), dask infers the dtype by calling
        ``func`` on tiny meta samples (matches NumPy promotion for the typical
        elementwise case, and reflects any in-func cast such as ``.astype``).
    null_value : scalar, optional
        Output null value.

        - ``None`` (default): if there is exactly one input raster and the
          output dtype matches its dtype, the output inherits that input's null
          value (preserves the sentinel for identity-like single-input ops).
          Otherwise, the value is a dtype-appropriate default from
          :func:`raster_tools.masking.get_default_null_value` against the
          resolved output dtype -- always representable, never overflows when
          the dtype changes.
        - scalar: used as-is.

        To force inheriting the first input's null value across other
        cases, pass it explicitly: ``null_value=r1.null_value``.
    meta : array-like, optional
        Empty array with the desired output array type (e.g. ``np.array((),
        dtype=np.float32)``, or a CuPy / sparse equivalent). Forwarded to
        :func:`dask.array.map_blocks`. When provided, dask uses this as the
        output meta and skips the 0-shape sample call it would otherwise make
        to derive one -- useful when ``func`` cannot tolerate 0-shape input.
        When ``None`` (default), dask derives a NumPy meta by calling ``func``
        on 0-shape inputs.
    out_bands : int, optional
        Number of bands in the output. ``None`` (default) is shape-preserving:
        the output has the same band count as the input and ``func`` returns
        same-shape blocks. A positive integer lets ``func`` change the band
        count (the y/x grid is unchanged), e.g. ``out_bands=3`` to emit three
        per-pixel statistics from a 1-band input, or ``out_bands=1`` to
        collapse a multi-band stack. ``func`` must return exactly ``out_bands``
        bands per block (a mismatch raises ``ValueError``). Consequences when
        set:

        - The input band axis is collapsed to a **single chunk** so ``func``
          sees every band of a spatial tile at once, and the y/x tiles are
          transparently re-sized so a block holds roughly one ``dask``
          ``array.chunk-size`` worth of data -- so ``func`` may see different
          spatial tile sizes than the input's native chunking.
        - The output Raster is restored to the input's **original** y/x
          chunking, with per-band band chunks and band coord
          ``np.arange(out_bands) + 1``.
        - Passing ``dtype=`` or ``meta=`` is recommended, since dask's 0-shape
          meta call still runs and a band-reshaping ``func`` may not produce
          the right dtype/shape at 0-shape.

        Setting ``out_bands`` to the input's band count is a valid way to give
        ``func`` an all-bands view of each spatial tile: the default per-band
        blocking calls ``func`` once per band, whereas the band collapse above
        hands it every band of a tile at once. The count is unchanged, but the
        exact-``out_bands`` return guard and the y/x re-tile still apply.
    return_mask : bool, optional
        If ``True``, ``func`` must return a ``(data, mask)`` pair instead of a
        single array. The returned ``mask`` -- not a sentinel comparison --
        defines the output Raster's null cells. ``mask`` is a boolean array the
        same shape as ``data``; masked cells are set to the resolved output
        null value (burned in). Use this to decouple *which* cells are null
        from *what value* they hold, avoiding the sentinel-collision pitfalls
        described above. The default is ``False`` (sentinel-derived mask).
        Notes:

        - The two arrays are carried through dask packed into a single NumPy
          structured-dtype block, then split apart again -- an internal detail;
          ``func`` just returns the plain pair.
        - Passing ``dtype=`` (or ``meta=``) describing the **data** dtype is
          recommended: it lets dask skip its 0-shape probe entirely. Without a
          hint the func must tolerate that probe (same caveat as
          ``out_bands``).
        - Requires NumPy-backed blocks (the structured-dtype carrier is a NumPy
          concept); cupy / sparse outputs are not supported with
          ``return_mask``. Composes with ``out_bands`` (the returned ``mask``
          must also have ``out_bands`` bands).
        - ``out_null_value`` injection is unnecessary (though harmless) when
          ``return_mask=True``.
    **kwargs
        Extra keyword arguments forwarded per-block to ``func``. The reserved
        names listed above are not allowed here.

    Returns
    -------
    Raster
        A new lazy Raster on the first input's grid (with ``out_bands`` bands
        when that argument is set).

    Notes
    -----
    Cross-input CRS or affine mismatches are not validated; the caller is
    responsible.

    The output mask is all-False if no null value is set (see the per-block
    contract above for how the mask is built when one is). To write the output
    mask directly, pass ``return_mask=True`` and return a ``(data, mask)`` pair
    from ``func`` (see ``return_mask``).

    Dask invokes ``func`` once on 0-shape inputs to derive the output array
    meta -- this happens whether or not ``dtype=`` is provided. Pass ``meta=``
    to skip the call entirely; ``dtype=`` only skips the additional sample call
    dask would otherwise make to infer the output dtype, not the 0-shape meta
    call. (Exception: with ``return_mask=True``, ``dtype=`` is folded into a
    structured ``meta=`` and so *does* skip the 0-shape meta call -- see
    ``return_mask``.) Most NumPy ops handle 0-shape inputs fine. If dask raises
    ``dtype inference failed in map_blocks. Please specify the dtype explicitly
    using the dtype kwarg``, that's the sample call -- ``dtype=`` per dask's
    hint is usually enough; pass ``meta=`` instead if your func also can't
    tolerate 0-shape inputs (dask silently swallows that crash, but your
    downstream output meta will be wrong).

    **Output null value resolution**

    When ``func`` opts in to ``out_null_value``, the wrapper resolves the
    scalar per-chunk without an extra dtype-inference pass:

    - With ``meta=`` or ``dtype=`` set, from that hint (dask leaves
      ``block_info[None]["dtype"]`` as ``None`` whenever ``meta=`` is set,
      which is why the hint is consulted first).
    - Otherwise, from ``block_info[None]["dtype"]``.

    During dask's meta inference call (where ``block_info`` is ``None``),
    ``out_null_value`` is a typed zero of the first input's dtype so funcs like
    ``np.where(m, out_null_value, d)`` infer the same dtype as their input
    rather than collapsing to object. If your func's *output dtype* depends on
    the specific ``out_null_value`` scalar, pass ``meta=`` to skip dask's
    0-shape meta call entirely; ``dtype=`` skips only the additional sample
    call, not the meta call.

    Examples
    --------
    Plain elementwise on one input (no special injection):

    >>> def double(d):
    ...     return d * 2
    >>> doubled = r.map_blocks(double)               # doctest: +SKIP

    Mask-aware multi-input -- write the null sentinel where either input was
    masked:

    >>> def add_skip_nulls(a, b, *, input_masks, **kwargs):
    ...     ma, mb = input_masks
    ...     out = a + b
    ...     out[ma | mb] = -9999
    ...     return out
    >>> summed = map_blocks(                          # doctest: +SKIP
    ...     add_skip_nulls, r1, r2, null_value=-9999,
    ... )

    Use dask's per-block info inside ``func``:

    >>> def by_chunk_id(d, *, block_info, **kwargs):
    ...     bi = block_info[None]["chunk-location"]
    ...     ...                                       # doctest: +SKIP

    See Also
    --------
    raster_tools.geo_map_blocks : Geo-aware variant that requires matching
        grids and hands ``func`` georeferenced ``xr.DataArray`` blocks.
    raster_tools.Raster.reproject : Per-input alignment to a target
        grid; pass ``r1.geobox`` to align ``r2`` to ``r1``.
    """
    if not rasters:
        raise ValueError("map_blocks requires at least one raster")
    _check_dtype_meta_agree(dtype, meta)
    _check_no_dask_kwargs(kwargs)
    _check_reserved_kwargs(kwargs, _MAP_BLOCKS_RESERVED_KWARGS, "map_blocks")
    rasters = [get_raster(r) for r in rasters]
    _check_shape_aligned(rasters)
    # Pre-rechunk template: keeps the output on the caller's original
    # y/x chunking even when out_bands re-tiles inputs for the compute.
    orig_ref = rasters[0]
    if out_bands is None:
        rasters = _align_chunks_to_ref(rasters)
        out_chunks = None
    else:
        _check_out_bands(out_bands)
        rasters = _align_chunks_single_band(rasters, out_bands)
        out_chunks = ((out_bands,), *rasters[0].data.chunks[1:])
    ref = rasters[0]
    data_hint = _out_dtype_hint(dtype, meta)
    wrapper, inputs = _build_map_blocks_wrapper(
        func,
        rasters,
        null_value=null_value,
        out_dtype_hint=data_hint,
        out_bands=out_bands,
        return_mask=return_mask,
        out_data_dtype=data_hint,
    )
    if return_mask:
        # The wrapper returns a structured (data, mask) array, so the
        # user's dtype=/meta= (which describe the data part) are folded
        # into a structured meta rather than forwarded raw -- they would
        # mis-describe the structured output. A structured meta also
        # skips dask's 0-shape probe entirely.
        struct_meta = (
            np.empty(
                0, dtype=np.dtype([("data", data_hint), ("mask", np.bool_)])
            )
            if data_hint is not None
            else None
        )
        struct = da.map_blocks(
            wrapper,
            *inputs,
            dtype=None,
            meta=struct_meta,
            chunks=out_chunks,
            **kwargs,
        )
        out_data = struct["data"]
        out_mask = struct["mask"]
    else:
        out_data = da.map_blocks(
            wrapper,
            *inputs,
            dtype=dtype,
            meta=meta,
            chunks=out_chunks,
            **kwargs,
        )
        out_mask = None
    out_nv = _resolve_out_null_value(
        null_value=null_value,
        ref_dtype=ref.dtype,
        ref_null_value=ref.null_value,
        n_rasters=len(rasters),
        out_dtype=out_data.dtype,
    )
    return data_to_raster_like(
        out_data, orig_ref, mask=out_mask, nv=out_nv, burn=return_mask
    )


def _build_map_blocks_wrapper(
    func,
    rasters,
    null_value=None,
    out_dtype_hint=None,
    out_bands=None,
    return_mask=False,
    out_data_dtype=None,
):
    """Build the per-block wrapper map_blocks passes to dask.

    Returns ``(wrapper, inputs)``. ``wrapper`` is the callable to
    pass to :func:`dask.array.map_blocks`. ``inputs`` is the list of
    dask arrays to spread positionally (data first; masks appended
    only when the user's ``func`` opts in via ``input_masks=``).

    When ``out_bands`` is set (a band-count change), the wrapper checks
    that each block ``func`` returns has exactly ``out_bands`` bands and
    raises otherwise -- dask silently accepts a mismatch between the
    func's actual band count and the ``chunks=`` we declare, which would
    otherwise yield a Raster whose metadata disagrees with its data.

    When ``return_mask`` is set, ``func`` returns a ``(data, mask)`` pair
    and the wrapper packs it into a single NumPy structured-dtype array
    ``[("data", out_data_dtype), ("mask", bool)]`` -- the only way to
    carry both through dask's single-array-per-block contract.
    ``map_blocks`` field-accesses the two back out. ``out_data_dtype``
    (the resolved data-dtype hint, or ``None``) fixes the struct's data
    field so it stays stable across blocks; ``None`` falls back to the
    returned data's dtype.

    The wrapper captures only small immutable values from the input
    rasters (per-input null values, ref dtype + null value, and the
    user's ``null_value`` parameter) so dask doesn't ship full
    Raster objects across workers.

    ``out_dtype_hint`` is the caller-resolved output dtype (from
    ``dtype=`` / ``meta=``; see :func:`_out_dtype_hint`) or ``None``.
    When the user's ``func`` opts in to ``out_null_value``, the wrapper
    resolves it per chunk from the hint when set, otherwise from
    ``block_info[None]["dtype"]``. The hint is needed because dask
    leaves ``block_info[None]["dtype"]`` as ``None`` when ``meta=`` is
    set, even though the meta's dtype is the real output dtype.
    """
    pass_masks = has_keyword(func, "input_masks")
    pass_block_info = has_keyword(func, "block_info")
    pass_block_id = has_keyword(func, "block_id")
    pass_input_nvs = has_keyword(func, "input_null_values")
    pass_out_nv = has_keyword(func, "out_null_value")

    inputs = [r.data for r in rasters]
    if pass_masks:
        inputs.extend(r.mask for r in rasters)

    # Capture only scalars / small immutables. No Raster references.
    nvs = tuple(r.null_value for r in rasters)
    ref = rasters[0]
    ref_dtype = ref.dtype
    ref_null_value = ref.null_value
    n_rasters = len(rasters)
    # Typed-zero placeholder used only during dask's meta inference
    # call (where block_info is None). Lets funcs like
    # ``np.where(m, out_null_value, d)`` infer the same dtype as
    # their input rather than collapsing to object dtype.
    meta_placeholder = (
        np.zeros((), dtype=ref_dtype)[()] if pass_out_nv else None
    )

    def _wrapper(*block_args, block_info=None, **inner_kwargs):
        if pass_masks:
            n = len(block_args)
            input_data = block_args[: n // 2]
            input_masks = block_args[n // 2 :]
            inner_kwargs["input_masks"] = input_masks
        else:
            input_data = block_args
        if pass_block_info:
            inner_kwargs["block_info"] = block_info
        if pass_block_id:
            # dask's block_id is the output block's chunk-location, i.e.
            # block_info[None]["chunk-location"]. Derive it from
            # block_info (rather than declaring block_id on the wrapper)
            # so the meta-inference call -- where block_info is None --
            # cleanly yields block_id=None too, matching dask.
            inner_kwargs["block_id"] = (
                None
                if block_info is None
                else block_info[None]["chunk-location"]
            )
        if pass_input_nvs:
            inner_kwargs["input_null_values"] = nvs
        if pass_out_nv:
            if block_info is None:
                inner_kwargs["out_null_value"] = meta_placeholder
            else:
                # Prefer the upfront hint (set when the caller passed
                # dtype= or meta=); fall back to dask's per-chunk
                # dtype. The hint is needed because dask leaves
                # block_info[None]["dtype"] as None when meta= is
                # set, even though the meta's dtype IS the output.
                chunk_dtype = block_info[None]["dtype"]
                resolved_dtype = (
                    out_dtype_hint
                    if out_dtype_hint is not None
                    else chunk_dtype
                )
                inner_kwargs["out_null_value"] = _resolve_out_null_value(
                    null_value=null_value,
                    ref_dtype=ref_dtype,
                    ref_null_value=ref_null_value,
                    n_rasters=n_rasters,
                    out_dtype=resolved_dtype,
                )
        result = func(*input_data, **inner_kwargs)
        if return_mask:
            return _pack_data_mask(
                result, block_info, out_bands, out_data_dtype
            )
        # Skip the check during dask's 0-shape meta call (block_info is
        # None): a raise there is swallowed and re-surfaced as dask's
        # opaque "dtype inference failed" message. The explicit chunks=
        # sets the output shape regardless of the meta return, so the
        # guard only needs to run on real blocks, where it raises a
        # clear error at compute time.
        if (
            out_bands is not None
            and block_info is not None
            and result.shape[0] != out_bands
        ):
            raise ValueError(
                f"func returned {result.shape[0]} band(s) but out_bands="
                f"{out_bands} was requested"
            )
        return result

    return _wrapper, inputs


def _pack_data_mask(result, block_info, out_bands, out_data_dtype):
    """Pack a func's ``(data, mask)`` return into a structured array.

    ``return_mask`` mode: ``func`` hands back the data block and an
    explicit boolean mask block. dask's map_blocks carries one array per
    block, so we pack both into a single NumPy structured-dtype array
    ``[("data", <dtype>), ("mask", bool)]``; :func:`map_blocks`
    field-accesses them back out after the graph is built. Field
    assignment casts ``data`` to the struct's data dtype and ``mask`` to
    bool.

    ``data`` and ``mask`` must be raw arrays; the geo wrappers strip any
    ``xr.DataArray`` to its ``.data`` before calling.

    Validation runs only on real blocks (``block_info`` is ``None``
    during dask's 0-shape meta call, where a raise is swallowed and
    re-surfaced as an opaque message).
    """
    if not (isinstance(result, (tuple, list)) and len(result) == 2):
        raise ValueError(
            "func must return a (data, mask) pair when return_mask=True"
        )
    data, mask = result
    if block_info is not None:
        if out_bands is not None and data.shape[0] != out_bands:
            raise ValueError(
                f"func returned {data.shape[0]} band(s) but out_bands="
                f"{out_bands} was requested"
            )
        if mask.shape != data.shape:
            raise ValueError(
                f"func returned a mask of shape {mask.shape} that does "
                f"not match the data shape {data.shape}"
            )
    ddt = (
        out_data_dtype
        if out_data_dtype is not None
        else np.asarray(data).dtype
    )
    packed = np.empty(
        data.shape, dtype=np.dtype([("data", ddt), ("mask", np.bool_)])
    )
    packed["data"] = data
    packed["mask"] = mask
    return packed


def _resolve_out_null_value(
    *, null_value, ref_dtype, ref_null_value, n_rasters, out_dtype
):
    """Resolve the output null value from precomputed scalars.

    Used both by :func:`map_blocks` (after dask returns the inferred
    dtype) and by the in-wrapper per-block path (to compute a
    chunk-local ``out_null_value`` from
    ``block_info[None]["dtype"]``). Takes only scalars / dtypes so
    the wrapper's closure stays free of Raster references.

    Resolution rules:

    - ``null_value`` is a scalar -> use it.
    - ``null_value`` is a string -> ``ValueError``.
    - ``null_value`` is ``None`` and there is exactly one input whose
      dtype matches ``out_dtype`` -> inherit that input's null value.
    - Otherwise -> :func:`get_default_null_value` against
      ``out_dtype``.
    """
    if null_value is None:
        if n_rasters == 1 and ref_dtype == out_dtype:
            return ref_null_value
        return get_default_null_value(out_dtype)
    if isinstance(null_value, str):
        raise ValueError(
            f"null_value must be None or a scalar, got {null_value!r}"
        )
    return null_value


def _check_asymmetric_depth_compatible_with_boundary(depth_dict, boundary):
    """Mirror dask's restriction: tuple depth requires boundary 'none'.

    ``da.overlap.map_overlap`` raises ``NotImplementedError`` deep in
    its graph when any per-axis depth is a tuple and the boundary on
    that axis isn't ``"none"``. Surface that as a ``ValueError`` at
    call time with a message that names the offending axis.
    """
    if boundary is None:
        return
    if isinstance(boundary, str) and boundary.lower() == "none":
        return
    for axis in (1, 2):
        d = depth_dict.get(axis, 0)
        if isinstance(d, tuple):
            raise ValueError(
                f"asymmetric depth {d!r} on axis {axis} is only "
                "supported with boundary=None or boundary='none'; "
                f"got boundary={boundary!r}"
            )


_NAMED_BOUNDARIES = frozenset({"reflect", "periodic", "nearest", "none"})
# Case-insensitive aliases for "fill outside cells with the raster's
# null value." Matched after .lower()-ing the input string.
_NULL_BOUNDARIES = frozenset({"null", "null_value", "nodata"})


def _normalize_depth_axis(value):
    if is_int(value):
        if value < 0:
            raise ValueError(f"depth must be non-negative, got {value}")
        return value
    if isinstance(value, tuple) and len(value) == 2:
        left, right = value
        if not (is_int(left) and is_int(right)):
            raise TypeError(
                f"asymmetric depth tuple must be ints, got {value!r}"
            )
        if left < 0 or right < 0:
            raise ValueError(f"depth must be non-negative, got {value!r}")
        return value
    raise TypeError(
        f"depth value must be an int or 2-tuple of ints, got {value!r}"
    )


def _normalize_depth(depth):
    """Normalize ``depth`` into a per-axis dict.

    Always returns a dict with all three axes (band/y/x) populated,
    keyed by axis index. Band axis is forced to 0.
    """
    if isinstance(depth, dict):
        out = {0: 0, 1: 0, 2: 0}
        for axis, value in depth.items():
            if axis not in (0, 1, 2):
                raise ValueError(
                    f"depth axis must be 0, 1, or 2; got {axis!r}"
                )
            out[axis] = _normalize_depth_axis(value)
        if out[0] != 0:
            raise ValueError("non-zero band-axis (0) depth is not supported")
        return out
    if is_int(depth):
        norm = _normalize_depth_axis(depth)
        return {0: 0, 1: norm, 2: norm}
    if isinstance(depth, tuple) and len(depth) == 2:
        dy, dx = depth
        return {
            0: 0,
            1: _normalize_depth_axis(dy),
            2: _normalize_depth_axis(dx),
        }
    raise TypeError(f"depth must be an int, 2-tuple, or dict; got {depth!r}")


def _align_chunks_to_ref(rasters):
    """Rechunk all inputs to the first raster's chunk structure.

    ``map_blocks`` / ``map_overlap`` / ``geo_map_blocks`` pass the inputs
    straight to dask, which needs identical block structure across the
    arrays. Inputs that share a grid but are chunked differently (e.g.
    after ``reproject``, which does not adopt the target's chunking)
    otherwise crash with an opaque dask ``IndexError``. Aligning to
    ``ref`` also keeps the per-chunk :class:`GeoBlockInfo` lookup (built
    from ``ref``) valid, and keeps the output -- which lands on ``ref``'s
    grid -- on ``ref``'s chunking. ``r.chunk`` rechunks each raster's
    data and mask together, so the two stay consistent. No-op for a
    single input.
    """
    ref_chunks = rasters[0].data.chunks
    return [
        r if r.data.chunks == ref_chunks else r.chunk(ref_chunks)
        for r in rasters
    ]


def _check_out_bands(out_bands):
    """Validate the ``out_bands`` argument: a positive integer or None."""
    if not is_int(out_bands) or isinstance(out_bands, bool) or out_bands < 1:
        raise ValueError(
            f"out_bands must be a positive integer, got {out_bands!r}"
        )


def _check_no_out_bands(kwargs, fname):
    """Reject a stray ``out_bands`` in the overlap variants' ``**kwargs``.

    ``out_bands`` is only a parameter of :func:`map_blocks` /
    :func:`geo_map_blocks`; on the overlap variants it would otherwise
    land in ``**kwargs`` and be silently forwarded to the user's func.
    """
    if "out_bands" in kwargs:
        raise ValueError(
            "out_bands is only supported by map_blocks / geo_map_blocks, "
            f"not {fname}; the overlap variants are strictly "
            "shape-preserving"
        )


def _align_chunks_single_band(rasters, out_bands):
    """Rechunk inputs to one band-chunk with memory-bounded y/x tiles.

    Used by ``map_blocks`` / ``geo_map_blocks`` when ``out_bands`` is set
    (a band-count change). The per-block ``func`` must see every band of
    a spatial tile at once to map the input bands to ``out_bands`` output
    bands, so the band axis is collapsed to a single chunk. Keeping the
    original y/x tile would then multiply per-block memory by the band
    count, so y/x are re-tiled via dask ``"auto"`` to stay within
    ``array.chunk-size``. The byte budget is divided by
    ``max(nbands_in, out_bands)`` so that both the all-bands input block
    (``nbands_in x tile``) and the output block (``out_bands x tile``)
    fit -- without this an expansion (e.g. 1 -> 100 bands) would produce
    a multi-GB output block. For a reduction (``out_bands <=
    nbands_in``) the budget equals ``array.chunk-size``, i.e. exactly
    ``(nbands, "auto", "auto")`` -- the same idiom ``local_stats`` uses.

    All inputs are aligned to ``ref``'s resulting concrete chunks (as in
    :func:`_align_chunks_to_ref`); ``ref``'s dtype drives the ``"auto"``
    sizing. ``r.chunk`` rechunks each raster's data and mask together.
    """
    ref = rasters[0]
    nbands = ref.shape[0]
    chunk_size = parse_bytes(dask.config.get("array.chunk-size"))
    limit = max(1, chunk_size * nbands // max(nbands, out_bands))
    target = ref.data.rechunk(
        (nbands, "auto", "auto"), block_size_limit=limit
    ).chunks
    return [r if r.data.chunks == target else r.chunk(target) for r in rasters]


def _ensure_chunks_for_overlap(rasters, depth_dict):
    """Grow chunks for ``depth`` and align all inputs to ``ref``.

    ``geo_map_overlap`` builds a per-chunk :class:`GeoBlockInfo` lookup
    from the input's pre-call chunking and keys it by
    ``block_info[0]['chunk-location']``. If dask auto-rechunks under
    ``da.overlap.map_overlap`` (because some chunk is smaller than the
    depth on that axis), the in-wrapper chunk-location refers to the
    rechunked grid and won't match the lookup. Pre-rechunking here
    keeps both in sync.

    The target chunking is ``ref``'s spatial chunks grown to satisfy the
    depth minimum on each axis. Every input (including ``ref``) whose
    chunks differ from that target is rechunked to it, so same-grid
    inputs with differing chunking are aligned in the same pass (see
    :func:`_align_chunks_to_ref`). ``ref`` is left untouched when it
    already matches.
    """

    def _depth_for_axis(axis):
        d = depth_dict.get(axis, 0)
        # Asymmetric (top, bottom) -> use the larger side as the
        # minimum-chunk requirement (safe over-grow).
        return max(d) if isinstance(d, tuple) else d

    ref = rasters[0]
    yx_chunks = ref.data.chunks[1:]
    new_yx = tuple(
        da.overlap.ensure_minimum_chunksize(d, c) if d else c
        for d, c in zip(
            (_depth_for_axis(1), _depth_for_axis(2)),
            yx_chunks,
            strict=True,
        )
    )
    target_chunks_3d = (ref.data.chunks[0], *new_yx)
    return [
        r if r.data.chunks == target_chunks_3d else r.chunk(target_chunks_3d)
        for r in rasters
    ]


def _resolve_boundary(boundary, raster):
    """Return ``(data_boundary, mask_boundary)`` for one raster.

    See the table in :func:`map_overlap`.
    """
    if boundary is None:
        return None, None
    if isinstance(boundary, str):
        # All boundary strings are matched case-insensitively
        # (so "NoData" / "REFLECT" / "None" all work). Return the
        # canonical lowercase form so dask's overlap module sees the
        # token it expects (it compares against the lowercase
        # literals).
        b = boundary.lower()
        if b in _NULL_BOUNDARIES:
            nv = raster.null_value
            if nv is None:
                nv = get_default_null_value(raster.dtype)
            return nv, True
        if b in _NAMED_BOUNDARIES:
            # 'none' -> dask treats as no padding for both arrays.
            # 'reflect'/'periodic'/'nearest' -> mask is padded the
            # same way so reflected/wrapped/copied data and mask stay
            # in sync.
            return b, b
        raise ValueError(
            f"unrecognized boundary string {boundary!r}; expected one "
            f"of {sorted(_NULL_BOUNDARIES | _NAMED_BOUNDARIES)} "
            "(case-insensitive) or None / a numeric scalar"
        )
    # Numeric scalar.
    nv = raster.null_value
    if nv is not None and nan_equal(boundary, nv):
        return boundary, True
    return boundary, False


def map_overlap(
    func,
    *rasters,
    depth,
    boundary=None,
    dtype=None,
    null_value=None,
    meta=None,
    return_mask=False,
    **kwargs,
):
    """Apply ``func`` block-wise with overlap across one or more rasters.

    Thin wrapper over :func:`dask.array.overlap.map_overlap`. Each call to
    ``func`` receives one block from each input raster, in input order,
    with ``depth`` extra cells of overlap on each side. dask trims the
    overlap from the result before it's wrapped back into a Raster on the
    first input's grid, so the user function returns same-shape
    (overlap-included) blocks and doesn't need to trim itself.

    Available both as a module function and as a
    :class:`~raster_tools.Raster` method: ``r1.map_overlap(func, r2,
    depth=1)`` is equivalent to ``map_overlap(func, r1, r2, depth=1)`` --
    in the method form the calling raster is the first input. References
    to the "first input" below mean ``r1`` in either spelling.

    **Per-block contract**

    Per-block kwargs (opt-in): ``input_masks``, ``input_null_values``,
    ``block_info``, ``block_id``, ``out_null_value``. Name any of these in
    ``func``'s signature to receive them per chunk; see below.

    By default the output Raster's mask is rebuilt from the output data and the
    resolved output null value (``out_data == null_value``, or
    ``np.isnan(out_data)`` for NaN nulls) -- write the sentinel only at cells
    you want masked. Cells your func happens to leave equal to the sentinel
    will appear masked even if you didn't intend them to; cells you wanted
    masked but didn't write the sentinel to will not. This is true regardless
    of the input rasters' masks: input masks do **not** carry through
    unchanged. To set the output mask explicitly instead -- decoupling nullness
    from the data values -- pass ``return_mask=True`` and have ``func`` return
    a ``(data, mask)`` pair (both overlap-included); see ``return_mask`` below.

    Always passed positionally::

        func(*input_data, **kwargs)

    where ``input_data`` is a tuple of N NumPy blocks, one per input raster, in
    input order. Each block already includes the overlap region.

    The user can also opt in to receive per-block extras by including named
    parameters in ``func``'s signature. Detection mirrors dask's own
    ``block_info=`` / ``block_id=`` mechanism and uses
    :func:`inspect.signature` (via :func:`dask.utils.has_keyword`). Recognized
    names (same set as :func:`~raster_tools.map_blocks`):

    - ``input_masks`` -- tuple of N ``np.ndarray`` (bool) per-block mask
      arrays, parallel to ``input_data`` and overlap-included.
    - ``input_null_values`` -- tuple of N scalars, each input raster's
      ``null_value`` (``None`` if unset).
    - ``block_info`` -- dask's standard per-block info dict.
    - ``block_id`` -- dask's standard per-block index tuple: the block's
      ``chunk-location`` (equivalently ``block_info[None]["chunk-location"]``).
      ``None`` during the meta inference call.
    - ``out_null_value`` -- scalar; the resolved output null value the wrapper
      will use to derive the output mask. Write this sentinel at cells you want
      masked. See "Output null value resolution" in Notes below.

    A function whose only kwargs absorber is ``**kwargs`` does NOT trigger any
    of these injections -- name the kwargs you want.

    **Reserved kwargs**

    ``input_masks``, ``input_null_values``, ``block_info``, ``block_id``, and
    ``out_null_value`` are reserved. Passing any of them via ``map_overlap``'s
    own ``**kwargs`` raises ``ValueError``.

    Parameters
    ----------
    func : callable
        Per-block function. See "Per-block contract" above. Must return an
        array-like that dask can ingest (NumPy ndarray, cupy ndarray, sparse
        array, etc.) with the same shape as a single (overlap-included) data
        block. For non-numpy backends, also pass ``meta=``.
    *rasters : Raster or str
        The input rasters. In the method form the calling raster is the
        first input and ``*rasters`` holds any additional ones; in the
        function form at least one is required. Path strings are accepted.
        Only the 3D shape is validated; CRS and affine are *not* checked.
        The caller is responsible for aligning inputs -- typically via
        ``r2.reproject(r1.geobox)``. Inputs are auto-rechunked to the first
        input's chunk structure, so same-shape inputs with differing
        chunking are handled. For a geo-aware variant that strictly
        requires matching grids, see :func:`~raster_tools.geo_map_overlap`.
    depth : int, tuple of int, or dict
        Number of overlap cells per spatial axis. ``int`` applies to both ``y``
        and ``x`` (band axis fixed at 0). ``(dy, dx)`` sets per-spatial-axis
        depth. ``dict`` maps axis index to depth (or to a ``(top, bottom)`` /
        ``(left, right)`` tuple for asymmetric depths). Asymmetric depths
        require ``boundary=None`` or ``boundary="none"`` per dask's
        restriction.
    boundary : optional
        How to fill cells outside the array's edges. Choices:

        - ``None`` (default): no padding (matches dask).
        - ``"null"`` / ``"null_value"`` / ``"nodata"`` (case-insensitive, so
          ``"NODATA"`` / ``"NoData"`` also work): fill with the input raster's
          null value (or :func:`get_default_null_value(dtype) <
          raster_tools.masking.get_default_null_value>` if unset). The
          corresponding mask cells are set to ``True``.
        - a numeric scalar: fill with that value. If the value matches the
          raster's null value, mask cells are set to ``True``; otherwise
          they're set to ``False``.
        - ``"reflect"`` / ``"periodic"`` / ``"nearest"``: dask's standard
          padding modes. The mask is padded the same way so
          reflected/wrapped/copied data and mask stay in sync at the source
          cell.
        - ``"none"``: explicit no-padding (same as ``None``).

        For multi-input, each raster's null value is consulted independently.

        The boundary -> mask rule only affects what the user's function sees in
        the mask block when it opts in to ``input_masks=``. The output Raster's
        mask is built independently from the output data (see Returns / Notes
        below); padded cells are trimmed off before the user sees them.
    dtype : dtype-like, optional
        Output dtype. When ``None`` (default), dask infers the dtype by calling
        ``func`` on tiny meta samples.
    null_value : scalar, optional
        Output null value.

        - ``None`` (default): if there is exactly one input raster and the
          output dtype matches its dtype, the output inherits that input's null
          value. Otherwise, the value is a dtype-appropriate default from
          :func:`raster_tools.masking.get_default_null_value`.
        - scalar: used as-is.

    meta : array-like, optional
        Empty array with the desired output array type. Forwarded to
        :func:`dask.array.overlap.map_overlap`. When provided, dask uses this
        as the output meta and skips the 0-shape sample call it would otherwise
        make to derive one -- useful when ``func`` cannot tolerate 0-shape
        input. When ``None`` (default), dask derives a NumPy meta by calling
        ``func`` on 0-shape inputs.
    return_mask : bool, optional
        If ``True``, ``func`` must return a ``(data, mask)`` pair instead of a
        single array. The returned ``mask`` -- not a sentinel comparison --
        defines the output Raster's null cells. ``mask`` is a boolean array the
        same shape as ``data`` (both overlap-included; dask trims them
        together). Masked cells are set to the resolved output null value
        (burned in). Use this to decouple *which* cells are null from *what
        value* they hold, avoiding the sentinel-collision pitfalls described
        above. The default is ``False`` (sentinel-derived mask). Notes:

        - The two arrays are carried through dask packed into a single NumPy
          structured-dtype block, then split apart again -- an internal detail;
          ``func`` just returns the plain pair.
        - Passing ``dtype=`` (or ``meta=``) describing the **data** dtype is
          recommended: it lets dask skip its 0-shape probe. Without a hint the
          func must tolerate that probe.
        - Requires NumPy-backed blocks (the structured-dtype carrier is a NumPy
          concept). Composes with ``input_masks`` and every ``boundary`` mode.
        - ``out_null_value`` injection is unnecessary (though harmless) when
          ``return_mask=True``.
    **kwargs
        Extra keyword arguments forwarded per-block to ``func``. The reserved
        names listed above are not allowed here.

    Returns
    -------
    Raster
        A new lazy Raster on the first input's grid.

    Notes
    -----
    The wrapper always trims overlap before returning so the result matches the
    input grid. If you need un-trimmed output, call
    :func:`dask.array.overlap.map_overlap` directly on ``raster.data``.

    The output mask is all-False if no null value is set (see the per-block
    contract above for how the mask is built when one is). To write the output
    mask directly, pass ``return_mask=True`` and return a ``(data, mask)`` pair
    from ``func`` (see ``return_mask``).

    Asymmetric per-side depths are only supported with no padding
    (``boundary=None`` or ``"none"``).

    Dask invokes ``func`` once on 0-shape inputs to derive the output array
    meta -- this happens whether or not ``dtype=`` is provided. Pass ``meta=``
    to skip the call entirely; ``dtype=`` only skips the additional sample call
    dask would otherwise make to infer the output dtype, not the 0-shape meta
    call. (Exception: with ``return_mask=True``, ``dtype=`` is folded into a
    structured ``meta=`` and so *does* skip the 0-shape meta call -- see
    ``return_mask``.) Most NumPy ops handle 0-shape inputs fine. If dask raises
    ``dtype inference failed in map_blocks. Please specify the dtype explicitly
    using the dtype kwarg``, that's the sample call (dask routes overlap
    through its inner map_blocks, so the message says map_blocks even from this
    function) -- ``dtype=`` per dask's hint is usually enough; pass ``meta=``
    instead if your func also can't tolerate 0-shape inputs (dask silently
    swallows that crash, but your downstream output meta will be wrong).

    **Output null value resolution**

    When ``func`` opts in to ``out_null_value``, the wrapper resolves the
    scalar per-chunk without an extra dtype-inference pass:

    - With ``meta=`` or ``dtype=`` set, from that hint (dask leaves
      ``block_info[None]["dtype"]`` as ``None`` whenever ``meta=`` is set,
      which is why the hint is consulted first).
    - Otherwise, from ``block_info[None]["dtype"]``.

    During dask's meta inference call (where ``block_info`` is ``None``),
    ``out_null_value`` is a typed zero of the first input's dtype so funcs like
    ``np.where(m, out_null_value, d)`` infer the same dtype as their input
    rather than collapsing to object. If your func's *output dtype* depends on
    the specific ``out_null_value`` scalar, pass ``meta=`` to skip dask's
    0-shape meta call entirely; ``dtype=`` skips only the additional sample
    call, not the meta call.

    Examples
    --------
    3x3 mean with reflected edges:

    >>> def mean_3x3(d):                                # doctest: +SKIP
    ...     pad = d[0]
    ...     out = np.zeros_like(pad, dtype=np.float32)
    ...     ... # convolve and write to out[1:-1, 1:-1]
    ...     return out[None]
    >>> smoothed = r.map_overlap(
    ...     mean_3x3, depth=1, boundary="reflect",
    ... )                                                # doctest: +SKIP

    See Also
    --------
    raster_tools.map_blocks : Block-wise without overlap; same per-block
        contract.
    raster_tools.geo_map_overlap : Geo-aware variant that hands ``func``
        georeferenced ``xr.DataArray`` blocks.
    """
    if not rasters:
        raise ValueError("map_overlap requires at least one raster")
    _check_dtype_meta_agree(dtype, meta)
    _check_no_dask_kwargs(kwargs)
    _check_no_out_bands(kwargs, "map_overlap")
    _check_reserved_kwargs(kwargs, _MAP_BLOCKS_RESERVED_KWARGS, "map_overlap")
    rasters = [get_raster(r) for r in rasters]
    _check_shape_aligned(rasters)
    rasters = _align_chunks_to_ref(rasters)
    ref = rasters[0]
    depth_dict = _normalize_depth(depth)
    _check_asymmetric_depth_compatible_with_boundary(depth_dict, boundary)

    data_hint = _out_dtype_hint(dtype, meta)
    wrapper, inputs = _build_map_blocks_wrapper(
        func,
        rasters,
        null_value=null_value,
        out_dtype_hint=data_hint,
        return_mask=return_mask,
        out_data_dtype=data_hint,
    )
    pass_masks = has_keyword(func, "input_masks")
    boundaries = [_resolve_boundary(boundary, r)[0] for r in rasters]
    if pass_masks:
        boundaries.extend(_resolve_boundary(boundary, r)[1] for r in rasters)
    depths = [depth_dict] * len(inputs)

    if return_mask:
        # The wrapper returns a structured (data, mask) array (see
        # map_blocks); fold the user's data-side dtype=/meta= into a
        # structured meta rather than forwarding them raw. dask trims the
        # structured block's overlap rim like any other array.
        struct_meta = (
            np.empty(
                0, dtype=np.dtype([("data", data_hint), ("mask", np.bool_)])
            )
            if data_hint is not None
            else None
        )
        struct = da.overlap.map_overlap(
            wrapper,
            *inputs,
            depth=depths,
            boundary=boundaries,
            dtype=None,
            meta=struct_meta,
            **kwargs,
        )
        out_data = struct["data"]
        out_mask = struct["mask"]
    else:
        out_data = da.overlap.map_overlap(
            wrapper,
            *inputs,
            depth=depths,
            boundary=boundaries,
            dtype=dtype,
            meta=meta,
            **kwargs,
        )
        out_mask = None
    out_nv = _resolve_out_null_value(
        null_value=null_value,
        ref_dtype=ref.dtype,
        ref_null_value=ref.null_value,
        n_rasters=len(rasters),
        out_dtype=out_data.dtype,
    )
    return data_to_raster_like(
        out_data, ref, mask=out_mask, nv=out_nv, burn=return_mask
    )


# Reserved kwargs for geo_map_blocks: superset of map_blocks's
# reserved set, plus the geo-only `geo_block_info` injection.
_GEO_MAP_BLOCKS_RESERVED_KWARGS = _MAP_BLOCKS_RESERVED_KWARGS | frozenset(
    {"geo_block_info"}
)


def _meta_dataarray(arr, nodata=None):
    """Build a placeholder DataArray for dask's meta-inference call.

    ``arr`` may have any shape (often ``(1, 0, 0)``); construct coords
    matching each axis length so xarray's validation passes. Attach
    ``nodata`` so the user's func sees the same metadata it will get
    on the real per-block call.
    """
    nb, ny, nx = arr.shape
    xda = xr.DataArray(
        arr,
        dims=("band", "y", "x"),
        coords={
            "band": np.arange(nb, dtype="int64"),
            "y": np.zeros(ny, dtype="float64"),
            "x": np.zeros(nx, dtype="float64"),
        },
    )
    if nodata is not None:
        xda = xda.rio.write_nodata(nodata)
    return xda


def _resolve_gbi_no_pad(block_info, block_args, gbi_lookup):
    """gbi resolver for geo_map_blocks: straight chunk-location lookup."""
    return gbi_lookup[block_info[0]["chunk-location"]]


def _resolve_gbi_with_overlap_pad(block_info, block_args, gbi_lookup):
    """gbi resolver for geo_map_overlap: pad the base gbi to match the
    overlap-included block shape.

    Distributes the per-axis overlap symmetrically. Exact for interior
    chunks and any non-``"none"`` boundary; approximate for
    ``boundary=None``/``"none"`` edge chunks (the per-side split may
    be slightly off-position).
    """
    base_gbi = gbi_lookup[block_info[0]["chunk-location"]]
    base_shape = base_gbi.shape
    actual_shape = block_args[0].shape
    tot_dy = actual_shape[1] - base_shape[1]
    tot_dx = actual_shape[2] - base_shape[2]
    top, bottom = tot_dy // 2, tot_dy - tot_dy // 2
    left, right = tot_dx // 2, tot_dx - tot_dx // 2
    gbi = base_gbi.pad_y(top, bottom) if (top or bottom) else base_gbi
    if left or right:
        gbi = gbi.pad_x(left, right)
    return gbi


def _as_array(x):
    """Extract ``.data`` from an ``xr.DataArray``, else return ``x``.

    The geo wrappers hand ``func`` georeferenced DataArrays and accept a
    DataArray (or bare array) back; this strips the xarray wrapper so the
    plumbing downstream sees a raw array. The non-geo wrappers are
    array-only and do not use this.
    """
    return x.data if isinstance(x, xr.DataArray) else x


def _build_geo_wrapper(
    func,
    rasters,
    *,
    null_value,
    gbi_resolver,
    out_dtype_hint=None,
    out_bands=None,
    return_mask=False,
    out_data_dtype=None,
):
    """Shared per-block wrapper builder for geo_map_blocks /
    geo_map_overlap.

    ``gbi_resolver(block_info, block_args, gbi_lookup) -> GeoBlockInfo``
    decides how to translate dask's chunk-location into the gbi the
    user sees. Pass ``_resolve_gbi_no_pad`` for the no-overlap
    flavor; ``_resolve_gbi_with_overlap_pad`` for overlap.

    ``out_dtype_hint`` is the caller-resolved output dtype (from
    ``dtype=`` / ``meta=``) or ``None``; it is preferred over
    ``block_info[None]["dtype"]`` when resolving a per-chunk
    ``out_null_value`` (same rule as :func:`_build_map_blocks_wrapper`).

    ``out_bands`` (geo_map_blocks only) enables the per-block output
    band-count check (see :func:`_build_map_blocks_wrapper`).

    ``return_mask`` / ``out_data_dtype`` enable the explicit-mask path:
    ``func`` returns a ``(data, mask)`` pair (each an ``xr.DataArray`` or
    ndarray) which the wrapper packs into a structured block via
    :func:`_pack_data_mask` (same as :func:`_build_map_blocks_wrapper`).

    Returns ``(wrapper, inputs)``. Captures only small immutables in
    the closure -- no Raster references.
    """
    pass_masks = has_keyword(func, "input_masks")
    pass_block_info = has_keyword(func, "block_info")
    pass_block_id = has_keyword(func, "block_id")
    pass_input_nvs = has_keyword(func, "input_null_values")
    pass_out_nv = has_keyword(func, "out_null_value")
    pass_geo_block_info = has_keyword(func, "geo_block_info")

    inputs = [r.data for r in rasters]
    if pass_masks:
        inputs.extend(r.mask for r in rasters)

    nvs = tuple(r.null_value for r in rasters)
    ref = rasters[0]
    ref_dtype = ref.dtype
    ref_null_value = ref.null_value
    n_rasters = len(rasters)
    # Per-chunk GeoBlockInfos, looked up by dask's chunk-location.
    gbi_lookup = geo_block_infos(ref)
    meta_placeholder = (
        np.zeros((), dtype=ref_dtype)[()] if pass_out_nv else None
    )

    def _wrapper(*block_args, block_info=None, **inner_kwargs):
        if pass_masks:
            n = len(block_args)
            data_blocks = block_args[: n // 2]
            mask_blocks = block_args[n // 2 :]
        else:
            data_blocks = block_args
            mask_blocks = None

        if block_info is None:
            # Meta-inference call: 0-shape DataArrays via _meta_dataarray.
            data_das = [
                _meta_dataarray(d, nodata=nvs[i])
                for i, d in enumerate(data_blocks)
            ]
            mask_das = (
                [_meta_dataarray(m) for m in mask_blocks]
                if pass_masks
                else None
            )
            gbi = None
        else:
            gbi = gbi_resolver(block_info, block_args, gbi_lookup)
            data_das = []
            mask_das = [] if pass_masks else None
            for i, d in enumerate(data_blocks):
                if pass_masks:
                    xd, xm = gbi.to_dataarray(
                        d, mask=mask_blocks[i], nodata=nvs[i]
                    )
                    data_das.append(xd)
                    mask_das.append(xm)
                else:
                    data_das.append(gbi.to_dataarray(d, nodata=nvs[i]))

        if pass_masks:
            inner_kwargs["input_masks"] = tuple(mask_das)
        if pass_input_nvs:
            inner_kwargs["input_null_values"] = nvs
        if pass_block_info:
            inner_kwargs["block_info"] = block_info
        if pass_block_id:
            # See _build_map_blocks_wrapper: derive from block_info so
            # the meta call (block_info None) yields block_id=None.
            inner_kwargs["block_id"] = (
                None
                if block_info is None
                else block_info[None]["chunk-location"]
            )
        if pass_geo_block_info:
            inner_kwargs["geo_block_info"] = gbi
        if pass_out_nv:
            if block_info is None:
                inner_kwargs["out_null_value"] = meta_placeholder
            else:
                chunk_dtype = block_info[None]["dtype"]
                resolved_dtype = (
                    out_dtype_hint
                    if out_dtype_hint is not None
                    else chunk_dtype
                )
                inner_kwargs["out_null_value"] = _resolve_out_null_value(
                    null_value=null_value,
                    ref_dtype=ref_dtype,
                    ref_null_value=ref_null_value,
                    n_rasters=n_rasters,
                    out_dtype=resolved_dtype,
                )

        result = func(*data_das, **inner_kwargs)
        if return_mask:
            # _pack_data_mask is array-only; strip any DataArray pair
            # elements here (the geo path's xarray support lives in the
            # wrappers). A non-pair return falls through to its raise.
            if isinstance(result, (tuple, list)) and len(result) == 2:
                result = (_as_array(result[0]), _as_array(result[1]))
            return _pack_data_mask(
                result, block_info, out_bands, out_data_dtype
            )
        arr = _as_array(result)
        # See _build_map_blocks_wrapper: only check real blocks
        # (block_info is None during dask's 0-shape meta call).
        if (
            out_bands is not None
            and block_info is not None
            and arr.shape[0] != out_bands
        ):
            raise ValueError(
                f"func returned {arr.shape[0]} band(s) but out_bands="
                f"{out_bands} was requested"
            )
        return arr

    return _wrapper, inputs


def _build_geo_map_blocks_wrapper(
    func,
    rasters,
    null_value=None,
    out_dtype_hint=None,
    out_bands=None,
    return_mask=False,
    out_data_dtype=None,
):
    """Per-block wrapper for :func:`geo_map_blocks`.

    Thin shim over ``_build_geo_wrapper`` with the no-pad gbi
    resolver.
    """
    return _build_geo_wrapper(
        func,
        rasters,
        null_value=null_value,
        gbi_resolver=_resolve_gbi_no_pad,
        out_dtype_hint=out_dtype_hint,
        out_bands=out_bands,
        return_mask=return_mask,
        out_data_dtype=out_data_dtype,
    )


def _build_geo_map_overlap_wrapper(
    func,
    rasters,
    null_value=None,
    out_dtype_hint=None,
    return_mask=False,
    out_data_dtype=None,
):
    """Per-block wrapper for :func:`geo_map_overlap`.

    Thin shim over ``_build_geo_wrapper`` with the overlap-pad
    gbi resolver.
    """
    return _build_geo_wrapper(
        func,
        rasters,
        null_value=null_value,
        gbi_resolver=_resolve_gbi_with_overlap_pad,
        out_dtype_hint=out_dtype_hint,
        return_mask=return_mask,
        out_data_dtype=out_data_dtype,
    )


def geo_map_blocks(
    func,
    *rasters,
    dtype=None,
    null_value=None,
    meta=None,
    out_bands=None,
    return_mask=False,
    **kwargs,
):
    """Apply ``func`` block-wise across one or more aligned rasters, handing it
    georeferenced :class:`xarray.DataArray` blocks.

    Same shape and contract as :func:`~raster_tools.map_blocks`, but each
    raster's data block is wrapped in a georeferenced ``xr.DataArray``
    (with ``band`` / ``y`` / ``x`` coords from the block's geobox, the
    raster's CRS, and the raster's null value attached as ``nodata``) via
    :meth:`~raster_tools.blocks.GeoBlockInfo.to_dataarray` before being
    passed to ``func``. Useful when the per-block function wants to
    operate in xarray-land (rio accessors, xr.where, etc.) without having
    to rebuild coordinates itself.

    Available both as a module function and as a
    :class:`~raster_tools.Raster` method: ``r1.geo_map_blocks(func, r2,
    ...)`` is equivalent to ``geo_map_blocks(func, r1, r2, ...)`` -- in
    the method form the calling raster is the first input. References to
    the "first input" below mean ``r1`` in either spelling.

    **Per-block contract**

    Per-block kwargs (opt-in): ``input_masks``, ``input_null_values``,
    ``block_info``, ``block_id``, ``out_null_value``, ``geo_block_info``. Name
    any of these in ``func``'s signature to receive them per chunk; see below.

    By default the output Raster's mask is rebuilt from the output data and the
    resolved output null value (``out_data == null_value``, or
    ``np.isnan(out_data)`` for NaN nulls) -- write the sentinel only at cells
    you want masked. Cells your func happens to leave equal to the sentinel
    will appear masked even if you didn't intend them to; cells you wanted
    masked but didn't write the sentinel to will not. This is true regardless
    of the input rasters' masks: input masks do **not** carry through
    unchanged. To set the output mask explicitly instead -- decoupling nullness
    from the data values -- pass ``return_mask=True`` and have ``func`` return
    a ``(data, mask)`` pair; see ``return_mask`` below.

    Always passed positionally::

        func(*data_dataarrays, **kwargs)

    where ``data_dataarrays`` is a tuple of N georeferenced
    :class:`xarray.DataArray` blocks, one per input raster, in input order.

    The user can also opt in to receive per-block extras by including named
    parameters in ``func``'s signature. Detection mirrors dask's
    ``block_info=`` mechanism (via :func:`dask.utils.has_keyword`). Recognized
    names:

    - ``input_masks`` -- tuple of N ``xr.DataArray`` (bool) per-block mask
      arrays, parallel to the data DataArrays. Same name as
      :func:`~raster_tools.map_blocks`, but elements are ``xr.DataArray``
      here (vs. ``np.ndarray`` there).
    - ``input_null_values`` -- tuple of N scalars, each input's ``null_value``
      (``None`` if unset).
    - ``block_info`` -- dask's standard per-block info dict.
    - ``block_id`` -- dask's standard per-block index tuple: the block's
      ``chunk-location`` (equivalently ``block_info[None]["chunk-location"]``).
      ``None`` during the meta inference call.
    - ``out_null_value`` -- scalar; the resolved output null value the wrapper
      will use to derive the output mask. Write this sentinel at cells you want
      masked. See "Output null value resolution" in Notes below.
    - ``geo_block_info`` -- the per-chunk
      :class:`~raster_tools.blocks.GeoBlockInfo` (the geo-aware analog of
      dask's ``block_info``). ``None`` during the meta inference call.

    A function whose only kwargs absorber is ``**kwargs`` does NOT trigger any
    of these injections -- name the kwargs you want.

    **Reserved kwargs**

    ``input_masks``, ``input_null_values``, ``block_info``,
    ``block_id``, ``out_null_value``, and ``geo_block_info`` are
    reserved. Passing any of them via ``geo_map_blocks``'s own
    ``**kwargs`` raises ``ValueError``.

    Parameters
    ----------
    func : callable
        Per-block function. See "Per-block contract" above. May return either
        an :class:`xarray.DataArray` (its ``.data`` is extracted, preserving
        backend) or any array-like that dask can ingest (NumPy ndarray, cupy
        ndarray, sparse array, etc.) with the same shape as a single data
        block. For non-numpy backends, also pass ``meta=``.
    *rasters : Raster or str
        The input rasters. In the method form the calling raster is the
        first input and ``*rasters`` holds any additional ones; in the
        function form at least one is required. Path strings are accepted.
        All inputs must be on the same grid (CRS, affine, shape) within the
        established sub-pixel tolerance; mismatched inputs raise
        ``ValueError``. Use ``r2.reproject(r1.geobox)`` to align inputs
        first if needed. Inputs are then auto-rechunked to the first
        input's chunk structure (``reproject`` does not adopt the target's
        chunking), so the output stays on the first input's grid and
        chunking.
    dtype : dtype-like, optional
        Output dtype. When ``None`` (default), dask infers the dtype by calling
        ``func`` on tiny meta samples.
    null_value : scalar, optional
        Output null value.

        - ``None`` (default): if there is exactly one input raster and the
          output dtype matches its dtype, the output inherits that input's null
          value. Otherwise, a dtype-appropriate default from
          :func:`raster_tools.masking.get_default_null_value`.
        - scalar: used as-is.

    meta : array-like, optional
        Empty array with the desired output array type. Forwarded to
        :func:`dask.array.map_blocks`. When provided, dask uses this as the
        output meta and skips the 0-shape sample call it would otherwise make
        to derive one -- useful when ``func`` cannot tolerate 0-shape DataArray
        inputs. The wrapper extracts ``.data`` from any returned
        :class:`xarray.DataArray`, so ``meta`` describes the wrapper's array
        output (whatever backend the user's func produces). When ``None``
        (default), dask derives a NumPy meta by calling ``func`` on 0-shape
        inputs.
    out_bands : int, optional
        Number of bands in the output. ``None`` (default) is shape-preserving.
        A positive integer lets ``func`` change the band count (the y/x grid is
        unchanged); ``func`` must return exactly ``out_bands`` bands per block
        (a mismatch raises ``ValueError``). See
        :func:`~raster_tools.map_blocks` for the full semantics: the input
        band axis is collapsed to a single chunk (so ``func`` sees all
        bands of a spatial tile, on possibly re-sized y/x tiles), the
        output is restored to the input's original y/x chunking with band
        coord ``np.arange(out_bands) + 1``, and passing ``dtype=`` /
        ``meta=`` is recommended. Setting ``out_bands`` to the input band
        count gives ``func`` an all-bands view of each tile (vs the
        default per-band blocks) without changing the count; see
        :func:`~raster_tools.map_blocks`.
    return_mask : bool, optional
        If ``True``, ``func`` must return a ``(data, mask)`` pair instead of a
        single array/DataArray. The returned ``mask`` -- not a sentinel
        comparison -- defines the output Raster's null cells. ``mask`` is a
        boolean array the same shape as ``data``; masked cells are set to the
        resolved output null value (burned in). Use this to decouple *which*
        cells are null from *what value* they hold, avoiding the
        sentinel-collision pitfalls described above. The default is ``False``
        (sentinel-derived mask). Notes:

        - Each element of the pair may be an :class:`xarray.DataArray` or a
          bare array; the wrapper uses each element's ``.data`` (any coords /
          CRS are discarded, same as for the data block).
        - The two arrays are carried through dask packed into a single NumPy
          structured-dtype block, then split apart again -- an internal detail;
          ``func`` just returns the plain pair.
        - Passing ``dtype=`` (or ``meta=``) describing the **data** dtype is
          recommended: it lets dask skip its 0-shape probe. Without a hint the
          func must tolerate that probe.
        - Requires NumPy-backed blocks. Composes with ``out_bands``,
          ``input_masks``, and ``geo_block_info``. ``out_null_value`` injection
          is unnecessary (though harmless) when ``return_mask=True``.
    **kwargs
        Extra keyword arguments forwarded per-block to ``func``. The
        reserved names listed above are not allowed here.

    Returns
    -------
    Raster
        A new lazy Raster on the first input's grid (with ``out_bands`` bands
        when that argument is set).

    Notes
    -----
    .. warning::
       ``func`` must not alter the grid. The output Raster lands on the
       **first input's** grid (its CRS / affine / x / y), regardless of
       any coords / CRS / nodata your func sets on a returned DataArray -- the
       wrapper extracts ``.data`` and discards everything else. Operations that
       change the grid (reproject, clip, coarsen, sel, manual coord assignment,
       etc.) will silently produce a Raster whose values were computed on a
       different grid than the one it claims, with no error at construction
       time. Use :meth:`Raster.reproject` / :meth:`Raster.clip` / etc. before
       or after this call instead.

    The output mask is all-False if no null value is set (see the per-block
    contract above for how the mask is built when one is). To write the output
    mask directly, pass ``return_mask=True`` and return a ``(data, mask)`` pair
    from ``func`` (see ``return_mask``).

    Dask invokes ``func`` once on 0-shape DataArrays (with zero-filled
    placeholder coords, not real geocoordinates) to derive the output array
    meta -- this happens whether or not ``dtype=`` is provided. Passing
    ``dtype=`` only skips the additional sample call dask would otherwise make
    to infer the output dtype; it does not skip the 0-shape meta call.
    (Exception: with ``return_mask=True``, ``dtype=`` is folded into a
    structured ``meta=`` and so *does* skip the 0-shape meta call -- see
    ``return_mask``.) During the meta call ``geo_block_info`` is ``None`` if
    the func opts in. Most NumPy / xarray ops handle 0-shape inputs fine. If
    dask raises ``dtype inference failed in map_blocks. Please specify the
    dtype explicitly using the dtype kwarg``, that's the sample call --
    ``dtype=`` per dask's hint is usually enough; pass ``meta=`` instead if
    your func also can't tolerate 0-shape inputs (dask silently swallows that
    crash, but your downstream output meta will be wrong).

    **Output null value resolution**

    When ``func`` opts in to ``out_null_value``, the wrapper resolves the
    scalar per-chunk without an extra dtype-inference pass:

    - With ``meta=`` or ``dtype=`` set, from that hint (dask leaves
      ``block_info[None]["dtype"]`` as ``None`` whenever ``meta=`` is set,
      which is why the hint is consulted first).
    - Otherwise, from ``block_info[None]["dtype"]``.

    During dask's meta inference call (where ``block_info`` is ``None``),
    ``out_null_value`` is a typed zero of the first input's dtype so funcs like
    ``np.where(m, out_null_value, d)`` infer the same dtype as their input
    rather than collapsing to object. If your func's *output dtype* depends on
    the specific ``out_null_value`` scalar, pass ``meta=`` to skip dask's
    0-shape meta call entirely; ``dtype=`` skips only the additional sample
    call, not the meta call.

    Examples
    --------
    Use the per-chunk geobox to clip a vector layer and rasterize into the
    chunk. The output dtype is uint8, not the input's float32 -- pass ``meta=``
    so dask uses the right dtype and skips the 0-shape meta call entirely (no
    ``geo_block_info is None`` guard needed):

    >>> def clip_and_rasterize(xda, *, geo_block_info, gdf):
    ...     bbox = geo_block_info.bbox
    ...     ...                                          # doctest: +SKIP
    ...     return burned_uint8
    >>> out = r.geo_map_blocks(                          # doctest: +SKIP
    ...     clip_and_rasterize,
    ...     meta=np.array((), dtype=np.uint8),
    ...     gdf=lines_gdf,
    ... )

    For dtype-preserving funcs (output dtype equals input dtype), you can skip
    ``meta=`` and guard on the meta call instead:

    >>> def scale(xda, *, geo_block_info=None):          # doctest: +SKIP
    ...     if geo_block_info is None:
    ...         return xda                # safe: same dtype, same shape
    ...     return xda * 2

    See Also
    --------
    raster_tools.map_blocks : Non-geo variant; permissive (shape-only
        check).
    raster_tools.geo_map_overlap : Geo-aware variant with overlap.
    raster_tools.Raster.reproject : Per-input alignment to a target
        grid; pass ``r1.geobox`` to align ``r2`` to ``r1``.
    """
    if not rasters:
        raise ValueError("geo_map_blocks requires at least one raster")
    _check_dtype_meta_agree(dtype, meta)
    _check_no_dask_kwargs(kwargs)
    _check_reserved_kwargs(
        kwargs, _GEO_MAP_BLOCKS_RESERVED_KWARGS, "geo_map_blocks"
    )
    rasters = [get_raster(r) for r in rasters]
    _check_shape_aligned(rasters)
    if not are_all_grids_same([r.geobox for r in rasters]):
        raise ValueError(
            "geo_map_blocks requires all input rasters to be on the "
            "same grid (CRS, affine, shape). Use "
            "Raster.reproject(crs_or_geobox=...) to align inputs "
            "first, e.g. r2.reproject(r1.geobox)."
        )
    # Pre-rechunk template: keeps the output on the caller's original
    # y/x chunking even when out_bands re-tiles inputs for the compute.
    orig_ref = rasters[0]
    if out_bands is None:
        rasters = _align_chunks_to_ref(rasters)
        out_chunks = None
    else:
        _check_out_bands(out_bands)
        rasters = _align_chunks_single_band(rasters, out_bands)
        out_chunks = ((out_bands,), *rasters[0].data.chunks[1:])
    ref = rasters[0]
    data_hint = _out_dtype_hint(dtype, meta)
    wrapper, inputs = _build_geo_map_blocks_wrapper(
        func,
        rasters,
        null_value=null_value,
        out_dtype_hint=data_hint,
        out_bands=out_bands,
        return_mask=return_mask,
        out_data_dtype=data_hint,
    )
    if return_mask:
        # The wrapper returns a structured (data, mask) array (see
        # map_blocks); fold the user's data-side dtype=/meta= into a
        # structured meta rather than forwarding them raw.
        struct_meta = (
            np.empty(
                0, dtype=np.dtype([("data", data_hint), ("mask", np.bool_)])
            )
            if data_hint is not None
            else None
        )
        struct = da.map_blocks(
            wrapper,
            *inputs,
            dtype=None,
            meta=struct_meta,
            chunks=out_chunks,
            **kwargs,
        )
        out_data = struct["data"]
        out_mask = struct["mask"]
    else:
        out_data = da.map_blocks(
            wrapper,
            *inputs,
            dtype=dtype,
            meta=meta,
            chunks=out_chunks,
            **kwargs,
        )
        out_mask = None
    out_nv = _resolve_out_null_value(
        null_value=null_value,
        ref_dtype=ref.dtype,
        ref_null_value=ref.null_value,
        n_rasters=len(rasters),
        out_dtype=out_data.dtype,
    )
    return data_to_raster_like(
        out_data, orig_ref, mask=out_mask, nv=out_nv, burn=return_mask
    )


def geo_map_overlap(
    func,
    *rasters,
    depth,
    boundary=None,
    dtype=None,
    null_value=None,
    meta=None,
    return_mask=False,
    **kwargs,
):
    """Apply ``func`` block-wise with overlap, handing it georeferenced
    :class:`xarray.DataArray` blocks.

    Same shape and contract as :func:`~raster_tools.geo_map_blocks` but
    adds the overlap machinery from :func:`~raster_tools.map_overlap`
    (``depth``, ``boundary``, the data/mask boundary correspondence rule).
    Each ``xr.DataArray`` block includes the overlap region, and its
    coords reflect the overlapped extent (top-left corner shifted outward
    by the per-side pad).

    Available both as a module function and as a
    :class:`~raster_tools.Raster` method: ``r1.geo_map_overlap(func, r2,
    depth=1)`` is equivalent to ``geo_map_overlap(func, r1, r2,
    depth=1)`` -- in the method form the calling raster is the first
    input. References to the "first input" below mean ``r1`` in either
    spelling.

    **Per-block contract**

    Per-block kwargs (opt-in): ``input_masks``, ``input_null_values``,
    ``block_info``, ``block_id``, ``out_null_value``, ``geo_block_info``. Name
    any of these in ``func``'s signature to receive them per chunk; see below.

    By default the output Raster's mask is rebuilt from the output data and the
    resolved output null value (``out_data == null_value``, or
    ``np.isnan(out_data)`` for NaN nulls) -- write the sentinel only at cells
    you want masked. Cells your func happens to leave equal to the sentinel
    will appear masked even if you didn't intend them to; cells you wanted
    masked but didn't write the sentinel to will not. This is true regardless
    of the input rasters' masks: input masks do **not** carry through
    unchanged. To set the output mask explicitly instead -- decoupling nullness
    from the data values -- pass ``return_mask=True`` and have ``func`` return
    a ``(data, mask)`` pair (both overlap-included); see ``return_mask`` below.

    Always passed positionally::

        func(*data_dataarrays, **kwargs)

    where ``data_dataarrays`` is a tuple of N georeferenced
    :class:`xarray.DataArray` blocks, one per input raster, in input order.
    Each block already includes the overlap region; the wrapper trims it after
    the function returns so the result lands on the first input's grid.

    The user can opt in to receive per-block extras by including named
    parameters in ``func``'s signature. Same set as
    :func:`~raster_tools.geo_map_blocks`:

    - ``input_masks`` -- tuple of N ``xr.DataArray`` (bool) per-block mask
      arrays, parallel to and overlap-included with the data DataArrays.
    - ``input_null_values`` -- tuple of N scalars, each input's ``null_value``
      (``None`` if unset).
    - ``block_info`` -- dask's standard per-block info dict.
    - ``block_id`` -- dask's standard per-block index tuple: the block's
      ``chunk-location`` (equivalently ``block_info[None]["chunk-location"]``).
      ``None`` during the meta inference call.
    - ``out_null_value`` -- scalar; the resolved output null value the wrapper
      will use to derive the output mask. Write this sentinel at cells you want
      masked. See "Output null value resolution" in Notes below.
    - ``geo_block_info`` -- the per-chunk
      :class:`~raster_tools.blocks.GeoBlockInfo`, reflecting the
      **overlapped** extent: ``shape`` matches the data block (including
      overlap), ``geobox`` extends to cover the overlap region, and
      ``row_slice`` / ``col_slice`` may have negative starts for top/left edge
      chunks. ``None`` during the meta call.

    A function whose only kwargs absorber is ``**kwargs`` does NOT trigger any
    of these injections -- name the kwargs you want.

    **Reserved kwargs**

    ``input_masks``, ``input_null_values``, ``block_info``, ``block_id``,
    ``out_null_value``, and ``geo_block_info`` are reserved. Passing any of
    them via ``geo_map_overlap``'s own ``**kwargs`` raises ``ValueError``.

    Parameters
    ----------
    func : callable
        Per-block function. See "Per-block contract" above. May return either
        an :class:`xarray.DataArray` (its ``.data`` is extracted, preserving
        backend) or any array-like that dask can ingest (NumPy ndarray, cupy
        ndarray, sparse array, etc.) with the same shape as a single
        (overlap-included) data block. For non-numpy backends, also pass
        ``meta=``.
    *rasters : Raster or str
        The input rasters. In the method form the calling raster is the
        first input and ``*rasters`` holds any additional ones; in the
        function form at least one is required. Path strings are accepted.
        All inputs must be on the same grid (CRS, affine, shape) within the
        established sub-pixel tolerance; mismatched inputs raise
        ``ValueError``. Use ``r2.reproject(r1.geobox)`` to align inputs
        first if needed. Inputs are then auto-rechunked to the first
        input's chunk structure (``reproject`` does not adopt the target's
        chunking), so the output stays on the first input's grid and
        chunking.
    depth : int, tuple of int, or dict
        Same semantics as :func:`~raster_tools.map_overlap`.
    boundary : optional
        Same semantics as :func:`~raster_tools.map_overlap` (None / scalar /
        ``"null"`` / ``"null_value"`` / ``"nodata"`` /
        ``"reflect"`` / ``"periodic"`` / ``"nearest"`` / ``"none"``).
    dtype : dtype-like, optional
        Output dtype. When ``None`` (default), dask infers the dtype by calling
        ``func`` on tiny meta samples.
    null_value : scalar, optional
        Output null value.

        - ``None`` (default): if there is exactly one input raster and the
          output dtype matches its dtype, the output inherits that input's null
          value. Otherwise, a dtype-appropriate default from
          :func:`raster_tools.masking.get_default_null_value`.
        - scalar: used as-is.

    meta : array-like, optional
        Empty array with the desired output array type. Forwarded to
        :func:`dask.array.overlap.map_overlap`. When provided, dask uses this
        as the output meta and skips the 0-shape sample call it would otherwise
        make to derive one -- useful when ``func`` cannot tolerate 0-shape
        DataArray inputs. The wrapper extracts ``.data`` from any returned
        :class:`xarray.DataArray`, so ``meta`` describes the wrapper's array
        output (whatever backend the user's func produces). When ``None``
        (default), dask derives a NumPy meta by calling ``func`` on 0-shape
        inputs.
    return_mask : bool, optional
        If ``True``, ``func`` must return a ``(data, mask)`` pair instead of a
        single array/DataArray. The returned ``mask`` -- not a sentinel
        comparison -- defines the output Raster's null cells. ``mask`` is a
        boolean array the same shape as ``data`` (both overlap-included; dask
        trims them together). Masked cells are set to the resolved output null
        value (burned in). Use this to decouple *which* cells are null from
        *what value* they hold, avoiding the sentinel-collision pitfalls
        described above. The default is ``False`` (sentinel-derived mask).
        Notes:

        - Each element of the pair may be an :class:`xarray.DataArray` or a
          bare array; the wrapper uses each element's ``.data`` (any coords /
          CRS are discarded, same as for the data block).
        - The two arrays are carried through dask packed into a single NumPy
          structured-dtype block, then split apart again -- an internal detail;
          ``func`` just returns the plain pair.
        - Passing ``dtype=`` (or ``meta=``) describing the **data** dtype is
          recommended: it lets dask skip its 0-shape probe. Without a hint the
          func must tolerate that probe.
        - Requires NumPy-backed blocks. Composes with ``input_masks``,
          ``geo_block_info``, and every ``boundary`` mode. ``out_null_value``
          injection is unnecessary (though harmless) when ``return_mask=True``.
    **kwargs
        Extra keyword arguments forwarded per-block to ``func``. The reserved
        names listed above are not allowed here.

    Returns
    -------
    Raster
        A new lazy Raster on the first input's grid.

    Notes
    -----
    .. warning::
       ``func`` must not alter the grid. The output Raster lands on the
       **first input's** grid (its CRS / affine / x / y), regardless of
       any coords / CRS / nodata your func sets on a returned DataArray -- the
       wrapper extracts ``.data`` and discards everything else. Operations that
       change the grid (reproject, clip, coarsen, sel, manual coord assignment,
       etc.) will silently produce a Raster whose values were computed on a
       different grid than the one it claims, with no error at construction
       time. Use :meth:`Raster.reproject` / :meth:`Raster.clip` / etc. before
       or after this call instead.

    The wrapper always trims overlap before returning so the result matches the
    input grid. If you need un-trimmed output, call
    :func:`dask.array.overlap.map_overlap` directly on ``raster.data``.

    The output mask is all-False if no null value is set (see the per-block
    contract above for how the mask is built when one is). To write the output
    mask directly, pass ``return_mask=True`` and return a ``(data, mask)`` pair
    from ``func`` (see ``return_mask``).

    The data/mask boundary correspondence rule from
    :func:`~raster_tools.map_overlap` applies
    (``"null"`` -> mask True; reflect/periodic/nearest -> mask same; constant
    matching null_value -> mask True; other constants -> mask False). This
    affects what ``func`` sees in the mask block when it opts in to
    ``input_masks=``; it does not affect how the *output* mask is built.

    Dask invokes ``func`` once on 0-shape DataArrays (with zero-filled
    placeholder coords, not real geocoordinates) to derive the output array
    meta -- this happens whether or not ``dtype=`` is provided. Pass ``meta=``
    to skip the call entirely. (Exception: with ``return_mask=True``,
    ``dtype=`` is folded into a structured ``meta=`` and so *does* skip the
    0-shape meta call -- see ``return_mask``.) During the meta call
    ``geo_block_info`` is ``None`` if the func opts in. Most NumPy / xarray ops
    handle 0-shape inputs fine. If dask raises ``dtype inference failed in
    map_blocks. Please specify the dtype explicitly using the dtype kwarg``,
    that's the sample call (dask routes overlap through its inner map_blocks,
    so the message says map_blocks even from this function) -- ``dtype=`` per
    dask's hint is usually enough; pass ``meta=`` instead if your func also
    can't tolerate 0-shape inputs (dask silently swallows that crash, but your
    downstream output meta will be wrong).

    With ``boundary=None`` or ``"none"``, edge chunks aren't padded on the
    array-boundary side. The per-side overlap split is computed from actual
    block shape vs base chunk shape and distributed symmetrically; for those
    edge cases the ``geo_block_info`` extent may be slightly off-position. For
    interior chunks and all non-``"none"`` boundaries this is exact.

    **Output null value resolution**

    When ``func`` opts in to ``out_null_value``, the wrapper resolves the
    scalar per-chunk without an extra dtype-inference pass:

    - With ``meta=`` or ``dtype=`` set, from that hint (dask leaves
      ``block_info[None]["dtype"]`` as ``None`` whenever ``meta=`` is set,
      which is why the hint is consulted first).
    - Otherwise, from ``block_info[None]["dtype"]``.

    During dask's meta inference call (where ``block_info`` is ``None``),
    ``out_null_value`` is a typed zero of the first input's dtype so funcs like
    ``np.where(m, out_null_value, d)`` infer the same dtype as their input
    rather than collapsing to object. If your func's *output dtype* depends on
    the specific ``out_null_value`` scalar, pass ``meta=`` to skip dask's
    0-shape meta call entirely; ``dtype=`` skips only the additional sample
    call, not the meta call.

    Examples
    --------
    Sum vector-line lengths within a per-cell radius. Pass
    ``meta=`` so dask skips the 0-shape meta call entirely (no
    ``geo_block_info is None`` guard needed):

    >>> def length_chunk(xda, *, geo_block_info, gdf, radius):
    ...     xc, yc = geo_block_info.x, geo_block_info.y
    ...     ...                                          # doctest: +SKIP
    ...     return out_arr_float32
    >>> out = r.geo_map_overlap(                         # doctest: +SKIP
    ...     length_chunk, depth=10, boundary=0,
    ...     meta=np.array((), dtype=np.float32),
    ...     gdf=lines_df, radius=radius,
    ... )

    For dtype-preserving funcs (output dtype equals input dtype),
    you can skip ``meta=`` and guard on the meta call instead:

    >>> def smooth(xda, *, geo_block_info=None):         # doctest: +SKIP
    ...     if geo_block_info is None:
    ...         return xda             # safe: same dtype, same shape
    ...     ...                        # apply per-pixel smoother

    See Also
    --------
    raster_tools.geo_map_blocks : No-overlap variant.
    raster_tools.map_overlap : Non-geo variant; permissive (shape-only
        check).
    raster_tools.Raster.reproject : Per-input alignment to a target
        grid; pass ``r1.geobox`` to align ``r2`` to ``r1``.
    """
    if not rasters:
        raise ValueError("geo_map_overlap requires at least one raster")
    _check_dtype_meta_agree(dtype, meta)
    _check_no_dask_kwargs(kwargs)
    _check_no_out_bands(kwargs, "geo_map_overlap")
    _check_reserved_kwargs(
        kwargs, _GEO_MAP_BLOCKS_RESERVED_KWARGS, "geo_map_overlap"
    )
    rasters = [get_raster(r) for r in rasters]
    _check_shape_aligned(rasters)
    ref = rasters[0]
    if not are_all_grids_same([r.geobox for r in rasters]):
        raise ValueError(
            "geo_map_overlap requires all input rasters to be on the "
            "same grid (CRS, affine, shape). Use "
            "Raster.reproject(crs_or_geobox=...) to align inputs "
            "first, e.g. r2.reproject(r1.geobox)."
        )

    depth_dict = _normalize_depth(depth)
    _check_asymmetric_depth_compatible_with_boundary(depth_dict, boundary)
    rasters = _ensure_chunks_for_overlap(rasters, depth_dict)
    ref = rasters[0]  # rechunk may have changed ref's chunking

    data_hint = _out_dtype_hint(dtype, meta)
    wrapper, inputs = _build_geo_map_overlap_wrapper(
        func,
        rasters,
        null_value=null_value,
        out_dtype_hint=data_hint,
        return_mask=return_mask,
        out_data_dtype=data_hint,
    )
    pass_masks = has_keyword(func, "input_masks")
    boundaries = [_resolve_boundary(boundary, r)[0] for r in rasters]
    if pass_masks:
        boundaries.extend(_resolve_boundary(boundary, r)[1] for r in rasters)
    depths = [depth_dict] * len(inputs)

    if return_mask:
        # The wrapper returns a structured (data, mask) array (see
        # map_blocks); fold the user's data-side dtype=/meta= into a
        # structured meta. dask trims the structured block's overlap rim
        # like any other array.
        struct_meta = (
            np.empty(
                0, dtype=np.dtype([("data", data_hint), ("mask", np.bool_)])
            )
            if data_hint is not None
            else None
        )
        struct = da.overlap.map_overlap(
            wrapper,
            *inputs,
            depth=depths,
            boundary=boundaries,
            dtype=None,
            meta=struct_meta,
            **kwargs,
        )
        out_data = struct["data"]
        out_mask = struct["mask"]
    else:
        out_data = da.overlap.map_overlap(
            wrapper,
            *inputs,
            depth=depths,
            boundary=boundaries,
            dtype=dtype,
            meta=meta,
            **kwargs,
        )
        out_mask = None
    out_nv = _resolve_out_null_value(
        null_value=null_value,
        ref_dtype=ref.dtype,
        ref_null_value=ref.null_value,
        n_rasters=len(rasters),
        out_dtype=out_data.dtype,
    )
    return data_to_raster_like(
        out_data, ref, mask=out_mask, nv=out_nv, burn=return_mask
    )


# The four public map functions above are mirrored as Raster methods.
# The methods are defined without docstrings in raster.py: raster.py
# cannot import this module at load time (this module imports raster.py
# at the top), so the shared docstrings are attached here instead. The
# top-of-module import fully loads raster.py first, so Raster and its
# methods always exist by the time these lines run.
Raster.map_blocks.__doc__ = map_blocks.__doc__
Raster.map_overlap.__doc__ = map_overlap.__doc__
Raster.geo_map_blocks.__doc__ = geo_map_blocks.__doc__
Raster.geo_map_overlap.__doc__ = geo_map_overlap.__doc__
