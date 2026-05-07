"""Block-mapping primitives for Rasters.

Provides :func:`map_blocks` and :func:`map_overlap` as thin wrappers over
the corresponding ``dask.array`` functions, plus :func:`geo_map_blocks`
and :func:`geo_map_overlap` which hand each user callback coordinated
:class:`xarray.DataArray` blocks (with band/y/x coords, CRS, nodata)
instead of raw NumPy blocks.

Also defines :class:`GeoBlockInfo`, the geo-aware analog of dask's
``block_info`` dict: a per-block metadata object carrying the block's
:class:`~odc.geo.geobox.GeoBox`, its band/row/col slices into the parent
array, and helpers for padding/shifting the block window and reconstructing
an :class:`xarray.DataArray` for the block's data. The future
``geo_map_blocks`` / ``geo_map_overlap`` wrappers will build on this
primitive: each user callback receives a coordinated ``DataArray`` (with
band/y/x coords, CRS, nodata) instead of a raw NumPy block.
"""

import dask.array as da
import numpy as np
import xarray as xr
from affine import Affine
from dask.array.core import apply_infer_dtype
from dask.utils import has_keyword
from odc.geo.geobox import GeoBox

from raster_tools._grids import are_all_grids_same
from raster_tools.dask_utils import chunks_to_array_locations
from raster_tools.dtypes import is_int
from raster_tools.masking import get_default_null_value
from raster_tools.raster import data_to_raster_like, get_raster
from raster_tools.utils import nan_equal

__all__ = [
    "GeoBlockInfo",
    "geo_block_infos_as_dask",
    "geo_map_blocks",
    "geo_map_overlap",
    "infer_output_dtype",
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
        Wrap a block's pixel data as a coordinated :class:`xarray.DataArray`.

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


def _resolve_null_value(null_value, ref_null_value, out_dtype):
    if null_value is None:
        return ref_null_value
    if isinstance(null_value, str):
        if null_value != "default":
            raise ValueError(
                f"null_value string must be 'default', got {null_value!r}"
            )
        return get_default_null_value(out_dtype)
    return null_value


# Kwarg names reserved for map_blocks's per-block injection. The user's
# func opts in to any of these by including them as named parameters.
# The caller of map_blocks may not pass them via **kwargs.
_MAP_BLOCKS_RESERVED_KWARGS = frozenset(
    {"input_masks", "input_null_values", "block_info", "out_null_value"}
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
    meta_dtype = np.asarray(meta).dtype
    if np.dtype(dtype) != meta_dtype:
        raise ValueError(
            f"dtype={dtype!r} conflicts with meta.dtype={meta_dtype!r}; "
            "pass one or the other, or matching values."
        )


def map_blocks(
    func, *rasters, dtype=None, null_value=None, meta=None, **kwargs
):
    """Apply ``func`` block-wise across one or more aligned rasters.

    Thin wrapper over :func:`dask.array.map_blocks`. Each call to
    ``func`` receives one block from each input raster, in the same
    order as ``*rasters``. The output :class:`~raster_tools.Raster`
    adopts its CRS, affine, and x/y coords from the first input;
    its mask is derived from the output data (see Notes).
    v1 is shape-preserving: ``func`` must return a NumPy array of
    the same shape as a single input data block.

    Per-block contract
    ------------------
    Always passed positionally:

        func(*input_data, **kwargs)

    where ``input_data`` is a tuple of N NumPy blocks, one per input
    raster, in caller order.

    The user can also opt in to receive per-block extras by including
    named parameters in ``func``'s signature. Detection mirrors dask's
    own ``block_info=`` / ``block_id=`` mechanism and uses
    :func:`inspect.signature` (via :func:`dask.utils.has_keyword`).
    Recognized names:

    - ``input_masks`` -- tuple of N ``np.ndarray`` (bool) per-block
      mask arrays, parallel to ``input_data``.
    - ``input_null_values`` -- tuple of N scalars, each input
      raster's ``null_value`` (``None`` if unset).
    - ``block_info`` -- dask's standard per-block info dict (see
      :func:`dask.array.map_blocks`).
    - ``out_null_value`` -- scalar; the resolved output null value
      the wrapper will use to derive the output mask. Useful for
      ``func`` to write at cells it wants masked. The wrapper
      resolves it per-chunk using ``block_info[None]["dtype"]`` so
      no extra dtype-inference pass is needed. During dask's own
      meta inference call (where ``block_info`` is ``None``),
      ``out_null_value`` is a typed zero of the first input's dtype,
      so funcs like ``np.where(m, out_null_value, d)`` infer the
      same dtype as their input rather than collapsing to object
      dtype. If your func's *output dtype* depends on the specific
      ``out_null_value`` scalar, pass ``dtype=`` explicitly to
      bypass dask's inference.

    A function whose only kwargs absorber is ``**kwargs`` (e.g.
    ``def f(*args, **kwargs):``) does NOT trigger any of these
    injections -- ``inspect.signature`` only sees explicit parameter
    names. Name the kwargs you want.

    Reserved kwargs
    ---------------
    ``input_masks``, ``input_null_values``, ``block_info``, and
    ``out_null_value`` are reserved. Passing any of them via
    ``map_blocks``'s own ``**kwargs`` raises ``ValueError`` --
    otherwise the wrapper's injection would silently clobber the
    caller's value.

    Parameters
    ----------
    func : callable
        Per-block function. See "Per-block contract" above for the
        full signature rules. Must return a NumPy array of the same
        shape as a single input data block.
    *rasters : Raster or str
        One or more aligned input rasters. Path strings are accepted.
        Only the 3D shape is validated; CRS, affine, and chunk
        alignment are *not* checked (mirroring ``gdal_calc.py``'s
        default behavior). The caller is responsible for aligning
        inputs -- typically via ``r2.reproject(r1.geobox)``. For a
        geo-aware variant that strictly requires matching grids, see
        :func:`geo_map_blocks`.
    dtype : dtype-like, optional
        Output dtype. When ``None`` (default), dask infers the dtype
        by calling ``func`` on tiny meta samples (matches NumPy
        promotion for the typical elementwise case, and reflects any
        in-func cast such as ``.astype``).
    null_value : scalar, optional
        Output null value.

        - ``None`` (default): if there is exactly one input raster and
          the output dtype matches its dtype, the output inherits
          that input's null value (preserves the sentinel for
          identity-like single-input ops). Otherwise, the value is
          a dtype-appropriate default from
          :func:`raster_tools.masking.get_default_null_value` against
          the resolved output dtype -- always representable, never
          overflows when the dtype changes.
        - scalar: used as-is.
        - strings (including the previously-supported ``"default"``)
          are no longer accepted.

        To force inherit-from-first-input across other cases, pass
        the value explicitly: ``null_value=r1.null_value``.
    meta : array-like, optional
        Empty array with the desired output array type (e.g.
        ``np.empty((), dtype=np.float32)``, or a CuPy / sparse
        equivalent). Forwarded to :func:`dask.array.map_blocks`. When
        provided, dask uses this as the output meta and skips the
        0-shape sample call it would otherwise make to derive one --
        useful when ``func`` cannot tolerate 0-shape input. When
        ``None`` (default), dask derives a NumPy meta by calling
        ``func`` on 0-shape inputs.
    **kwargs
        Extra keyword arguments forwarded per-block to ``func``. The
        reserved names listed above are not allowed here.

    Returns
    -------
    Raster
        A new lazy Raster on the first input's grid.

    Notes
    -----
    Cross-input CRS or affine mismatches are not validated; the caller
    is responsible.

    The output Raster's mask is **derived from the output data and
    the resolved output null value** -- ``out_data == null_value``,
    or ``np.isnan(out_data)`` for NaN nulls; all-False if no null
    value is set. This matches the rest of raster_tools but means
    a function that changes which cells equal the null sentinel
    will shift which cells appear masked -- it does not carry the
    first input's mask through unchanged. Writing a new mask via
    ``func`` is not supported in v1.

    Examples
    --------
    Plain elementwise on one input (no special injection):

    >>> def double(d):
    ...     return d * 2
    >>> doubled = map_blocks(double, r)              # doctest: +SKIP

    Mask-aware multi-input -- write the null sentinel where either
    input was masked:

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
    geo_map_blocks : Geo-aware variant that requires matching grids
        and hands ``func`` coordinated ``xr.DataArray`` blocks.
    raster_tools.Raster.reproject : Per-input alignment to a target
        grid; pass ``r1.geobox`` to align ``r2`` to ``r1``.
    """
    if not rasters:
        raise ValueError("map_blocks requires at least one raster")
    _check_dtype_meta_agree(dtype, meta)
    reserved_collision = _MAP_BLOCKS_RESERVED_KWARGS & kwargs.keys()
    if reserved_collision:
        raise ValueError(
            f"these kwargs are reserved by map_blocks for per-block "
            f"injection and cannot be passed by the caller: "
            f"{sorted(reserved_collision)}"
        )
    rasters = [get_raster(r) for r in rasters]
    ref = rasters[0]
    for i, r in enumerate(rasters[1:], 1):
        if r.shape != ref.shape:
            raise ValueError(
                f"raster {i} shape {r.shape} does not match raster 0 "
                f"shape {ref.shape}"
            )
    wrapper, inputs = _build_map_blocks_wrapper(
        func, rasters, null_value=null_value
    )
    out_data = da.map_blocks(
        wrapper, *inputs, dtype=dtype, meta=meta, **kwargs
    )
    out_nv = _resolve_out_null_value(
        null_value=null_value,
        ref_dtype=ref.dtype,
        ref_null_value=ref.null_value,
        n_rasters=len(rasters),
        out_dtype=out_data.dtype,
    )
    return data_to_raster_like(out_data, ref, nv=out_nv)


def _build_map_blocks_wrapper(func, rasters, null_value=None):
    """Build the per-block wrapper map_blocks passes to dask.

    Returns ``(wrapper, inputs)``. ``wrapper`` is the callable to
    pass to :func:`dask.array.map_blocks`. ``inputs`` is the list of
    dask arrays to spread positionally (data first; masks appended
    only when the user's ``func`` opts in via ``input_masks=``).

    The wrapper captures only small immutable values from the input
    rasters (per-input null values, ref dtype + null value, and the
    user's ``null_value`` parameter) so dask doesn't ship full
    Raster objects across workers. ``out_null_value`` is resolved
    inside the wrapper at call time using
    ``block_info[None]["dtype"]``, so no pre-resolution is needed.
    """
    pass_masks = has_keyword(func, "input_masks")
    pass_block_info = has_keyword(func, "block_info")
    pass_input_nvs = has_keyword(func, "input_null_values")
    pass_out_nv = has_keyword(func, "out_null_value")

    inputs = [r.data for r in rasters]
    if pass_masks:
        inputs.extend(r.mask for r in rasters)

    # Capture only scalars / small immutables. No Raster references.
    nvs = tuple(r.null_value for r in rasters) if pass_input_nvs else None
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
        if pass_input_nvs:
            inner_kwargs["input_null_values"] = nvs
        if pass_out_nv:
            if block_info is None:
                inner_kwargs["out_null_value"] = meta_placeholder
            else:
                inner_kwargs["out_null_value"] = _resolve_out_null_value(
                    null_value=null_value,
                    ref_dtype=ref_dtype,
                    ref_null_value=ref_null_value,
                    n_rasters=n_rasters,
                    out_dtype=block_info[None]["dtype"],
                )
        return func(*input_data, **inner_kwargs)

    return _wrapper, inputs


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


def infer_output_dtype(func, *rasters, **kwargs):
    """Infer the output dtype ``func`` will produce on these rasters.

    Mirrors :func:`map_blocks`'s contract: applies the same
    introspection-based wrapping (``input_masks``,
    ``input_null_values``, ``block_info`` are injected if named in
    ``func``'s signature), then runs
    :func:`dask.array.core.apply_infer_dtype` on tiny meta samples to
    derive the output dtype without computing real data.

    Useful for users who want to know what dtype their per-block
    function will produce -- e.g. to pre-resolve a null value sentinel
    or build downstream graphs.

    Parameters
    ----------
    func : callable
        Per-block function; same contract as :func:`map_blocks`.
    *rasters : Raster or str
        One or more input rasters (path strings accepted).
    **kwargs
        Extra keyword arguments forwarded per-block to ``func``.

    Returns
    -------
    numpy.dtype
        The inferred output dtype.

    Raises
    ------
    ValueError
        If no rasters are provided, or if dask cannot infer the dtype
        from a sample call (in which case dask suggests passing
        ``dtype=`` explicitly to :func:`map_blocks`).
    """
    if not rasters:
        raise ValueError("infer_output_dtype requires at least one raster")
    reserved_collision = _MAP_BLOCKS_RESERVED_KWARGS & kwargs.keys()
    if reserved_collision:
        raise ValueError(
            f"these kwargs are reserved by map_blocks for per-block "
            f"injection and cannot be passed by the caller: "
            f"{sorted(reserved_collision)}"
        )
    rasters = [get_raster(r) for r in rasters]
    wrapper, inputs = _build_map_blocks_wrapper(func, rasters)
    return apply_infer_dtype(wrapper, inputs, kwargs, "infer_output_dtype")


def _check_asymmetric_depth_compatible_with_boundary(depth_dict, boundary):
    """Mirror dask's restriction: tuple depth requires boundary 'none'.

    ``da.overlap.map_overlap`` raises ``NotImplementedError`` deep in
    its graph when any per-axis depth is a tuple and the boundary on
    that axis isn't ``"none"``. Surface that as a ``ValueError`` at
    call time with a message that names the offending axis.
    """
    if boundary is None or boundary == "none":
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


def _ensure_chunks_for_overlap(rasters, depth_dict):
    """Rechunk inputs so each spatial chunk is at least ``depth`` wide.

    ``geo_map_overlap`` builds a per-chunk :class:`GeoBlockInfo` lookup
    from the input's pre-call chunking and keys it by
    ``block_info[0]['chunk-location']``. If dask auto-rechunks under
    ``da.overlap.map_overlap`` (because some chunk is smaller than the
    depth on that axis), the in-wrapper chunk-location refers to the
    rechunked grid and won't match the lookup. Pre-rechunking here
    keeps both in sync.
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
    if new_yx == yx_chunks:
        return rasters
    new_chunks_3d = (ref.data.chunks[0], *new_yx)
    return [r.chunk(new_chunks_3d) for r in rasters]


def _resolve_boundary(boundary, raster):
    """Return ``(data_boundary, mask_boundary)`` for one raster.

    See the table in :func:`map_overlap`.
    """
    if boundary is None:
        return None, None
    if isinstance(boundary, str):
        # "null" / "null_value" / "nodata" / "NoData" / "NODATA" all
        # match here -- compare case-insensitively.
        if boundary.lower() in _NULL_BOUNDARIES:
            nv = raster.null_value
            if nv is None:
                nv = get_default_null_value(raster.dtype)
            return nv, True
        if boundary in _NAMED_BOUNDARIES:
            # 'none' -> dask treats as no padding for both arrays.
            # 'reflect'/'periodic'/'nearest' -> mask is padded the
            # same way so reflected/wrapped/copied data and mask stay
            # in sync.
            return boundary, boundary
        raise ValueError(
            f"unrecognized boundary string {boundary!r}; expected one "
            f"of {sorted(_NULL_BOUNDARIES | _NAMED_BOUNDARIES)} or "
            "None / a numeric scalar"
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
    **kwargs,
):
    """Apply ``func`` block-wise with overlap across one or more rasters.

    Thin wrapper over :func:`dask.array.overlap.map_overlap`. Each call
    to ``func`` receives one block from each input raster, in the same
    order as ``*rasters``, with ``depth`` extra cells of overlap on
    each side. dask trims the overlap from the result before it's
    wrapped back into a Raster on the input's grid, so the user
    function returns same-shape (overlap-included) blocks and doesn't
    need to trim itself.

    Per-block contract
    ------------------
    Always passed positionally:

        func(*input_data, **kwargs)

    where ``input_data`` is a tuple of N NumPy blocks, one per input
    raster, in caller order. Each block already includes the overlap
    region.

    The user can also opt in to receive per-block extras by including
    named parameters in ``func``'s signature. Detection mirrors dask's
    own ``block_info=`` / ``block_id=`` mechanism and uses
    :func:`inspect.signature` (via :func:`dask.utils.has_keyword`).
    Recognized names (same set as :func:`map_blocks`):

    - ``input_masks`` -- tuple of N ``np.ndarray`` (bool) per-block
      mask arrays, parallel to ``input_data`` and overlap-included.
    - ``input_null_values`` -- tuple of N scalars, each input
      raster's ``null_value`` (``None`` if unset).
    - ``block_info`` -- dask's standard per-block info dict.
    - ``out_null_value`` -- scalar; the resolved output null value
      the wrapper will use to derive the output mask. Resolved per-
      chunk via ``block_info[None]["dtype"]``. During dask's meta
      inference call, this is a typed zero of the first input's
      dtype (so funcs like ``np.where(m, out_null_value, d)`` infer
      the same dtype as their input).

    A function whose only kwargs absorber is ``**kwargs`` does NOT
    trigger any of these injections -- name the kwargs you want.

    Reserved kwargs
    ---------------
    ``input_masks``, ``input_null_values``, ``block_info``, and
    ``out_null_value`` are reserved. Passing any of them via
    ``map_overlap``'s own ``**kwargs`` raises ``ValueError``.

    Parameters
    ----------
    func : callable
        Per-block function. See "Per-block contract" above. Must
        return a NumPy array of the same shape as a single
        (overlap-included) data block.
    *rasters : Raster or str
        One or more aligned input rasters. Path strings are accepted.
        Only the 3D shape is validated; CRS, affine, and chunk
        alignment are *not* checked. The caller is responsible for
        aligning inputs -- typically via ``r2.reproject(r1.geobox)``.
        For a geo-aware variant that strictly requires matching grids,
        see :func:`geo_map_overlap`.
    depth : int, tuple of int, or dict
        Number of overlap cells per spatial axis. ``int`` applies to
        both ``y`` and ``x`` (band axis fixed at 0).
        ``(dy, dx)`` sets per-spatial-axis depth.
        ``dict`` maps axis index to depth (or to a ``(top, bottom)`` /
        ``(left, right)`` tuple for asymmetric depths). Asymmetric
        depths require ``boundary=None`` or ``boundary="none"`` per
        dask's restriction.
    boundary : optional
        How to fill cells outside the array's edges. Choices:

        - ``None`` (default): no padding (matches dask).
        - ``"null"`` / ``"null_value"`` / ``"nodata"`` (case-insensitive,
          so ``"NODATA"`` / ``"NoData"`` also work): fill with the
          input raster's null value (or
          :func:`get_default_null_value(dtype) <
          raster_tools.masking.get_default_null_value>` if unset). The
          corresponding mask cells are set to ``True``.
        - a numeric scalar: fill with that value. If the value matches
          the raster's null value, mask cells are set to ``True``;
          otherwise they're set to ``False``.
        - ``"reflect"`` / ``"periodic"`` / ``"nearest"``: dask's
          standard padding modes. The mask is padded the same way so
          reflected/wrapped/copied data and mask stay in sync at the
          source cell.
        - ``"none"``: explicit no-padding (same as ``None``).

        For multi-input, each raster's null value is consulted
        independently.

        The boundary -> mask rule only affects what the user's function
        sees in the mask block when it opts in to ``input_masks=``.
        The output Raster's mask is built independently from the
        output data (see Returns / Notes below); padded cells are
        trimmed off before the user sees them.
    dtype : dtype-like, optional
        Output dtype. When ``None`` (default), dask infers the dtype
        by calling ``func`` on tiny meta samples.
    null_value : scalar, optional
        Output null value.

        - ``None`` (default): if there is exactly one input raster and
          the output dtype matches its dtype, the output inherits
          that input's null value. Otherwise, the value is a
          dtype-appropriate default from
          :func:`raster_tools.masking.get_default_null_value`.
        - scalar: used as-is.
        - strings (including the previously-supported ``"default"``)
          are no longer accepted.
    meta : array-like, optional
        Empty array with the desired output array type. Forwarded to
        :func:`dask.array.overlap.map_overlap`. When provided, dask
        uses this as the output meta and skips the 0-shape sample
        call it would otherwise make to derive one -- useful when
        ``func`` cannot tolerate 0-shape input. When ``None``
        (default), dask derives a NumPy meta by calling ``func`` on
        0-shape inputs.
    **kwargs
        Extra keyword arguments forwarded per-block to ``func``. The
        reserved names listed above are not allowed here.

    Returns
    -------
    Raster
        A new lazy Raster on the first input's grid.

    Notes
    -----
    The wrapper always trims overlap before returning so the result
    matches the input grid. If you need un-trimmed output, call
    :func:`dask.array.overlap.map_overlap` directly on
    ``raster.data``.

    The output Raster's mask is **derived from the output data and
    the resolved output null value** -- ``out_data == null_value``,
    or ``np.isnan(out_data)`` for NaN nulls; all-False if no null
    value is set. Writing a new mask via ``func`` is not supported.

    Asymmetric per-side depths are only supported with no padding
    (``boundary=None`` or ``"none"``).

    See Also
    --------
    map_blocks : Block-wise without overlap; same per-block contract.
    geo_map_overlap : Geo-aware variant that hands ``func`` coordinated
        ``xr.DataArray`` blocks.
    """
    if not rasters:
        raise ValueError("map_overlap requires at least one raster")
    _check_dtype_meta_agree(dtype, meta)
    reserved_collision = _MAP_BLOCKS_RESERVED_KWARGS & kwargs.keys()
    if reserved_collision:
        raise ValueError(
            f"these kwargs are reserved by map_overlap for per-block "
            f"injection and cannot be passed by the caller: "
            f"{sorted(reserved_collision)}"
        )
    rasters = [get_raster(r) for r in rasters]
    ref = rasters[0]
    for i, r in enumerate(rasters[1:], 1):
        if r.shape != ref.shape:
            raise ValueError(
                f"raster {i} shape {r.shape} does not match raster 0 "
                f"shape {ref.shape}"
            )
    depth_dict = _normalize_depth(depth)
    _check_asymmetric_depth_compatible_with_boundary(depth_dict, boundary)

    wrapper, inputs = _build_map_blocks_wrapper(
        func, rasters, null_value=null_value
    )
    pass_masks = has_keyword(func, "input_masks")
    boundaries = [_resolve_boundary(boundary, r)[0] for r in rasters]
    if pass_masks:
        boundaries.extend(_resolve_boundary(boundary, r)[1] for r in rasters)
    depths = [depth_dict] * len(inputs)

    out_data = da.overlap.map_overlap(
        wrapper,
        *inputs,
        depth=depths,
        boundary=boundaries,
        dtype=dtype,
        meta=meta,
        **kwargs,
    )
    out_nv = _resolve_out_null_value(
        null_value=null_value,
        ref_dtype=ref.dtype,
        ref_null_value=ref.null_value,
        n_rasters=len(rasters),
        out_dtype=out_data.dtype,
    )
    return data_to_raster_like(out_data, ref, nv=out_nv)


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


def _validate_aligned_rasters(rasters, fname):
    if not rasters:
        raise ValueError(f"{fname} requires at least one raster")
    rasters = [get_raster(r) for r in rasters]
    ref = rasters[0]
    for i, r in enumerate(rasters[1:], 1):
        if r.shape != ref.shape:
            raise ValueError(
                f"raster {i} shape {r.shape} does not match raster 0 "
                f"shape {ref.shape}"
            )
    return rasters, ref


def _build_geo_map_blocks_wrapper(func, rasters, null_value=None):
    """Build the per-block wrapper geo_map_blocks passes to dask.

    Returns ``(wrapper, inputs)``. ``wrapper`` is the callable to
    pass to :func:`dask.array.map_blocks`. ``inputs`` is the list of
    dask arrays to spread positionally (data first; masks appended
    only when the user's ``func`` opts in via ``input_masks=``).

    Mirrors :func:`_build_map_blocks_wrapper` but produces coordinated
    :class:`xarray.DataArray` blocks instead of NumPy. Captures only
    small immutables in the closure -- no Raster references.
    """
    pass_masks = has_keyword(func, "input_masks")
    pass_block_info = has_keyword(func, "block_info")
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
            chunk_loc = block_info[0]["chunk-location"]
            gbi = gbi_lookup[chunk_loc]
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
        if pass_geo_block_info:
            inner_kwargs["geo_block_info"] = gbi
        if pass_out_nv:
            if block_info is None:
                inner_kwargs["out_null_value"] = meta_placeholder
            else:
                inner_kwargs["out_null_value"] = _resolve_out_null_value(
                    null_value=null_value,
                    ref_dtype=ref_dtype,
                    ref_null_value=ref_null_value,
                    n_rasters=n_rasters,
                    out_dtype=block_info[None]["dtype"],
                )

        result = func(*data_das, **inner_kwargs)
        if hasattr(result, "values"):
            return np.asarray(result.values)
        return np.asarray(result)

    return _wrapper, inputs


def geo_map_blocks(
    func,
    *rasters,
    dtype=None,
    null_value=None,
    meta=None,
    **kwargs,
):
    """Apply ``func`` block-wise across one or more aligned rasters,
    handing it coordinated :class:`xarray.DataArray` blocks.

    Same shape and contract as :func:`map_blocks`, but each raster's
    data block is wrapped in a coordinated ``xr.DataArray`` (with
    ``band`` / ``y`` / ``x`` coords from the block's geobox, the
    raster's CRS, and the raster's null value attached as ``nodata``)
    via :meth:`GeoBlockInfo.to_dataarray` before being passed to
    ``func``. Useful when the per-block function wants to operate in
    xarray-land (rio accessors, xr.where, etc.) without having to
    rebuild coordinates itself.

    Per-block contract
    ------------------
    Always passed positionally:

        func(*data_dataarrays, **kwargs)

    where ``data_dataarrays`` is a tuple of N coordinated
    :class:`xarray.DataArray` blocks, one per input raster, in caller
    order.

    The user can also opt in to receive per-block extras by including
    named parameters in ``func``'s signature. Detection mirrors dask's
    ``block_info=`` mechanism (via :func:`dask.utils.has_keyword`).
    Recognized names:

    - ``input_masks`` -- tuple of N ``xr.DataArray`` (bool) per-block
      mask arrays, parallel to the data DataArrays. Same name as
      :func:`map_blocks` (NumPy there); per-function the element
      type matches what data is in that function.
    - ``input_null_values`` -- tuple of N scalars, each input's
      ``null_value`` (``None`` if unset).
    - ``block_info`` -- dask's standard per-block info dict.
    - ``out_null_value`` -- scalar; the resolved output null value
      the wrapper will use to derive the output mask. Resolved
      per-chunk via ``block_info[None]["dtype"]``. During dask's
      meta inference call, this is a typed zero of the first
      input's dtype.
    - ``geo_block_info`` -- the per-chunk :class:`GeoBlockInfo`
      (the geo-aware analog of dask's ``block_info``). ``None``
      during the meta inference call.

    A function whose only kwargs absorber is ``**kwargs`` does NOT
    trigger any of these injections -- name the kwargs you want.

    Reserved kwargs
    ---------------
    ``input_masks``, ``input_null_values``, ``block_info``,
    ``out_null_value``, and ``geo_block_info`` are reserved. Passing
    any of them via ``geo_map_blocks``'s own ``**kwargs`` raises
    ``ValueError``.

    Parameters
    ----------
    func : callable
        Per-block function. See "Per-block contract" above. May
        return either an ``xr.DataArray`` (its ``.values`` are
        extracted) or a NumPy array of the same shape as a single
        data block.
    *rasters : Raster or str
        One or more input rasters. Path strings are accepted. All
        inputs must be on the same grid (CRS, affine, shape) within
        the established sub-pixel tolerance; mismatched inputs raise
        ``ValueError``. Use ``r2.reproject(r1.geobox)`` to align
        inputs first if needed.
    dtype : dtype-like, optional
        Output dtype. When ``None`` (default), dask infers the dtype
        by calling ``func`` on tiny meta samples.
    null_value : scalar, optional
        Output null value.

        - ``None`` (default): if there is exactly one input raster and
          the output dtype matches its dtype, the output inherits
          that input's null value. Otherwise, a dtype-appropriate
          default from
          :func:`raster_tools.masking.get_default_null_value`.
        - scalar: used as-is.
        - strings (including the previously-supported ``"default"``)
          are no longer accepted.
    meta : array-like, optional
        Empty array with the desired output array type. Forwarded to
        :func:`dask.array.map_blocks`. When provided, dask uses this
        as the output meta and skips the 0-shape sample call it
        would otherwise make to derive one -- useful when ``func``
        cannot tolerate 0-shape DataArray inputs. Note that the
        wrapper always returns a NumPy array (it extracts ``.values``
        from any returned :class:`xarray.DataArray`), so ``meta``
        describes the wrapper's NumPy output, not the user func's
        xarray output. When ``None`` (default), dask derives a NumPy
        meta by calling ``func`` on 0-shape inputs.
    **kwargs
        Extra keyword arguments forwarded per-block to ``func``. The
        reserved names listed above are not allowed here.

    Returns
    -------
    Raster
        A new lazy Raster on the first input's grid.

    Notes
    -----
    Coords / CRS / nodata on a returned DataArray are not validated
    against the input -- they're discarded; the output Raster's grid
    comes from ``rasters[0]``.

    The output Raster's mask is **derived from the output data and
    the resolved output null value** -- ``out_data == null_value``,
    or ``np.isnan(out_data)`` for NaN nulls; all-False if no null
    value is set. Writing a new mask via ``func`` is not supported.

    Dask invokes ``func`` once on 0-shape DataArrays (no coords) to
    derive the output array meta -- this happens whether or not
    ``dtype=`` is provided. Passing ``dtype=`` only skips the
    additional sample call dask would otherwise make to infer the
    output dtype; it does not skip the 0-shape meta call. During the
    meta call ``geo_block_info`` is ``None`` if the func opts in.
    Most NumPy / xarray ops handle 0-shape inputs fine.

    See Also
    --------
    map_blocks : Non-geo variant; permissive (shape-only check).
    geo_map_overlap : Geo-aware variant with overlap.
    raster_tools.Raster.reproject : Per-input alignment to a target
        grid; pass ``r1.geobox`` to align ``r2`` to ``r1``.
    """
    if not rasters:
        raise ValueError("geo_map_blocks requires at least one raster")
    _check_dtype_meta_agree(dtype, meta)
    reserved_collision = _GEO_MAP_BLOCKS_RESERVED_KWARGS & kwargs.keys()
    if reserved_collision:
        raise ValueError(
            f"these kwargs are reserved by geo_map_blocks for per-block "
            f"injection and cannot be passed by the caller: "
            f"{sorted(reserved_collision)}"
        )
    rasters, ref = _validate_aligned_rasters(rasters, "geo_map_blocks")
    if not are_all_grids_same([r.geobox for r in rasters]):
        raise ValueError(
            "geo_map_blocks requires all input rasters to be on the "
            "same grid (CRS, affine, shape). Use "
            "Raster.reproject(crs_or_geobox=...) to align inputs "
            "first, e.g. r2.reproject(r1.geobox)."
        )
    wrapper, inputs = _build_geo_map_blocks_wrapper(
        func, rasters, null_value=null_value
    )
    out_data = da.map_blocks(
        wrapper, *inputs, dtype=dtype, meta=meta, **kwargs
    )
    out_nv = _resolve_out_null_value(
        null_value=null_value,
        ref_dtype=ref.dtype,
        ref_null_value=ref.null_value,
        n_rasters=len(rasters),
        out_dtype=out_data.dtype,
    )
    return data_to_raster_like(out_data, ref, nv=out_nv)


def geo_map_overlap(
    func,
    *rasters,
    depth,
    boundary=None,
    pass_mask=False,
    dtype=None,
    null_value=None,
    **kwargs,
):
    """Apply ``func`` block-wise with overlap, handing it coordinated
    :class:`xarray.DataArray` blocks.

    Same shape and contract as :func:`geo_map_blocks` but adds the
    overlap machinery from :func:`map_overlap` (``depth``,
    ``boundary``, the data/mask boundary correspondence rule). Each
    ``xr.DataArray`` block includes the overlap region, and its
    coords reflect the overlapped extent (top-left corner shifted
    outward by the per-side pad).

    Parameters
    ----------
    func : callable
        Per-block function. With ``pass_mask=False`` (default) it
        receives ``(xda1, ..., xdaN, geo_block_info=gbi, **kwargs)``.
        With ``pass_mask=True`` each input's mask DataArray is
        interleaved immediately after its data DataArray:
        ``(xda1, xma1, xda2, xma2, ..., xdaN, xmaN,
        geo_block_info=gbi, **kwargs)``. Each block already includes
        the overlap region; the wrapper trims it after the function
        returns so the result lands on the input's grid. May return
        either an ``xr.DataArray`` (its ``.values`` are extracted) or
        a NumPy array of the same shape as a single
        (overlap-included) data block.
    *rasters : Raster or str
        One or more input rasters. Path strings are accepted. All
        inputs must be on the same grid (CRS, affine, shape) within
        the established sub-pixel tolerance; mismatched inputs raise
        ``ValueError``. Use ``r2.reproject(r1.geobox)`` to align
        inputs first if needed.
    depth : int, tuple of int, or dict
        Same semantics as :func:`map_overlap`.
    boundary : optional
        Same semantics as :func:`map_overlap` (None / scalar /
        ``"null"`` / ``"null_value"`` / ``"nodata"`` /
        ``"reflect"`` / ``"periodic"`` / ``"nearest"`` / ``"none"``).
    pass_mask : bool, optional
        If ``True``, each input's boolean mask DataArray is passed to
        ``func`` immediately after its data DataArray (interleaved).
        Default ``False``.
    dtype : dtype-like, optional
        Output dtype. When ``None`` (default), dask infers the dtype
        by calling ``func`` on tiny meta samples (matches NumPy
        promotion for the typical elementwise case, and reflects any
        in-func cast such as ``.astype``).
    null_value : scalar or str, optional
        Output null value. ``None`` (default) inherits from the first
        input. A scalar is used as-is. The string ``"default"`` selects
        a dtype-appropriate default via
        :func:`raster_tools.masking.get_default_null_value`.
    **kwargs
        Extra keyword arguments forwarded per-block to ``func``.

    Returns
    -------
    Raster
        A new lazy Raster on the first input's grid.

    Notes
    -----
    The wrapper always trims overlap before returning so the result
    matches the input grid. If you need un-trimmed output, call
    :func:`dask.array.overlap.map_overlap` directly on
    ``raster.data``.

    The per-block :class:`GeoBlockInfo` is always passed to ``func`` as
    ``geo_block_info`` and reflects the **overlapped** extent: its
    ``shape`` matches the data block (including overlap), ``geobox``
    extends to cover the overlap region, and ``row_slice`` /
    ``col_slice`` may have negative starts for top/left edge chunks.

    The output Raster's mask is **derived from the output data and
    the resolved output null value** -- ``out_data == null_value``,
    or ``np.isnan(out_data)`` for NaN nulls; all-False if no null
    value is set. This matches the rest of raster_tools but means
    a function that changes which cells equal the null sentinel
    will shift which cells appear masked -- it does not carry the
    first input's mask through unchanged. Writing a new mask via
    ``func`` is not supported in v1.

    The data/mask boundary correspondence rule from :func:`map_overlap`
    applies (``"null"`` -> mask True; reflect/periodic/nearest -> mask
    same; constant matching null_value -> mask True; other constants
    -> mask False). This affects what ``func`` sees in the mask block
    when ``pass_mask=True``; it does not affect how the *output*
    mask is built.

    Dask invokes ``func`` once on 0-shape DataArrays (no coords) to
    derive the output array meta -- this happens whether or not
    ``dtype=`` is provided. Passing ``dtype=`` only skips the
    additional sample call dask would otherwise make to infer the
    output dtype; it does not skip the 0-shape meta call. During the
    meta call ``geo_block_info`` is ``None``. Most NumPy / xarray ops
    handle 0-shape inputs fine.

    Per-input null values are not passed to ``func``; prefer
    ``pass_mask=True`` over comparing data values to a null sentinel.

    With ``boundary=None`` or ``"none"``, edge chunks aren't padded on
    the array-boundary side. The per-side overlap split is computed
    from actual block shape vs base chunk shape and distributed
    symmetrically; for those edge cases the ``geo_block_info`` extent
    may be slightly off-position. For interior chunks and all
    non-``"none"`` boundaries this is exact.

    See Also
    --------
    geo_map_blocks : No-overlap variant.
    map_overlap : Non-geo variant; permissive (shape-only check).
    raster_tools.Raster.reproject : Per-input alignment to a target
        grid; pass ``r1.geobox`` to align ``r2`` to ``r1``.
    """
    # TODO: add a `meta=` kwarg forwarded to da.overlap.map_overlap
    # (matching map_blocks / map_overlap / geo_map_blocks). Deferred
    # until this function is migrated to the introspection-based
    # contract; do it as part of that refactor to avoid churning the
    # signature twice.
    rasters, ref = _validate_aligned_rasters(rasters, "geo_map_overlap")
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
    nvs = [r.null_value for r in rasters]
    n = len(rasters)
    gbi_lookup = geo_block_infos(ref)

    if pass_mask:
        boundaries = []
        for r in rasters:
            data_b, mask_b = _resolve_boundary(boundary, r)
            boundaries.extend([data_b, mask_b])
    else:
        boundaries = [_resolve_boundary(boundary, r)[0] for r in rasters]

    if pass_mask:
        inputs = [arr for r in rasters for arr in (r.data, r.mask)]
    else:
        inputs = [r.data for r in rasters]
    depths = [depth_dict] * len(inputs)

    def _wrapper(*block_args, block_info=None, **inner_kwargs):
        # block_info is None during dask's meta-inference call.
        if block_info is None:
            if pass_mask:
                das = []
                for i in range(n):
                    d = block_args[2 * i]
                    m = block_args[2 * i + 1]
                    das.extend(
                        [
                            _meta_dataarray(d, nodata=nvs[i]),
                            _meta_dataarray(m),
                        ]
                    )
            else:
                das = [
                    _meta_dataarray(block_args[i], nodata=nvs[i])
                    for i in range(n)
                ]
            result = func(*das, geo_block_info=None, **inner_kwargs)
            if hasattr(result, "values"):
                return np.asarray(result.values)
            return np.asarray(result)

        chunk_loc = block_info[0]["chunk-location"]
        base_gbi = gbi_lookup[chunk_loc]
        base_shape = base_gbi.shape
        actual_shape = block_args[0].shape
        # Distribute the per-axis overlap symmetrically. Exact for
        # interior chunks and any non-'none' boundary; approximate for
        # boundary=None/'none' edge chunks (see Notes).
        tot_dy = actual_shape[1] - base_shape[1]
        tot_dx = actual_shape[2] - base_shape[2]
        top, bottom = tot_dy // 2, tot_dy - tot_dy // 2
        left, right = tot_dx // 2, tot_dx - tot_dx // 2
        gbi = base_gbi.pad_y(top, bottom) if (top or bottom) else base_gbi
        if left or right:
            gbi = gbi.pad_x(left, right)

        if pass_mask:
            das = []
            for i in range(n):
                d = block_args[2 * i]
                m = block_args[2 * i + 1]
                xd, xm = gbi.to_dataarray(d, mask=m, nodata=nvs[i])
                das.extend([xd, xm])
        else:
            das = [
                gbi.to_dataarray(block_args[i], nodata=nvs[i])
                for i in range(n)
            ]
        result = func(*das, geo_block_info=gbi, **inner_kwargs)
        if hasattr(result, "values"):
            return np.asarray(result.values)
        return np.asarray(result)

    out_data = da.overlap.map_overlap(
        _wrapper,
        *inputs,
        depth=depths,
        boundary=boundaries,
        dtype=dtype,
        **kwargs,
    )
    out_nv = _resolve_null_value(null_value, ref.null_value, out_data.dtype)
    return data_to_raster_like(out_data, ref, nv=out_nv)
