import numpy as np
from numba import generated_jit, jit, types

__all__ = [
    "nan_unique_count_jit",
    "nanargmax_jit",
    "nanargmin_jit",
    "nanasm_jit",
    "nanentropy_jit",
    "nanmax_jit",
    "nanmean_jit",
    "nanmedian_jit",
    "nanmin_jit",
    "nanmode_jit",
    "nanstd_jit",
    "nansum_jit",
    "nanvar_jit",
]


ngjit = jit(nopython=True, nogil=True)


# Numba has a long standing bug where scalars are not handled properly in
# several numpy functions. This is a workaround. It generates the proper
# implementation based on the input type. This func can be used to properly
# handle scalar inputs in funcs that would otherwise fail to compile for scalar
# inputs like np.nanmean.
# See: https://github.com/numba/numba/issues/4202
@generated_jit(nopython=True, nogil=True)
def _atleast_1d(x):
    if x in types.number_domain:
        return lambda x: np.array([x])
    return lambda x: np.atleast_1d(x)


@ngjit
def nanmin_jit(x):
    return np.nanmin(x)


@ngjit
def nanmax_jit(x):
    return np.nanmax(x)


@ngjit
def nanmean_jit(x):
    x = _atleast_1d(x)
    return np.nanmean(x)


@ngjit
def nanmedian_jit(x):
    x = _atleast_1d(x)
    return np.nanmedian(x)


@ngjit
def nansum_jit(x):
    x = _atleast_1d(x)
    return np.nansum(x)


@ngjit
def nanvar_jit(x):
    x = _atleast_1d(x)
    return np.nanvar(x)


@ngjit
def nanstd_jit(x):
    x = _atleast_1d(x)
    return np.nanstd(x)


@ngjit
def nan_unique_count_jit(x):
    x = _atleast_1d(x)
    # Create set of floats. {1.0} is a hack to tell numba.jit what type the set
    # contains
    s = {1.0}
    s.clear()
    for v in x.ravel():
        if not np.isnan(v):
            s.add(v)
    return len(s)


@ngjit
def nanmode_jit(x):
    # Returns NaN if all values are NaN.
    x = _atleast_1d(x)
    c = {}
    x = x.ravel()
    j = 0
    for i in range(x.size):
        v = x[i]
        if not np.isnan(v):
            if v in c:
                c[v] += 1
            else:
                c[v] = 1
            x[j] = v
            j += 1
    vals = x[:j]
    if len(vals) == 0:
        return np.nan
    vals.sort()
    cnts = np.empty(len(vals), dtype=types.uint64)
    for i in range(len(vals)):
        cnts[i] = c[vals[i]]
    return vals[np.argmax(cnts)]


@ngjit
def nanentropy_jit(x):
    x = _atleast_1d(x)
    c = {}
    n = 0
    for v in x.ravel():
        if not np.isnan(v):
            if v in c:
                c[v] += 1
            else:
                c[v] = 1
            n += 1
    entr = 0.0
    if n > 0:
        frac = 1.0 / n
        for cnt in c.values():
            p = cnt * frac
            entr -= p * np.log(p)
    return entr


@ngjit
def nanasm_jit(x):
    x = _atleast_1d(x)
    c = {}
    n = 0
    for v in x.ravel():
        if not np.isnan(v):
            if v in c:
                c[v] += 1
            else:
                c[v] = 1
            n += 1
    asm = 0.0
    if n > 0:
        frac = 1.0 / n
        for cnt in c.values():
            p = cnt * frac
            asm += p * p
    return asm


@ngjit
def nanargmin_jit(x):
    # Get argmin. If empty give -1, if all nan, give -2. This differs from
    # numpy which throws an error.
    x = _atleast_1d(x)
    if x.size == 0:
        return -1
    nan_mask = np.isnan(x)
    if nan_mask.all():
        return -2
    x = x.ravel()
    nan_mask = nan_mask.ravel()
    j = 0
    min_val = np.inf
    for i in range(x.size):
        v = x[i]
        if nan_mask[i]:
            continue
        if v < min_val:
            j = i
            min_val = v
    return j


@ngjit
def nanargmax_jit(x):
    # Get argmax. If empty give -1, if all nan, give -2. This differs from
    # numpy which throws an error.
    x = _atleast_1d(x)
    if x.size == 0:
        return -1
    nan_mask = np.isnan(x)
    if nan_mask.all():
        return -2
    x = x.ravel()
    nan_mask = nan_mask.ravel()
    j = 0
    max_val = -np.inf
    for i in range(x.size):
        v = x[i]
        if nan_mask[i]:
            continue
        if v > max_val:
            j = i
            max_val = v
    return j
