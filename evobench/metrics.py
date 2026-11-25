"""Quality indicators for multi-objective optimization: hypervolume and IGD.

Hypervolume is computed exactly for 2 objectives (sweep over the sorted
front) and for 3 objectives (exact slab decomposition over the third
objective, reusing the 2-D routine).  More than 3 objectives raise
``NotImplementedError`` — use Monte-Carlo sampling externally if you need
higher dimensions.

IGD (inverted generational distance) measures the average distance from
points of a reference set (usually the true Pareto front) to the closest
point of an approximation set; smaller is better.
"""

import math

__all__ = ["hypervolume", "igd", "nondominated"]


def dominates(a, b):
    """True if ``a`` weakly dominates ``b`` and is strictly better somewhere."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def nondominated(points):
    """Return the non-dominated subset of ``points`` (minimization)."""
    result = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i != j and dominates(q, p):
                dominated = True
                break
        if not dominated:
            result.append(list(p))
    return result


def hypervolume(points, ref_point):
    """Exact hypervolume of ``points`` w.r.t. ``ref_point`` (minimization).

    Points that do not strictly dominate the reference point are ignored.
    Supported for 2 and 3 objectives.
    """
    ref = [float(v) for v in ref_point]
    m = len(ref)
    if m not in (2, 3):
        raise NotImplementedError(
            "exact hypervolume is implemented for 2 and 3 objectives only"
        )
    pts = [
        [float(v) for v in p]
        for p in points
        if all(p[i] < ref[i] for i in range(m))
    ]
    if not pts:
        return 0.0
    pts = nondominated(pts)
    if m == 2:
        return _hypervolume_2d(pts, ref)
    return _hypervolume_3d(pts, ref)


def _hypervolume_2d(points, ref):
    """Exact 2-D hypervolume of a non-dominated set."""
    ordered = sorted(points)  # f1 ascending; f2 descending for a front
    volume = 0.0
    prev_f2 = ref[1]
    for f1, f2 in ordered:
        volume += (ref[0] - f1) * (prev_f2 - f2)
        prev_f2 = f2
    return volume


def _hypervolume_3d(points, ref):
    """Exact 3-D hypervolume by slab decomposition along the 3rd objective."""
    levels = sorted({p[2] for p in points})
    volume = 0.0
    for i, lo in enumerate(levels):
        hi = levels[i + 1] if i + 1 < len(levels) else ref[2]
        active = [[p[0], p[1]] for p in points if p[2] <= lo + 1e-12]
        volume += (hi - lo) * _hypervolume_2d(nondominated(active), ref[:2])
    return volume


def igd(points, reference_points):
    """Inverted generational distance from ``reference_points`` to ``points``.

    For each reference point, take the Euclidean distance to the closest
    approximation point; return the mean.  Smaller is better; an
    approximation that contains the reference set scores 0.
    """
    approx = [list(map(float, p)) for p in points]
    refs = [list(map(float, r)) for r in reference_points]
    if not approx:
        raise ValueError("igd needs at least one approximation point")
    if not refs:
        raise ValueError("igd needs at least one reference point")
    total = 0.0
    for r in refs:
        total += min(math.dist(r, p) for p in approx)
    return total / len(refs)
