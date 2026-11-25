"""Benchmark test problems for multi-objective optimization.

Implements the ZDT family (ZDT1, ZDT2, ZDT3) and DTLZ2.  Every problem
exposes the same small interface:

    problem.n_var        number of decision variables
    problem.n_obj        number of objectives
    problem.lower        lower bounds (list, one per variable)
    problem.upper        upper bounds (list, one per variable)
    problem.evaluate(x)  objective vector (list) for decision vector ``x``
    problem.pareto_front(n)  ``n`` points sampled from the true Pareto front
    problem.ref_point    reference point used for hypervolume

All objectives are minimized.
"""

import math
import random

__all__ = ["Problem", "PROBLEMS", "get_problem", "zdt1", "zdt2", "zdt3", "dtlz2"]


class Problem:
    """A multi-objective test problem."""

    def __init__(self, name, n_var, n_obj, lower, upper, func, front_func, ref_point):
        self.name = name
        self.n_var = n_var
        self.n_obj = n_obj
        self.lower = list(lower)
        self.upper = list(upper)
        self._func = func
        self._front_func = front_func
        self.ref_point = list(ref_point)

    def evaluate(self, x):
        """Return the objective vector for decision vector ``x``."""
        if len(x) != self.n_var:
            raise ValueError(
                "%s expects %d variables, got %d" % (self.name, self.n_var, len(x))
            )
        return self._func(list(x))

    def pareto_front(self, n=200):
        """Return ``n`` points sampled from the analytical Pareto front."""
        return self._front_func(n)

    def __repr__(self):  # pragma: no cover - cosmetic
        return "Problem(%s, n_var=%d, n_obj=%d)" % (self.name, self.n_var, self.n_obj)


# ---------------------------------------------------------------------------
# ZDT family: 2 objectives, n variables in [0, 1], f1 = x1, g depends on rest.
# ---------------------------------------------------------------------------

def _zdt_g(x):
    return 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)


def _zdt1_func(x):
    f1 = x[0]
    g = _zdt_g(x)
    return [f1, g * (1.0 - math.sqrt(f1 / g))]


def _zdt2_func(x):
    f1 = x[0]
    g = _zdt_g(x)
    return [f1, g * (1.0 - (f1 / g) ** 2)]


def _zdt3_func(x):
    f1 = x[0]
    g = _zdt_g(x)
    return [f1, g * (1.0 - math.sqrt(f1 / g) - (f1 / g) * math.sin(10.0 * math.pi * f1))]


def _front_zdt1(n):
    return [[t, 1.0 - math.sqrt(t)] for t in _linspace(0.0, 1.0, n)]


def _front_zdt2(n):
    return [[t, 1.0 - t * t] for t in _linspace(0.0, 1.0, n)]


# Disconnected segments of the ZDT3 front (values of f1).
_ZDT3_SEGMENTS = [
    (0.0, 0.0830015349),
    (0.182228780, 0.257762363),
    (0.409313675, 0.453882104),
    (0.618396794, 0.652511703),
    (0.823331798, 0.851832865),
]


def _front_zdt3(n):
    points = []
    per = max(1, n // len(_ZDT3_SEGMENTS))
    for lo, hi in _ZDT3_SEGMENTS:
        for t in _linspace(lo, hi, per):
            points.append([t, 1.0 - math.sqrt(t) - t * math.sin(10.0 * math.pi * t)])
    return points


# ---------------------------------------------------------------------------
# DTLZ2: 3 objectives, n = (M - 1) + k variables in [0, 1] with k = 10.
# The Pareto front is the positive octant of the unit sphere.
# ---------------------------------------------------------------------------

def _dtlz2_func(x, n_obj=3):
    g = sum((xi - 0.5) ** 2 for xi in x[n_obj - 1:])
    objectives = []
    for i in range(n_obj):
        value = 1.0 + g
        for j in range(n_obj - 1 - i):
            value *= math.cos(x[j] * math.pi / 2.0)
        if i > 0:
            value *= math.sin(x[n_obj - 1 - i] * math.pi / 2.0)
        objectives.append(value)
    return objectives


def _front_dtlz2(n):
    rng = random.Random(12345)
    points = []
    while len(points) < n:
        v = [abs(rng.gauss(0.0, 1.0)) for _ in range(3)]
        norm = math.sqrt(sum(c * c for c in v))
        if norm == 0.0:
            continue
        points.append([c / norm for c in v])
    return points


def _linspace(a, b, n):
    if n == 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


# ---------------------------------------------------------------------------
# Problem factories and registry.
# ---------------------------------------------------------------------------

def zdt1(n_var=30):
    """ZDT1: 2 objectives, ``n_var`` variables in [0, 1], convex front."""
    return Problem("ZDT1", n_var, 2, [0.0] * n_var, [1.0] * n_var,
                   _zdt1_func, _front_zdt1, [1.1, 1.1])


def zdt2(n_var=30):
    """ZDT2: 2 objectives, ``n_var`` variables in [0, 1], non-convex front."""
    return Problem("ZDT2", n_var, 2, [0.0] * n_var, [1.0] * n_var,
                   _zdt2_func, _front_zdt2, [1.1, 1.1])


def zdt3(n_var=30):
    """ZDT3: 2 objectives, ``n_var`` variables in [0, 1], disconnected front."""
    return Problem("ZDT3", n_var, 2, [0.0] * n_var, [1.0] * n_var,
                   _zdt3_func, _front_zdt3, [1.1, 1.1])


def dtlz2(n_var=12):
    """DTLZ2: 3 objectives, ``n_var`` variables in [0, 1], spherical front."""
    return Problem("DTLZ2", n_var, 3, [0.0] * n_var, [1.0] * n_var,
                   _dtlz2_func, _front_dtlz2, [1.1, 1.1, 1.1])


PROBLEMS = {
    "zdt1": zdt1,
    "zdt2": zdt2,
    "zdt3": zdt3,
    "dtlz2": dtlz2,
}


def get_problem(name):
    """Look up a problem by (case-insensitive) name, e.g. ``"zdt1"``."""
    try:
        return PROBLEMS[name.lower()]()
    except KeyError:
        raise ValueError(
            "unknown problem %r; available: %s" % (name, ", ".join(sorted(PROBLEMS)))
        )
