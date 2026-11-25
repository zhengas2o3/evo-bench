"""evo-bench: a lightweight benchmark harness for multi-objective
evolutionary optimization algorithms.  Pure Python, zero dependencies.
"""

from .problems import Problem, PROBLEMS, get_problem
from . import metrics, nsga2

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Problem",
    "PROBLEMS",
    "get_problem",
    "metrics",
    "nsga2",
]
