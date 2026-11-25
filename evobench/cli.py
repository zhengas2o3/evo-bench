"""Command-line interface for evo-bench.

Usage::

    evobench run --problem zdt1 --pop 100 --gens 100 --seed 42 --out results/zdt1

Runs NSGA-II on the chosen problem, prints a metric summary, and writes
``front.csv`` and ``summary.json`` into the output directory.
"""

import argparse
import csv
import json
import os
import sys
import time

from . import metrics, nsga2
from .problems import PROBLEMS, get_problem

__all__ = ["main", "build_parser"]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="evobench",
        description="Lightweight benchmark harness for multi-objective "
                    "evolutionary optimization algorithms.",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="run NSGA-II on a benchmark problem")
    run_p.add_argument(
        "--problem",
        required=True,
        choices=sorted(PROBLEMS),
        help="benchmark problem to solve",
    )
    run_p.add_argument("--pop", type=int, default=100, help="population size")
    run_p.add_argument("--gens", type=int, default=100, help="number of generations")
    run_p.add_argument("--seed", type=int, default=42, help="random seed")
    run_p.add_argument(
        "--out",
        default="results",
        help="output directory for front.csv and summary.json",
    )
    return parser


def _run(args):
    problem = get_problem(args.problem)
    start = time.time()
    objectives = nsga2.run(
        problem, pop_size=args.pop, generations=args.gens, seed=args.seed
    )
    elapsed = time.time() - start

    front = metrics.nondominated(objectives)
    front.sort()
    hv = metrics.hypervolume(front, problem.ref_point)
    reference = problem.pareto_front(500)
    igd_value = metrics.igd(front, reference)

    os.makedirs(args.out, exist_ok=True)
    front_path = os.path.join(args.out, "front.csv")
    header = ["f%d" % (i + 1) for i in range(problem.n_obj)]
    with open(front_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for point in front:
            writer.writerow(["%.8f" % v for v in point])

    summary = {
        "problem": problem.name,
        "algorithm": "NSGA-II",
        "n_var": problem.n_var,
        "n_obj": problem.n_obj,
        "pop_size": args.pop,
        "generations": args.gens,
        "seed": args.seed,
        "front_size": len(front),
        "hypervolume": hv,
        "ref_point": problem.ref_point,
        "igd": igd_value,
        "runtime_seconds": round(elapsed, 3),
    }
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("Problem     : %s (vars=%d, objs=%d)" % (problem.name, problem.n_var, problem.n_obj))
    print("Population  : %d" % args.pop)
    print("Generations : %d" % args.gens)
    print("Seed        : %d" % args.seed)
    print("Front size  : %d" % len(front))
    print("Hypervolume : %.6f  (ref = %s)" % (hv, problem.ref_point))
    print("IGD         : %.6f" % igd_value)
    print("Runtime     : %.2f s" % elapsed)
    print("Wrote %s" % front_path)
    print("Wrote %s" % summary_path)
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
