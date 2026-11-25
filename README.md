# evo-bench

A lightweight benchmark harness for multi-objective evolutionary optimization algorithms — pure Python, zero dependencies.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-unittest-orange.svg)](tests/)

**evo-bench** provides a small, hackable testbed for multi-objective evolutionary algorithms (MOEAs). It ships with the classic ZDT and DTLZ test problems, a clean-room NSGA-II implementation, and the two most widely used quality indicators — hypervolume and IGD — all in dependency-free Python. It is meant for teaching, quick prototyping, sanity-checking new algorithm ideas, and producing reproducible baseline numbers before you scale up to heavier frameworks.

## Features

- **Zero dependencies** — runs anywhere Python 3.9+ runs; no NumPy, no build step.
- **Classic test problems** — ZDT1, ZDT2, ZDT3 (bi-objective) and DTLZ2 (three-objective), each with its analytical Pareto front for reference-point sampling.
- **NSGA-II out of the box** — fast non-dominated sorting, crowding distance, binary tournament selection, SBX crossover and polynomial mutation.
- **Quality indicators** — exact hypervolume for 2 and 3 objectives, and inverted generational distance (IGD) against sampled true fronts.
- **Reproducible runs** — every run is driven by an explicit seed; results are exported as CSV + JSON for painless post-processing.
- **Simple CLI** — one command runs a full benchmark and prints a metric summary.

## Installation

From source:

```bash
git clone https://github.com/zhengas2o3/evo-bench.git
cd evo-bench
pip install .
```

Or simply run it in place — there is nothing to install:

```bash
python -m evobench --help
```

## Quickstart

Run NSGA-II on ZDT1 with a population of 100 for 100 generations:

```bash
evobench run --problem zdt1 --pop 100 --gens 100 --seed 42 --out results/zdt1
```

or, without installing:

```bash
python -m evobench run --problem zdt1 --pop 100 --gens 100 --seed 42 --out results/zdt1
```

Example output:

```text
Problem     : ZDT1 (vars=30, objs=2)
Population  : 100
Generations : 100
Seed        : 42
Front size  : 100
Hypervolume : 0.847002  (ref = [1.1, 1.1])
IGD         : 0.016872
Wrote results/zdt1/front.csv
Wrote results/zdt1/summary.json
```

Each run writes two files into `--out`:

- `front.csv` — the final non-dominated set, one objective vector per row (`f1,f2[,f3]`).
- `summary.json` — run configuration plus the computed indicators, ready for aggregation scripts.

## Supported problems

| Problem | Objectives | Variables (default) | Bounds | Pareto front shape |
|---------|-----------|---------------------|--------|--------------------|
| ZDT1    | 2         | 30                  | [0, 1] | Convex             |
| ZDT2    | 2         | 30                  | [0, 1] | Non-convex         |
| ZDT3    | 2         | 30                  | [0, 1] | Disconnected       |
| DTLZ2   | 3         | 12                  | [0, 1] | Concave (sphere)   |

## Supported algorithms

| Algorithm | Selection | Crossover | Mutation |
|-----------|-----------|-----------|----------|
| NSGA-II   | Binary tournament on (rank, crowding distance) | Simulated binary crossover (SBX, η = 20) | Polynomial mutation (η = 20) |

New algorithms only need to expose a `run(problem, pop_size, generations, seed)` function returning the final population's objective vectors — see `evobench/nsga2.py` as a template.

## Project structure

```text
evo-bench/
├── evobench/
│   ├── __init__.py      # package metadata and public API
│   ├── __main__.py      # python -m evobench entry point
│   ├── problems.py      # ZDT1, ZDT2, ZDT3, DTLZ2 + true-front sampling
│   ├── nsga2.py         # NSGA-II implementation
│   ├── metrics.py       # hypervolume and IGD
│   └── cli.py           # command-line interface
├── tests/
│   └── test_evobench.py # unittest suite
├── examples/
│   └── example_run.md   # a worked example with sample output
├── pyproject.toml
├── LICENSE
└── README.md
```

## Roadmap

- [ ] Add DTLZ1/DTLZ3 and the WFG problem suite
- [ ] Add MOEA/D and NSGA-III reference implementations
- [ ] Generational distance (GD) and spacing indicators
- [ ] Multi-run statistics (median/IQR tables, attainment surfaces)
- [ ] Optional CSV-to-LaTeX report export

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Ruizhao Zheng.
