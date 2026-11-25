# Example run

A small, fast run you can reproduce in well under a second:

```bash
python -m evobench run --problem zdt1 --pop 40 --gens 25 --seed 1 --out /tmp/evobench-demo
```

Actual console output of that command:

```text
Problem     : ZDT1 (vars=30, objs=2)
Population  : 40
Generations : 25
Seed        : 1
Front size  : 30
Hypervolume : 0.105527  (ref = [1.1, 1.1])
IGD         : 0.648457
Runtime     : 0.27 s
Wrote /tmp/evobench-demo/front.csv
Wrote /tmp/evobench-demo/summary.json
```

25 generations is only a smoke test — the population is still on its way
to the front. A more typical budget converges much further:

```bash
python -m evobench run --problem zdt1 --pop 100 --gens 100 --seed 42 --out results/zdt1
```

```text
Problem     : ZDT1 (vars=30, objs=2)
Population  : 100
Generations : 100
Seed        : 42
Front size  : 100
Hypervolume : 0.847002  (ref = [1.1, 1.1])
IGD         : 0.016872
```

(For a fixed seed the indicators are fully reproducible; only the
reported runtime varies by machine.)

## Output files

`front.csv` — the final non-dominated set, one row per solution:

```csv
f1,f2
0.00007103,2.73774812
0.00011331,2.37764890
0.00063096,2.37485538
...
```

`summary.json` — configuration and indicators:

```json
{
  "problem": "ZDT1",
  "algorithm": "NSGA-II",
  "n_var": 30,
  "n_obj": 2,
  "pop_size": 40,
  "generations": 25,
  "seed": 1,
  "front_size": 30,
  "hypervolume": 0.10552665370972683,
  "ref_point": [1.1, 1.1],
  "igd": 0.6484567457756453,
  "runtime_seconds": 0.273
}
```

## Try the other problems

```bash
python -m evobench run --problem zdt2  --pop 100 --gens 100 --seed 42 --out results/zdt2
python -m evobench run --problem zdt3  --pop 100 --gens 100 --seed 42 --out results/zdt3
python -m evobench run --problem dtlz2 --pop 100 --gens 100 --seed 42 --out results/dtlz2
```
