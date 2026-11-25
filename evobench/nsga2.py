"""NSGA-II: a fast elitist multi-objective genetic algorithm.

Deb, Pratap, Agarwal & Meyarivan, "A Fast and Elitist Multiobjective
Genetic Algorithm: NSGA-II", IEEE Trans. Evol. Comput. 6(2), 2002.

The implementation uses only the Python standard library:
  * fast non-dominated sorting          -- O(M N^2)
  * crowding-distance assignment
  * binary tournament selection on (rank, crowding distance)
  * simulated binary crossover (SBX)
  * polynomial mutation

Public entry point::

    objectives = run(problem, pop_size=100, generations=100, seed=42)

which returns the objective vectors of the final population.
"""

import math
import random

__all__ = [
    "run",
    "dominates",
    "fast_non_dominated_sort",
    "crowding_distance",
    "sbx_crossover",
    "polynomial_mutation",
]


def dominates(a, b):
    """True if ``a`` Pareto-dominates ``b`` (minimization)."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def fast_non_dominated_sort(objectives):
    """Partition indices into Pareto fronts; front 0 is the best."""
    n = len(objectives)
    dominates_list = [[] for _ in range(n)]
    domination_count = [0] * n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominates_list[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dominates_list[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    nxt.append(q)
        fronts.append(nxt)
        i += 1
    fronts.pop()  # drop the trailing empty front
    return fronts


def crowding_distance(objectives, front):
    """Crowding distance of each member of ``front`` (dict index -> float)."""
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            distance[i] = math.inf
        return distance
    n_obj = len(objectives[0])
    for k in range(n_obj):
        ordered = sorted(front, key=lambda i: objectives[i][k])
        distance[ordered[0]] = math.inf
        distance[ordered[-1]] = math.inf
        f_min = objectives[ordered[0]][k]
        f_max = objectives[ordered[-1]][k]
        if f_max == f_min:
            continue
        for j in range(1, len(ordered) - 1):
            distance[ordered[j]] += (
                objectives[ordered[j + 1]][k] - objectives[ordered[j - 1]][k]
            ) / (f_max - f_min)
    return distance


def sbx_crossover(parent1, parent2, lower, upper, rng, eta_c=20.0, prob=0.9):
    """Simulated binary crossover; returns two children (decision vectors)."""
    child1, child2 = list(parent1), list(parent2)
    if rng.random() > prob:
        return child1, child2
    for i in range(len(parent1)):
        if rng.random() > 0.5:
            continue
        x1, x2 = parent1[i], parent2[i]
        if abs(x1 - x2) < 1e-14:
            continue
        lo, hi = lower[i], upper[i]
        if x1 > x2:
            x1, x2 = x2, x1
        rand = rng.random()
        # Child 1
        beta = 1.0 + 2.0 * (x1 - lo) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
        c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
        # Child 2
        beta = 1.0 + 2.0 * (hi - x2) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
        c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
        child1[i] = min(max(c1, lo), hi)
        child2[i] = min(max(c2, lo), hi)
        if rng.random() <= 0.5:
            child1[i], child2[i] = child2[i], child1[i]
    return child1, child2


def polynomial_mutation(x, lower, upper, rng, eta_m=20.0, prob=None):
    """Polynomial mutation; returns a mutated copy of ``x``."""
    n = len(x)
    if prob is None:
        prob = 1.0 / n
    y = list(x)
    for i in range(n):
        if rng.random() > prob:
            continue
        lo, hi = lower[i], upper[i]
        if hi == lo:
            continue
        delta1 = (y[i] - lo) / (hi - lo)
        delta2 = (hi - y[i]) / (hi - lo)
        rand = rng.random()
        mut_pow = 1.0 / (eta_m + 1.0)
        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta_m + 1.0)
            deltaq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta_m + 1.0)
            deltaq = 1.0 - val ** mut_pow
        y[i] = min(max(y[i] + deltaq * (hi - lo), lo), hi)
    return y


def _tournament(rng, size, rank, crowd):
    a = rng.randrange(size)
    b = rng.randrange(size)
    if rank[a] != rank[b]:
        return a if rank[a] < rank[b] else b
    if crowd[a] != crowd[b]:
        return a if crowd[a] > crowd[b] else b
    return a if rng.random() < 0.5 else b


def run(problem, pop_size=100, generations=100, seed=None):
    """Run NSGA-II on ``problem``.

    Returns the objective vectors of the final population (a list of
    ``pop_size`` lists, each of length ``problem.n_obj``).
    """
    rng = random.Random(seed)
    lower, upper = problem.lower, problem.upper
    n_var = problem.n_var

    pop = [
        [rng.uniform(lower[i], upper[i]) for i in range(n_var)]
        for _ in range(pop_size)
    ]
    objs = [problem.evaluate(x) for x in pop]

    for _ in range(generations):
        fronts = fast_non_dominated_sort(objs)
        rank, crowd = {}, {}
        for fi, front in enumerate(fronts):
            cd = crowding_distance(objs, front)
            for idx in front:
                rank[idx] = fi
                crowd[idx] = cd[idx]

        offspring = []
        while len(offspring) < pop_size:
            p1 = _tournament(rng, pop_size, rank, crowd)
            p2 = _tournament(rng, pop_size, rank, crowd)
            c1, c2 = sbx_crossover(pop[p1], pop[p2], lower, upper, rng)
            offspring.append(polynomial_mutation(c1, lower, upper, rng))
            if len(offspring) < pop_size:
                offspring.append(polynomial_mutation(c2, lower, upper, rng))
        off_objs = [problem.evaluate(x) for x in offspring]

        combined = pop + offspring
        combined_objs = objs + off_objs
        fronts = fast_non_dominated_sort(combined_objs)
        new_pop, new_objs = [], []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                for idx in front:
                    new_pop.append(combined[idx])
                    new_objs.append(combined_objs[idx])
            else:
                cd = crowding_distance(combined_objs, front)
                ordered = sorted(front, key=lambda i: cd[i], reverse=True)
                for idx in ordered[: pop_size - len(new_pop)]:
                    new_pop.append(combined[idx])
                    new_objs.append(combined_objs[idx])
                break
        pop, objs = new_pop, new_objs

    return objs
