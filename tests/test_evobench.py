"""Unit tests for evo-bench (standard-library unittest only)."""

import math
import unittest

from evobench import get_problem, metrics, nsga2


class TestProblems(unittest.TestCase):
    def test_zdt1_known_point(self):
        # x = (0.5, 0, ..., 0) -> g = 1, f = (0.5, 1 - sqrt(0.5))
        problem = get_problem("zdt1")
        x = [0.5] + [0.0] * (problem.n_var - 1)
        f = problem.evaluate(x)
        self.assertAlmostEqual(f[0], 0.5)
        self.assertAlmostEqual(f[1], 1.0 - math.sqrt(0.5))

    def test_zdt2_known_point(self):
        # x = (0.5, 0, ..., 0) -> g = 1, f = (0.5, 1 - 0.25)
        problem = get_problem("zdt2")
        x = [0.5] + [0.0] * (problem.n_var - 1)
        f = problem.evaluate(x)
        self.assertAlmostEqual(f[0], 0.5)
        self.assertAlmostEqual(f[1], 0.75)

    def test_zdt3_origin(self):
        # x = 0 vector -> f = (0, 1)
        problem = get_problem("zdt3")
        f = problem.evaluate([0.0] * problem.n_var)
        self.assertAlmostEqual(f[0], 0.0)
        self.assertAlmostEqual(f[1], 1.0)

    def test_dtlz2_on_sphere_when_g_zero(self):
        # With x_i = 0.5 for the position variables beyond the first M-1,
        # g = 0 and the objective vector lies on the unit sphere.
        problem = get_problem("dtlz2")
        x = [0.5] * problem.n_var
        f = problem.evaluate(x)
        self.assertEqual(len(f), 3)
        self.assertAlmostEqual(sum(v * v for v in f), 1.0)
        # f1 = f2 = cos(pi/4)^2 = 0.5, f3 = sin(pi/4)
        self.assertAlmostEqual(f[0], 0.5)
        self.assertAlmostEqual(f[1], 0.5)
        self.assertAlmostEqual(f[2], math.sin(math.pi / 4.0))

    def test_problem_dimensions_and_bounds(self):
        for name, n_obj in [("zdt1", 2), ("zdt2", 2), ("zdt3", 2), ("dtlz2", 3)]:
            problem = get_problem(name)
            self.assertEqual(problem.n_obj, n_obj)
            self.assertEqual(len(problem.lower), problem.n_var)
            self.assertEqual(len(problem.upper), problem.n_var)
            mid = [(lo + hi) / 2.0 for lo, hi in zip(problem.lower, problem.upper)]
            self.assertEqual(len(problem.evaluate(mid)), n_obj)
            self.assertGreater(len(problem.pareto_front(50)), 0)


class TestNonDominatedSort(unittest.TestCase):
    def test_sort_fronts(self):
        objectives = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0], [2.0, 0.5]]
        fronts = nsga2.fast_non_dominated_sort(objectives)
        self.assertEqual(sorted(fronts[0]), [0, 1, 2])
        self.assertEqual(sorted(fronts[1]), [3, 4])
        self.assertEqual(len(fronts), 2)

    def test_dominates(self):
        self.assertTrue(nsga2.dominates([0.1, 0.2], [0.2, 0.3]))
        self.assertTrue(nsga2.dominates([0.1, 0.3], [0.2, 0.3]))
        self.assertFalse(nsga2.dominates([0.1, 0.4], [0.2, 0.3]))
        self.assertFalse(nsga2.dominates([0.2, 0.3], [0.2, 0.3]))

    def test_crowding_distance_boundaries_infinite(self):
        objectives = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
        dist = nsga2.crowding_distance(objectives, [0, 1, 2])
        self.assertEqual(dist[0], math.inf)
        self.assertEqual(dist[2], math.inf)
        self.assertTrue(math.isfinite(dist[1]))
        self.assertGreater(dist[1], 0.0)


class TestMetrics(unittest.TestCase):
    def test_hypervolume_single_point(self):
        # One point (0.5, 0.5), ref (1, 1) -> 0.5 * 0.5 = 0.25
        self.assertAlmostEqual(metrics.hypervolume([[0.5, 0.5]], [1.0, 1.0]), 0.25)

    def test_hypervolume_two_points(self):
        # Rectangles [0.25,1]x[0.75,1] and [0.75,1]x[0.25,1]:
        # 0.1875 + 0.1875 - overlap 0.0625 = 0.3125
        hv = metrics.hypervolume([[0.25, 0.75], [0.75, 0.25]], [1.0, 1.0])
        self.assertAlmostEqual(hv, 0.3125)

    def test_hypervolume_3d_single_point(self):
        # (0.5, 0.5, 0.5) vs ref (1, 1, 1) -> 0.125
        hv = metrics.hypervolume([[0.5, 0.5, 0.5]], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(hv, 0.125)

    def test_igd_zero_for_identical_sets(self):
        refs = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
        self.assertAlmostEqual(metrics.igd(refs, refs), 0.0)

    def test_igd_known_value(self):
        # Reference {(0,1), (1,0)}, approximation {(0,1)}:
        # distances 0 and sqrt(2) -> mean sqrt(2)/2
        value = metrics.igd([[0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]])
        self.assertAlmostEqual(value, math.sqrt(2.0) / 2.0)


class TestNSGA2Run(unittest.TestCase):
    def test_small_run_produces_nondominated_solutions(self):
        problem = get_problem("zdt1")
        objectives = nsga2.run(problem, pop_size=20, generations=10, seed=7)
        self.assertEqual(len(objectives), 20)
        for f in objectives:
            self.assertEqual(len(f), 2)
            self.assertTrue(all(math.isfinite(v) for v in f))
            self.assertGreaterEqual(f[0], 0.0)
            self.assertLessEqual(f[0], 1.0)
        # The final population must contain non-dominated points, and the
        # first front from the sort must match the independent filter.
        front = metrics.nondominated(objectives)
        self.assertGreater(len(front), 0)
        fronts = nsga2.fast_non_dominated_sort(objectives)
        first_front_points = sorted(tuple(objectives[i]) for i in fronts[0])
        self.assertEqual(first_front_points, sorted(tuple(p) for p in front))

    def test_run_is_reproducible_with_seed(self):
        problem = get_problem("zdt2")
        first = nsga2.run(problem, pop_size=12, generations=5, seed=123)
        second = nsga2.run(problem, pop_size=12, generations=5, seed=123)
        self.assertEqual(first, second)

    def test_run_on_dtlz2(self):
        problem = get_problem("dtlz2")
        objectives = nsga2.run(problem, pop_size=16, generations=5, seed=3)
        self.assertEqual(len(objectives), 16)
        for f in objectives:
            self.assertEqual(len(f), 3)


if __name__ == "__main__":
    unittest.main()
