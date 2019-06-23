import unittest as ut
import numpy as np
from pyqumo import chains


class TestCaseBase(ut.TestCase):
    PRECISION = 8

    def assertAllClose(self, lvalue, rvalue, msg=None, places=PRECISION):
        lv = np.asarray(lvalue)
        rv = np.asarray(rvalue)
        try:
            self.assertAlmostEqual(lv.item(), rv.item(), places, msg)
        except ValueError:
            tol = pow(10.0, -places)
            try:
                self.assertTrue(
                    np.allclose(lv, rv, tol, tol),
                    "{} != {} {}(tol={})".format(
                        lv.tolist(), rv.tolist(),
                        msg + " " if msg is not None else "", tol))
            except TypeError as err:
                raise TypeError("{}: {}".format(repr(err), msg))
        except Exception as err:
            raise RuntimeError("{} -- {}".format(repr(err), msg))


class DTMCTest(TestCaseBase):
    def test_correct_creation(self):
        t1 = [[1.0]]
        t2 = [[0.5, 0.5], [0.8, 0.2]]
        dtmc1 = chains.DTMC(t1)
        dtmc2 = chains.DTMC(t2)
        self.assertAllClose(dtmc1.matrix, t1)
        self.assertAllClose(dtmc2.matrix, t2)

    # noinspection PyPropertyAccess
    def test_matrix_immutable(self):
        t = [[0, 1], [1, 0]]
        dtmc = chains.DTMC(t)
        with self.assertRaises(AttributeError):
            dtmc.matrix = [[1, 0], [0, 1]]

    # noinspection PyPropertyAccess
    def test_order_property_value_and_immutable(self):
        dtmc1 = chains.DTMC([[1.0]])
        dtmc2 = chains.DTMC([[0.5, 0.5], [0.8, 0.2]])
        dtmc3 = chains.DTMC(np.eye(3))
        self.assertEqual(dtmc1.order, 1)
        self.assertEqual(dtmc2.order, 2)
        self.assertEqual(dtmc3.order, 3)
        with self.assertRaises(AttributeError):
            dtmc2.order = 1

    def test_illegal_matrices_could_not_be_used(self):
        with self.assertRaises(ValueError):
            chains.DTMC([[0.5]])
        with self.assertRaises(ValueError):
            chains.DTMC([1.0])
        with self.assertRaises(ValueError):
            chains.DTMC([[1.5]])
        with self.assertRaises(ValueError):
            chains.DTMC([[0, 1], [0.7, 0.7]])
        with self.assertRaises(ValueError):
            chains.DTMC([[-0.1, 1.1], [0.5, 0.5]])
        with self.assertRaises(ValueError):
            chains.DTMC([[0.5, 0.5], [1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            chains.DTMC([[0.3, 0.3, 0.4], [0.5, 0.5, 0]])

    def test_illegal_matrices_could_be_used_if_check_is_false(self):
        t = [[1, 0.5], [-0.1, 1.1]]
        dtmc = chains.DTMC(t, check=False)
        self.assertAllClose(dtmc.matrix, t)

    def test_illegal_matrices_could_be_used_with_tolerance(self):
        t1 = [[1.005]]
        t2 = [[-0.05, 1.03], [0.43, 0.53]]

        dtmc1 = chains.DTMC(t1, tol=0.01)
        dtmc2 = chains.DTMC(t2, tol=0.05)

        self.assertAllClose(dtmc1.matrix, t1, places=2)
        self.assertAllClose(dtmc2.matrix, t2, places=1)
        with self.assertRaises(ValueError):
            chains.DTMC(t1, tol=0.001)
        with self.assertRaises(ValueError):
            chains.DTMC(t2, tol=0.01)

    def test_steady_pmf(self):
        t1 = [[1.0]]
        t2 = [[0, 1], [1, 0]]
        t9_wiki = [[.00, .50, .00, .50, .00, .00, .00, .00, .00],
                   [.25, .00, .25, .00, .25, .00, .00, .25, .00],
                   [.00, .50, .00, .00, .00, .50, .00, .00, .00],
                   [1/3, .00, .00, .00, 1/3, .00, 1/3, .00, .00],
                   [.00, .25, .00, .25, .00, .25, .00, .25, .00],
                   [.00, .00, 1/3, .00, 1/3, .00, .00, .00, 1/3],
                   [.00, .00, .00, .50, .00, .00, .00, .50, .00],
                   [.00, .25, .00, .00, .25, .00, .25, .00, .25],
                   [.00, .00, .00, .00, .00, .50, .00, .50, .00]]
        ctmc1 = chains.DTMC(t1)
        ctmc2 = chains.DTMC(t2)
        ctmc9 = chains.DTMC(t9_wiki)
        pmf1 = [1.0]
        pmf2 = [0.5, 0.5]
        pmf9 = [.077, .154, .077, .115, 0.154, 0.115, 0.077, 0.154, 0.077]

        self.assertAllClose(ctmc1.steady_pmf(), pmf1)
        self.assertAllClose(ctmc2.steady_pmf(), pmf2)
        self.assertAllClose(ctmc9.steady_pmf(), pmf9, places=3)
        # Test that the second call returns the same thing (if cache is used)
        self.assertAllClose(ctmc9.steady_pmf(), pmf9, places=3)

    def test_trace_correctly_interprets_pmf0_and_state0_inputs(self):
        t = [[0, 1], [1, 0]]
        dtmc = chains.DTMC(t)

        # 1) check that if state0 is specified, then the trace will always
        #    start with it, no matter whether pmf0 is provided or not
        for i in range(100):
            step = next(dtmc.trace(1, state0=0, pmf0=[0, 1]))
            self.assertEqual(step[0], 0)
        for i in range(100):
            step = next(dtmc.trace(1, state0=1))
            self.assertEqual(step[0], 1)

        n_iterations = 1000
        hits = [0] * dtmc.order

        # 2) check that if state0 is NOT specified, then pmf0 is used
        pmf0 = [0.3, 0.7]
        for i in range(n_iterations):
            step = next(dtmc.trace(1, pmf0=pmf0))
            hits[step[0]] += 1
        probs = np.asarray(hits) / n_iterations
        self.assertAllClose(probs, pmf0, places=1)

        # 3) check that if neither step0, nor pmf0 specified, uniform
        #    distribution is used
        hits = [0] * dtmc.order
        for i in range(n_iterations):
            step = next(dtmc.trace(1))
            hits[step[0]] += 1
        probs = np.asarray(hits) / n_iterations
        self.assertAllClose(probs, [1. / dtmc.order] * dtmc.order, places=1)

    def test_trace_visits_states_in_time_average_as_steady_state_pmf(self):
        n_steps = 10000

        def estimate_rates(dtmc):
            hits_prev = [0.0] * dtmc.order
            hits_next = [0.0] * dtmc.order
            for step in dtmc.trace(n_steps):
                hits_prev[step[0]] += 1
                hits_next[step[1]] += 1
            return (np.asarray(hits_prev) / n_steps,
                    np.asarray(hits_next) / n_steps)

        dtmc1 = chains.DTMC([[0, 1.], [1., 0]])
        dtmc2 = chains.DTMC([[.00, .50, .00, .50, .00, .00, .00, .00, .00],
                             [.25, .00, .25, .00, .25, .00, .00, .25, .00],
                             [.00, .50, .00, .00, .00, .50, .00, .00, .00],
                             [1/3, .00, .00, .00, 1/3, .00, 1/3, .00, .00],
                             [.00, .25, .00, .25, .00, .25, .00, .25, .00],
                             [.00, .00, 1/3, .00, 1/3, .00, .00, .00, 1/3],
                             [.00, .00, .00, .50, .00, .00, .00, .50, .00],
                             [.00, .25, .00, .00, .25, .00, .25, .00, .25],
                             [.00, .00, .00, .00, .00, .50, .00, .50, .00]])

        pmf1 = dtmc1.steady_pmf()
        pmf2 = dtmc2.steady_pmf()
        rates1_prev, rates1_next = estimate_rates(dtmc1)
        rates2_prev, rates2_next = estimate_rates(dtmc2)

        self.assertAllClose(pmf1, rates1_prev, places=1)
        self.assertAllClose(pmf1, rates1_next, places=1)
        self.assertAllClose(pmf2, rates2_prev, places=1)
        self.assertAllClose(pmf2, rates2_next, places=1)

    def test_path_with_destination_set_ends_at_that_destination(self):
        dtmc = chains.DTMC([[.00, .50, .00, .50, .00, .00, .00, .00, .00],
                            [.25, .00, .25, .00, .25, .00, .00, .25, .00],
                            [.00, .50, .00, .00, .00, .50, .00, .00, .00],
                            [1/3, .00, .00, .00, 1/3, .00, 1/3, .00, .00],
                            [.00, .25, .00, .25, .00, .25, .00, .25, .00],
                            [.00, .00, 1/3, .00, 1/3, .00, .00, .00, 1/3],
                            [.00, .00, .00, .50, .00, .00, .00, .50, .00],
                            [.00, .25, .00, .00, .25, .00, .25, .00, .25],
                            [.00, .00, .00, .00, .00, .50, .00, .50, .00]])
        n_attempts = 100
        max_path_length = 100
        for i in range(n_attempts):
            target = np.random.choice(dtmc.order)
            path = list(dtmc.generate_path(max_path_length, ends={target}))
            self.assertGreater(len(path), 0)
            self.assertTrue(target not in path[1:-1])
            self.assertTrue(len(path) == max_path_length or
                            (len(path) < max_path_length and
                             path[-1] == target))

    def test_paths_for_single_state_chain(self):
        dtmc = chains.DTMC([[1.0]])
        paths_no_ends = list(dtmc.paths(0))
        paths_to_0 = list(dtmc.paths(0, ends={0}))
        paths_to_1 = list(dtmc.paths(0, ends={1}))
        self.assertEqual(len(paths_no_ends), 1)
        self.assertEqual(len(paths_to_0), 1)
        self.assertEqual(paths_no_ends[0], [0])
        self.assertEqual(paths_to_0[0], [0])
        self.assertEqual(len(paths_to_1), 0)

    def _test_paths(self, chain, source, ends, expected_paths, ptol=1e-8,
                    debug=False):
        paths = list(chain.paths(source, ends=ends, ptol=ptol, debug=debug))
        self.assertEqual(len(paths), len(expected_paths))
        for path in expected_paths:
            self.assertIn(path, paths)

    def test_paths_for_line_chain(self):
        dtmc = chains.DTMC([[0, 1, 0, 0], [0, 0, 1, 0],
                            [0, 0, 0, 1], [0, 0, 0, 1]])
        self._test_paths(dtmc, 0, None, [[0, 1, 2, 3]])
        self._test_paths(dtmc, 0, {3}, [[0, 1, 2, 3]])
        self._test_paths(dtmc, 0, {2}, [[0, 1, 2]])
        self._test_paths(dtmc, 0, {1}, [[0, 1]])
        self._test_paths(dtmc, 0, {0}, [[0]])
        self._test_paths(dtmc, 0, {1, 2}, [[0, 1, 2], [0, 1]])
        self._test_paths(dtmc, 0, {0, 1, 2}, [[0], [0, 1], [0, 1, 2]])
        self._test_paths(dtmc, 0, {0, 1, 2, 3}, [[0], [0, 1], [0, 1, 2],
                                                 [0, 1, 2, 3]])

    def test_paths_for_circle_chain(self):
        dtmc = chains.DTMC([[0, 1, 0, 0], [0, 0, 1, 0],
                            [0, 0, 0, 1], [1, 0, 0, 0]])
        self._test_paths(dtmc, 0, None, [])
        self._test_paths(dtmc, 0, {3}, [[0, 1, 2, 3]])
        self._test_paths(dtmc, 0, {2}, [[0, 1, 2]])
        self._test_paths(dtmc, 0, {1}, [[0, 1]])
        self._test_paths(dtmc, 0, {0}, [[0]])
        self._test_paths(dtmc, 0, {1, 2}, [[0, 1, 2], [0, 1]])
        self._test_paths(dtmc, 0, {0, 1, 2}, [[0], [0, 1], [0, 1, 2]])
        self._test_paths(dtmc, 0, {0, 1, 2, 3}, [[0], [0, 1], [0, 1, 2],
                                                 [0, 1, 2, 3]])
        self._test_paths(dtmc, 2, {0, 1}, [[2, 3, 0, 1], [2, 3, 0]])

    def test_paths_for_tree(self):
        dtmc = chains.DTMC([[0., .5, .5, 0., 0., 0.],
                            [0., 0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., .5, .5],
                            [0., 0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0., 1.]])
        self._test_paths(dtmc, 0, {5}, [[0, 2, 5]])
        self._test_paths(dtmc, 0, None, [[0, 1, 3], [0, 2, 4], [0, 2, 5]])

    def test_paths_chain_with_cycle_with_ptol(self):
        dtmc = chains.DTMC([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.9, 0.0, 0.1, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.8, 0.0, 0.0, 0.0, 0.0, 0.2],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        self._test_paths(dtmc, 0, None, [[0, 1, 4], [0, 1, 2, 3, 5]])
        self._test_paths(dtmc, 0, None, [[0, 1, 2, 3, 5]], ptol=0.15)
        self._test_paths(dtmc, 0, None, [], ptol=0.25)


class CTMCTest(TestCaseBase):
    def test_correct_creation(self):
        t1 = [[0.0]]
        t2 = [[-1.0, 1.0], [1.0, -1.0]]
        ctmc1 = chains.CTMC(t1)
        ctmc2 = chains.CTMC(t2)
        self.assertAllClose(ctmc1.matrix, t1)
        self.assertAllClose(ctmc2.matrix, t2)

    # noinspection PyPropertyAccess
    def test_matrix_immutable(self):
        t = [[-1.0, 1.0], [1.0, -1.0]]
        ctmc = chains.CTMC(t)
        with self.assertRaises(AttributeError):
            ctmc.matrix = [[-2.0, 2.0], [3.0, -3.0]]

    def test_illegal_matrices_could_not_be_used(self):
        with self.assertRaises(ValueError):
            chains.CTMC([[1.0]])
        with self.assertRaises(ValueError):
            chains.CTMC([[-1.0]])
        with self.assertRaises(ValueError):
            chains.CTMC([[1.0, 0.0], [1.0, -1.0]])
        with self.assertRaises(ValueError):
            chains.CTMC([[0, 0], [1, -0.5]])
        with self.assertRaises(ValueError):
            chains.CTMC([[-1, 0.5], [1, -1]])
        with self.assertRaises(ValueError):
            chains.CTMC([[-0.5, 1, -0.5], [1, -1, 0], [0, 1, -1]])
        with self.assertRaises(ValueError):
            chains.CTMC([[-1.0, 0.5, 0.5], [0, -1, 1]])
        with self.assertRaises(ValueError):
            chains.CTMC([[-1, 1], [2, -2], [0, 0]])
        with self.assertRaises(ValueError):
            chains.CTMC([0.0])

    def test_illegal_matrices_could_be_used_when_check_is_false(self):
        ctmc = chains.CTMC([[1.0]], check=False)
        self.assertAllClose(ctmc.matrix, [[1.]])

    def test_order(self):
        t1 = [[0.0]]
        t2 = [[-1.0, 1.0], [1.0, -1.0]]
        ctmc1 = chains.CTMC(t1)
        ctmc2 = chains.CTMC(t2)
        self.assertEqual(ctmc1.order, 1)
        self.assertEqual(ctmc2.order, 2)

    def test_illegal_matrices_could_be_used_with_tolerance(self):
        t1 = [[0.005]]
        t2 = [[-1.0, 0.95], [1.01, -0.98]]
        ctmc1 = chains.CTMC(t1, tol=0.01)
        ctmc2 = chains.CTMC(t2, tol=0.1)

        self.assertTrue(np.allclose(ctmc1.matrix, t1))
        self.assertTrue(np.allclose(ctmc2.matrix, t2))

        with self.assertRaises(ValueError):
            chains.CTMC(t1, tol=0.001)
        with self.assertRaises(ValueError):
            chains.CTMC(t2, tol=0.01)

    def test_steady_distribution(self):
        t1 = [[0.]]
        t2 = [[-1., 1.], [1., -1.]]
        t3 = [[-.025, .02, .005], [.3, -.5, .2], [.02, .4, -.42]]
        ctmc1 = chains.CTMC(t1)
        ctmc2 = chains.CTMC(t2)
        ctmc3 = chains.CTMC(t3)
        self.assertAllClose(ctmc1.steady_pmf(), [1.], "ctmc1.steady_pmf()")
        self.assertAllClose(ctmc2.steady_pmf(), [0.5, 0.5])
        self.assertAllClose(ctmc3.steady_pmf(), [.885, .071, .044], places=3)
        # Test that the second call returns the same thing (if cache is used)
        self.assertAllClose(ctmc3.steady_pmf(), [.885, .071, .044], places=3)

    def test_trace_correctly_interprets_pmf0_and_state0_inputs(self):
        t = [[-1, 1], [2, -2]]
        ctmc = chains.CTMC(t)

        # 1) check that if state0 is specified, then the trace will always
        #    start with it, no matter whether pmf0 is provided or not
        for i in range(100):
            step = next(ctmc.trace(1, state0=0, pmf0=[0, 1]))
            self.assertEqual(step[0], 0)
        for i in range(100):
            step = next(ctmc.trace(1, state0=1))
            self.assertEqual(step[0], 1)

        n_iterations = 1000
        hits = [0] * ctmc.order

        # 2) check that if state0 is NOT specified, then pmf0 is used
        pmf0 = [0.3, 0.7]
        for i in range(n_iterations):
            step = next(ctmc.trace(1, pmf0=pmf0))
            hits[step[0]] += 1
        probs = np.asarray(hits) / n_iterations
        self.assertAllClose(probs, [0.3, 0.7], places=1)

        # 3) check that if neither step0, nor pmf0 specified, uniform
        #    distribution is used
        hits = [0] * ctmc.order
        for i in range(n_iterations):
            step = next(ctmc.trace(1))
            hits[step[0]] += 1
        probs = np.asarray(hits) / n_iterations
        self.assertAllClose(probs, [1. / ctmc.order] * ctmc.order, places=1)

    def test_trace_visits_states_in_time_average_as_steady_state_pmf(self):
        n_steps = 10000

        def estimate_rates(ctmc):
            hits = [0.0] * ctmc.order
            t = 0.0
            for step in ctmc.trace(n_steps):
                dt = step[-1]
                hits[step[0]] += dt
                t += dt
            return np.asarray(hits) / t

        ctmc1 = chains.CTMC([[-1., 1.], [1., -1.]])
        ctmc2 = chains.CTMC([[-.025, .02, .005],
                             [.3, -.5, .2],
                             [.02, .4, -.42]])
        pmf1 = ctmc1.steady_pmf()
        pmf2 = ctmc2.steady_pmf()
        rates1 = estimate_rates(ctmc1)
        rates2 = estimate_rates(ctmc2)

        self.assertAllClose(pmf1, rates1, places=1)
        self.assertAllClose(pmf2, rates2, places=1)

    def test_embedded_dtmc(self):
        # def arrival_intensity(generator, pmf):
        #     # D1 is a generator without diagonal elements (each transition
        #     # is treated as generating)
        #     generator = np.asarray(generator)
        #     pmf = np.asarray(pmf)
        #     d1 = generator - np.diag(generator.diagonal())
        #     ones = np.ones((order_of(generator), 1))
        #     return pmf.dot(d1).dot(ones)
        t1 = [[-2., 2.], [2., -2.]]
        t2 = [[-2, 1., 1.], [2, -2., 0.], [0, 0, 0]]
        ctmc1 = chains.CTMC(t1)
        ctmc2 = chains.CTMC(t2)

        dtmc1 = ctmc1.embedded_dtmc()
        dtmc2 = ctmc2.embedded_dtmc()

        self.assertAllClose(dtmc1.matrix, [[0., 1.], [1., 0.]])
        self.assertAllClose(dtmc2.matrix, [[0, .5, .5], [1, 0, 0], [0, 0, 1]])
