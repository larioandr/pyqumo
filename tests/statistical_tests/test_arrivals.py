import unittest as ut
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

from pyqumo import arrivals as ar
from pyqumo.distributions import Erlang
from pyqumo import stats


class TestPoisson(ut.TestCase):

    def test_valid_creation(self):
        p1 = ar.PoissonProcess(1.0)
        p2 = ar.PoissonProcess(2.0)
        self.assertAlmostEqual(p1.rate, 1.0)
        self.assertAlmostEqual(p2.rate, 2.0)

    def test_invalid_creation(self):
        with self.assertRaises(ValueError):
            ar.PoissonProcess(0.0)
        with self.assertRaises(ValueError):
            ar.PoissonProcess(-1.0)
        with self.assertRaises(TypeError):
            ar.PoissonProcess('1.0')
        with self.assertRaises(TypeError):
            ar.PoissonProcess([1, 2])

    def test_moments(self):
        p = ar.PoissonProcess(2.0)
        self.assertAlmostEqual(p.mean(), 0.5)
        self.assertAlmostEqual(p.std(), 0.5)
        self.assertAlmostEqual(p.var(), 0.25)
        self.assertAlmostEqual(p.moment(1), 0.5)
        self.assertAlmostEqual(p.moment(2), 0.5)
        self.assertAlmostEqual(p.moment(3), 0.75)
        self.assertAlmostEqual(p.cv(), 1.0)

    def test_lag_is_zero(self):
        p = ar.PoissonProcess(2.0)
        self.assertAlmostEqual(p.lag(1), 0.0)
        self.assertAlmostEqual(p.lag(2), 0.0)

    # noinspection PyTypeChecker
    def test_generate(self):
        p = ar.PoissonProcess(3.0)
        samples = list(p.generate(20000))
        self.assertAlmostEqual(np.mean(samples), p.mean(), 1)
        self.assertAlmostEqual(np.std(samples), p.std(), 1)


class TestMAP(ut.TestCase):

    def test_valid_creation(self):
        d0 = [[-1, 0.5], [0.5, -1]]
        d1 = [[0, 0.5], [0.2, 0.3]]
        m1 = ar.MAP(d0, d1)
        assert_allclose(m1.generator, [[-1, 1], [0.7, -0.7]])

    def test_invalid_creation(self):
        # Off-diagonal D0 elements must be positive
        with self.assertRaises(ValueError):
            ar.MAP(
                d0=[[-1.0, -0.1], [0, -1]],
                d1=[[0, 1.1], [1., 0.]]
            )
        # D1 elements must be positive
        with self.assertRaises(ValueError):
            ar.MAP(
                d0=[[-1., 1.], [1.5, -1.]],
                d1=[[0., 0.], [-0.5, 0.0]]
            )
        # D0 + D1 must give infinitesimal matrix
        with self.assertRaises(ValueError):
            ar.MAP(
                d0=[[-1., 1.], [0.5, -1.0]],
                d1=[[0.01, 0.0], [0, 0.5]]
            )
        with self.assertRaises(ValueError):
            ar.MAP(
                d0=[[-1., 1.], [0.5, -1.0]],
                d1=[[0.0, 0.0], [0, 0.49]]
            )

    def test_erlang_constructor(self):
        m1 = ar.MAP.erlang(1, 1.0)
        m2 = ar.MAP.erlang(2, 5.0)
        m3 = ar.MAP.erlang(3, 10.0)

        assert_allclose(m1.D0, [[-1.0]])
        assert_allclose(m1.D1, [[1.0]])
        assert_allclose(m2.D0, [[-5, 5], [0, -5]])
        assert_allclose(m2.D1, [[0, 0], [5, 0]])
        assert_allclose(m3.D0, [[-10, 10, 0], [0, -10, 10], [0, 0, -10]])
        assert_allclose(m3.D1, [[0, 0, 0], [0, 0, 0], [10, 0, 0]])

    def test_exponential_constructor(self):
        m1 = ar.MAP.exponential(1.0)
        m2 = ar.MAP.exponential(2.0)

        assert_almost_equal(m1.D0, [[-1.0]])
        assert_almost_equal(m1.D1, [[1.0]])
        assert_almost_equal(m2.D0, [[-2.0]])
        assert_almost_equal(m2.D1, [[2.0]])

        with self.assertRaises(ValueError):
            ar.MAP.exponential(-1)

        with self.assertRaises(ValueError):
            ar.MAP.exponential(0.0)

    def test_moments_like_erlang(self):
        e1 = Erlang(1, 1.0)
        e2 = Erlang(2, 5.0)
        e3 = Erlang(3, 10.0)
        m1 = ar.MAP.erlang(e1.shape, e1.rate)
        m2 = ar.MAP.erlang(e2.shape, e2.rate)
        m3 = ar.MAP.erlang(e3.shape, e3.rate)

        for k in range(10):
            self.assertAlmostEqual(m1.moment(k), e1.moment(k))
            self.assertAlmostEqual(m2.moment(k), e2.moment(k))
            self.assertAlmostEqual(m3.moment(k), e3.moment(k))

    # noinspection PyTypeChecker
    def test_generate(self):
        D0 = [
            [-9.0,  0.0,  0.0,   0.0],
            [0.0,  -9.0,  9.0,   0.0],
            [0.0,   0.0, -0.1,   0.0],
            [0.1,   0.0,  0.0,  -0.1],
        ]
        D1 = [
            [8.0, 1.0, 0.00, 0.00],
            [0.0, 0.0, 0.00, 0.00],
            [0.0, 0.0, 0.09, 0.01],
            [0.0, 0.0, 0.00, 0.00],
        ]
        m = ar.MAP(D0, D1, check=True)
        NUM_SAMPLES = 25000
        samples = list(m.generate(NUM_SAMPLES))

        self.assertEqual(len(samples), NUM_SAMPLES)
        assert_allclose(np.mean(samples), m.mean(), rtol=0.1)
        assert_allclose(np.std(samples), m.std(), rtol=0.1)
        assert_allclose(np.var(samples), m.var(), rtol=0.1)
        assert_allclose(stats.lag(samples, 2), [m.lag(1), m.lag(2)], rtol=0.1)

    # noinspection PyTypeChecker
    def test_call(self):
        D0 = [
            [-99.0,  0.0,   0.0,   0.0],
            [0.0,  -99.0,  99.0,   0.0],
            [0.0,    0.0, -0.01,   0.0],
            [0.01,   0.0,   0.0, -0.01],
        ]
        D1 = [
            [98.0, 1.00, 0.000, 0.000],
            [0.00, 0.00, 0.000, 0.000],
            [0.00, 0.00, 0.009, 0.001],
            [0.00, 0.00, 0.000, 0.000],
        ]
        m = ar.MAP(D0, D1, check=True)
        NUM_SAMPLES = 25000
        samples = [m() for _ in range(NUM_SAMPLES)]

        self.assertEqual(len(samples), NUM_SAMPLES)
        assert_allclose(np.mean(samples), m.mean(), rtol=0.2)
        assert_allclose(np.std(samples), m.std(), rtol=0.2)
        assert_allclose(np.var(samples), m.var(), rtol=0.2)
        assert_allclose(stats.lag(samples, 2), [m.lag(1), m.lag(2)], rtol=0.2)
