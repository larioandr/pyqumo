import unittest as ut
import numpy as np
from numpy.testing import assert_almost_equal

import pyqumo.stats as stats
from pyqumo.distributions import Exp, PhaseType
from pyqumo.arrivals import PoissonProcess, MAP


class TestMoments(ut.TestCase):

    def test_moments_const_samples(self):
        samples_1 = [1.0] * 10
        samples_2 = [2.0] * 10

        self.assertIsInstance(stats.moment(samples_1, 1), np.ndarray)
        self.assertIsInstance(stats.moment(samples_1, 2), np.ndarray)
        self.assertIsInstance(stats.moment(samples_2, 1), np.ndarray)
        self.assertIsInstance(stats.moment(samples_2, 2), np.ndarray)

        assert_almost_equal(stats.moment(samples_1, 1), [1.0])
        assert_almost_equal(stats.moment(samples_1, 2), [1, 1])
        assert_almost_equal(stats.moment(samples_2, 1), [2.0])
        assert_almost_equal(stats.moment(samples_2, 2), [2, 4])

    def test_moments_exp_samples(self):
        samples = np.random.exponential(1 / 5.0, 20000)

        self.assertIsInstance(stats.moment(samples, 1), np.ndarray)
        assert_almost_equal(stats.moment(samples, 1), [1 / 5], 2)
        assert_almost_equal(stats.moment(samples, 3), [0.2, 0.08, 6 / 125], 2)

    def test_moments_exp_distribution(self):
        source = Exp(3.0)

        self.assertIsInstance(stats.moment(source, 1), np.ndarray)
        self.assertIsInstance(stats.moment(source, 2), np.ndarray)
        assert_almost_equal(stats.moment(source, 1), [1 / 3], 10)
        assert_almost_equal(stats.moment(source, 2), [1 / 3, 2 / 9], 10)

    def test_moments_ph_distribution(self):
        source = PhaseType.exponential(3.0)

        self.assertIsInstance(stats.moment(source, 1), np.ndarray)
        self.assertIsInstance(stats.moment(source, 2), np.ndarray)
        assert_almost_equal(stats.moment(source, 1), [1 / 3], 10)
        assert_almost_equal(stats.moment(source, 3), [1 / 3, 2 / 9, 2 / 9], 10)

    def test_moments_poisson_arrival(self):
        source = PoissonProcess(3.0)

        self.assertIsInstance(stats.moment(source, 1), np.ndarray)
        self.assertIsInstance(stats.moment(source, 2), np.ndarray)
        assert_almost_equal(stats.moment(source, 1), [1 / 3], 10)
        assert_almost_equal(stats.moment(source, 2), [1 / 3, 2 / 9], 10)

    def test_moments_map_arrival(self):
        source = MAP.exponential(3.0)

        self.assertIsInstance(stats.moment(source, 1), np.ndarray)
        self.assertIsInstance(stats.moment(source, 2), np.ndarray)
        assert_almost_equal(stats.moment(source, 1), [1 / 3], 10)
        assert_almost_equal(stats.moment(source, 2), [1 / 3, 2 / 9], 10)


class TestLags(ut.TestCase):

    def test_map_equivalence(self):
        m = MAP([[-10., 1], [2., -3.]], [[9, 0], [0.1, 0.9]])
        samples = list(m.generate(20000))
        self.assertIsInstance(stats.lag(m, 1), np.ndarray)
        self.assertIsInstance(stats.lag(m, 2), np.ndarray)
        self.assertIsInstance(stats.lag(samples, 1), np.ndarray)
        self.assertIsInstance(stats.lag(samples, 2), np.ndarray)
        assert_almost_equal(stats.lag(m, 2), stats.lag(samples, 2), 2)
