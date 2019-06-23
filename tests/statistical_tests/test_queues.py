import unittest as ut
import numpy as np
from numpy.testing import assert_almost_equal

from pyqumo.queues import MM1, MM1N, MapPh1N
from pyqumo.arrivals import PoissonProcess, MAP
from pyqumo.distributions import Exp, PhaseType


class TestMM1(ut.TestCase):
    def test_valid_creation(self):
        queue_2_5 = MM1(arrival_rate=2.0, service_rate=5.0)
        queue_1_10 = MM1(arrival_rate=1.0, service_rate=10.0)
        self.assertAlmostEqual(queue_2_5.arrival_rate, 2.0)
        self.assertAlmostEqual(queue_2_5.service_rate, 5.0)
        self.assertAlmostEqual(queue_1_10.arrival_rate, 1.0)
        self.assertAlmostEqual(queue_1_10.service_rate, 10.0)

    def test_invalid_creation(self):
        with self.assertRaises(ValueError):
            MM1(0.0, 1)
        with self.assertRaises(ValueError):
            MM1(-1, 1)
        with self.assertRaises(ValueError):
            MM1(1, 0)
        with self.assertRaises(ValueError):
            MM1(1, -1)
        with self.assertRaises(TypeError):
            MM1('1', 1)
        with self.assertRaises(TypeError):
            MM1(1, '1')
        with self.assertRaises(TypeError):
            MM1([1, 2], 1)
        with self.assertRaises(TypeError):
            MM1(1, [1, 2])

    def test_arrival_process(self):
        queue = MM1(1.0, 2.0)
        self.assertIsInstance(queue.arrival, PoissonProcess)
        self.assertAlmostEqual(queue.arrival.rate, queue.arrival_rate)

    def test_service_distribution(self):
        queue = MM1(1, 2)
        self.assertIsInstance(queue.service, Exp)
        self.assertAlmostEqual(queue.service.rate, queue.service_rate)

    def test_departure_process(self):
        queue = MM1(2.0, 5.0)
        self.assertIsInstance(queue.departure, PoissonProcess)
        # According to Burke theorem, M/M/1 departure process is the same
        # as arrival process
        self.assertAlmostEqual(queue.departure.rate, queue.arrival_rate)

    def test_utilization(self):
        q1 = MM1(2, 5)
        q2 = MM1(1, 10)
        q3 = MM1(5, 5)
        q4 = MM1(10, 4)
        self.assertAlmostEqual(q1.utilization, 0.4)
        self.assertAlmostEqual(q2.utilization, 0.1)
        self.assertAlmostEqual(q3.utilization, 1.0)
        self.assertAlmostEqual(q4.utilization, 2.5)

    def test_system_size_pmf(self):
        q1 = MM1(1, 2)
        q2 = MM1(2, 5)
        assert_almost_equal(q1.system_size_pmf(size=3), [0.5, 0.25, 0.125])
        assert_almost_equal(q2.system_size_pmf(size=3), [0.6, 0.24, 0.096])

    def test_system_size_avg(self):
        q1 = MM1(1, 2)
        q2 = MM1(2, 5)
        q3 = MM1(1, 1)
        q4 = MM1(2, 1)
        self.assertAlmostEqual(q1.system_size_avg, 1.0)
        self.assertAlmostEqual(q2.system_size_avg, 2 / 3)
        self.assertAlmostEqual(q3.system_size_avg, np.inf)
        self.assertAlmostEqual(q4.system_size_avg, np.inf)

    def test_system_size_var(self):
        q1 = MM1(1, 2)
        q2 = MM1(2, 5)
        q3 = MM1(1, 1)
        q4 = MM1(2, 1)
        self.assertAlmostEqual(q1.system_size_var, 2.0)
        self.assertAlmostEqual(q2.system_size_var, 10 / 9)
        self.assertAlmostEqual(q3.system_size_var, np.inf)
        self.assertAlmostEqual(q4.system_size_var, np.inf)

    def test_system_size_std(self):
        q1 = MM1(1, 2)
        q2 = MM1(2, 5)
        q3 = MM1(1, 1)
        q4 = MM1(2, 1)
        self.assertAlmostEqual(q1.system_size_std, np.sqrt(2.0))
        self.assertAlmostEqual(q2.system_size_std, np.sqrt(10) / 3)
        self.assertAlmostEqual(q3.system_size_std, np.inf)
        self.assertAlmostEqual(q4.system_size_std, np.inf)

    def test_response_time(self):
        q1 = MM1(1, 2)
        q2 = MM1(2, 5)
        q3 = MM1(1, 1)
        q4 = MM1(2, 1)
        self.assertAlmostEqual(q1.response_time, 1.0)
        self.assertAlmostEqual(q2.response_time, 1 / 3)
        self.assertAlmostEqual(q3.response_time, np.inf)
        self.assertAlmostEqual(q4.response_time, np.inf)

    def test_wait_time(self):
        q1 = MM1(1, 2)
        q2 = MM1(2, 5)
        q3 = MM1(1, 1)
        q4 = MM1(2, 1)
        self.assertAlmostEqual(q1.wait_time, 0.5)
        self.assertAlmostEqual(q2.wait_time, 2 / 15)
        self.assertAlmostEqual(q3.wait_time, np.inf)
        self.assertAlmostEqual(q4.wait_time, np.inf)


class TestMM1N(ut.TestCase):

    def test_valid_creation(self):
        queue_2_5 = MM1N(arrival_rate=2.0, service_rate=5.0, capacity=1)
        queue_1_10 = MM1N(arrival_rate=1.0, service_rate=10.0, capacity=5)
        self.assertAlmostEqual(queue_2_5.arrival_rate, 2.0)
        self.assertAlmostEqual(queue_2_5.service_rate, 5.0)
        self.assertEqual(queue_2_5.capacity, 1)
        self.assertAlmostEqual(queue_1_10.arrival_rate, 1.0)
        self.assertAlmostEqual(queue_1_10.service_rate, 10.0)
        self.assertAlmostEqual(queue_1_10.capacity, 5)

    def test_invalid_creation(self):
        with self.assertRaises(ValueError):
            MM1N(0.0, 1, 1)
        with self.assertRaises(ValueError):
            MM1N(-1, 1, 1)
        with self.assertRaises(ValueError):
            MM1N(1, 0, 1)
        with self.assertRaises(ValueError):
            MM1N(1, -1, 1)
        with self.assertRaises(TypeError):
            MM1N('1', 1, 1)
        with self.assertRaises(TypeError):
            MM1N(1, '1', 1)
        with self.assertRaises(TypeError):
            MM1N([1, 2], 1, 1)
        with self.assertRaises(TypeError):
            MM1N(1, [1, 2], 1)
        with self.assertRaises(ValueError):
            MM1N(1, 1, -1)
        with self.assertRaises(ValueError):
            MM1N(1, 1, 0)
        with self.assertRaises(TypeError):
            MM1N(1, 1, '1')
        with self.assertRaises(TypeError):
            MM1N(1, 1, 1.2)
        with self.assertRaises(TypeError):
            MM1N(1, 1, [1, 2])

    def test_arrival_process(self):
        queue = MM1N(1.0, 2.0, 5)
        self.assertIsInstance(queue.arrival, PoissonProcess)
        self.assertAlmostEqual(queue.arrival.rate, queue.arrival_rate)

    def test_service_distribution(self):
        queue = MM1N(1, 2, 5)
        self.assertIsInstance(queue.service, Exp)
        self.assertAlmostEqual(queue.service.rate, queue.service_rate)

    def test_departure_process(self):
        queue = MM1N(2.0, 5.0, 200)
        self.assertIsInstance(queue.departure, MAP)
        # Since queue is very long, arrival rate will be close to departure
        # rate.
        self.assertAlmostEqual(queue.departure.rate, queue.arrival_rate, 5)

    def test_utilization(self):
        q1 = MM1N(2, 5, 4)
        q2 = MM1N(1, 10, 3)
        q3 = MM1N(5, 5, 2)
        q4 = MM1N(10, 4, 1)
        self.assertAlmostEqual(q1.utilization, 0.4)
        self.assertAlmostEqual(q2.utilization, 0.1)
        self.assertAlmostEqual(q3.utilization, 1.0)
        self.assertAlmostEqual(q4.utilization, 2.5)

    def test_system_size_pmf(self):
        q1 = MM1N(1, 2, 10)
        q2 = MM1N(2, 5, 5)
        assert_almost_equal(
            q1.system_size_pmf(size=3), [0.500244, 0.250122, 0.125061], 6)
        assert_almost_equal(
            q2.system_size_pmf(size=3), [0.602468, 0.240987, 0.096395], 6)

    def test_system_size_avg(self):
        q1 = MM1N(1, 2, 5)
        q2 = MM1N(2, 5, 5)
        q3 = MM1N(1, 1, 10)
        q4 = MM1N(2, 1, 100)
        self.assertAlmostEqual(q1.system_size_avg, 0.9047619047619048)
        self.assertAlmostEqual(q2.system_size_avg, 0.6419895893580104)
        self.assertAlmostEqual(q3.system_size_avg, np.inf)
        self.assertAlmostEqual(q4.system_size_avg, np.inf)

    def test_system_size_var(self):
        # TODO: implement first
        pass

    def test_system_size_std(self):
        # TODO: implement first
        pass

    def test_response_time(self):
        q1 = MM1N(1, 2, 5)
        q2 = MM1N(2, 5, 6)
        q3 = MM1N(1, 1, 10)
        q4 = MM1N(2, 1, 100)
        self.assertAlmostEqual(q1.response_time, 0.9047619047619048)
        self.assertAlmostEqual(q2.response_time, 0.3275895226739491)
        self.assertAlmostEqual(q3.response_time, np.inf)
        self.assertAlmostEqual(q4.response_time, np.inf)

    def test_wait_time(self):
        q1 = MM1N(1, 2, 5)
        q2 = MM1N(2, 5, 6)
        q3 = MM1N(1, 1, 10)
        q4 = MM1N(2, 1, 100)
        self.assertAlmostEqual(q1.wait_time, 0.40476190476190477)
        self.assertAlmostEqual(q2.wait_time, 0.12758952267394907)
        self.assertAlmostEqual(q3.wait_time, np.inf)
        self.assertAlmostEqual(q4.wait_time, np.inf)


class TestMapPh1N(ut.TestCase):

    def test_valid_creation(self):
        q1 = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 5)
        q2 = MapPh1N(MAP.exponential(2.0), PhaseType.exponential(5.0), 4)
        self.assertAlmostEqual(q1.arrival_rate, 1)
        self.assertAlmostEqual(q1.service_rate, 2)
        self.assertEqual(q1.capacity, 5)
        self.assertAlmostEqual(q2.arrival_rate, 2)
        self.assertAlmostEqual(q2.service_rate, 5)
        self.assertAlmostEqual(q2.capacity, 4)

    def test_invalid_creation(self):
        ar1 = MAP.exponential(1.0)
        srv1 = PhaseType.exponential(2.0)

        with self.assertRaises(ValueError):
            MapPh1N(ar1, srv1, -1)
        with self.assertRaises(ValueError):
            MapPh1N(ar1, srv1, 0)
        with self.assertRaises(TypeError):
            MapPh1N(ar1, srv1, '1')
        with self.assertRaises(TypeError):
            MapPh1N(ar1, srv1, 1.2)
        with self.assertRaises(TypeError):
            MapPh1N(ar1, srv1, [1, 2])

    def test_arrival_process(self):
        q = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 5)
        self.assertIsInstance(q.arrival, MAP)
        self.assertAlmostEqual(q.arrival.rate, q.arrival_rate)

    def test_service_distribution(self):
        q = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 5)
        self.assertIsInstance(q.service, PhaseType)
        self.assertAlmostEqual(q.service.rate, q.service_rate)

    def test_departure_process(self):
        q1 = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 20)
        q2 = MapPh1N(MAP.erlang(4, 1.0), PhaseType.erlang(3, 2.0), 20)
        self.assertIsInstance(q1.departure, MAP)
        self.assertIsInstance(q2.departure, MAP)
        # Since queue is very long, arrival rate will be close to departure
        # rate.
        self.assertAlmostEqual(q1.departure.rate, q1.arrival_rate, 5)
        self.assertAlmostEqual(q2.departure.rate, q2.arrival_rate, 5)

    def test_utilization(self):
        q1 = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 20)
        q2 = MapPh1N(MAP.erlang(4, 1.0), PhaseType.erlang(3, 2.0), 20)
        self.assertAlmostEqual(q1.utilization, 1 / 2)
        self.assertAlmostEqual(q2.utilization, 3 / 8)

    def test_system_size_pmf(self):
        q1 = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 5)
        q2 = MapPh1N(MAP.exponential(2.0), PhaseType.exponential(5.0), 10)
        exp_q1 = MM1N(1.0, 2.0, 5)
        exp_q2 = MM1N(2, 5, 10)
        assert_almost_equal(q1.system_size_pmf(10), exp_q1.system_size_pmf(10))
        self.assertEqual(len(q1.system_size_pmf()), q1.capacity + 1)
        assert_almost_equal(q2.system_size_pmf(), exp_q2.system_size_pmf())

    def test_system_size_avg(self):
        q1 = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 5)
        q2 = MapPh1N(MAP.exponential(2.0), PhaseType.exponential(5.0), 10)
        exp_q1 = MM1N(1.0, 2.0, 5)
        exp_q2 = MM1N(2, 5, 10)
        self.assertAlmostEqual(q1.system_size_avg, exp_q1.system_size_avg)
        self.assertAlmostEqual(q2.system_size_avg, exp_q2.system_size_avg)

    #
    # def test_system_size_var(self):
    #     # TODO: implement first
    #     pass
    #
    # def test_system_size_std(self):
    #     # TODO: implement first
    #     pass
    #
    def test_response_time(self):
        q1 = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 5)
        q2 = MapPh1N(MAP.exponential(2.0), PhaseType.exponential(5.0), 10)
        exp_q1 = MM1N(1.0, 2.0, 5)
        exp_q2 = MM1N(2, 5, 10)
        self.assertAlmostEqual(q1.response_time, exp_q1.response_time)
        self.assertAlmostEqual(q2.response_time, exp_q2.response_time)

    def test_wait_time(self):
        q1 = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 5)
        q2 = MapPh1N(MAP.exponential(2.0), PhaseType.exponential(5.0), 10)
        exp_q1 = MM1N(1.0, 2.0, 5)
        exp_q2 = MM1N(2, 5, 10)
        self.assertAlmostEqual(q1.wait_time, exp_q1.wait_time)
        self.assertAlmostEqual(q2.wait_time, exp_q2.wait_time)
