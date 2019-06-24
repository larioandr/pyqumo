import numpy as np
import pytest
from numpy.testing import assert_allclose
from pydesim import simulate

from pyqumo.distributions import Exponential, PhaseType
from pyqumo.qsim import QueueingSystem, QueueingTandemNetwork


@pytest.mark.parametrize('arrival,service,stime_limit', [
    (Exponential(5), Exponential(1), 8000),
    (Exponential(3), Exponential(2), 8000),
    (PhaseType.exponential(10.0), PhaseType.exponential(48.0), 50),
])
def test_mm1_model(arrival, service, stime_limit):
    ret = simulate(QueueingSystem, stime_limit=stime_limit, params={
        'arrival': arrival,
        'service': service,
        'queue_capacity': None,
    })

    busy_rate = ret.data.server.busy_trace.timeavg()
    system_size = ret.data.system_size_trace.timeavg()
    est_arrival_mean = ret.data.source.intervals.statistic().mean()
    est_departure_mean = ret.data.sink.arrival_intervals.statistic().mean()
    est_service_mean = ret.data.server.service_intervals.mean()
    est_delay = ret.data.source.delays.mean()

    mean_service = service.mean()
    mean_arrival = arrival.mean()
    rho = mean_service / mean_arrival

    assert_allclose(est_service_mean, mean_service, rtol=0.25)
    assert_allclose(busy_rate, rho, rtol=0.25)
    assert_allclose(system_size, rho / (1 - rho), rtol=0.25)
    assert_allclose(est_arrival_mean, mean_arrival, rtol=0.25)
    assert_allclose(est_departure_mean, mean_arrival, rtol=0.25)
    assert_allclose(est_delay, mean_arrival * rho / (1 - rho), rtol=0.25)


@pytest.mark.parametrize('arrival,service,stime_limit', [
    (Exponential(5), Exponential(1), 5000),
    (Exponential(3), Exponential(2), 5000),
    (PhaseType.exponential(10.0), PhaseType.exponential(48.0), 50),
])
def test_mm1_single_hop_tandem_model(arrival, service, stime_limit):
    ret = simulate(QueueingTandemNetwork, stime_limit=stime_limit, params={
        'arrivals': [arrival],
        'services': [service],
        'queue_capacity': None,
        'num_stations': 1,
    })

    busy_rate = ret.data.servers[0].busy_trace.timeavg()
    system_size = ret.data.system_size_trace[0].timeavg()
    est_arrival_mean = ret.data.sources[0].intervals.statistic().mean()
    est_service_mean = ret.data.servers[0].service_intervals.mean()
    est_delay = ret.data.sources[0].delays.mean()
    est_departure_mean = ret.data.sink.arrival_intervals.statistic().mean()

    mean_service = service.mean()
    mean_arrival = arrival.mean()
    rho = mean_service / mean_arrival

    assert_allclose(est_service_mean, mean_service, rtol=0.25)
    assert_allclose(busy_rate, rho, rtol=0.25)
    assert_allclose(system_size, rho / (1 - rho), atol=0.05, rtol=0.25)
    assert_allclose(est_arrival_mean, mean_arrival, rtol=0.25)
    assert_allclose(est_departure_mean, mean_arrival, rtol=0.25)
    assert_allclose(est_delay, mean_arrival * rho / (1 - rho), rtol=0.25)


@pytest.mark.parametrize('arrival,service,stime_limit,num_stations', [
    (Exponential(5), Exponential(1), 12000, 3),
    (Exponential(30), Exponential(2), 12000, 10),
    (PhaseType.exponential(10.0), PhaseType.exponential(48.0), 200, 4),
])
def test_mm1_multihop_tandem_model_with_cross_traffic(
        arrival, service, stime_limit, num_stations):
    ret = simulate(QueueingTandemNetwork, stime_limit=stime_limit, params={
        'arrivals': [arrival for _ in range(num_stations)],
        'services': [service for _ in range(num_stations)],
        'queue_capacity': None,
        'num_stations': num_stations,
    })

    n = num_stations

    mean_service = service.mean()
    mean_arrival = arrival.mean()
    rho = mean_service / mean_arrival

    expected_node_delays = []
    for i in range(num_stations):
        server = ret.data.servers[i]
        queue = ret.data.queues[i]

        est_busy_rate = server.busy_trace.timeavg()
        est_system_size = ret.data.system_size_trace[i].timeavg()
        est_arrival_mean = queue.arrival_intervals.statistic().mean()
        est_service_mean = server.service_intervals.mean()
        est_departure_mean = server.departure_intervals.statistic().mean()

        expected_busy_rate = rho * (i + 1)
        expected_service_mean = mean_service
        expected_system_size = expected_busy_rate / (1 - expected_busy_rate)
        expected_arrival_mean = mean_arrival / (i + 1)
        expected_departure_mean = expected_arrival_mean
        expected_node_delays.append(
            expected_system_size * expected_arrival_mean)

        assert_allclose(est_busy_rate, expected_busy_rate, rtol=0.25)
        assert_allclose(est_service_mean, expected_service_mean, rtol=0.25)
        assert_allclose(est_system_size, expected_system_size, rtol=0.25)
        assert_allclose(est_arrival_mean, expected_arrival_mean, rtol=0.25)
        assert_allclose(est_departure_mean, expected_departure_mean, rtol=0.25)

    est_delays = [ret.data.sources[i].delays.mean() for i in range(n)]
    expected_delays = [0.0] * n
    for i in range(n-1, -1, -1):
        expected_delays[i] = expected_node_delays[i] + (
            expected_delays[i + 1] if i < n - 1 else 0
        )

    assert_allclose(est_delays, expected_delays, rtol=0.25)


@pytest.mark.parametrize('arrival,service,stime_limit,num_stations', [
    (Exponential(5), Exponential(1), 12000, 3),
    (Exponential(30), Exponential(2), 12000, 10),
    (PhaseType.exponential(10.0), PhaseType.exponential(48.0), 200, 4),
])
def test_mm1_multihop_tandem_model_without_cross_traffic(
        arrival, service, stime_limit, num_stations):
    n = num_stations

    # noinspection PyTypeChecker
    ret = simulate(QueueingTandemNetwork, stime_limit=stime_limit, params={
        'arrivals': ([arrival] + [None] * (n - 1)),
        'services': [service for _ in range(num_stations)],
        'queue_capacity': None,
        'num_stations': num_stations,
    })

    mean_service = service.mean()
    mean_arrival = arrival.mean()
    rho = mean_service / mean_arrival
    expected_busy_rate = rho
    expected_service_mean = mean_service
    expected_system_size = expected_busy_rate / (1 - expected_busy_rate)
    expected_arrival_mean = mean_arrival
    expected_departure_mean = expected_arrival_mean

    for i in range(num_stations):
        server = ret.data.servers[i]
        queue = ret.data.queues[i]

        est_busy_rate = server.busy_trace.timeavg()
        est_system_size = ret.data.system_size_trace[i].timeavg()
        est_arrival_mean = queue.arrival_intervals.statistic().mean()
        est_service_mean = server.service_intervals.mean()
        est_departure_mean = server.departure_intervals.statistic().mean()

        assert_allclose(est_busy_rate, expected_busy_rate, rtol=0.25)
        assert_allclose(est_service_mean, expected_service_mean, rtol=0.25)
        assert_allclose(est_system_size, expected_system_size, rtol=0.25)
        assert_allclose(est_arrival_mean, expected_arrival_mean, rtol=0.25)
        assert_allclose(est_departure_mean, expected_departure_mean, rtol=0.25)

    expected_delay = expected_system_size * mean_arrival * n
    est_delay = ret.data.sources[0].delays.mean()
    assert_allclose(est_delay, expected_delay, rtol=0.25)


def test_tandem_with_different_services():
    stime_limit = 10000
    n = 3
    services = [Exponential(5), Exponential(8), Exponential(4)]
    arrivals = [Exponential(10), None, None]

    ret = simulate(QueueingTandemNetwork, stime_limit=stime_limit, params={
        'arrivals': arrivals,
        'services': services,
        'queue_capacity': None,
        'num_stations': n,
    })

    rhos = [services[i].mean() / arrivals[0].mean() for i in range(n)]
    sizes = [r / (1 - r) for r in rhos]
    delays = [sz * arrivals[0].mean() for sz in sizes]
    end_to_end_delay = sum(delays)

    assert_allclose(
        ret.data.sources[0].delays.mean(), end_to_end_delay, rtol=0.25
    )
