import pytest
from numpy.testing import assert_allclose
from pydesim import simulate

from pyqumo.distributions import Exponential, PhaseType
from pyqumo.qsim import QueueingSystem, QueueingTandemNetwork, \
    tandem_queue_network, tandem_queue_network_with_fixed_service


@pytest.mark.parametrize('arrival,service,stime_limit', [
    (Exponential(5), Exponential(1), 12000),
    (Exponential(3), Exponential(2), 12000),
    (PhaseType.exponential(10.0), PhaseType.exponential(48.0), 4000),
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
    est_sys_wait = ret.data.system_wait_intervals.mean()
    est_queue_wait = ret.data.queue.wait_intervals.mean()

    mean_service = service.mean()
    mean_arrival = arrival.mean()
    rho = mean_service / mean_arrival
    expected_delay = mean_arrival * rho / (1 - rho)

    assert_allclose(est_service_mean, mean_service, rtol=0.25)
    assert_allclose(busy_rate, rho, rtol=0.25)
    assert_allclose(system_size, rho / (1 - rho), rtol=0.25)
    assert_allclose(est_arrival_mean, mean_arrival, rtol=0.25)
    assert_allclose(est_departure_mean, mean_arrival, rtol=0.25)
    assert_allclose(est_delay, expected_delay, rtol=0.25)
    assert_allclose(est_sys_wait, expected_delay, rtol=0.25)
    assert_allclose(est_queue_wait, expected_delay - mean_service, 0.25)
    
    assert ret.data.queue.drop_ratio == 0
    assert ret.data.queue.num_dropped == 0
    assert ret.data.queue.num_arrived > 0
    assert ret.data.server.num_served > 0


@pytest.mark.parametrize('arrival,service,stime_limit', [
    (Exponential(5), Exponential(1), 15000),
    (Exponential(3), Exponential(2), 15000),
    (PhaseType.exponential(10.0), PhaseType.exponential(48.0), 4000),
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
    est_sys_wait = ret.data.system_wait_intervals[0].mean()
    est_queue_wait = ret.data.queues[0].wait_intervals.mean()

    mean_service = service.mean()
    mean_arrival = arrival.mean()
    rho = mean_service / mean_arrival
    expected_delay = mean_arrival * rho / (1 - rho)

    assert_allclose(est_service_mean, mean_service, rtol=0.25)
    assert_allclose(busy_rate, rho, rtol=0.25)
    assert_allclose(system_size, rho / (1 - rho), atol=0.05, rtol=0.25)
    assert_allclose(est_arrival_mean, mean_arrival, rtol=0.25)
    assert_allclose(est_departure_mean, mean_arrival, rtol=0.25)
    assert_allclose(est_delay, expected_delay, rtol=0.25)
    assert_allclose(est_sys_wait, expected_delay, rtol=0.25)
    assert_allclose(est_queue_wait, expected_delay - mean_service, 0.25)

    assert ret.data.queues[0].drop_ratio == 0
    assert ret.data.queues[0].num_dropped == 0
    assert ret.data.queues[0].num_arrived > 0
    assert ret.data.servers[0].num_served > 0


@pytest.mark.parametrize('arrival,service,stime_limit,num_stations', [
    (Exponential(5), Exponential(1), 48000, 3),
    (Exponential(30), Exponential(2), 48000, 10),
    (PhaseType.exponential(10.0), PhaseType.exponential(48.0), 4000, 4),
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
        est_system_wait = ret.data.system_wait_intervals[i].mean()
        est_queue_wait = queue.wait_intervals.mean()

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
        assert_allclose(est_system_wait, expected_node_delays[-1], rtol=0.25)
        assert_allclose(
            est_queue_wait, 
            expected_node_delays[-1] - expected_service_mean, 
            rtol=0.25
        )

    est_delays = [ret.data.sources[i].delays.mean() for i in range(n)]
    expected_delays = [0.0] * n
    for i in range(n-1, -1, -1):
        expected_delays[i] = expected_node_delays[i] + (
            expected_delays[i + 1] if i < n - 1 else 0
        )

    assert_allclose(est_delays, expected_delays, rtol=0.35)


@pytest.mark.parametrize('arrival,service,stime_limit,num_stations', [
    (Exponential(5), Exponential(1), 24000, 3),
    (Exponential(30), Exponential(2), 24000, 10),
    (PhaseType.exponential(10.0), PhaseType.exponential(48.0), 2000, 4),
])
def test_mm1_multihop_tandem_model_without_cross_traffic(
        arrival, service, stime_limit, num_stations):
    n = num_stations

    # noinspection PyTypeChecker
    arrivals = [arrival] + [None] * (n - 1)
    services = [service for _ in range(num_stations)]
    simret = tandem_queue_network(arrivals, services, None, stime_limit)

    mean_service = service.mean()
    mean_arrival = arrival.mean()
    rho = mean_service / mean_arrival
    expected_busy_rate = rho
    expected_service_mean = mean_service
    expected_system_size = expected_busy_rate / (1 - expected_busy_rate)
    expected_arrival_mean = mean_arrival
    expected_departure_mean = expected_arrival_mean

    for i in range(num_stations):
        node = simret.nodes[0]
        est_busy_rate = node.busy.timeavg()
        est_system_size = node.system_size.timeavg()
        est_arrival_mean = node.arrivals.mean()
        est_service_mean = node.service.mean()
        est_departure_mean = node.departures.mean()

        assert_allclose(est_busy_rate, expected_busy_rate, rtol=0.25)
        assert_allclose(est_service_mean, expected_service_mean, rtol=0.25)
        assert_allclose(est_system_size, expected_system_size, rtol=0.25)
        assert_allclose(est_arrival_mean, expected_arrival_mean, rtol=0.25)
        assert_allclose(est_departure_mean, expected_departure_mean, rtol=0.25)

    expected_delay = expected_system_size * mean_arrival * n
    est_delay = simret.nodes[0].delay.mean()
    assert_allclose(est_delay, expected_delay, rtol=0.25)


def test_tandem_with_different_services():
    stime_limit = 20000
    n = 3
    services = [Exponential(5), Exponential(8), Exponential(4)]
    arrivals = [Exponential(10), None, None]

    simret = tandem_queue_network(arrivals, services, None, stime_limit)

    rhos = [services[i].mean() / arrivals[0].mean() for i in range(n)]
    sizes = [r / (1 - r) for r in rhos]
    delays = [sz * arrivals[0].mean() for sz in sizes]
    end_to_end_delay = sum(delays)

    assert_allclose(simret.nodes[0].delay.mean(), end_to_end_delay, rtol=0.25)


def test_tandem_with_fixed_service():
    stime_limit = 20000
    n = 3
    service = Exponential(5)
    arrivals = [Exponential(10), None, None]

    simret = tandem_queue_network_with_fixed_service(
        arrivals, service, None, stime_limit)
    
    print(simret)
    service_means = [simret.nodes[i].service.mean() for i in range(n)]

    n_services = len(simret.nodes[-1].service.as_tuple())
    assert_allclose(
        simret.nodes[0].service.as_tuple()[:n_services],
        simret.nodes[1].service.as_tuple()[:n_services],
    )
    assert_allclose(
        simret.nodes[0].service.as_tuple()[:n_services],
        simret.nodes[2].service.as_tuple()[:n_services],
    )

    assert_allclose(service_means[0], service.mean(), rtol=0.1)
