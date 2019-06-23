import numpy as np

from pyqumo.matrix import cached_method, cbdiag
from pyqumo.distributions import Exp
from pyqumo.arrivals import PoissonProcess, MAP
from pyqumo.chains import CTMC


class QueueingSystem(object):
    @property
    def arrival_rate(self): raise NotImplementedError

    @property
    def service_rate(self): raise NotImplementedError

    @property
    def service(self): raise NotImplementedError

    @property
    def arrival(self): raise NotImplementedError

    @property
    def departure(self): raise NotImplementedError

    @property
    def utilization(self): raise NotImplementedError

    def system_size_prob(self, k): raise NotImplementedError

    def system_size_pmf(self, size): raise NotImplementedError

    @property
    def system_size_avg(self): raise NotImplementedError

    @property
    def system_size_var(self): raise NotImplementedError

    @property
    def system_size_std(self): raise NotImplementedError

    @property
    def response_time(self): raise NotImplementedError

    @property
    def wait_time(self): raise NotImplementedError


class MM1(QueueingSystem):
    def __init__(self, arrival_rate, service_rate):
        self._arrival = PoissonProcess(arrival_rate)
        self._service = Exp(service_rate)
        self.__cache__ = {}

    @property
    def arrival_rate(self):
        return self._arrival.rate

    @property
    def service_rate(self):
        return self._service.rate

    @property
    def service(self):
        return self._service

    @property
    def arrival(self):
        return self._arrival

    @property
    @cached_method('departure')
    def departure(self):
        return PoissonProcess(self.arrival_rate)

    def __str__(self):
        return "M/M/1 with arrivals rate = {:g}, service rate = {:g}".format(
            self.arrival_rate, self.service_rate)

    @property
    @cached_method('utilization')
    def utilization(self):
        return self.arrival_rate / self.service_rate

    @cached_method('system_size_prob', 1)
    def system_size_prob(self, k):
        ro = self.utilization
        if 0 <= ro <= 1:
            if k == 0:
                return 1 - ro
            elif k > 0:
                return self.system_size_prob(k - 1) * ro
            else:
                return 0.0
        else:
            return 0.0

    def system_size_pmf(self, size=10):
        # ro = self.utilization
        return np.asarray([self.system_size_prob(i) for i in range(size)])

    @property
    @cached_method('system_size_avg')
    def system_size_avg(self):
        ro = float(self.utilization)
        if 1 - ro < 1e-10:
            return np.inf
        return ro / (1 - ro)

    @property
    @cached_method('system_size_var')
    def system_size_var(self):
        ro = float(self.utilization)
        if 1 - ro < 1e-10:
            return np.inf
        return ro / pow(1 - ro, 2)

    @property
    @cached_method('system_size_std')
    def system_size_std(self):
        return np.sqrt(self.system_size_var)

    @property
    @cached_method('response_time')
    def response_time(self):
        if self.service_rate - self.arrival_rate < 1e-10:
            return np.inf
        return 1.0 / (self.service_rate - self.arrival_rate)

    @property
    @cached_method('wait_time')
    def wait_time(self):
        return self.response_time - 1 / self.service_rate


class MM1N(QueueingSystem):
    def __init__(self, arrival_rate, service_rate, capacity):
        if not isinstance(capacity, int):
            raise TypeError("integer expected, '{}' found".format(capacity))
        if capacity <= 0:
            raise ValueError("positive integer expected, '{}' found".format(
                capacity))
        self._arrival = PoissonProcess(arrival_rate)
        self._service = Exp(service_rate)
        self._capacity = capacity
        self.__cache__ = {}

    @property
    def arrival_rate(self):
        return self._arrival.rate

    @property
    def service_rate(self):
        return self._service.rate

    @property
    def capacity(self):
        return self._capacity

    @property
    def service(self):
        return self._service

    @property
    def arrival(self):
        return self._arrival

    @property
    @cached_method('departure')
    def departure(self):
        n = self.capacity
        a = self.arrival_rate
        b = self.service_rate
        d0 = cbdiag(n + 1, [(0, [[-(a + b)]]), (1, [[a]])])
        d0[0, 0] += b
        d0[n, n] += a
        d1 = cbdiag(n + 1, [(-1, [[b]])])
        return MAP(d0, d1)

    @property
    @cached_method('utilization')
    def utilization(self):
        return self.arrival_rate / self.service_rate

    @cached_method('system_size_prob', 1)
    def system_size_prob(self, k):
        ro = self.utilization
        if np.abs(ro - 1) < 1e-10:
            return 1 / (self.capacity + 1)
        else:
            if k == 0:
                return (1 - ro) / (1 - pow(ro, self.capacity + 1))
            elif 0 < k <= self.capacity:
                return self.system_size_prob(0) * pow(ro, k)
            else:
                return 0.0

    def system_size_pmf(self, size=None):
        if size is None:
            size = self.capacity + 1
        return np.asarray([self.system_size_prob(i) for i in range(size)])

    @property
    @cached_method('system_size_avg')
    def system_size_avg(self):
        ro = float(self.utilization)
        if 1 - ro < 1e-10:
            return np.inf
        n = self.capacity
        ro_n = pow(ro, n)
        numerator = ro * (1 - (n + 1) * ro_n + n * ro_n * ro)
        denominator = (1 - ro) * (1 - ro_n * ro)
        return numerator / denominator

    @property
    def system_size_var(self):
        raise NotImplementedError

    @property
    def system_size_std(self):
        raise NotImplementedError

    @property
    @cached_method('response_time')
    def response_time(self):
        ro = float(self.utilization)
        if 1 - ro < 1e-10:
            return np.inf
        mu = self.service_rate
        n = self.capacity
        ro_n = pow(ro, n)
        numerator = 1 / mu * (1 - (n + 1) * ro_n + n * ro_n * ro)
        denominator = (1 - ro) * (1 - ro_n * ro)
        return numerator / denominator

    @property
    @cached_method('wait_time')
    def wait_time(self):
        return self.response_time - 1 / self.service_rate


class MapPh1N(QueueingSystem):
    def __init__(self, arrival, service, capacity):
        super().__init__()
        if not isinstance(capacity, int):
            raise TypeError("integer expected, '{}' found".format(capacity))
        if capacity <= 0:
            raise ValueError("positive integer expected, '{}' found".format(
                capacity))
        if not isinstance(arrival, MAP):
            raise TypeError("MAP arrival expected")
        # FIXME: suddenly this check doesn't pass in Notebook!
        # if not isinstance(service, pyqunet.distributions.PH):
        #     raise TypeError("PH service expected, '{}' found".format(
        #         type(service)))
        self._arrival = arrival
        self._service = service
        self._capacity = capacity
        self.__cache__ = {}

    @property
    def arrival_rate(self):
        return self._arrival.rate

    @property
    def service_rate(self):
        return self._service.rate

    @property
    def service(self):
        return self._service

    @property
    def arrival(self):
        return self._arrival

    @property
    def capacity(self):
        return self._capacity

    # noinspection PyPep8Naming
    @property
    @cached_method('departure')
    def departure(self):
        # Aliasing matrices from arrival MAP and service PH
        D0 = self.arrival.D0
        D1 = self.arrival.D1
        W = self.arrival.order
        Iw = np.eye(W)
        S = self.service.subgenerator
        tau = self.service.pmf0
        V = self.service.order
        Iv = np.eye(V)
        Ev = np.ones((V, 1))
        M = self.capacity - 1
        B = V * W
        Ob = np.zeros((B, B))

        # Building blocks
        D0_Iv = np.kron(D0, Iv)
        D1_Iv = np.kron(D1, Iv)
        D0_S = np.kron(D0, Iv) + np.kron(Iw, S)
        Ct = np.kron(-S.dot(Ev), tau)
        Iw_Ct = np.kron(Iw, Ct)
        R0 = np.kron(D1, np.kron(tau, Ev))
        Ra = np.kron(D0 + D1, Iv) + np.kron(Iw, S)

        # Building departure D0 and D1
        D0_dep = cbdiag(self.capacity, ((0, D0_S), (1, D1_Iv)))
        D0_dep[M * B:, M * B:] = Ra
        D0_left_col = np.vstack((D0_Iv,) + (Ob,) * self.capacity)
        D0_top_row = np.hstack((R0,) + (Ob,) * M)
        D0_dep = np.hstack((D0_left_col, np.vstack((D0_top_row, D0_dep))))
        D1_dep = cbdiag(self.capacity + 1, ((-1, Iw_Ct),))

        return MAP(D0_dep, D1_dep)

    @property
    def utilization(self):
        return self.arrival.rate / self.service.rate

    @cached_method('system_size_prob', 1)
    def system_size_prob(self, k):
        if k < 0:
            return 0.0
        dep = self.departure
        b = self.arrival.order * self.service.order
        ctmc = dep.background_ctmc()
        assert isinstance(ctmc, CTMC)
        pmf = ctmc.steady_pmf()
        return sum(pmf[k * b: (k + 1) * b])

    def system_size_pmf(self, size=None):
        if size is None:
            size = self.capacity + 1
        return np.asarray([self.system_size_prob(i) for i in range(size)])

    @property
    @cached_method('system_size_avg')
    def system_size_avg(self):
        pmf = self.system_size_pmf()
        a = [i * pmf[i] for i in range(len(pmf))]
        return sum(a)

    @property
    @cached_method('system_size_var')
    def system_size_var(self):
        pmf = self.system_size_pmf()
        m2_elements = [(i ** 2) * pmf[i] for i in range(len(pmf))]
        m2 = sum(m2_elements)
        m1 = self.system_size_avg
        return m2 - m1 ** 2

    @property
    @cached_method('system_size_std')
    def system_size_std(self):
        return np.sqrt(self.system_size_avg)

    @property
    @cached_method('response_time')
    def response_time(self):
        return self.system_size_avg / self.arrival.rate

    @property
    @cached_method('wait_time')
    def wait_time(self):
        return self.response_time - 1 / self.service.rate
