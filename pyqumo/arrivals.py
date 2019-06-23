import numpy as np

from pyqumo import chains
from pyqumo.matrix import is_infinitesimal, cached_method, cbdiag, order_of
from pyqumo.distributions import Exp


class ArrivalProcess(object):
    def __init__(self): super().__init__()

    def mean(self): raise NotImplementedError

    def var(self): raise NotImplementedError

    def std(self): raise NotImplementedError

    def cv(self): raise NotImplementedError

    def lag(self, k): raise NotImplementedError

    def moment(self, k): raise NotImplementedError

    def generate(self, size): raise NotImplementedError


class PoissonProcess(ArrivalProcess):
    def __init__(self, rate):
        super().__init__()
        if rate <= 0.0:
            raise ValueError("positive rate expected, '{}' found".format(rate))
        self._rate = rate
        self._dist = Exp(rate)
        self.__cache__ = {}

    @property
    def rate(self):
        return self._rate

    @property
    def intervals_distribution(self):
        return self._dist

    def mean(self): return self._dist.mean()

    def var(self): return self._dist.var()

    def std(self): return self._dist.std()

    def cv(self): return self._dist.std() / self._dist.mean()

    def lag(self, k):
        return 0.0

    def moment(self, k): return self._dist.moment(k)

    def generate(self, size):
        return self._dist.generate(size)


class MAP(ArrivalProcess):
    @staticmethod
    def erlang(shape, rate):
        d0 = cbdiag(shape, [(0, [[-rate]]), (1, [[rate]])])
        d1 = np.zeros((shape, shape))
        d1[shape-1, 0] = rate
        return MAP(d0, d1)

    @staticmethod
    def exponential(rate):
        if rate <= 0:
            raise ValueError("positive rate expected, '{}' found".format(rate))
        d0 = [[-rate]]
        d1 = [[rate]]
        return MAP(d0, d1)

    def __init__(self, d0, d1, check=True, rtol=1e-5, atol=1e-6):
        super().__init__()
        d0 = np.asarray(d0)
        d1 = np.asarray(d1)
        if check:
            if not is_infinitesimal(d0 + d1, rtol=rtol, atol=atol):
                raise ValueError("D0 + D1 must be infinitesimal")
            if not np.all(d1 >= -atol):
                raise ValueError("all D1 elements must be non-negative")
            if not (np.all(d0 - np.diag(d0.diagonal()) >= -atol)):
                raise ValueError("all non-diagonal D0 elements must be "
                                 "non-negative")

        self._d0 = d0
        self._d1 = d1
        self.__cache__ = {}

    # noinspection PyPep8Naming
    @property
    def D0(self):
        return self._d0

    # noinspection PyPep8Naming
    @property
    def D1(self):
        return self._d1

    # noinspection PyPep8Naming
    def D(self, n):
        if n == 0:
            return self.D0
        elif n == 1:
            return self.D1
        else:
            raise ValueError("illegal n={} found".format(n))

    @property
    @cached_method('generator')
    def generator(self):
        return self.D0 + self.D1

    @property
    @cached_method('order')
    def order(self):
        return order_of(self.D0)

    @cached_method('d0n', 1)
    def d0n(self, k):
        """Returns $(-D0)^{k}$."""
        delta = k - round(k)
        if np.abs(delta) > 1e-10:
            print("DELTA = {}".format(delta))
            raise TypeError("illegal degree='{}', integer expected".format(k))
        k = int(k)
        if k == 0:
            return np.eye(self.order)
        elif k > 0:
            return self.d0n(k - 1).dot(-self.D0)
        elif k == -1:
            return -np.linalg.inv(self.D0)
        else:  # degree <= -2
            return self.d0n(k + 1).dot(self.d0n(-1))

    @cached_method('moment', index_arg=1)
    def moment(self, k):
        pi = self.embedded_dtmc().steady_pmf()
        x = np.math.factorial(k) * pi.dot(self.d0n(-k)).dot(np.ones(self.order))
        return x.item()

    @property
    @cached_method('rate')
    def rate(self):
        return 1.0 / self.moment(1)

    @cached_method('mean')
    def mean(self):
        return self.moment(1)

    @cached_method('var')
    def var(self):
        return self.moment(2) - pow(self.moment(1), 2)

    @cached_method('std')
    def std(self):
        return self.var() ** 2

    @cached_method('cv')
    def cv(self):
        return self.std() / self.mean()

    @cached_method('lag', index_arg=1)
    def lag(self, k):
        # TODO: write unit test
        #
        # Computing lag-k as:
        #
        #   r^2 * pi * (-D0)^(-1) * P^k * (-D0)^(-1) * 1s - 1
        #   -------------------------------------------------- ,
        #   2 * r^2 * pi * (-D0)^(-2) * 1s - 1
        #
        # where r - rate (\lambda), pi - stationary distribution of the
        # embedded DTMC, 1s - vector of ones of MAP order
        #
        dtmc_matrix_k = self._pow_dtmc_matrix(k)
        pi = self.embedded_dtmc().steady_pmf()
        rate2 = pow(self.rate, 2.0)
        e = np.ones(self.order)
        d0ni = self.d0n(-1)
        d0ni2 = self.d0n(-2)

        numerator = (rate2 *
                     pi.dot(d0ni).dot(dtmc_matrix_k).dot(d0ni).dot(e)) - 1
        denominator = (2 * rate2 * pi.dot(d0ni2).dot(e) - 1)
        return numerator / denominator

    @cached_method('background_ctmc')
    def background_ctmc(self):
        return chains.CTMC(self.generator, check=False)

    @cached_method('embedded_dtmc')
    def embedded_dtmc(self):
        return chains.DTMC(self.d0n(-1).dot(self.D1), check=False)

    def generate(self, size, init=None):
        # Building P - a transition probabilities matrix of size Nx(2N), where:
        # - P(i, j), j < N, is a probability to move i -> j without arrival;
        # - P(i, j), N <= j < 2N is a probability to move i -> (j - N) with
        #   arrival
        rates = -self.D0.diagonal()
        means = np.diag(np.power(rates, -1))
        p0 = means.dot(self.D0 + np.diag(rates))
        p1 = means.dot(self.D1)
        p = np.hstack([p0, p1])

        # Looking for the first state.
        # - if init is None or a vector, it is treated as initial PMF
        # - if init is an `int`, it is treated as the first state
        state = None
        init_pmf = None
        if init is None:
            init_pmf = np.asarray([1. / self.order] * self.order)
        elif init is int:
            state = init
        elif order_of(init) == self.order:
            init_pmf = init
        else:
            raise ValueError("unexpected init argument = '{}'".format(init))
        if state is None:
            state = np.random.choice(range(self.order), p=init_pmf)

        # Yielding random intervals
        arrival_interval = 0.0
        for i in range(size):
            arrival_interval += np.random.exponential(1 / rates[state])
            next_state = np.random.choice(range(2 * self.order), p=p[state])
            if next_state >= self.order:
                next_state -= self.order
                yield arrival_interval
                arrival_interval = 0.0
            state = next_state

    def compose(self, other):
        # TODO:  write unit tests
        if not isinstance(other, MAP):
            raise TypeError
        self_eye = np.eye(self.order)
        other_eye = np.eye(other.order)
        d0_out = np.kron(self.D0, other_eye) + np.kron(other.D0, self_eye)
        d1_out = np.kron(self.D1, other_eye) + np.kron(other.D1, self_eye)
        return MAP(d0_out, d1_out)

    @cached_method('_pow_dtmc_matrix', index_arg=1)
    def _pow_dtmc_matrix(self, k):
        if k == 0:
            return np.eye(self.order)
        elif k > 0:
            return self._pow_dtmc_matrix(k - 1).dot(self.embedded_dtmc().matrix)
        else:
            raise ValueError("k='{}' must be non-negative".format(k))
