from functools import lru_cache

import numpy as np

from pyqumo import chains
from pyqumo.matrix import is_infinitesimal, cached_method, cbdiag, order_of
from pyqumo.distributions import Exp


class ArrivalProcess:
    def mean(self):
        raise NotImplementedError

    @property
    @lru_cache
    def rate(self):
        return 1.0 / self.mean()

    def var(self):
        raise NotImplementedError

    def std(self):
        raise NotImplementedError

    def cv(self):
        raise NotImplementedError

    def lag(self, k):
        raise NotImplementedError

    def moment(self, k):
        raise NotImplementedError

    def generate(self, size):
        raise NotImplementedError

    def next(self):
        return self.generate(1)


class GIProcess(ArrivalProcess):
    def __init__(self, dist):
        if dist is None:
            raise ValueError('distribution required')
        self._dist = dist

    @property
    def dist(self):
        return self._dist

    def mean(self):
        return self._dist.mean()

    def var(self):
        return self._dist.var()

    def std(self):
        return self._dist.std()

    def cv(self):
        return self._dist.std() / self._dist.mean()

    def lag(self, k):
        return 0.0

    def moment(self, k):
        return self._dist.moment(k)

    def generate(self, size):
        return self._dist.generate(size)


class PoissonProcess(GIProcess):
    def __init__(self, rate):
        if rate <= 0.0:
            raise ValueError("positive rate expected, '{}' found".format(rate))
        super().__init__(Exp(rate))


class MAP(ArrivalProcess):
    @staticmethod
    def erlang(shape, rate):
        """MAP representation of Erlang process with the given shape and rate.

        :param shape number of phases
        :param rate rate at each phase
        """
        d0 = cbdiag(shape, [(0, [[-rate]]), (1, [[rate]])])
        d1 = np.zeros((shape, shape))
        d1[shape-1, 0] = rate
        return MAP(d0, d1)

    @staticmethod
    def exponential(rate):
        """MAP representation of a Poisson process with the given rate.

        :param rate exponential distribution rate ($1/\mu$)
        """
        if rate <= 0:
            raise ValueError("positive rate expected, '{}' found".format(rate))
        d0 = [[-rate]]
        d1 = [[rate]]
        return MAP(d0, d1)

    def __init__(self, d0, d1, check=True, rtol=1e-5, atol=1e-6):
        """Create MAP process with the given D0 and D1 matrices.

        If `check == True` is set (this is default), then we validate matrices
        D0 and D1. These matrices must comply three requirements:

        1) Sum `D0 + D1` must be an infinitesimal matrix
        2) All elements of D1 must be non-negative
        3) All elements of D0 except the main diagonal must be non-negative

        To avoid problems with "1e-9 is not zero", constructor accepts
        parameters `rtol` (relative tolerance) and `atol` (absolute tolerance):

        - any value `x: |x| <= atol` is treated as `0`
        - any value `x: x >= -atol` is treated as non-negative
        - any value `x: x <= atol` is treated as non-positive

        Sometimes it is useful to avoid matrices validation (e.g., when MAP
        matrices are obtained as a result of another computation). To disable
        validation, set `check = False`.

        :param d0 matrix of "invisible" transitions
        :param d1 matrix of "visible" transitions
        :param check set to `False` to skip matrices D0 and D1 validation
        :param rtol relative tolerance used when validating D0 and D1
        :param atol absolute tolerance used when validating D0 and D1
        """
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
        self.__state = None  # used in __call__()
    
    def copy(self):
        """Build a new MAP with the same matrices D0 and D1 without validation.
        """
        return MAP(self.D0, self.D1, check=False)

    # noinspection PyPep8Naming
    @property
    def D0(self):
        """Get matrix D0."""
        return self._d0

    # noinspection PyPep8Naming
    @property
    def D1(self):
        """Get matrix D1."""
        return self._d1

    # noinspection PyPep8Naming
    def D(self, n):
        """Get matrix Dn for n = 0 or n = 1."""
        if n == 0:
            return self.D0
        elif n == 1:
            return self.D1
        else:
            raise ValueError("illegal n={} found".format(n))

    @property
    @lru_cache
    def generator(self):
        return self.D0 + self.D1

    @property
    @lru_cache
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
        return self.var() ** 0.5

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

    def generate(self, size, max_iters_per_sample=None):
        # Building P - a transition probabilities matrix of size Nx(2N), where:
        # - P(i, j), j < N, is a probability to move i -> j without arrival;
        # - P(i, j), N <= j < 2N is a probability to move i -> (j - N) with
        #   arrival
        rates = self.__cache__.get('$generate__rates', -self.D0.diagonal())
        means = self.__cache__.get(
            '$generate__means', np.diag(np.power(rates, -1)))
        p0 = self.__cache__.get(
            '$generate__p0', means.dot(self.D0 + np.diag(rates)))
        p1 = self.__cache__.get('$generate__p1', means.dot(self.D1))
        p = self.__cache__.get('$generate__p', np.hstack([p0, p1]))

        self.__cache__.update({
            '$generate__rates': rates,
            '$generate__means': means,
            '$generate__p0': p0,
            '$generate__p1': p1,
            '$generate__p': p,
        })

        if self.__state is None:
            self.__state = np.random.choice(
                range(self.order), p=(np.ones(self.order) / self.order)
            )

        # Yielding random intervals
        arrival_interval = 0.0
        num_generated = 0
        all_states = list(range(2 * self.order))

        it = 0
        while (num_generated < size and 
               (max_iters_per_sample is None or it < max_iters_per_sample)):
            arrival_interval += np.random.exponential(1 / rates[self.__state])
            next_state = np.random.choice(all_states, p=p[self.__state])
            if next_state >= self.order:
                self.__state = next_state - self.order
                num_generated += 1
                yield arrival_interval
                it = 0
                arrival_interval = 0.0
            else:
                self.__state = next_state
                it += 1
        if num_generated < size:
            raise RuntimeError(
                f'{num_generated}/{size} generated in {it} iterations')
    
    def __call__(self):
        return next(self.generate(1))
    
    def reset_generate_state(self):
        self.__state = None

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
