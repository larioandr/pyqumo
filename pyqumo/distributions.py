from math import factorial

import numpy as np
import scipy
from scipy import linalg

from pyqumo.matrix import is_pmf, order_of, cached_method, cbdiag


def vectorize(
        otypes=None, doc=None, excluded=None, cache=False, signature=None):
    def wrapper(f):
        return np.vectorize(f, otypes=otypes, doc=doc, excluded=excluded,
                            cache=cache, signature=signature)
    return wrapper


class Distribution:
    """Base class for all continuous distributions"""
    def mean(self) -> float: raise NotImplementedError

    def var(self) -> float: raise NotImplementedError

    def std(self): return np.sqrt(self.var())

    def moment(self, n: int) -> float: raise NotImplementedError

    def generate(self, num): raise NotImplementedError

    def pdf(self, x): raise NotImplementedError

    def cdf(self, x): raise NotImplementedError

    def __call__(self) -> float:
        return next(self.generate(1))

    def sample(self, shape):
        size = np.prod(shape)
        return np.asarray(list(self.generate(size))).reshape(shape)


class Constant(Distribution):
    def __init__(self, value):
        self.__value = value

    def __call__(self):
        return self.__value

    def mean(self):
        return self.__value

    def std(self):
        return 0

    def var(self):
        return 0

    def pdf(self, x):
        return np.inf if x == self.__value else 0

    def cdf(self, x):
        return 1 if x >= self.__value else 0

    def moment(self, k):
        if k <= 0 or np.abs(k - np.round(k)) > 0:
            raise ValueError('positive integer expected')
        return self.__value ** k

    def generate(self, size=1):
        if size > 1:
            return np.asarray([self.__value] * size)
        return self.__value

    def __str__(self):
        return f'const({self.__value})'

    def __repr__(self):
        return str(self)


class Uniform(Distribution):
    def __init__(self, a, b):
        self.a, self.b = a, b
    
    @property
    def min(self):
        return self.a if self.a < self.b else self.b

    @property
    def max(self):
        return self.b if self.a < self.b else self.a
    
    def mean(self):
        return 0.5 * (self.a + self.b)
    
    def std(self):
        return (self.max - self.min) / (12 ** 0.5)
    
    def var(self):
        return (self.max - self.min) ** 2 / 12
    
    def moment(self, k):
        if k == 1:
            return self.mean()
        elif k == 2:
            return self.var() + self.mean() ** 2
        else:
            if k <= 0 or np.abs(k - np.round(k)) > 0:
                raise ValueError('positive integer expected')
            raise ValueError('two moments supported')
    
    def generate(self, num):
        return np.random.uniform(self.min, self.max, size=num)
    
    def pdf(self, x):
        return 1 / (self.max - self.min) if self.min <= x <= self.max else 0
    
    def cdf(self, x):
        return 0 if x < self.min else (
            1 / (self.max - self.min) if x < self.max else 1
        )
    
    def __call__(self):
        return self.generate(None)
    
    def sample(self, shape):
        return np.random.uniform(self.min, self.max, size=shpe)
    
    def __str__(self):
        return f'U({self.min},{self.max})'


class Normal(Distribution):
    def __init__(self, mean, std):
        self.__mean, self.__std = mean, std

    def mean(self):
        return self.__mean

    def std(self):
        return self.__std

    def var(self):
        return self.__std ** 2

    def pdf(self, x):
        return 1 / np.sqrt(2 * np.pi * self.var()) * np.exp(
            -(x - self.mean()) ** 2 / (2 * self.var())
        )

    def cdf(self, x):
        return 0.5 * (1 + np.math.erf((x - self.mean())/(self.std() * 2**0.5)))

    def moment(self, k):
        if k <= 0 or np.abs(k - np.round(k)) > 0:
            raise ValueError('positive integer expected')
        if k > 4:
            raise ValueError('only first four moments supported')
        m, s = self.__mean, self.__std
        if k == 1:
            return m
        elif k == 2:
            return m ** 2 + s ** 2
        elif k == 3:
            return m ** 3 + 3 * m * (s ** 2)
        elif k == 4:
            return m ** 4 + 6 * (m ** 2) * (s ** 2) + 3 * (s ** 4)

    def generate(self, size=1):
        return np.random.normal(self.__mean, self.__std, size=size)

    def __call__(self):
        return np.random.normal(self.__mean, self.__std)

    def __str__(self):
        return f'N({round(self.__mean, 9)},{round(self.__std, 9)})'


class Exponential:
    def __init__(self, mean):
        self.__mean = mean

    def __call__(self):
        return np.random.exponential(self.__mean)

    def generate(self, size=1):
        return np.random.exponential(self.__mean, size=size)

    def mean(self):
        return self.__mean
    
    @property
    def rate(self):
        return 1 / self.__mean

    def std(self):
        return self.__mean

    def var(self):
        return self.__mean ** 2

    def moment(self, k):
        if k <= 0 or np.abs(k - np.round(k)) > 0:
            raise ValueError('positive integer expected')
        return factorial(k) * (self.__mean ** k)

    @property
    def rate(self):
        return 1 / self.__mean

    def __str__(self):
        return f'exp({self.__mean})'

    def __repr__(self):
        return str(self)

    def pdf(self, x):
        return Exp.f(x, self.rate)

    def cdf(self, x):
        return Exp.F(x, self.rate)



class Discrete:
    def __init__(self, values, weights=None):
        if not values:
            raise ValueError('expected non-empty values')
        _weights, _values = [], []
        try:
            # First we assume that values is dictionary. In this case we
            # expect that it stores values in pairs like `value: weight` and
            # iterate through it using `items()` method to fill value and
            # weights arrays:
            for key, weight in values.items():
                _values.append(key)
                _weights.append(weight)
            _weights = np.asarray(_weights)
        except AttributeError:
            # If `values` doesn't have `items()` attribute, we treat as an
            # iterable holding values only. Then we check whether its size
            # matches weights (if provided), and fill weights it they were
            # not provided:
            _values = values
            if weights:
                if len(values) != len(weights):
                    raise ValueError('values and weights size mismatch')
            else:
                weights = (1. / len(values),) * len(values)
            _weights = np.asarray(weights)

        # Check that all weights are non-negative and their sum is positive:
        if np.any(_weights < 0):
            raise ValueError('weights must be non-negative')
        ws = sum(_weights)
        if np.allclose(ws, 0):
            raise ValueError('weights sum must be positive')

        # Normalize weights to get probabilities:
        _probs = tuple(x / ws for x in _weights)

        # Store values and probabilities
        self._values = tuple(_values)
        self._probs = _probs

    def __call__(self):
        return np.random.choice(self._values, p=self.prob)

    @property
    def values(self):
        return self._values

    @property
    def prob(self):
        return self._probs

    def getp(self, value):
        try:
            index = self._values.index(value)
            return self._probs[index]
        except ValueError:
            return 0

    def mean(self):
        return sum(v * p for v, p in zip(self._values, self._probs))

    def moment(self, k):
        if k <= 0 or np.abs(k - np.round(k)) > 0:
            raise ValueError('positive integer expected')
        return sum((v**k) * p for v, p in zip(self._values, self._probs))

    def std(self):
        return self.var() ** 0.5

    def var(self):
        return self.moment(2) - self.mean() ** 2

    def generate(self, size=1):
        return np.random.choice(self._values, p=self.prob, size=size)

    def __str__(self):
        s = '{' + ', '.join(
            [f'{value}: {self.getp(value)}' for value in sorted(self._values)]
        ) + '}'
        return s

    def __repr__(self):
        return str(self)


class LinComb:
    def __init__(self, dists, w=None):
        self._dists = dists
        self._w = w if w is not None else np.ones(len(dists))
        assert len(self._w) == len(dists)

    def mean(self):
        acc = 0
        for w, d in zip(self._w, self._dists):
            try:
                x = w * d.mean()
            except AttributeError:
                x = w * d
            acc += x
        return acc

    def var(self):
        acc = 0
        for w, d in zip(self._w, self._dists):
            try:
                x = (w ** 2) * d.var()
            except AttributeError:
                x = 0
            acc += x
        return acc

    def std(self):
        return self.var() ** 0.5

    def __call__(self):
        acc = 0
        for w, d in zip(self._w, self._dists):
            try:
                x = w * d()
            except TypeError:
                x = w * d
            acc += x
        return acc

    def generate(self, size=None):
        if size is None:
            return self()
        acc = np.zeros(size)
        for w, d in zip(self._w, self._dists):
            try:
                row = d.generate(size)
            except AttributeError:
                row = d * np.ones(size)
            acc += w * row
        return acc

    def __str__(self):
        return ' + '.join(f'{w}*{d}' for w, d in zip(self._w, self._dists))

    def __repr__(self):
        return str(self)


class VarChoice:
    def __init__(self, dists, w=None):
        assert w is None or len(w) == len(dists)
        if w is not None:
            w = np.asarray([round(wi, 5) for wi in w])
            if not np.all(np.asarray(w) >= 0):
                print(w)
                assert False
            assert np.all(np.asarray(w) >= 0)
            assert sum(w) > 0

        self._dists = dists
        self._w = np.asarray(w) if w is not None else np.ones(len(dists))
        self._p = self._w / sum(self._w)

    @property
    def order(self):
        return len(self._dists)

    def mean(self):
        acc = 0
        for p, d in zip(self._p, self._dists):
            try:
                x = p * d.mean()
            except AttributeError:
                x = p * d
            acc += x
        return acc

    def var(self):
        return self.moment(2) - self.mean() ** 2

    def std(self):
        return self.var() ** 0.5

    def moment(self, k):
        acc = 0
        for p, d in zip(self._p, self._dists):
            try:
                x = p * d.moment(k)
            except AttributeError:
                x = p * (d ** k)
            acc += x
        return acc

    def __call__(self):
        index = np.random.choice(self.order, p=self._p)
        try:
            return self._dists[index]()
        except TypeError:
            return self._dists[index]

    def generate(self, size=None):
        if size is None:
            return self()
        return np.asarray([self() for _ in range(size)])

    def __str__(self):
        return (
                '{{' +
                ', '.join(f'{w}: {d}' for w, d in zip(self._p, self._dists)) +
                '}}')

    def __repr__(self):
        return str(self)


class SemiMarkovAbsorb:
    MAX_ITER = 100000

    def __init__(self, mat, time, p0=None):
        mat = np.asarray(mat)
        order = mat.shape[0]
        p0 = np.asarray(p0 if p0 else ([1] + [0] * (order - 1)))

        # Validate initial probabilities and time shapes:
        assert mat.shape == (order, order)
        assert len(p0) == order
        assert len(time) == order

        # Build transitional matrix:
        self._trans_matrix = np.vstack((
            np.hstack((
                mat,
                np.ones((order, 1)) - mat.sum(axis=1).reshape((order, 1))
            )),
            np.asarray([[0] * order + [1]]),
        ))
        assert np.all(self._trans_matrix >= 0)

        # Store values:
        self._mat = np.asarray(mat)
        self._time = time
        self._order = order
        self._p0 = p0

    @property
    def trans_matrix(self):
        return self._trans_matrix

    @property
    def order(self):
        return self._order

    @property
    def absorbing_state(self):
        return self.order

    @property
    def p0(self):
        return self._p0

    def __call__(self):
        order = self._order
        state = np.random.choice(order, p=self._p0)
        it = 0
        time_acc = 0
        while state != self.absorbing_state and it < self.MAX_ITER:
            time_acc += self._time[state]()
            state = np.random.choice(order + 1, p=self._trans_matrix[state])
            it += 1

        if state != self.absorbing_state:
            raise RuntimeError('loop inside semi-markov chain')

        return time_acc

    def generate(self, size=None):
        if size is not None:
            return np.asarray([self() for _ in range(size)])
        return self()


class Exp(Distribution):
    """Exponential random distribution.

    To create a distribution its parameter must be specified. The object is
    immutable.
    """

    def __init__(self, rate):
        super().__init__()
        if rate <= 0.0:
            raise ValueError("exponential parameter must be positive")
        self._param = rate
        self.__cache__ = {}

    @property
    def rate(self):
        """Distribution parameter"""
        return self._param

    def mean(self):
        return 1. / self._param

    def var(self):
        return 1. / (self._param ** 2)

    def std(self):
        return 1. / self._param

    @staticmethod
    @np.vectorize
    def m(n, rate):
        return np.math.factorial(n) / pow(rate, n)

    @cached_method('moment', 1)
    def moment(self, n):
        return Exp.m(n, self.rate)

    @staticmethod
    @np.vectorize
    def f(x, rate):
        if rate <= 0:
            raise ValueError("rate <= 0")
        return rate * pow(np.e, -rate * x) if x >= 0 else 0.0

    def pdf(self, x):
        return self.f(x, self.rate)

    # noinspection PyPep8Naming
    @staticmethod
    @np.vectorize
    def F(x, rate):
        if rate <= 0:
            raise ValueError("rate <= 0")
        return 1.0 - pow(np.e, -rate * x) if x >= 0 else 0.0

    def cdf(self, x):
        return self.F(x, self.rate)

    def generate(self, num):
        for i in range(num):
            yield np.random.exponential(1 / self.rate)

    def __str__(self):
        return "exp({})".format(self.rate)


class Erlang(Distribution):
    """Erlang random distribution.

    To create a distribution its shape (k) and rate (lambda) parameters must
    be specified. Its density function is defined as:

    f(x;k,\\lambda) = \\lambda^k x^(k-1) e^(-\\lambda * x) / (k-1)!
    """

    def __init__(self, shape, rate, tol=1e-08):
        super().__init__()
        if (shape <= tol or shape == np.inf or
                np.abs(np.round(shape) - shape) > tol):
            raise ValueError("shape must be natural integer")
        if rate <= 0.0:
            raise ValueError("rate <= 0")
        self._shape, self._rate = int(np.round(shape)), rate
        self.__cache__ = {}

    @property
    def shape(self):
        return self._shape

    @property
    def rate(self):
        return self._rate

    @cached_method('mean')
    def mean(self):
        return self._shape / self._rate

    @cached_method('var')
    def var(self):
        return self._shape / (self._rate ** 2)

    @cached_method('std')
    def std(self):
        return self.var() ** 0.5

    @staticmethod
    @np.vectorize
    def m(n, shape, rate):
        factors = [i / rate for i in range(shape, shape + n)]
        return np.prod(factors)

    @cached_method('moment', 1)
    def moment(self, n):
        return Erlang.m(n, self.shape, self.rate)

    @staticmethod
    @np.vectorize
    def f(x, shape, rate):
        if rate < 0:
            raise ValueError("rate < 0")
        if shape < 1:
            raise ValueError("shape < 1")
        return (rate * pow(rate * x, shape - 1) * pow(np.e, -rate * x) /
                np.math.factorial(shape - 1)) if 0 <= x < np.inf else 0.0

    def pdf(self, x):
        return Erlang.f(x, self.shape, self.rate)

    # noinspection PyPep8Naming
    @staticmethod
    @np.vectorize
    def F(x, shape, rate):
        if rate < 0:
            raise ValueError("rate < 0")
        if shape < 1:
            raise ValueError("shape < 1")
        elif shape == 1:
            return 1 - pow(np.e, -rate * x)
        else:
            return (Erlang.F(x, shape - 1, rate) -
                    1 / rate * Erlang.f(x, shape, rate))

    def cdf(self, x):
        return Erlang.F(x, self.shape, self.rate)

    def generate(self, num):
        for i in range(num):
            yield sum(np.random.exponential(1 / self.rate, size=self.shape))

    def __str__(self):
        return "Erlang(rate={}, shape={})".format(self.rate, self.shape)


class HyperExp(Distribution):
    """Hyper-exponential distribution.

    Hyper-exponential distribution is defined by:

    - a vector of rates (a1, ..., aN)
    - probabilities mass function (p1, ..., pN)

    Then the resulting probability is a weighted sum of exponential
    distributions Exp(ai) with weights pi:

    $X = \\sum_{i=1}^{N}{p_i X_i}$, where $X_i ~ Exp(ai)$
    """

    def __init__(self, rates, probs):
        rates, probs = np.asarray(rates), np.asarray(probs)
        if len(rates) != len(probs):
            raise ValueError("rates and probs vectors order mismatch")
        if not is_pmf(probs):
            raise ValueError("probs must define a PMF")
        # noinspection PyTypeChecker
        if not (np.all(rates >= 0.0)):
            raise ValueError("rates must be non-negative")
        self._rates, self._pmf0 = rates, probs
        self.__cache__ = {}

    @property
    def rates(self):
        return self._rates

    @property
    def pmf0(self):
        return self._pmf0

    @cached_method('mean')
    def mean(self):
        return sum(self._pmf0 / self._rates)

    @cached_method('var')
    def var(self):
        return sum(self._pmf0 / (self._rates ** 2))

    @cached_method('std')
    def std(self):
        return np.sqrt(self.var())

    @staticmethod
    @vectorize(excluded={1, 2})
    def m(n, rates, pmf0):
        """Compute n-th moment of the hyperexponential distribution.

        Notes:
            this version doesn't support n vectors

        Args:
            n: integer scalar - moment index
            rates: any iterable with positive floats
            pmf0: initial probabilities

        Returns: n-th moment value
        """
        return sum(np.math.factorial(n) * pmf0 / pow(rates, n))

    @cached_method('moment', 1)
    def moment(self, n):
        return HyperExp.m(n, self.rates, self.pmf0)

    @property
    def order(self):
        return len(self._rates)

    def generate(self, size):
        for i in range(size):
            state = np.random.choice(self.order, 1, p=self._pmf0)
            yield np.random.exponential(1. / self._rates[state])

    @staticmethod
    @vectorize(excluded={1, 2})
    def f(x, rates, pmf0):
        state = np.random.choice(len(pmf0), 1, p=pmf0)
        return Exp.f(x, rates[state])

    def pdf(self, x):
        return self.f(x, self.rates, self.pmf0)

    # noinspection PyPep8Naming
    @staticmethod
    @vectorize(excluded={1, 2})
    def F(x, rates, pmf0):
        state = np.random.choice(len(pmf0), 1, p=pmf0)
        return Exp.F(x, rates[state])

    def cdf(self, x):
        return self.F(x, self.rates, self.pmf0)

    def __str__(self):
        return "HyperExp(rates={}, pmf0={})".format(self._rates, self._pmf0)


class PhaseType(Distribution):
    @staticmethod
    def exponential(rate):
        return PhaseType([[-rate]], [1.0])

    @staticmethod
    def erlang(shape, rate):
        s = cbdiag(shape, ((0, [[-rate]]), (1, [[rate]])))
        pmf0 = np.asarray((1.0,) + (0.0,) * (shape - 1))
        return PhaseType(s, pmf0)

    def __init__(self, subgenerator, pmf0):
        self._subgenerator = np.asarray(subgenerator)
        self._pmf0 = np.asarray(pmf0)
        self.__cache__ = {}

    @property
    def order(self):
        return order_of(self.S)

    # noinspection PyPep8Naming
    @property
    def S(self):
        return self._subgenerator

    @property
    def subgenerator(self):
        return self.S

    @property
    def pmf0(self):
        return self._pmf0

    @property
    @cached_method('sni')
    def sni(self):
        return -np.linalg.inv(self.S)

    @property
    def rate(self):
        return 1 / self.mean()

    @cached_method('mean')
    def mean(self):
        return self.moment(1)

    @cached_method('var')
    def var(self):
        return self.moment(2) - self.mean() ** 2

    @cached_method('moment', 1)
    def moment(self, n):
        sni_powered = np.linalg.matrix_power(self.sni, n)
        ones = np.ones(shape=(self.order, 1))
        x = np.math.factorial(n) * self.pmf0.dot(sni_powered).dot(ones)
        return x.item()

    def generate(self, num, max_hop_count=100):
        # Building P - a transition probabilities matrix of size Nx(2N), where:
        # - P(i, j), j < N, is a probability to move i -> j without arrival;
        # - P(i, j), N <= j < 2N is a probability to move i -> (j - N) with
        #   arrival
        n = self.order
        ones = np.ones(n).reshape((n, 1))
        rcol = -self.subgenerator.dot(ones)
        rates = -self.subgenerator.diagonal()
        generator = np.hstack([self.subgenerator + np.diag(rates), rcol])
        means = np.diag(np.power(rates, -1))
        p = means.dot(generator)
        assert isinstance(p, np.ndarray)
        p_line = np.asarray([x if x > 1e-8 else 0 for x in p.flatten()])
        p = p_line.reshape((n, n + 1))
        pmf0 = self.pmf0.flatten()

        # Yielding random intervals
        for i in range(num):
            arrival_interval = 0.0
            hop_count = 0
            state = np.random.choice(range(self.order), p=pmf0)
            while (state < n and
                   (max_hop_count is None or hop_count < max_hop_count)):
                next_state = np.random.choice(range(n + 1), p=p[state])
                arrival_interval += np.random.exponential(1 / rates[state])
                hop_count += 1
                state = next_state
            if state == n:
                yield arrival_interval
            else:
                raise ValueError("failed to get to absorbing state in {} hops"
                                 "".format(hop_count))

    @staticmethod
    @vectorize(excluded={1, 2})
    def f(x, subgenerator, pmf0):
        pmf0 = np.asarray(pmf0)
        ones = np.ones(shape=(order_of(subgenerator)))
        s = np.asarray(subgenerator)
        if 0 <= x < np.inf:
            return pmf0.dot(linalg.expm(x * s)).dot(-s).dot(ones)
        else:
            return 0.0

    def pdf(self, x):
        return PhaseType.f(x, self.S, self.pmf0)

    # noinspection PyPep8Naming
    @staticmethod
    @vectorize(excluded={1, 2})
    def F(x, subgenerator, pmf0):
        pmf0 = np.asarray(pmf0)
        ones = np.ones(shape=(order_of(subgenerator)))
        s = np.asarray(subgenerator)
        if 0 <= x < np.inf:
            return 1 - pmf0.dot(linalg.expm(x * s)).dot(ones)
        elif x <= 0:
            return 0
        else:
            return 1.0

    def cdf(self, x):
        return PhaseType.F(x, self.S, self.pmf0)

    def __str__(self):
        return "PH(S={}, pmf0={})".format(self.S.tolist(), self.pmf0.tolist())


class LinearTransform(Distribution):
    def __init__(self, xi, k, b):
        super().__init__()
        self.__xi = xi
        self.__k = k
        self.__b = b
    
    @property
    def xi(self):
        return self.__xi
    
    @property
    def k(self):
        return self.__k
    
    @property
    def b(self):
        return self.__b

    def mean(self):
        return self.k * self.xi.mean() + self.b

    def var(self):
        return (self.k ** 2) * self.xi.var()

    def std(self):
        return self.k * self.xi.std()

    def moment(self, n: int):
        if n <= 0 or np.abs(n - np.round(n)) > 0:
            raise ValueError('positive integer expected')
        if n == 1:
            return self.mean()
        elif n == 2:
            return self.mean() ** 2 + self.var()
        else:
            raise ValueError('two moments supported')

    def generate(self, num):
        for i in range(num):
            yield self.xi() * self.k + self.b

    def pdf(self, x):
        return self.xi.pdf((x - self.b) / self.k) / self.k

    def cdf(self, x):
        return self.xi.cdf((x - self.b) / self.k)

    def __call__(self) -> float:
        return self.xi() * self.k + self.b

    def sample(self, shape):
        size = np.prod(shape)
        return np.asarray(list(self.generate(size))).reshape(shape)
