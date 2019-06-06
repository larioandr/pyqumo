from math import factorial

import numpy as np


class Constant:
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


class Normal:
    def __init__(self, mean, std):
        self.__mean, self.__std = mean, std

    def mean(self):
        return self.__mean

    def std(self):
        return self.__std

    def var(self):
        return self.__std ** 2

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


class PhaseType:
    def __init__(self, s, p):
        pass  # TODO


class MarkovProcess:
    def __init__(self, d0, d1):
        pass  # TODO


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
