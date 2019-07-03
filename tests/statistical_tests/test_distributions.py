import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_allclose

from pyqumo.distributions import Constant, Exponential, SemiMarkovAbsorb, \
    LinComb, VarChoice, Normal, Uniform, LinearTransform


@pytest.mark.parametrize('value', [34, 42])
def test_constant_distribution(value):
    const = Constant(value)

    # First we check that calling constants return the values as expected:
    assert const() == value

    # Then we check that we can get mean, std and moments:
    assert const.mean() == value
    assert const.std() == 0
    assert const.var() == 0
    assert const.moment(1) == value
    assert const.moment(2) == value ** 2
    assert const.moment(3) == value ** 3

    # To be consistent, validate that moment argument is a natural number:
    for k in (0, -1, 2.3):
        with pytest.raises(ValueError) as excinfo:
            const.moment(k)
        assert 'positive integer expected' in str(excinfo.value).lower()

    # Then we make sure that subsequent calls still return the correct values:
    _values = list(set([const() for _ in range(10)]))
    assert _values == [value]

    # Also make sure that the distribution provides generate() method ...
    _values = const.generate(size=13)
    assert  list(_values) == [value] * 13
    assert isinstance(_values, np.ndarray)

    # ... and by default it generates a single value:
    assert const.generate() == value

    # Finally, we make sure that str is implemented and contains value:
    assert str(value) in str(const)


@pytest.mark.parametrize('mean', [34, 42])
def test_exponential_distribution(mean):
    dist = Exponential(mean)

    # First we validate the distribution properties:
    assert_almost_equal(dist.mean(), mean)
    assert_almost_equal(dist.std(), mean)
    assert_almost_equal(dist.var(), mean ** 2)
    assert_almost_equal(dist.moment(1), mean)
    assert_almost_equal(dist.moment(2), 2 * (mean ** 2))
    assert_almost_equal(dist.moment(3), 6 * (mean ** 3))

    # To be consistent, validate that moment argument is a natural number:
    for k in (0, -1, 2.3):
        with pytest.raises(ValueError) as excinfo:
            dist.moment(k)
        assert 'positive integer expected' in str(excinfo.value).lower()

    # Check also this distribution has rate property:
    assert_almost_equal(dist.rate, 1 / mean)

    # Generate enough items using call method, validate they have the expected
    # statistical properties:
    _values = np.asarray([dist() for _ in range(500)])
    assert_allclose(_values.mean(), mean, rtol=0.25)
    assert_allclose(_values.std(), mean, rtol=0.25)

    # Check that we also have generate() method:
    _values = dist.generate(size=500)
    assert isinstance(_values, np.ndarray)
    assert_allclose(_values.mean(), mean, rtol=0.25)
    assert_allclose(_values.std(), mean, rtol=0.25)

    # Finally, we make sure that str is implemented and contains value:
    assert str(mean) in str(dist)

    r = 1 / mean
    assert_allclose(dist.cdf(mean * np.log(2)), 0.5)
    assert_allclose(dist.cdf(0), 0)
    assert_allclose(dist.cdf(1), 1 - np.exp(-r))
    
    assert_allclose(dist.pdf(mean * np.log(2)), r/2)
    assert_allclose(dist.pdf(0), r)
    assert_allclose(dist.pdf(1), r * np.exp(-r))


def test_semi_markov_absorb():
    mat = [[0, 0.5, 0.5], [0, 0, 0], [0, 0, 0]]
    time = [Constant(10), Exponential(5), Exponential(20)]
    initial_prob = [1, 0, 0]
    dist = SemiMarkovAbsorb(mat=mat, time=time, p0=initial_prob)

    # First we make sure that the (full) transitional matrix and absorbing
    # state are calculated properly:
    assert_allclose(
        dist.trans_matrix,
        [[0, 0.5, 0.5, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
    )
    assert dist.absorbing_state == 3
    assert dist.order == 3

    # Check that the mean value matches the expected:
    expected_mean = 10 + 0.5 * 5 + 0.5 * 20
    # assert_almost_equal(dist.mean(), expected_mean)

    # Then we check that we can generate samples and estimate their properties:
    samples = np.asarray([dist() for _ in range(500)])
    assert_allclose(samples.mean(), expected_mean, rtol=0.15)


@pytest.mark.parametrize('dists, w, mean, var, std', [
    ([Exponential(10)], None, 10, 100, 10),
    ([Exponential(3), Exponential(4)], None, 7, 25, 5),
    ([Constant(1), 2, Exponential(3)], [4, 8, 2], 26, 36, 6),
])
def test_lin_comb(dists, w, mean, var, std):
    lc = LinComb(dists, w)

    assert_allclose(lc.mean(), mean, rtol=0.05)
    assert_allclose(lc.var(), var, rtol=0.05)
    assert_allclose(lc.std(), std, rtol=0.05)

    samples = lc.generate(2000)
    assert_allclose(samples.mean(), mean, rtol=0.2)
    assert_allclose(samples.var(), var, rtol=0.2)
    assert_allclose(samples.std(), std, rtol=0.2)


@pytest.mark.parametrize('dists, w', [
    ([Exponential(1)], None),
    ([Exponential(2), Exponential(3)], None),
    ([Exponential(2), 4, Constant(3)], [0.1, 0.4, 0.5]),
    ([Exponential(3), Normal(10, 1)], [10, 2])
])
def test_var_choice(dists, w):
    vc = VarChoice(dists, w)

    n = len(dists)
    p = [1/n] * n if w is None else list(np.asarray(w) / sum(w))
    d = [dist if hasattr(dist, 'moment') else Constant(dist) for dist in dists]
    m1 = sum(p[i] * d[i].mean() for i in range(n))
    m2 = sum(p[i] * d[i].moment(2) for i in range(n))
    m3 = sum(p[i] * d[i].moment(3) for i in range(n))
    m4 = sum(p[i] * d[i].moment(4) for i in range(n))
    var = m2 - (m1 ** 2)

    assert_allclose(vc.mean(), m1)
    assert_allclose(vc.var(), var)
    assert_allclose(vc.std(), var ** 0.5)

    assert_allclose(vc.moment(1), m1)
    assert_allclose(vc.moment(2), m2)
    assert_allclose(vc.moment(3), m3)
    assert_allclose(vc.moment(4), m4)

    # To be consistent, validate that moment argument is a natural number:
    for k in (0, -1, 2.3):
        with pytest.raises(ValueError) as excinfo:
            vc.moment(k)
        assert 'positive integer expected' in str(excinfo.value).lower()

    samples = vc.generate(1000)
    assert_allclose(samples.mean(), m1, rtol=0.1)


def test_var_choice_generate_provides_values_from_distributions_only():
    vc = VarChoice([Constant(34), Constant(42)])
    samples = [vc() for _ in range(100)]
    values = set(samples)
    assert values == {34, 42}


@pytest.mark.parametrize('mean,std', [(10, 2), (2.3, 1.1)])
def test_normal_distribution(mean, std):
    dist = Normal(mean, std)

    m1 = mean
    m2 = mean ** 2 + std ** 2
    m3 = mean ** 3 + 3 * mean * (std ** 2)
    m4 = mean ** 4 + 6 * (mean ** 2) * (std ** 2) + 3 * (std ** 4)

    assert_allclose(dist.mean(), mean)
    assert_allclose(dist.var(), std**2)
    assert_allclose(dist.std(), std)
    assert_allclose(dist.moment(1), m1)
    assert_allclose(dist.moment(2), m2)
    assert_allclose(dist.moment(3), m3)
    assert_allclose(dist.moment(4), m4)

    with pytest.raises(ValueError) as excinfo:
        dist.moment(5)
    assert 'four moments supported' in str(excinfo.value).lower()

    # To be consistent, validate that moment argument is a natural number:
    for k in (0, -1, 2.3):
        with pytest.raises(ValueError) as excinfo:
            dist.moment(k)
        assert 'positive integer expected' in str(excinfo.value).lower()

    samples = dist.generate(2000)
    assert_allclose(samples.mean(), mean, rtol=0.2)
    assert_allclose(samples.std(), std, rtol=0.2)

    samples = np.asarray([dist() for _ in range(1000)])
    assert_allclose(samples.mean(), mean, rtol=0.2)
    assert_allclose(samples.std(), std, rtol=0.2)

    assert str(dist) == f'N({mean},{std})'


@pytest.mark.parametrize('a,b', [(0, 10), (23, 24), (30, 20)])
def test_uniform_distribution(a, b):
    dist = Uniform(a, b)

    m1 = 0.5 * (a + b)
    var = np.abs(b - a) ** 2 / 12
    std = var ** 0.5
    m2 = m1 ** 2 + var

    assert_allclose(dist.mean(), m1)
    assert_allclose(dist.var(), var)
    assert_allclose(dist.std(), std)
    assert_allclose(dist.moment(1), m1)
    assert_allclose(dist.moment(2), m2)

    with pytest.raises(ValueError) as excinfo:
        dist.moment(3)
    assert 'two moments supported' in str(excinfo.value).lower()
    
    # To be consistent, validate that moment argument is a natural number:
    for k in (0, -1, 2.3):
        with pytest.raises(ValueError) as excinfo:
            dist.moment(k)
        assert 'positive integer expected' in str(excinfo.value).lower()
    
    samples = dist.generate(2000)
    assert_allclose(samples.mean(), m1, rtol=0.2)
    assert_allclose(samples.std(), std, rtol=0.2)

    samples = np.asarray([dist() for _ in range(1000)])
    assert_allclose(samples.mean(), m1, rtol=0.2)
    assert_allclose(samples.std(), std, rtol=0.2)

    assert str(dist) == f'U({min(a, b)},{max(a, b)})'


@pytest.mark.parametrize('xi,koef,offset', [
    (Exponential(1), 1.0, 0.0),
    (Exponential(2), 4.0, 3.0),
])
def test_linear_transform(xi, koef, offset):
    dist = LinearTransform(xi, koef, offset)

    mean, std = xi.mean() * koef + offset, xi.std() * koef
    moments = [mean, mean ** 2 + std ** 2]
    
    assert_allclose(dist.mean(), mean)
    assert_allclose(dist.std(), std)
    assert_allclose(dist.var(), std ** 2)
    assert_allclose([dist.moment(1), dist.moment(2)], moments)

    with pytest.raises(ValueError) as excinfo:
        dist.moment(3)
    assert 'two moments supported' in str(excinfo.value).lower()
    
    # To be consistent, validate that moment argument is a natural number:
    for k in (0, -1, 2.3):
        with pytest.raises(ValueError) as excinfo:
            dist.moment(k)
        assert 'positive integer expected' in str(excinfo.value).lower()

    assert_allclose(dist.cdf(offset), xi.cdf(0))
    assert_allclose(dist.cdf(offset + koef), xi.cdf(1))

    print(f'dist.pdf(b) = {dist.pdf(offset)}')
    print(f'xi.pdf(0) = {xi.pdf(0)}, k = {koef}')

    assert_allclose(dist.pdf(offset), xi.pdf(0) / koef)
    assert_allclose(dist.pdf(offset + koef), xi.pdf(1) / koef)

    samples = np.asarray(list(dist.generate(50000)))
    assert_allclose(samples.mean(), dist.mean(), rtol=0.1)
    assert_allclose(samples.std(), dist.std(), rtol=0.1)

    samples = np.asarray([dist() for _ in range(50000)])
    assert_allclose(samples.mean(), dist.mean(), rtol=0.1)
    assert_allclose(samples.std(), dist.std(), rtol=0.1)

