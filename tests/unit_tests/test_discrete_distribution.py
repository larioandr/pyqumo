import numpy as np
import pytest

from pyqumo.distributions import Discrete


#
# Validate distribution creation
#
def test_creation_with_empty_values_raises_error():
    with pytest.raises(ValueError) as excinfo:
        Discrete([])
    assert 'expected non-empty values' in str(excinfo.value).lower()


@pytest.mark.parametrize('values,weights', [
    ([10], [0.5, 0.5]),
    ([10, 20], [1.0]),
])
def test_values_and_weights_sizes_must_match(values, weights):
    with pytest.raises(ValueError) as excinfo:
        Discrete(values, weights=weights)
    assert 'values and weights size mismatch' in str(excinfo.value).lower()


def test_weights_must_be_non_negative():
    with pytest.raises(ValueError) as excinfo:
        Discrete([10, 20], [0.5, -0.1])
    assert 'weights must be non-negative' in str(excinfo.value).lower()


def test_weights_sum_must_be_positive():
    with pytest.raises(ValueError) as excinfo:
        Discrete([10, 20], [0, 0])
    assert 'weights sum must be positive' in str(excinfo.value).lower()


@pytest.mark.parametrize('values', [[10], [10, 20], [10, 20, 30]])
def test_values_get_equal_probability_if_weights_not_given(values):
    disc = Discrete(values)
    expected_p = np.asarray([1. / len(values)] * len(values))
    np.testing.assert_almost_equal(disc.values, values)
    np.testing.assert_almost_equal(disc.prob, expected_p)


def test_creating_discrete_distribution_with_non_normalized_weights():
    values = [10, 20, 30]
    weights = [5, 0, 15]
    disc = Discrete(values, weights)
    np.testing.assert_almost_equal(disc.getp(10), 0.25)
    np.testing.assert_almost_equal(disc.getp(20), 0)
    np.testing.assert_almost_equal(disc.getp(30), 0.75)


@pytest.mark.parametrize('values, probs', [
    ({42: 99}, (1.0,)),
    ({42: 13, 34: 13}, (0.5, 0.5)),
    ({10: 0.2, 20: 0.3, 30: 0.5}, (0.2, 0.3, 0.5)),
])
def test_creating_discrete_distribution_with_dictionary(values, probs):
    disc = Discrete(values)
    assert set(disc.values) == set(values.keys())
    np.testing.assert_almost_equal(disc.prob, probs)


#
# Getting probabilities
#
def test_getp_returns_zero_for_any_nonexisting_value():
    disc = Discrete([10, 20])
    assert disc.getp(13) == 0
    assert disc.getp(0) == 0


#
# Counting mean, moment, stddev and variance
#
@pytest.mark.parametrize('values, weights, expected', [
    ([10, 20], None, 15),
    ([10, 20], [1, 4], 18),
    ([10, 20, 30], [0.2, 0.3, 0.5], 23),
])
def test_mean_estimation(values, weights, expected):
    disc = Discrete(values, weights)
    assert disc.mean() == expected


@pytest.mark.parametrize('values, weights, m1, m2, m3', [
    ([10, 20], None, 15, 250, 4500),
    ([10, 20], [1, 4], 18, 340, 6600),
    ([10, 20, 30], [0.2, 0.3, 0.5], 23, 590, 16100),
])
def test_moment_estimation(values, weights, m1, m2, m3):
    disc = Discrete(values, weights)
    assert disc.moment(1) == m1
    assert disc.moment(2) == m2
    assert disc.moment(3) == m3


@pytest.mark.parametrize('k', [-1, 0, 1.1])
def test_negative_or_float_moment_parameter_raises_error(k):
    disc = Discrete([10, 20, 30])
    with pytest.raises(ValueError) as excinfo:
        disc.moment(k)
    assert 'positive integer expected' in str(excinfo.value).lower()


@pytest.mark.parametrize('values, weights, var', [
    ([10, 20], None, 25.0),
    ([10, 20], [1, 4], 16.0),
    ([10, 20, 30], [0.2, 0.3, 0.5], 61.0),
])
def test_stddev_and_var(values, weights, var):
    disc = Discrete(values, weights)
    np.testing.assert_almost_equal(disc.var(), var)
    np.testing.assert_almost_equal(disc.std(), var ** 0.5)


#
# Test values generation
#
@pytest.mark.parametrize('value, n', [(1, 5), (10, 23)])
def test_calling_discrete_distribution_with_single_values_returns_it(value, n):
    disc = Discrete([value])
    samples = [disc() for _ in range(n)]
    assert samples == [value] * n


@pytest.mark.parametrize('values, weights, n', [
    ([10, 20], None, 1000),
    ([10, 20], [1, 4], 1000),
    ([10, 20, 30], [0.2, 0.3, 0.5], 1000),
])
def test_calling_generates_with_given_pmf(values, weights, n):
    disc = Discrete(values, weights)
    samples = np.asarray([disc() for _ in range(n)])
    est_prob = [sum(samples == i) / len(samples) for i in values]

    np.testing.assert_almost_equal(est_prob, disc.prob, 1)
    assert set(samples) == set(values)


@pytest.mark.parametrize('values, weights, size', [
    ([10, 20], None, 1000),
    ([10, 20], [1, 4], 1000),
    ([10, 20, 30], [0.2, 0.3, 0.5], 1000),
])
def test_generate_produces_np_array_of_given_size(values, weights, size):
    disc = Discrete(values, weights)
    samples = disc.generate(size)
    est_prob = [sum(samples == i) / len(samples) for i in values]

    np.testing.assert_almost_equal(est_prob, disc.prob, 1)
    assert set(samples) == set(values)
