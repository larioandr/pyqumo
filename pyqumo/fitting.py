import numpy as np
import numpy.matlib as ml
import math
import sys
import scipy.optimize

from IPython.display import clear_output

from pyqumo.arrivals import MAP
from pyqumo.chains import DTMC
from pyqumo.matrix import cbdiag
from pyqumo import stats
from pyqumo.distributions import PhaseType


def fit_ph(source, order, method='opt', verbose=False, options=None):
    """Fit a PH distribution of a given order from a trace or from another
    distribution (in the latter case, it must provide methods `moment(k)`
    and `generate(n)`).

    Two methods are supported:
    - non-linear optimization
    - G-Fit

    If `source` is a distribution (another `pyqunet.distributions.PhaseType`),
    then in the first case, the PH will be reduced using analytically computed
    moments, while in the latter case a trace will be generated and the
    EM algorithm will be used.

    Args:
        source (array-like): a list of samples
        order: an order of PH to fit
        method: 'opt' or 'gfit'
        verbose: if `True`, progress will be printed to the screen
        options (dict): provides additional method-related options:
            - x0: initial guess, default: `None` (OPT, GFIT)
            - loss: loss function, see `scipy.optimize.least_squares` (OPT)
            - numMoments: number of moments to use for fitting, default: 3 (OPT)
            - weights: sample weights vector, default: `None` (GFIT)
            - maxIter: max number of iterations, default: 200 (GFIT)
            - stopCond: stop condition, default: 1e-7 (GFIT)
            - numSamples: number of samples to generate into a trace,
                default: 20'000 (GFIT, used when source is a distribution)

    Returns:
        phase-type distribution, `pyqunet.distributions.PH`
    """
    if options is None:
        options = {}
    if method == 'opt':
        x0 = options.get('x0', None)         # initial guess
        loss = options.get('loss', None)     # 'cauchy', ...
        maxn = options.get('numMoments', 3)  # number of moments
        moments = stats.moment(source, maxn)
        return fit_ph_nonlinear_opt(moments, order, x0, loss)
    elif method == 'gfit':
        x0 = options.get('x0', None)
        weights = options.get('weights', None)
        max_iter = options.get('maxIter', 200)
        stop_cond = options.get('stopCond', 1e-7)
        num_samples = options.get('numSamples', 20000)
        if hasattr(source, 'generate'):
            trace = list(source.generate(num_samples))
        else:
            trace = list(source)
        tau, s = PHFromTrace(trace, order, weights, max_iter, stop_cond, x0,
                             result='vecmat', retlogli=False, verbose=verbose)
        return PhaseType(s, tau)
    else:
        raise ValueError("method '{}' not supported, use 'opt', 'gfit'".format(
            method))


def fit_ph_moments(moments, order, method='opt', options=None):
    """
    Fit a PH distribution given a set of the first $k$ moments

    Args:
        moments (array-like): a list (or tuple, ndarray) of the first k
            moments.
        order: an order of PH to fit
        method: only 'opt' is supported
        options (dict): provides additional method-related options:
            - x0: initial guess, default: `None` (OPT)
            - loss: loss function, see `scipy.optimize.least_squares` (OPT)
            - numMoments: number of moments to use for fitting; if `None`,
                then all the given moments will be used, default: `None` (OPT)

    Returns:
        phase-type distribution, `pyqunet.distributions.PH`
    """
    if options is None:
        options = {}
    if method == 'opt':
        x0 = options.get('x0', None)         # initial guess
        loss = options.get('loss', None)     # 'cauchy', ...
        maxn = options.get('numMoments', len(moments))  # number of moments
        return fit_ph_nonlinear_opt(moments[:maxn], order, x0, loss)
    else:
        raise ValueError("method '{}' not supported, use 'opt'".format(method))


def fit_ph_nonlinear_opt(moments, order=3, x0=None, loss=None):
    """
    Fits PH for the given moments using non-linear optimization.

    Args:
        moments:
        order:
        x0:
        loss:

    Returns: PH distribution
    """
    def decompose(x, n):
        tau = x[:n]
        s = np.zeros((n, n))
        for i in range(n):
            row = x[(i + 1) * n: (i + 2) * n]
            s[i] = np.concatenate((row[:i], [-np.sum(row)], row[i:n - 1]))
        return s, tau

    def residual(x, n, input_moments):
        ph = PhaseType(*decompose(x, n))
        estimated_moments = [ph.moment(i + 1)
                             for i in range(len(input_moments))]
        estimated_moments = np.asarray(estimated_moments)
        return np.r_[estimated_moments - input_moments,
                     10 * (1 - ph.pmf0.sum())]

    def _x(v1, v2, a_order):
        return [v1] * a_order + [v2] * (a_order * a_order)

    moments = np.asarray(moments)
    normalized_moments, mu = stats.normalize_moments(moments)

    params = {
        'fun': residual,
        'x0': x0 if x0 is not None else np.array(_x(1. / order, 1., order)),
        'bounds': (_x(0., 0., order), _x(1., np.inf, order)),
        'kwargs': {'input_moments': normalized_moments, 'n': order, },
    }
    if loss is not None:
        params['loss'] = loss

    result = scipy.optimize.least_squares(**params)
    # noinspection PyUnresolvedReferences
    normalized_subgenerator, pmf0 = decompose(result.x, order)

    # Normalizing PMF0 (why could they be greater than 1???)
    sum_prob = sum(pmf0)
    if np.abs(sum_prob - 1.0) > 1e-18:
        pmf0 = pmf0 / sum_prob

    subgenerator = normalized_subgenerator / mu
    return PhaseType(subgenerator, pmf0)


# noinspection PyPep8Naming,PyIncorrectDocstring,PyUnusedLocal
def PHFromTrace(trace, orders, weights=None, maxIter=200, stopCond=1e-7,
                initial=None, result="vecmat", retlogli=True, verbose=False,
                searchOptions=None):
    """
    Performs PH distribution fitting using the EM algorithm
    (G-FIT, [1]_).

    This function is a part of BuTools.
    @author: Gabor Horvath

    Parameters
    ----------
    trace : column vector, length K
        The samples of the trace
    orders : list of int, length(N), or int
        The length of the list determines the number of
        Erlang branches to use in the fitting method.
        The entries of the list are the orders of the
        Erlang distributions. If this parameter is a
        single integer, all possible branch number - order
        combinations are tested where the total number of
        states is "orders".
    weights : TODO
    maxIter : int, optional
        Maximum number of iterations. The default value is
        200
    stopCond : double, optional
        The algorithm stops if the relative improvement of
        the log likelihood falls below stopCond. The
        default value is 1e-7
    initial : tuple of two vectors, optional
        The initial values of the branch probabilities and
        rate parameters is given by this tuple. If not
        given, a default initial guess is determined and
        the algorithm starts from there.
    result : {"vecmat", "vecvec"}, optional
        The result can be returned two ways. If "vecmat" is
        selected, the result is returned in the classical
        representation of phase-type distributions, thus the
        initial vector and the generator matrix.
        If "vecvec" is selected, two vectors are returned,
        one holds the branch probabilities, and the second
        holds the rate parameters of the Erlang branches.
        The default value is "vecmat"
    retlogli: TODO
    verbose: TODO
    searchOptions (dict):
        - mode: 'precise', 'fast', 'adaptive' (default)
        - adaptiveOrder: if `orders` is a scalar and less or equal to this
            value, all combinations will be examined; otherwise fast technique
            will be used.
        - fastMaxIter: 100 (default)
        - fastStopCond: 1e-3 (default)

    Returns
    -------
    (alpha, A) : tuple of matrix, shape (1,M) and matrix, shape (M,M)
        If the "vecmat" result format is chosen, the function
        returns the initial probability vector and the
        generator matrix of the phase type distribution.
    (pi, lambda) : tuple of vector, length N and vector, length N
        If the "vecvec" result format is chosen, the function
        returns the vector of branch probabilities and the
        vector of branch rates in a tuple.
    logli : double
        The log-likelihood divided by the trace length

    Notes
    -----
    This procedure is quite fast in the supported
    mathematical frameworks. If the maximum speed is
    needed, please use the multi-core optimized c++
    implementation called SPEM-FIT_.

    .. _SPEM-FIT: https://bitbucket.org/ghorvath78/spemfit

    References
    ----------
    .. [1] Thummler, Axel, Peter Buchholz, and Miklós Telek.
           A novel approach for fitting probability
           distributions to real trace data with the EM
           algorithm. Dependable Systems and Networks, 2005.
    """

    if weights is None:
        weights = []

    def allorders(branches, sumorders):
        if branches == 1:
            return [[sumorders]]
        else:
            o = []
            # noinspection PyShadowingNames
            for i in range(sumorders - branches + 1):
                x = allorders(branches - 1, sumorders - i - 1)
                for j in range(len(x)):
                    xt = x[j]
                    xt.append(i + 1)
                    xt.sort()
                    # check if we have it already
                    if o.count(xt) == 0:
                        o.append(xt)
            return o

    if searchOptions is None:
        searchOptions = {}
    # TODO: implement search options

    if type(orders) is int:
        bestres = ([], [], -np.inf)
        for br in range(2, orders + 1):
            allord = allorders(br, orders)
            for ordk in allord:
                res = PHFromTrace(trace, ordk, weights, maxIter, stopCond,
                                  initial, result, True)
                if res[2] > bestres[2]:
                    bestres = res
        if retlogli:
            return bestres
        else:
            return bestres[0], bestres[1]

    trace = np.asarray(trace)

    M = len(orders)
    K = len(trace)

    # initial alpha and lambda is such that the mean is matched
    if initial is None:
        alphav = np.ones(M) / M
        lambd = orders * np.linspace(1, M, M)
        trm = np.sum(trace) / len(trace)
        inim = np.sum(alphav / np.linspace(1, M, M))
        lambd = lambd * inim / trm
    elif len(initial) == 2:
        if len(initial[0]) == M and len(initial[1]) == M:
            alphav = initial[0]
            lambd = initial[1]
        else:
            raise Exception(
                "The length of the initial branch probability and rate vectors "
                "is not consistent with the length of the orders vector!")
    else:
        raise Exception("Invalid initial branch probability and rate vectors!")

    if len(weights) == 0:
        weights = np.ones(len(trace)) / K
    W = weights / np.sum(weights)

    Q = np.zeros((M, K))
    logli, ologli = 1e-14, 1
    steps = 1
    while abs((ologli - logli) / logli) > stopCond and steps <= maxIter:
        ologli = logli
        # E-step:
        for i in range(M):
            Q[i, :] = (alphav[i] * (lambd[i] * trace) ** (
                orders[i] - 1) / math.factorial(orders[i] - 1) *
                       lambd[i]) * np.exp(-lambd[i] * trace)
        nor = np.sum(Q, 0)
        for i in range(M):
            Q[i, :] /= nor
        logli = np.log(nor).dot(W)  # np.sum(np.log(nor)) / K
        if verbose and steps % 10 == 0:
            clear_output()
            print("iteration: ", steps, ", logli: ", logli)
            sys.stdout.flush()
        # M-step:
        v1 = Q.dot(W)  # np.sum(Q,1)
        v2 = Q.dot(trace * W)  # Q.dot(trace)
        alphav = v1  # v1/K
        lambd = orders * v1 / v2
        steps += 1

    if verbose:
        clear_output()
        print("EM algorithm terminated.", orders)
        print("Num of iterations: ", steps, ", logli: ", logli)
        sys.stdout.flush()

    if result == "vecvec":
        if retlogli:
            return alphav, lambd, logli
        else:
            return alphav, lambd
    elif result == "vecmat":
        # construct the vector and the matrix representation
        N = sum(orders)
        alpha = ml.zeros((1, N))
        A = ml.zeros((N, N))
        ix = 0
        for i in range(M):
            alpha[0, ix] = alphav[i]
            if orders[i] == 1:
                A[ix, ix] = -lambd[i]
            else:
                A[ix:ix + orders[i], ix:ix + orders[i]] = lambd[i] * (
                    np.diag(np.ones(orders[i] - 1), 1) -
                    np.diag(np.ones(orders[i])))
            ix += orders[i]
        if retlogli:
            return alpha, A, logli
        else:
            return alpha, A
    else:
        raise Exception(
            "Unknown result format given! (valid are: vecmat and vecvec)")


def fit_map(source, order, method='opt', verbose=False, options=None, **kwargs):
    """Fit a MAP of a given order from a trace or from another arrival process
    (in the latter case, it must provide methods `moment(k)`, `lag(k)` and
    `generate(n)`).

    Two methods are supported:
    - non-linear optimization
    - independent fitting of D0 as PH using moments and D1 using lag-k
    - EM-procedure

    If `source` is an arrival process, e.g. another `pyqunet.arrivals.MAP`,
    then in the first case, the MAP will be reduced using analytically computed
    moments and lag-k correlations, while in the latter case a trace will be
    generated and the EM algorithm will be used. Behaviour in the second case
    depends on the selected PH-fitting method, see options.

    Args:
        source (array-like): a list of samples
        order: an order of PH to fit
        method: 'opt', 'indi' or 'em'
        verbose: if `True`, progress will be printed to the screen
        options (dict): provides additional method-related options:
            - x0: initial guess, default: `None` (OPT)
            - loss: loss function, see `scipy.optimize.least_squares`
                (OPT, INDI)
            - numMoments: number of moments to use for fitting, default: 3
                (OPT, INDI)
            - numLags: number of lag-k to use for fitting, default: 2
                (OPT, INDI)
            - maxIter: max number of iterations, default: 200 (GFIT, INDI)
            - stopCond: stop condition, default: 1e-7 (GFIT, INDI)
            - numSamples: number of samples to generate into a trace,
                default: 20'000 (GFIT and INDI when source is a distribution)
            - phFitMethod: 'opt' or 'gfit', default: 'opt' (INDI)

    Returns:
        Markovian arrival process, `pyqunet.arrivals.MAP`
    """
    if options is None:
        options = {}
    if method == 'opt':
        x0 = options.get('x0', None)         # initial guess
        loss = options.get('loss', None)     # 'cauchy', ...
        num_moments = options.get('numMoments', 3)
        num_lags = options.get('numLags', 2)
        moments = stats.moment(source, num_moments)
        lags = stats.lag(source, num_lags)
        return fit_map_nonlinear_opt(moments, lags, order, x0, loss)
    elif method == 'em':
        max_iter = options.get('maxIter', 200)
        stop_cond = options.get('stopCond', 1e-7)
        num_samples = options.get('numSamples', 20000)
        if hasattr(source, 'generate'):
            trace = list(source.generate(num_samples))
        else:
            trace = list(source)
        d0, d1 = MAPFromTrace(trace, order, max_iter, stop_cond, initial=None,
                              retlogli=False, verbose=verbose)
        return MAP(d0, d1)
    elif method == 'indi':
        ph_fit_method = options.get('phFitMethod', 'opt')
        num_lags = options.get('numLags', kwargs.get('numLags', 2))
        lags = stats.lag(source, num_lags)
        ph = fit_ph(source, order, ph_fit_method, verbose, options)
        return fit_map_horvath(ph, lags)
    elif method == 'opt-cdf':
        x0 = options.get('x0', None)         # initial guess
        num_components = options.get('numComponents', 3)
        weights = options.get('weights', None)
        return fit_map_cdf(source, num_components, weights, order, x0=x0)
    else:
        raise ValueError("method '{}' not supported, use 'opt', 'gfit'".format(
            method))


def fit_map_nonlinear_opt(moments, lags, order=3, x0=None, loss=None):
    """
    Fits MAP for the given moments using non-linear optimization.

    Args:
        moments:
        lags:
        order:
        x0:
        loss:

    Returns: MAP
    """

    def decompose(x, n):
        a_d0 = np.zeros((n, n))
        a_d1 = x[n * (n - 1):].reshape((n, n))
        for i in range(n):
            row = x[i * (n - 1): (i + 1) * (n - 1)]
            a_d0[i] = np.concatenate(
                (row[:i], [-np.sum(row) - np.sum(a_d1[i])], row[i:n - 1]))
        return a_d0, a_d1

    def residual(x, n, input_moments, input_lags):
        m = MAP(*decompose(x, n), check=False)
        estimated_moments = [m.moment(i + 1) for i in range(len(input_moments))]
        estimated_lags = [m.lag(i + 1) for i in range(len(input_lags))]
        estimated_moments_lags = np.r_[estimated_moments, estimated_lags]
        input_moments_lags = np.r_[input_moments, input_lags]
        return estimated_moments_lags - input_moments_lags

    def _x(v1, v2, a_order):
        return [v1] * a_order * (a_order - 1) + [v2] * (a_order * a_order)

    moments = np.asarray(moments)
    lags = np.asarray(lags)
    normalized_moments, mu = stats.normalize_moments(moments)

    params = {
        'fun': residual,
        'x0': x0 if x0 is not None else np.array(_x(1., 1., order)),
        'bounds': (_x(0., 0, order), _x(np.inf, np.inf, order)),
        'kwargs': {'input_moments': normalized_moments,
                   'input_lags': lags, 'n': order, },
    }
    if loss is not None:
        params['loss'] = loss

    normalized_result = scipy.optimize.least_squares(**params)
    # noinspection PyUnresolvedReferences
    normalized_matrices = decompose(normalized_result.x, order)
    d0, d1 = normalized_matrices[0] / mu, normalized_matrices[1] / mu
    return MAP(d0, d1)


# noinspection PyPep8Naming
def fit_map_cdf(source, num_components, weights=None, order=3, x0=None):
    """
    Fits MAP for the given moments using non-linear optimization.

    Args:
        source:
        num_components:
        weights:
        order:
        x0:

    Returns: MAP
    """

    # noinspection PyPep8Naming
    def pdf_components(a_D0, a_D1, n, K=5):
        one = np.ones(n)
        eye = np.eye(n)
        P = -np.linalg.inv(a_D0).dot(a_D1)

        # Firstly, compute steady-state probability vector
        P = P.T - eye
        P[0] = one
        pi = np.zeros(n)
        pi[0] = 1

        pi = np.linalg.solve(P, pi)

        D0_power = eye
        a_result = [0] * K

        for k in range(K):
            D0_power = D0_power.dot(a_D0)
            a_result[k] = pi.dot(D0_power).dot(one)

        return np.array(a_result)

    def get_weights(t=1.0, k=5):
        w = [0] * k
        factorial = 2

        for ki in range(k):
            # noinspection PyTypeChecker
            w[ki] = t ** (ki + 2) / factorial
            factorial *= (ki + 3)

        return np.array(w)

    def decompose(x, n):
        a_d0 = np.zeros((n, n))
        a_d1 = x[n * (n - 1):].reshape((n, n))
        for i in range(n):
            row = x[i * (n - 1): (i + 1) * (n - 1)]
            a_d0[i] = np.concatenate(
                (row[:i], [-np.sum(row) - np.sum(a_d1[i])], row[i:n - 1]))
        return a_d0, a_d1

    # noinspection PyPep8Naming
    def residual(x, n, components, ws):
        a_D0, a_D1 = decompose(x, n)
        estimated = pdf_components(a_D0, a_D1, n, components.size)
        return (estimated - components) * ws

    assert isinstance(source, MAP)
    x_size = (2 * order - 1) * order
    map_components = pdf_components(
        source.D0, source.D1, source.order, num_components)

    if weights is None:
        weights = get_weights(1, num_components)
    elif isinstance(weights, int) or isinstance(weights, float):
        weights = get_weights(float(weights), num_components)

    params = {
        'fun': residual,
        'x0': x0 if x0 is not None else np.array([1] * x_size),
        'bounds': ([0] * x_size, [np.inf] * x_size),
        'kwargs': {'components': map_components, 'ws': weights, 'n': order, },
    }

    result = scipy.optimize.least_squares(**params)
    # noinspection PyUnresolvedReferences
    ret_D0, ret_D1 = decompose(result.x, order)
    return MAP(ret_D0, ret_D1)


# noinspection PyPep8Naming
def fit_map_horvath(ph, lags):
    """Find D1 matrix using a D0 as subgenerator of the given PH and lags.

    Args:
        ph (`pyqunet.distributions.PH`): a PH distribution approximating D0
        lags (array-like): a list of lag-k auto-correlation coefficients
    """
    if len(lags) > 1:
        return fit_map_horvath(ph, [lags[0]])

    N = ph.order
    D0 = ph.subgenerator
    En = np.ones((N, 1))
    pi = ph.pmf0

    num_lags = len(lags)
    if num_lags == 0:
        D1_row_sum = (-D0).dot(En).reshape(N)
        D1 = np.asarray([D1_row_sum[i] * pi for i in range(N)])
        return MAP(D0, D1)

    ph_moments = stats.moment(ph, 2)
    ph_moments, mu = stats.normalize_moments(ph_moments)

    D0ni = np.linalg.inv(-D0) / mu
    D0ni2 = D0ni.dot(D0ni)
    rate = ph.rate * mu
    lag1 = lags[0]

    d = (-D0 * mu).dot(En).reshape((N, 1))
    gamma = pi.dot(D0ni).reshape((1, N))
    block_gamma = cbdiag(N, [(0, gamma)])
    block_eye = np.hstack([np.eye(N)] * N)
    A = np.vstack([block_eye, block_gamma])
    b = np.vstack([d, pi.reshape((N, 1))])

    delta = pow(rate, 2) * pi.dot(D0ni2)
    f = D0ni.dot(En)
    # noinspection PyTypeChecker
    v = lag1 * (2 * pow(rate, 2) * pi.dot(D0ni2).dot(En).reshape(1)[0] - 1) + 1
    c = np.hstack([f[i] * delta for i in range(N)])

    if num_lags == 1:
        A = np.vstack([A, c])
        b = np.vstack([b, [[v]]]).reshape(2 * N + 1)
        ret = scipy.optimize.lsq_linear(A, b, (0, np.inf), tol=1e-10,
                                        method='bvls')
        # noinspection PyUnresolvedReferences
        x = ret.x
        assert isinstance(x, np.ndarray)
        D1 = x.reshape((N, N)).transpose() / mu

        try:
            return MAP(D0, D1, rtol=1e-3, atol=1e-4)
        except ValueError:
            if np.abs(lags[0] < 1e-5):
                return fit_map_horvath(ph, [])
            else:
                return fit_map_horvath(ph, [lags[0] * 0.5])
    else:

        def residual(xi, n, input_lags):
            d1i = xi.reshape(n, n).transpose() / mu
            m = MAP(D0, d1i, check=False)
            estimated_lags = [m.lag(i + 1) for i in range(len(input_lags))]
            system_diff = np.asarray(A.dot(xi) - b).flatten() * 1000
            lags_diff = np.asarray(input_lags - estimated_lags)
            diff = np.hstack((system_diff, lags_diff))
            return diff

        lags = np.asarray(lags)

        params = {
            'fun': residual,
            'x0': np.asarray([d[i] * pi for i in range(N)]
                             ).transpose().flatten(),
            'bounds': (0, np.inf),
            'kwargs': {'input_lags': lags, 'n': N, },
        }

        result = scipy.optimize.least_squares(**params)
        # noinspection PyUnresolvedReferences
        D1 = result.x.reshape(N, N).transpose() / mu
        return MAP(D0, D1, check=False)


# noinspection PyPep8Naming
def MAPFromTrace(trace, orders, maxIter=200, stopCond=1e-7, initial=None,
                 result="matmat", retlogli=True, verbose=False):
    """
    Performs MAP fitting using the EM algorithm (ErCHMM,
    [1]_, [2]_).

    This function is a part of BuTools.
    @author: Gabor Horvath

    Parameters
    ----------
    trace : column vector, length K
        The samples of the trace
    orders : list of int, length(N), or int
        The length of the list determines the number of
        Erlang branches to use in the fitting method.
        The entries of the list are the orders of the
        Erlang distributions. If this parameter is a
        single integer, all possible branch number - order
        combinations are tested where the total number of
        states is "orders".
    maxIter : int, optional
        Maximum number of iterations. The default value is
        200
    stopCond : double, optional
        The algorithm stops if the relative improvement of
        the log likelihood falls below stopCond. The
        default value is 1e-7
    initial : tuple of a vector and a matrix, shape(N,N), optional
        The rate parameters of the Erlang distributions
        and the branch transition probability matrix to be
        used initially. If not given, a default initial
        guess is determined and the algorithm starts from
        there.
    result : {"vecmat", "matmat"}, optional
        The result can be returned two ways. If "matmat" is
        selected, the result is returned in the classical
        representation of MAPs, thus the D0 and D1 matrices.
        If "vecmat" is selected, the rate parameters of the
        Erlang branches and the branch transition probability
        matrix are returned. The default value is "matmat"
    retlogli: TODO
    verbose: TODO

    Returns
    -------
    (D0, D1) : tuple of matrix, shape (M,M) and matrix, shape (M,M)
        If the "matmat" result format is chosen, the function
        returns the D0 and D1 matrices of the MAP
    (lambda, P) : tuple of vector, length N and matrix, shape (M,M)
        If the "vecmat" result format is chosen, the function
        returns the vector of the Erlang rate parameters of
        the branches and the branch transition probability
        matrix
    logli : double
        The log-likelihood divided by the trace length

    Notes
    -----
    This procedure is quite slow in the supported
    mathematical frameworks. If the maximum speed is
    needed, please use the multi-core optimized c++
    implementation called SPEM-FIT_.

    .. _SPEM-FIT: https://bitbucket.org/ghorvath78/spemfit

    References
    ----------
    .. [1] Okamura, Hiroyuki, and Tadashi Dohi. Faster
           maximum likelihood estimation algorithms for
           Markovian arrival processes. Quantitative
           Evaluation of Systems, 2009. QEST'09. Sixth
           International Conference on the. IEEE, 2009.

    .. [2] Horváth, Gábor, and Hiroyuki Okamura. A Fast EM
           Algorithm for Fitting Marked Markovian Arrival
           Processes with a New Special Structure. Computer
           Performance Engineering. Springer Berlin
           Heidelberg, 2013. 119-133.
    """

    # noinspection PyShadowingNames
    def allorders(branches, sumorders):
        if branches == 1:
            return [[sumorders]]
        else:
            o = []
            for index in range(sumorders - branches + 1):
                x = allorders(branches - 1, sumorders - index - 1)
                for j in range(len(x)):
                    xt = x[j]
                    xt.append(index + 1)
                    xt.sort()
                    # check if we have it already
                    if o.count(xt) == 0:
                        o.append(xt)
                        #                    for ok in o:
                        #                        if ok==xt
                        #                            break;
                        #                    else:
                        #                        o.append(xt)
            return o

    if type(orders) is int:
        bestres = ([], [], -np.inf)
        bestOrders = []
        for br in range(2, orders + 1):
            allord = allorders(br, orders)
            for ordk in allord:
                if verbose:
                    print("Trying orders ", ordk)
                res = MAPFromTrace(trace, ordk, maxIter, stopCond, initial,
                                   result, True)
                # noinspection PyUnresolvedReferences
                if res[2] > bestres[2]:
                    bestres = res
                    bestOrders = ordk
        if verbose:
            print("Best solution: logli =", bestres[2], "orders =", bestOrders)
        if retlogli:
            return bestres
        else:
            return bestres[0], bestres[1]

    M = len(orders)
    K = len(trace)
    trace = np.asarray(trace)

    # initial alpha and lambda is such that the mean is matched
    if initial is None:
        alphav = np.ones(M) / M
        lambd = orders * np.linspace(1, M, M)
        trm = np.sum(trace) / len(trace)
        inim = np.sum(alphav / np.linspace(1, M, M))
        lambd = lambd * inim / trm
        # noinspection PyUnresolvedReferences,PyDeprecation
        P = (ml.ones((M, 1)) * ml.matrix([alphav])).A
    elif len(initial) == 2:
        if len(initial[0]) == M and initial[1].shape == (M, M):
            lambd = initial[0]
            P = np.array(initial[1])
            # NOTE: original line was as follows:
            # alphav = DTMCSolve(ml.matrix(P))
            dtmc = DTMC(P)
            alphav = dtmc.steady_pmf()
        else:
            raise Exception(
                "The length of the initial branch probability and rate vectors "
                "is not consistent with the length of the orders vector!")
    else:
        raise Exception("Invalid initial branch probability and rate vectors!")

    Q = np.zeros((M, K))
    A = np.zeros((K, M))
    B = np.zeros((M, K))
    Ascale = np.zeros(K)
    Bscale = np.zeros(K)
    logli, ologli = 1e-14, 0
    steps = 1
    while abs((ologli - logli) / logli) > stopCond and steps <= maxIter:
        ologli = logli
        # E-step:
        for i in range(M):
            Q[i, :] = ((lambd[i] * trace) ** (orders[i] - 1) / math.factorial(
                orders[i] - 1) * lambd[i]) * np.exp(-lambd[i] * trace)
        # forward likelihood vectors:
        prev = alphav
        scprev = 0
        for k in range(K):
            prev = prev.dot(np.diag(Q[:, k])).dot(P)
            scale = math.log2(np.sum(prev))
            prev = prev * 2 ** -scale
            Ascale[k] = scprev + scale
            A[k, :] = prev
            scprev = Ascale[k]
        Av = np.vstack((alphav, A[0:-1, :]))
        Ascalev = np.hstack(([0], Ascale[0:-1]))
        # backward likelihood vectors:
        nnext = np.ones(M)
        scprev = 0
        for k in range(K - 1, -1, -1):
            nnext = np.diag(Q[:, k]).dot(P).dot(nnext)
            scale = math.log2(np.sum(nnext))
            nnext = nnext * 2 ** -scale
            Bscale[k] = scprev + scale
            B[:, k] = nnext
            scprev = Bscale[k]
        Bv = np.hstack((B[:, 1:], np.ones((M, 1))))
        Bscalev = np.hstack((Bscale[1:], [0]))

        llh = alphav.dot(B[:, 0])
        logli = (math.log(llh) + Bscale[0] * math.log(2)) / K
        illh = 1.0 / llh

        # M-step:
        # Calculate new estimates for the parameters
        AB = Av * B.T
        nor = np.sum(AB, 1)
        for m in range(M):
            AB[:, m] /= nor
        v1 = np.sum(AB, 0)
        v2 = AB.T.dot(trace).T
        alphav = v1 / K
        lambd = (orders * v1 / v2).T

        Avv = Av * Q.T
        nor = illh * 2 ** (Ascalev + Bscalev - Bscale[0]).T
        for m in range(M):
            Avv[:, m] *= nor
        P = (Avv.T.dot(Bv.T)) * P
        for m in range(M):
            P[m, :] /= np.sum(P[m, :])

        steps += 1
        if verbose and steps % 10 == 0:
            clear_output()
            print("iteration: ", steps, ", logli: ", logli)
            sys.stdout.flush()

    if verbose:
        clear_output()
        print("Num of iterations: ", steps, ", logli: ", logli)
        print("EM algorithm terminated.", orders)
        sys.stdout.flush()

    if result == "vecmat":
        if retlogli:
            return lambd, P, logli
        else:
            return lambd, P
    elif result == "matmat":
        N = int(np.sum(orders).item())
        D0 = ml.zeros((N, N))
        ix = 0
        for i in range(M):
            if orders[i] == 1:
                D0[ix, ix] = -lambd[i]
            else:
                D0[ix:ix + orders[i], ix:ix + orders[i]] = lambd[i] * (
                    np.diag(np.ones(orders[i] - 1), 1) - np.diag(
                        np.ones(orders[i])))
            ix += orders[i]

        D1 = ml.zeros((N, N))
        indicesTo = np.hstack(([0], np.cumsum(orders[0:-1])))
        indicesFrom = np.cumsum(orders) - 1
        X = np.diag(lambd).dot(P)
        ix = 0
        # noinspection PyTypeChecker
        for i in indicesFrom:
            jx = 0
            for j in indicesTo:
                D1[i, j] = X[ix, jx]
                jx += 1
            ix += 1
        if retlogli:
            return D0, D1, logli
        else:
            return D0, D1
    else:
        raise Exception(
            "Unknown result format given! (valid are: vecmat and vecvec)")
