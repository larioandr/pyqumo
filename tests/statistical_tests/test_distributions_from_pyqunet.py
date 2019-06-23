import pyqumo.distributions as cd

import unittest
import numpy


class TestBase(unittest.TestCase):
    """Base class for all distributions unit tests. Provides an array of
    descriptors and a number of test methods, each of which inspects a part
    of each descriptor.

    Each descriptor is a dictionary containing the following fields:

    - 'distribution': an object of pyqumo.distributions.Distribution,
        which is to be inspected

    Additionally a descriptor may contain other fields (e.g. 'rate', 'pmf' etc.)
    which are specific to a particular distribution.
    """
    tests = []  # list of descriptors

    DEFAULT_GENERATE_SIZE = 10000
    DEFAULT_GENERATE_PRECISION = 1
    DEFAULT_PRECISION = 8
    DEFAULT_GRID_FUNCTION_VECTORIZED = False

    def assertAllClose(self, lvalue, rvalue,
                       places=DEFAULT_PRECISION, msg=None):
        lv = numpy.asarray(lvalue)
        rv = numpy.asarray(rvalue)
        try:
            self.assertAlmostEqual(lv.item(), rv.item(), places, msg)
        except ValueError:
            tol = pow(10.0, -places)
            try:
                self.assertTrue(numpy.allclose(lv, rv, tol, tol),
                                msg + " (tol={})".format(tol))
            except TypeError as err:
                raise TypeError("{}: {}".format(repr(err), msg))
        except Exception as err:
            raise RuntimeError("{} -- {}".format(repr(err), msg))

    def _check_test_descriptor(self, test_descriptor):

        def test_property(dist, descriptor):
            name = descriptor['name']
            expected = descriptor['value']
            precision = descriptor.get('precision', self.DEFAULT_PRECISION)
            try:
                value = getattr(dist, name)
            except Exception as err:
                raise RuntimeError("{} when testing {}.{}".format(
                    repr(err), dist, name))
            self.assertAllClose(
                value, expected, precision, "{}.{}={}, expected={}".format(
                    dist, name, value, expected))

        def test_simple_function(dist, descriptor):
            name = descriptor['name']
            expected = descriptor['value']
            precision = descriptor.get('precision', self.DEFAULT_PRECISION)
            try:
                value = getattr(dist, name)()
            except Exception as err:
                raise RuntimeError("{} when testing {}.{}()".format(
                    repr(err), dist, name))
            self.assertAllClose(
                value, expected, precision, "{}.{}()={}, expected={}".format(
                    dist, name, value, expected))

        def test_grid_function(dist, descriptor):
            name = descriptor['name']
            grid = descriptor['grid']
            precision = descriptor.get('precision', self.DEFAULT_PRECISION)
            vectorized = descriptor.get('vectorized',
                                        self.DEFAULT_GRID_FUNCTION_VECTORIZED)
            f = getattr(dist, name)

            # Check point-by-point (scalar style)
            for point in grid:
                args, expected = point[:-1], point[-1]
                try:
                    value = f(*args)
                except Exception as err:
                    raise RuntimeError("{} when testing {}.{}({})".format(
                        repr(err), dist, name, args))
                self.assertAllClose(value, expected, precision,
                                    "{}.{}({})={}, expected={}".format(
                                        dist, name, args, value, expected))

            if vectorized:
                arg = numpy.asarray([point[:-1] for point in grid])
                expected = numpy.asarray([[point[-1]]for point in grid])
                try:
                    value = f(arg)
                except Exception as err:
                    raise RuntimeError("{} when testing {}.{}({})".format(
                        repr(err), dist, name, arg))
                self.assertAllClose(value, expected, precision,
                                    "{}.{}({})={}, expected={}".format(
                                        dist, name, arg, value, expected))

        def test_complex_function(dist, descriptor):
            name = descriptor['name']
            args = descriptor.get('args', ())
            kwargs = descriptor.get('kwargs', {})
            calls = descriptor['calls']
            generator = descriptor.get('generator', False)
            default_precision = descriptor.get(
                'precision', self.DEFAULT_PRECISION)

            try:
                ret = getattr(dist, name)(*args, **kwargs)
            except Exception as err:
                raise RuntimeError(
                    "{} when testing {}.{}(...) with args={}, kwargs={}".format(
                        repr(err), dist, name, args, kwargs))

            value = list(ret) if generator else ret
            for call_descriptor in calls:
                call_value = call_descriptor['call'](value)
                precision = call_descriptor.get('precision', default_precision)
                if 'as_value' in call_descriptor:
                    expected = call_descriptor['as_value']
                elif 'as_property' in call_descriptor:
                    property_name = call_descriptor['as_property']
                    expected = getattr(dist, property_name)
                elif 'as_function' in call_descriptor:
                    function_name = call_descriptor['as_function']
                    expected = getattr(dist, function_name)()
                elif 'as_call' in call_descriptor:
                    expected = call_descriptor['as_call'](dist)
                else:
                    print(call_descriptor)
                    raise KeyError
                self.assertAllClose(
                    call_value, expected, precision,
                    "{}({}.{}({}))={}, expected={}".format(
                        str(call_descriptor['call']), dist, name, args,
                        call_value, expected))

        def test_like_property(dist, alias, descriptor):
            name = descriptor['name']
            alias_name = descriptor.get('alias', name)
            precision = descriptor.get('precision', self.DEFAULT_PRECISION)
            try:
                value = getattr(dist, name)
            except Exception as err:
                raise RuntimeError("{} when testing {}.{}".format(
                    repr(err), dist, name))
            expected = getattr(alias, alias_name)
            self.assertAlmostEqual(
                value, expected, precision,
                "{}.{}={}, expected={} [alias: {}.{}]".format(
                    dist, name, value, expected, alias, alias_name))

        def test_like_simple_function(dist, alias, descriptor):
            name = descriptor['name']
            alias_name = descriptor.get('alias', name)
            precision = descriptor.get('precision', self.DEFAULT_PRECISION)
            value = getattr(dist, name)()
            expected = getattr(alias, alias_name)()
            self.assertAllClose(
                value, expected, precision,
                "{}.{}()={}, expected={} [alias: {}.{}()]".format(
                    dist, name, value, expected, alias, alias_name))

        def test_like_grid_function(dist, alias, descriptor):
            name = descriptor['name']
            alias_name = descriptor.get('alias', name)
            grid = descriptor['grid']
            precision = descriptor.get('precision', self.DEFAULT_PRECISION)
            vectorized = descriptor.get('vectorized',
                                        self.DEFAULT_GRID_FUNCTION_VECTORIZED)
            f = getattr(dist, name)
            alias_f = getattr(alias, alias_name)

            # Check point-by-point (scalar style)
            for point in grid:
                arg = numpy.asarray(point)
                try:
                    value = f(*arg)
                except TypeError:
                    arg = point
                    try:
                        value = f(arg)
                    except Exception as err:
                        raise RuntimeError("{} when testing {}.{}({})".format(
                            repr(err), dist, name, arg))
                    else:
                        expected = alias_f(arg)
                except Exception as err:
                    raise RuntimeError("{} when testing {}.{}({})".format(
                        repr(err), dist, name, arg))
                else:
                    expected = alias_f(*arg)
                self.assertAllClose(
                    value, expected, precision,
                    "{}.{}({})={}, expected={} [alias: {}.{}(...)]".format(
                        dist, name, arg, value, expected, alias, alias_name))

            if vectorized:
                value = f(grid)
                expected = alias_f(grid)
                self.assertAllClose(
                    value, expected, precision,
                    "{}.{}({})={}, expected={} [alias: {}.{}(...)]".format(
                        dist, name, grid, value, expected, alias, alias_name))

        def test_like(dist, descriptor):
            for alias in descriptor['alias']:
                for prop in descriptor.get('properties', []):
                    test_like_property(dist, alias, prop)
                for fun in descriptor.get('functions', []):
                    try:
                        test_like_grid_function(dist, alias, fun)
                    except KeyError:
                        test_like_simple_function(dist, alias, fun)

        distribution = test_descriptor['distribution']
        for each in test_descriptor.get('properties', []):
            test_property(distribution, each)
        for each in test_descriptor.get('functions', []):
            try:
                test_grid_function(distribution, each)
            except KeyError:
                try:
                    test_simple_function(distribution, each)
                except KeyError:
                    test_complex_function(distribution, each)
        for each in test_descriptor.get('like', []):
            test_like(distribution, each)

    def test_descriptors(self):
        for descriptor in self.tests:
            self._check_test_descriptor(descriptor)


class TestExp(TestBase):
    def setUp(self):
        self.tests = [{
            'distribution': cd.Exp(1.0),
            'properties': [
                {'name': 'rate', 'value': 1.0},
            ],
            'functions': [
                {'name': 'mean', 'value': 1.0},
                {'name': 'std', 'value': 1.0},
                {'name': 'var', 'value': 1.0},
                {'name': 'moment', 'grid': [(1, 1.0), (2, 2.0)]},
                {'name': 'pdf', 'precision': 4,
                 'grid': [(0.0, 1.0), (1.0, 0.36788), (2.0, 0.13533),
                          (numpy.inf, 0.0)]},
                {'name': 'cdf', 'precision': 4,
                 'grid': [(0.0, 0.0), (1.0, 0.63212), (2.0, 0.86466),
                          (numpy.inf, 1.0)]},
                {'name': 'generate', 'args': (25000,), 'precision': 1,
                 'generator': True,
                 'calls': [
                    {'call': numpy.mean, 'as_value': 1.0, 'precision': 1},
                    {'call': numpy.std, 'as_value': 1.0},
                    {'call': numpy.var, 'as_function': 'mean'},
                    {'call': lambda x: 1./numpy.mean(x), 'as_property': 'rate'},
                    {'call': numpy.mean, 'as_call': lambda d: d.mean()}
                ]},
                {'name': 'sample', 'kwargs': {'shape': (25000,)},
                 'precision': 1, 'calls': [
                    {'call': numpy.mean, 'as_function': 'mean'},
                    {'call': numpy.std, 'as_function': 'std'},
                    {'call': lambda x: x.shape, 'as_value': (25000,)}
                ]},
                {'name': 'sample', 'kwargs': {'shape': (2, 1)},
                 'precision': 1, 'calls': [
                    {'call': lambda x: x.shape, 'as_value': (2, 1)}
                ]},
                {'name': 'sample', 'kwargs': {'shape': (5, 6, 7)},
                 'precision': 1, 'calls': [
                    {'call': lambda x: x.shape, 'as_value': (5, 6, 7)}
                ]},
            ]
        }, {
            'distribution': cd.Exp(2.0),
            'properties': [
                {'name': 'rate', 'value': 2.0},
            ],
            'functions': [
                {'name': 'mean', 'value': 0.5}, {'name': 'std', 'value': 0.5},
                {'name': 'var', 'value': 0.25},
                {'name': 'moment', 'grid': [(1, 0.5), (2, 0.5)]},
                {'name': 'pdf', 'precision': 4,
                 'grid': [(0.0, 2.0), (1.0, 0.27067), (2.0, 0.03663),
                          (numpy.inf, 0.0)]},
                {'name': 'cdf', 'precision': 4,
                 'grid': [(0.0, 0.0), (1.0, 0.86467), (2.0, 0.98168),
                          (numpy.inf, 1.0)]},
                {'name': 'generate', 'args': (25000,), 'precision': 1,
                 'generator': True,
                 'calls': [
                     {'call': numpy.mean, 'as_function': 'mean'},
                     {'call': numpy.std, 'as_function': 'std'},
                     {'call': numpy.var, 'as_function': 'var'},
                 ]},
            ]
        }]

    def test_fail_creation_with_zero_or_negative_rate(self):
        rates = [0, -1, -2, -numpy.inf]
        for rate in rates:
            with self.assertRaises(ValueError):
                cd.Exp(rate)


class TestErlang(TestBase):
    def setUp(self):
        self.tests = [{
            'distribution': cd.Erlang(shape=1, rate=1.0),
            'like': [{
                'alias': [cd.Exp(1.0)],
                'functions': [
                    {'name': 'mean'}, {'name': 'var'}, {'name': 'std'},
                    {'name': 'moment', 'grid': [1, 2, 3]},
                    {'name': 'pdf', 'grid': [0.0, 1.0, 2.0, numpy.inf]},
                    {'name': 'cdf', 'grid': [0.0, 1.0, 2.0, numpy.inf]}
                ]
            }],
            'properties': [
                {'name': 'shape', 'value': 1},
                {'name': 'rate', 'value': 1.0}
            ],
            'functions': [
                {'name': 'generate', 'args': (25000,), 'precision': 1,
                 'generator': True,
                 'calls': [
                     {'call': numpy.mean, 'as_function': 'mean'},
                     {'call': numpy.std, 'as_function': 'std'},
                     {'call': numpy.var, 'as_function': 'var'},
                 ]},
                {'name': 'sample', 'kwargs': {'shape': (25000,)},
                 'precision': 1, 'calls': [
                    {'call': numpy.mean, 'as_function': 'mean'},
                    {'call': numpy.std, 'as_function': 'std'},
                    {'call': lambda x: x.shape, 'as_value': (25000,)}
                ]},
                {'name': 'sample', 'kwargs': {'shape': (5, 1)},
                 'precision': 1, 'calls': [
                    {'call': lambda x: x.shape, 'as_value': (5, 1)}
                ]},
                {'name': 'sample', 'kwargs': {'shape': (3, 4, 5)},
                 'precision': 1, 'calls': [
                    {'call': lambda x: x.shape, 'as_value': (3, 4, 5)}
                ]},
            ]
        }, {
            'distribution': cd.Erlang(shape=1, rate=2.0),
            'like': [{
                'alias': [cd.Exp(2.0)],
                'functions': [
                    {'name': 'mean'}, {'name': 'var'}, {'name': 'std'},
                    {'name': 'moment', 'grid': [1, 2, 3]},
                    {'name': 'pdf', 'grid': [0.0, 1.0, 2.0, numpy.inf]},
                    {'name': 'cdf', 'grid': [0.0, 1.0, 2.0, numpy.inf]},
                ]
            }],
            'properties': [
                {'name': 'shape', 'value': 1},
                {'name': 'rate', 'value': 2.0},
            ]
        }, {
            'distribution': cd.Erlang(shape=2, rate=4.0),
            'properties': [
                {'name': 'shape', 'value': 2},
                {'name': 'rate', 'value': 4.0},
            ],
            'functions': [
                {'name': 'mean', 'value': 0.5},
                {'name': 'std', 'value': 0.125 ** 0.5},
                {'name': 'var', 'value': 0.125},
                {'name': 'moment', 'grid': [(1, 0.5), (2, 0.375)]},
                {'name': 'pdf', 'precision': 4,
                 'grid': [(0.0, 0.0), (1.0, 0.29305), (2.0, 0.01073),
                          (numpy.inf, 0)],
                 },
                {'name': 'cdf', 'precision': 4,
                 'grid': [(0.0, 0.0), (1.0, 0.90842), (2.0, 0.99698),
                          (numpy.inf, 1.0)]},
                {'name': 'generate', 'args': (25000,), 'precision': 1,
                 'generator': True,
                 'calls': [
                     {'call': numpy.mean, 'as_function': 'mean'},
                     {'call': numpy.std, 'as_function': 'std'},
                     {'call': numpy.var, 'as_function': 'var'},
                 ]},
            ],
        }]

    def test_fail_creation_wht_zero_or_negative_shape(self):
        shapes = [0, -1, -2, -numpy.inf]
        for shape in shapes:
            with self.assertRaises(ValueError):
                cd.Erlang(shape=shape, rate=1.0)

    def test_fail_creation_with_non_integer_shape(self):
        shapes = [1.5, 2.5, numpy.inf]
        for shape in shapes:
            with self.assertRaises(ValueError):
                cd.Erlang(shape=shape, rate=1.0)

    def test_fail_creation_with_zero_or_negative_rate(self):
        rates = [0.0, -1.0, -2.0, -numpy.inf]
        for rate in rates:
            with self.assertRaises(ValueError):
                cd.Erlang(shape=1, rate=rate)


class TestHyperExp(TestBase):
    def setUp(self):
        self.tests = [{
            'distribution': cd.HyperExp([1.0], [1.0]),
            'like': [{
                'alias': [cd.HyperExp([1.0, 1.0], [0.5, 0.5]),
                          cd.HyperExp([1.0] * 4, [0.25] * 4),
                          cd.Exp(1.0),
                          cd.Erlang(1, 1.0)],
                'functions': [
                    {'name': 'mean'}, {'name': 'var'}, {'name': 'std'},
                    {'name': 'moment', 'grid': [1, 2, 3]},
                    {'name': 'pdf', 'grid': [0.0, 1.0, 2.0, numpy.inf]},
                    {'name': 'cdf', 'grid': [0.0, 1.0, 2.0, numpy.inf]},
                ]
            }],
            'properties': [
                {'name': 'rates', 'value': [1.0]},
                {'name': 'pmf0', 'value': [1.0]},
            ],
            'functions': [
                {'name': 'generate', 'args': (25000,), 'precision': 1,
                 'generator': True,
                 'calls': [
                     {'call': numpy.mean, 'as_function': 'mean'},
                     {'call': numpy.std, 'as_function': 'std'},
                     {'call': numpy.var, 'as_function': 'var'},
                 ]},
                {'name': 'sample', 'kwargs': {'shape': (25000,)},
                 'precision': 1, 'calls': [
                    {'call': numpy.mean, 'as_function': 'mean'},
                    {'call': numpy.std, 'as_function': 'std'},
                    {'call': lambda x: x.shape, 'as_value': (25000,)}
                ]},
                {'name': 'sample', 'kwargs': {'shape': (4, 1)},
                 'precision': 1, 'calls': [
                    {'call': lambda x: x.shape, 'as_value': (4, 1)}
                ]},
                {'name': 'sample', 'kwargs': {'shape': (3, 4, 5)},
                 'precision': 1, 'calls': [
                    {'call': lambda x: x.shape, 'as_value': (3, 4, 5)}
                ]},
            ]
        }, {
            'distribution': cd.HyperExp([2.0], [1.0]),
            'like': [{
                'alias': [cd.HyperExp([2.0, 2.0], [0.5, 0.5]),
                          cd.HyperExp([2.0] * 4, [0.25] * 4),
                          cd.Exp(2.0),
                          cd.Erlang(1, 2.0)],
                'functions': [
                    {'name': 'mean'}, {'name': 'var'}, {'name': 'std'},
                    {'name': 'moment', 'grid': [1, 2, 3], 'vectorized': False},
                    {'name': 'pdf', 'grid': [0.0, 1.0, 2.0, numpy.inf]},
                    {'name': 'cdf', 'grid': [0.0, 1.0, 2.0, numpy.inf]},
                ]
            }],
            'properties': [
                {'name': 'rates', 'value': [2.0]},
                {'name': 'pmf0', 'value': [1.0]},
            ],
            'functions': [
                {'name': 'generate', 'args': (10000,), 'precision': 1,
                 'generator': True,
                 'calls': [
                     {'call': numpy.mean, 'as_function': 'mean'},
                     {'call': numpy.std, 'as_function': 'std'},
                     {'call': numpy.var, 'as_function': 'var'},
                 ]},
            ]
        },
        ]


class TestPhaseType(TestBase):
    def setUp(self):
        self.tests = [{
            'distribution': cd.PhaseType([[-1.0]], [1.0]),
            'like': [{
                'alias': [cd.HyperExp([1.0, 1.0], [0.5, 0.5]),
                          cd.HyperExp([1.0] * 4, [0.25] * 4),
                          cd.Exp(1.0),
                          cd.Erlang(1, 1.0)],
                'functions': [
                    {'name': 'mean'}, {'name': 'var'}, {'name': 'std'},
                    {'name': 'moment', 'grid': [1, 2, 3], 'vectorized': False},
                    {'name': 'pdf', 'grid': [0.0, 1.0, 2.0, numpy.inf]},
                    {'name': 'cdf', 'grid': [0.0, 1.0, 2.0, numpy.inf]},
                ]
            }],
            'properties': [
                {'name': 'S', 'value': [[-1.0]]},
                {'name': 'subgenerator', 'value': [[-1.0]]},
                {'name': 'pmf0', 'value': [1.0]},
                {'name': 'order', 'value': 1},
            ],
            'functions': [
                # {'name': 'generate', 'args': (10000,), 'precision': 1,
                #  'generator': True,
                #  'calls': [
                #      {'call': numpy.mean, 'as_function': 'mean'},
                #      {'call': numpy.std, 'as_function': 'std'},
                #      {'call': numpy.var, 'as_function': 'var'},
                #  ]},
                # {'name': 'sample', 'kwargs': {'shape': (20000,)},
                #  'precision': 1, 'calls': [
                #     {'call': numpy.mean, 'as_function': 'mean'},
                #     {'call': numpy.std, 'as_function': 'std'},
                #     {'call': lambda x: x.shape, 'as_value': (20000,)}
                # ]},
                # {'name': 'sample', 'kwargs': {'shape': (4, 1)},
                #  'precision': 1, 'calls': [
                #     {'call': lambda x: x.shape, 'as_value': (4, 1)}
                # ]},
                # {'name': 'sample', 'kwargs': {'shape': (3, 4, 5)},
                #  'precision': 1, 'calls': [
                #     {'call': lambda x: x.shape, 'as_value': (3, 4, 5)}
                # ]},
            ],
        }, {
            'distribution': cd.PhaseType([[-1.0, 1.0], [0.0, -1.0]], [1.0, 0]),
            'like': [{
                'alias': [cd.Erlang(2, 1.0)],
                'functions': [
                    {'name': 'mean'}, {'name': 'var'}, {'name': 'std'},
                    {'name': 'moment', 'grid': [1, 2, 3], 'vectorized': False},
                    {'name': 'pdf', 'grid': [0.0, 1.0, 2.0, numpy.inf]},
                    {'name': 'cdf', 'grid': [0.0, 1.0, 2.0, numpy.inf]},
                ]
            }],
            'properties': [
                {'name': 'S', 'value': [[-1.0, 1.0], [0.0, -1.0]]},
                {'name': 'subgenerator', 'value': [[-1.0, 1.0], [0.0, -1.0]]},
                {'name': 'pmf0', 'value': [1.0, 0]},
                {'name': 'order', 'value': 2},
            ],
        },
        ]
