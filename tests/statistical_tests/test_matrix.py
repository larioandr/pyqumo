from unittest import TestCase
import pyqumo.matrix as qumo
import numpy as np


def assert_all_close(tc, lvalue, rvalue, places=6, msg=""):
    lv = np.asarray(lvalue)
    rv = np.asarray(rvalue)
    try:
        tc.assertAlmostEqual(lv.item(), rv.item(), places, msg)
    except ValueError:
        tol = pow(10.0, -places)
        try:
            tc.assertTrue(np.allclose(lv, rv, tol, tol),
                          msg + " (tol={})".format(tol))
        except TypeError as err:
            raise TypeError("{}: {}".format(repr(err), msg))
    except Exception as err:
        raise RuntimeError("{} -- {}".format(repr(err), msg))


# noinspection PyUnresolvedReferences
class TestMatrixFunctions(TestCase):
    def setUp(self):
        self.l1 = [1]
        self.l2 = [1, 2]
        self.l11 = [[1]]
        self.l12 = [[1, 2]]
        self.l21 = [[1], [2]]
        self.l22 = [[1, 2], [3, 4]]
        self.l23 = [[1, 2, 3], [4, 5, 6]]
        self.m1 = np.array(self.l1)
        self.m2 = np.array(self.l2)
        self.m11 = np.array(self.l11)
        self.m12 = np.array(self.l12)
        self.m21 = np.array(self.l21)
        self.m22 = np.array(self.l22)
        self.m23 = np.array(self.l23)

    def test_is_square_ndarray(self):
        self.assertTrue(qumo.is_square(self.m11))
        self.assertTrue(qumo.is_square(self.m22))
        self.assertFalse(qumo.is_square(self.m12))
        self.assertFalse(qumo.is_square(self.m21))
        self.assertFalse(qumo.is_square(self.m23))

    def test_is_square_lists(self):
        self.assertTrue(qumo.is_square(self.l11))
        self.assertTrue(qumo.is_square(self.l22))
        self.assertFalse(qumo.is_square(self.l12))
        self.assertFalse(qumo.is_square(self.l21))

    def test_is_vector_ndarray(self):
        self.assertTrue(qumo.is_vector(self.m1))
        self.assertTrue(qumo.is_vector(self.m11))
        self.assertTrue(qumo.is_vector(self.m2))
        self.assertTrue(qumo.is_vector(self.m12))
        self.assertTrue(qumo.is_vector(self.m21))
        self.assertFalse(qumo.is_vector(self.m23))
        self.assertFalse(qumo.is_vector(self.m22))

    def test_is_vector_lists(self):
        self.assertTrue(qumo.is_vector(self.l1))
        self.assertTrue(qumo.is_vector(self.l2))
        self.assertTrue(qumo.is_vector(self.l12))
        self.assertTrue(qumo.is_vector(self.l21))
        self.assertFalse(qumo.is_vector(self.l22))
        self.assertFalse(qumo.is_vector(self.l23))

    def test_order_of_ndarray(self):
        self.assertEqual(qumo.order_of(self.m1), 1)
        self.assertEqual(qumo.order_of(self.m2), 2)
        self.assertEqual(qumo.order_of(self.m11), 1)
        self.assertEqual(qumo.order_of(self.m12), 2)
        self.assertEqual(qumo.order_of(self.m21), 2)
        self.assertEqual(qumo.order_of(self.m22), 2)
        with self.assertRaises(ValueError):
            qumo.order_of(self.m23)

    def test_order_of_numpy_lists(self):
        self.assertEqual(qumo.order_of(self.l1), 1)
        self.assertEqual(qumo.order_of(self.l2), 2)
        self.assertEqual(qumo.order_of(self.l11), 1)
        self.assertEqual(qumo.order_of(self.l12), 2)
        self.assertEqual(qumo.order_of(self.l21), 2)
        self.assertEqual(qumo.order_of(self.l22), 2)
        with self.assertRaises(ValueError):
            qumo.order_of(self.l23)


class TestIsStochastic(TestCase):
    def test_stochastic_matrices(self):
        a1 = [[1.0]]
        a2 = [[0.1, 0.9], [0.2, 0.8]]
        a2_with_tol = [[0.1, 0.95], [0.2, 0.75]]
        self.assertTrue(qumo.is_stochastic(a1))
        self.assertTrue(qumo.is_stochastic(a2))
        self.assertFalse(qumo.is_stochastic(a2_with_tol))
        self.assertTrue(qumo.is_stochastic(a2_with_tol, atol=0.05))

    def test_returns_false_when_matrix_has_negative_elements(self):
        a1 = [[1.01]]
        a2 = [[-0.1, 1.1], [0.0, 1.0]]
        self.assertFalse(qumo.is_stochastic(a1))
        self.assertTrue(qumo.is_stochastic(a1, atol=0.01))
        self.assertFalse(qumo.is_stochastic(a2))
        self.assertTrue(qumo.is_stochastic(a2, atol=0.1))

    def test_returns_false_when_row_sum_not_one(self):
        a1 = [[0.99]]
        a2 = [[0.1, 0.95], [0.2, 0.8]]
        self.assertFalse(qumo.is_stochastic(a1))
        self.assertTrue(qumo.is_stochastic(a1, atol=0.01))
        self.assertFalse(qumo.is_stochastic(a2))
        self.assertTrue(qumo.is_stochastic(a2, atol=0.05))


class TestIsInfinitesimal(TestCase):
    def test_correct_matrices(self):
        a11 = [[0.0]]
        a22 = [[-1.0, 1.0], [0.1, -0.1]]
        a33 = [[-1.0, 0.2, 0.8], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]]
        self.assertTrue(qumo.is_infinitesimal(a11))
        self.assertTrue(qumo.is_infinitesimal(a22))
        self.assertTrue(qumo.is_infinitesimal(a33))

    def test_returns_false_when_nondiagonal_elements_negative(self):
        a = [[-1.0, 1.1, -0.1], [1.0, -1.0, 0.0], [2.0, 10.0, -12.0]]
        self.assertFalse(qumo.is_infinitesimal(a))
        self.assertTrue(qumo.is_infinitesimal(a, atol=0.11))

    def test_returns_false_when_sum_not_zero(self):
        a = [[-1.0, 0.9], [0.1, -0.3]]
        self.assertFalse(qumo.is_infinitesimal(a))
        self.assertFalse(qumo.is_infinitesimal(a, atol=0.11))
        self.assertTrue(qumo.is_infinitesimal(a, atol=0.21))


class TestIsSubInfinitesimal(TestCase):
    def test_correct_matrices(self):
        a11 = [[-1.0]]
        a22 = [[-2.0, 1.0], [0.1, -0.1]]
        self.assertFalse(qumo.is_infinitesimal(a11))
        self.assertFalse(qumo.is_infinitesimal(a22))
        self.assertTrue(qumo.is_subinfinitesimal(a11))
        self.assertTrue(qumo.is_subinfinitesimal(a22))

    def test_returns_false_when_nondiagonal_elements_negative(self):
        a = [[-2.0, 1.1, -0.1], [1.0, -1.1, 0.0], [2.0, 10.0, -12.0]]
        self.assertFalse(qumo.is_subinfinitesimal(a))
        self.assertTrue(qumo.is_subinfinitesimal(a, atol=0.11))

    def test_returns_false_when_row_sum_greater_than_zero(self):
        a = [[-1.0, 1.09], [0.1, -0.3]]
        self.assertFalse(qumo.is_subinfinitesimal(a))
        self.assertTrue(qumo.is_subinfinitesimal(a, atol=0.11))

    def test_returns_false_when_all_row_sums_equal_to_zero(self):
        a = [[-1.0, 1.0], [0.1, -0.1]]
        self.assertFalse(qumo.is_subinfinitesimal(a))


class TestIsPMF(TestCase):
    def test_correct_list_matrices(self):
        a1 = [1.0]
        a2 = [0.2, 0.8]
        a32 = [[0.1, 0.9], [0.2, 0.8], [0.0, 1.0]]
        self.assertTrue(qumo.is_pmf(a1))
        self.assertTrue(qumo.is_pmf(a2))
        self.assertTrue(qumo.is_pmf(a32))

    def test_correct_ndarray_matrices(self):
        a1 = np.array([1.0])
        a2 = np.array([0.2, 0.8])
        a32 = np.array([[0.1, 0.9], [0.2, 0.8], [0.0, 1.0]])
        self.assertTrue(qumo.is_pmf(a1))
        self.assertTrue(qumo.is_pmf(a2))
        self.assertTrue(qumo.is_pmf(a32))

    def test_returns_false_when_some_elements_are_negative(self):
        a2 = [-0.1, 1.0]
        a32 = [[-0.1, 1.0], [0.2, 0.8], [0.0, 1.0]]
        self.assertFalse(qumo.is_pmf(a2))
        self.assertTrue(qumo.is_pmf(a2, atol=0.11))
        self.assertFalse(qumo.is_pmf(a32))
        self.assertTrue(qumo.is_pmf(a32, atol=0.11))

    def test_returns_false_when_any_row_sum_is_not_one(self):
        a1 = [1.05]
        a2_underflow = [0.1, 0.8]
        a2_overflow = [0.1, 1.0]
        a22_underflow = [[0.2, 0.6], [0.2, 0.8]]
        a22_overflow = [[0.3, 1.0], [0.3, 0.7]]
        self.assertFalse(qumo.is_pmf(a1))
        self.assertTrue(qumo.is_pmf(a1, atol=0.1))
        self.assertFalse(qumo.is_pmf(a2_underflow))
        self.assertTrue(qumo.is_pmf(a2_underflow, atol=0.11))
        self.assertFalse(qumo.is_pmf(a2_overflow))
        self.assertTrue(qumo.is_pmf(a2_overflow, atol=0.11))
        self.assertFalse(qumo.is_pmf(a22_underflow))
        self.assertTrue(qumo.is_pmf(a22_underflow, atol=0.21))
        self.assertFalse(qumo.is_pmf(a22_overflow))
        self.assertTrue(qumo.is_pmf(a22_overflow, atol=0.31))


class TestIsPDF(TestCase):
    def test_correct_list_matrices(self):
        a1 = [1.0]
        a2 = [0.5, 1.0]
        a23 = [[0.2, 0.7, 1.0], [0.1, 0.9, 1.0]]
        self.assertTrue(qumo.is_pdf(a1))
        self.assertTrue(qumo.is_pdf(a2))
        self.assertTrue(qumo.is_pdf(a23))

    def test_correct_ndarray_matrices(self):
        a1 = np.array([1.0])
        a2 = np.array([0.5, 1.0])
        a23 = np.array([[0.2, 0.7, 1.0], [0.1, 0.9, 1.0]])
        self.assertTrue(qumo.is_pdf(a1))
        self.assertTrue(qumo.is_pdf(a2))
        self.assertTrue(qumo.is_pdf(a23))

    def test_returns_false_when_some_elements_are_negative(self):
        a2 = [-0.1, 0.2, 1.0]
        a23 = [[0.0, 0.5, 1.0], [-0.2, 0.4, 1.0]]
        self.assertFalse(qumo.is_pdf(a2))
        self.assertTrue(qumo.is_pdf(a2, atol=0.11))
        self.assertFalse(qumo.is_pdf(a23))
        self.assertTrue(qumo.is_pdf(a23, atol=0.21))

    def test_returns_false_when_not_ascending(self):
        a = [0.5, 0.4, 1.0]
        self.assertFalse(qumo.is_pdf(a))
        self.assertTrue(qumo.is_pdf(a, atol=0.11))

    def test_returns_false_when_rightmost_element_less_than_one(self):
        a = [0.5, 0.6, 0.95]
        self.assertFalse(qumo.is_pdf(a))
        self.assertTrue(qumo.is_pdf(a, atol=0.1))

    def test_returns_false_when_rightmost_element_greater_than_one(self):
        a = [0.5, 0.6, 1.05]
        self.assertFalse(qumo.is_pdf(a))
        self.assertTrue(qumo.is_pdf(a, atol=0.1))


class TestAlmostEqual(TestCase):
    def test_compare_lists(self):
        a1 = [1]
        a2 = [1, 2]
        a12 = [[1, 2]]
        a22 = [[1, 2], [3, 4]]
        self.assertTrue(qumo.almost_equal(a1, [1]))
        self.assertTrue(qumo.almost_equal(a1, [0.95], 0.1))
        self.assertTrue(qumo.almost_equal(a2, [1, 2]))
        self.assertTrue(qumo.almost_equal(a2, [0.9, 2.1], 0.11))
        self.assertTrue(qumo.almost_equal(a12, [[1, 2]]))
        self.assertTrue(qumo.almost_equal(a12, [[0.9, 2.1]], 0.11))
        self.assertTrue(qumo.almost_equal(a22, [[1, 2], [3, 4]]))
        self.assertTrue(qumo.almost_equal(a22, [[1.2, 2], [3.1, 3.8]], 0.21))

    def test_compare_ndarrays_and_lists(self):
        a1 = np.array([1])
        a2 = np.array([1, 2])
        a12 = np.array([[1, 2]])
        a22 = np.array([[1, 2], [3, 4]])
        self.assertTrue(qumo.almost_equal(a1, [1]))
        self.assertTrue(qumo.almost_equal(a1, [0.95], 0.1))
        self.assertTrue(qumo.almost_equal(a2, [1, 2]))
        self.assertTrue(qumo.almost_equal(a2, [0.9, 2.1], 0.11))
        self.assertTrue(qumo.almost_equal(a12, [[1, 2]]))
        self.assertTrue(qumo.almost_equal(a12, [[0.9, 2.1]], 0.11))
        self.assertTrue(qumo.almost_equal(a22, [[1, 2], [3, 4]]))
        self.assertTrue(qumo.almost_equal(a22, [[1.2, 2], [3.1, 3.8]], 0.21))

    def test_compare_ndarrays(self):
        a1 = np.array([1])
        a2 = np.array([1, 2])
        a12 = np.array([[1, 2]])
        a22 = np.array([[1, 2], [3, 4]])
        self.assertTrue(qumo.almost_equal(a1, np.array([1])))
        self.assertTrue(qumo.almost_equal(a1, np.array([0.95]), 0.1))
        self.assertTrue(qumo.almost_equal(a2, np.array([1, 2])))
        self.assertTrue(qumo.almost_equal(a2, np.array([0.9, 2.1]), 0.11))
        self.assertTrue(qumo.almost_equal(a12, np.array([[1, 2]])))
        self.assertTrue(qumo.almost_equal(a12, np.array([[0.9, 2.1]]), 0.11))
        self.assertTrue(qumo.almost_equal(a22, np.array([[1, 2], [3, 4]])))
        self.assertTrue(qumo.almost_equal(
            a22, np.array([[1.2, 2], [3.1, 3.8]]), 0.21))


class TestPmfPdfConverters(TestCase):
    def test_pmf2pdf_list_matrices(self):
        p1 = [1.0]
        p2 = [0.1, 0.9]
        p3 = [0.4, 0.0, 0.3, 0.3]
        p23 = [[0.2, 0.1, 0.7], [0.4, 0.0, 0.6]]
        self.assertTrue(qumo.almost_equal(qumo.pmf2pdf(p1), [1.0]))
        self.assertTrue(qumo.almost_equal(qumo.pmf2pdf(p2), [0.1, 1.0]))
        self.assertTrue(qumo.almost_equal(qumo.pmf2pdf(p3),
                                          [0.4, 0.4, 0.7, 1.0]))
        self.assertTrue(qumo.almost_equal(qumo.pmf2pdf(p23),
                                          [[0.2, 0.3, 1.0], [0.4, 0.4, 1.0]]))

    def test_pmf2pdf_ndarray_matrices(self):
        p1 = np.array([1.0])
        p2 = np.array([0.1, 0.9])
        p3 = np.array([0.4, 0.0, 0.3, 0.3])
        p23 = np.array([[0.2, 0.1, 0.7], [0.4, 0.0, 0.6]])
        self.assertTrue(qumo.almost_equal(qumo.pmf2pdf(p1), [1.0]))
        self.assertTrue(qumo.almost_equal(qumo.pmf2pdf(p2), [0.1, 1.0]))
        self.assertTrue(qumo.almost_equal(qumo.pmf2pdf(p3),
                                          [0.4, 0.4, 0.7, 1.0]))
        self.assertTrue(qumo.almost_equal(qumo.pmf2pdf(p23),
                                          [[0.2, 0.3, 1.0], [0.4, 0.4, 1.0]]))

    def test_pdf2pmf_list_matrices(self):
        p1 = [1.0]
        p2 = [0.1, 1.0]
        p3 = [0.4, 0.4, 0.7, 1.0]
        p23 = [[0.2, 0.3, 1.0], [0.4, 0.4, 1.0]]
        self.assertTrue(qumo.almost_equal(qumo.pdf2pmf(p1), [1.0]))
        self.assertTrue(qumo.almost_equal(qumo.pdf2pmf(p2), [0.1, 0.9]))
        self.assertTrue(qumo.almost_equal(qumo.pdf2pmf(p3),
                                          [0.4, 0.0, 0.3, 0.3]))
        self.assertTrue(qumo.almost_equal(qumo.pdf2pmf(p23),
                                          [[0.2, 0.1, 0.7], [0.4, 0.0, 0.6]]))

    def test_pdf2pmf_ndarray_matrices(self):
        p1 = np.array([1.0])
        p2 = np.array([0.1, 1.0])
        p3 = np.array([0.4, 0.4, 0.7, 1.0])
        p23 = np.array([[0.2, 0.3, 1.0], [0.4, 0.4, 1.0]])
        self.assertTrue(qumo.almost_equal(qumo.pdf2pmf(p1), [1.0]))
        self.assertTrue(qumo.almost_equal(qumo.pdf2pmf(p2), [0.1, 0.9]))
        self.assertTrue(qumo.almost_equal(qumo.pdf2pmf(p3),
                                          [0.4, 0.0, 0.3, 0.3]))
        self.assertTrue(qumo.almost_equal(qumo.pdf2pmf(p23),
                                          [[0.2, 0.1, 0.7], [0.4, 0.0, 0.6]]))


class TestArrays2StringConverters(TestCase):
    def test_row2string_list(self):
        a1 = [1.0]
        a2 = [1.0, 2.0]
        a2int = [1, 2]
        a0 = []
        self.assertEqual(qumo.row2string(a1, ','), "1.0")
        self.assertEqual(qumo.row2string(a2, ','), "1.0,2.0")
        self.assertEqual(qumo.row2string(a2int, ';'), '1;2')
        self.assertEqual(qumo.row2string(a0, ','), "")

    def test_row2string_ndarray(self):
        a1 = np.array([1.0])
        a2 = np.array([1.0, 2.0])
        a2int = np.array([1, 2])
        a0 = np.array([])
        self.assertEqual(qumo.row2string(a1, ','), "1.0")
        self.assertEqual(qumo.row2string(a2, ','), "1.0,2.0")
        self.assertEqual(qumo.row2string(a2int, ';'), '1;2')
        self.assertEqual(qumo.row2string(a0, ','), "")

    def test_matrix2string_list(self):
        m0 = [[]]
        m1 = [[1.0]]
        m2 = [[1, 2], [3, 4]]
        m3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        self.assertEqual(qumo.matrix2string(m0, ',', ';'), "")
        self.assertEqual(qumo.matrix2string(m1, ',', ';'), "1.0")
        self.assertEqual(qumo.matrix2string(m2, ',', ';'), "1,2;3,4")
        self.assertEqual(qumo.matrix2string(m3, '-', ':'),
                         "1.0-2.0-3.0:4.0-5.0-6.0")

    def test_matrix2string_ndarray(self):
        m0 = np.array([[]])
        m1 = np.array([[1.0]])
        m2 = np.array([[1, 2], [3, 4]])
        m3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(qumo.matrix2string(m0, ',', ';'), "")
        self.assertEqual(qumo.matrix2string(m1, ',', ';'), "1.0")
        self.assertEqual(qumo.matrix2string(m2, ',', ';'), "1,2;3,4")
        self.assertEqual(qumo.matrix2string(m3, '-', ':'),
                         "1.0-2.0-3.0:4.0-5.0-6.0")

    def test_array2string_ndarray(self):
        v0 = np.array([])
        v1 = np.array([1, 2])
        m0 = np.array([[]])
        m1 = np.array([[1.0, 2.0], [10.0, 20.0]])
        self.assertEqual(qumo.array2string(v0), '')
        self.assertEqual(qumo.array2string(v1, ',', ';'), '1,2')
        self.assertEqual(qumo.array2string(m0), '')
        self.assertEqual(qumo.array2string(m1, ' ', ':'), '1.0 2.0:10.0 20.0')

    def test_parse_array_with_correct_strings(self):
        s0 = ''
        s1 = '1,2'
        s1ws = '    1   ,\n\t  2\t'
        s21 = '1;2'
        s21ws = '\n 1\t \n;\n \t 2\n'
        s22 = '1;2 : 3; 4.5'
        self.assertTrue(qumo.almost_equal(qumo.parse_array(s0), []))
        self.assertTrue(qumo.almost_equal(qumo.parse_array(s1, ','), [1, 2]))
        self.assertTrue(qumo.almost_equal(qumo.parse_array(s1ws, ','), [1, 2]))
        self.assertTrue(qumo.almost_equal(qumo.parse_array(s21, ',', ';'),
                                          [[1], [2]]))
        self.assertTrue(qumo.almost_equal(qumo.parse_array(s21ws, ',', ';'),
                                          [[1], [2]]))
        self.assertTrue(qumo.almost_equal(qumo.parse_array(s22, ';', ':'),
                                          [[1.0, 2.0], [3.0, 4.5]]))

    def test_parse_array_with_wrong_strings(self):
        s_with_wrong_symbol = 'x'
        s_with_wrong_shape = '1, 2; 3, 4, 5'
        s_with_missing_items = '1, ,3; 4, 5, 6'
        with self.assertRaises(Exception):
            qumo.parse_array(s_with_wrong_symbol)
        with self.assertRaises(Exception):
            qumo.parse_array(s_with_wrong_shape)
        with self.assertRaises(Exception):
            qumo.parse_array(s_with_missing_items)


class TestBlockMatrixBuilders(TestCase):

    def test_cbmat_1x1(self):
        b0 = [[0.0]]
        b1 = [[1.0]]
        b2 = [[2.0]]
        b3 = [[3.0]]
        b4 = [[4.0]]
        blocks1 = [(0, 0, b0), (0, 1, b1), (1, 0, b2), (1, 1, b3)]
        blocks2 = [(0, 0, b3), (0, 1, b2), (1, 0, b1), (1, 1, b0)]
        blocks3 = [(0, 0, b1), (0, 2, b2), (3, 0, b3), (4, 2, b4)]

        m1 = qumo.cbmat(blocks1)
        m2 = qumo.cbmat(blocks2)
        m3 = qumo.cbmat(blocks3)

        np.testing.assert_allclose(m1, [[0., 1.], [2., 3.]])
        np.testing.assert_allclose(m2, [[3., 2.], [1., 0.]])
        np.testing.assert_allclose(m3, [[1, 0, 2], [0, 0, 0],
                                        [0, 0, 0], [3, 0, 0],
                                        [0, 0, 4]])

    def test_cbmat_2x2(self):
        b1 = [[1, 1], [1, 1]]
        b2 = [[2, 2], [2, 2]]
        b3 = [[3, 3], [3, 3]]
        b4 = [[4, 4], [4, 4]]
        blocks1 = [(0, 0, b1), (0, 1, b2), (1, 0, b3), (1, 1, b4)]
        blocks2 = [(0, 0, b4), (0, 1, b3), (1, 0, b2), (1, 1, b1)]
        blocks3 = [(0, 0, b1), (0, 3, b2), (2, 0, b3), (2, 1, b4)]

        m1 = qumo.cbmat(blocks1)
        m2 = qumo.cbmat(blocks2)
        m3 = qumo.cbmat(blocks3)

        np.testing.assert_allclose(
            m1, [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        np.testing.assert_allclose(
            m2, [[4, 4, 3, 3], [4, 4, 3, 3], [2, 2, 1, 1], [2, 2, 1, 1]])
        np.testing.assert_allclose(
            m3, [[1, 1, 0, 0, 0, 0, 2, 2],
                 [1, 1, 0, 0, 0, 0, 2, 2],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [3, 3, 4, 4, 0, 0, 0, 0],
                 [3, 3, 4, 4, 0, 0, 0, 0]])

    def test_cbdiag_1x2(self):
        b1 = [[1, 1]]
        b2 = [[2, 2]]
        b3 = [[3, 3]]

        m1 = qumo.cbdiag(1, [(0, b1), (1, b2), (-1, b3)])
        m2 = qumo.cbdiag(2, [(0, b1), (1, b2), (-1, b3)])
        m3 = qumo.cbdiag(3, [(0, b1), (1, b2), (-1, b3)])
        m4 = qumo.cbdiag(3, [(-2, b2), (1, b3), (2, b1)])

        np.testing.assert_allclose(m1, [[1, 1]])
        np.testing.assert_allclose(m2, [[1, 1, 2, 2], [3, 3, 1, 1]])
        np.testing.assert_allclose(m3, [[1, 1, 2, 2, 0, 0],
                                        [3, 3, 1, 1, 2, 2],
                                        [0, 0, 3, 3, 1, 1]])
        np.testing.assert_allclose(m4, [[0, 0, 3, 3, 1, 1],
                                        [0, 0, 0, 0, 3, 3],
                                        [2, 2, 0, 0, 0, 0]])

    def test_cbdiag_2x2(self):
        b1 = [[1, 1], [1, 1]]
        b2 = [[2, 2], [2, 2]]
        b3 = [[3, 3], [3, 3]]

        m1 = qumo.cbdiag(1, [(0, b1), (1, b2), (-1, b3)])
        m2 = qumo.cbdiag(2, [(0, b1), (1, b2), (-1, b3)])
        m3 = qumo.cbdiag(3, [(0, b1), (1, b2), (-1, b3)])
        m4 = qumo.cbdiag(3, [(0, b2), (1, b3), (2, b1)])

        np.testing.assert_allclose(m1, [[1, 1], [1, 1]])
        np.testing.assert_allclose(m2, [[1, 1, 2, 2], [1, 1, 2, 2],
                                        [3, 3, 1, 1], [3, 3, 1, 1]])
        np.testing.assert_allclose(m3, [[1, 1, 2, 2, 0, 0],
                                        [1, 1, 2, 2, 0, 0],
                                        [3, 3, 1, 1, 2, 2],
                                        [3, 3, 1, 1, 2, 2],
                                        [0, 0, 3, 3, 1, 1],
                                        [0, 0, 3, 3, 1, 1]])
        np.testing.assert_allclose(m4, [[2, 2, 3, 3, 1, 1],
                                        [2, 2, 3, 3, 1, 1],
                                        [0, 0, 2, 2, 3, 3],
                                        [0, 0, 2, 2, 3, 3],
                                        [0, 0, 0, 0, 2, 2],
                                        [0, 0, 0, 0, 2, 2]])
