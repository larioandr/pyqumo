import numpy as np
from scipy import sparse
import functools


def cached_method(name, index_arg=None, cache_attr='__cache__'):
    def cached_method_decorator(func):
        @functools.wraps(func)
        def func_wrapper(self, *args, **kwargs):
            cache = getattr(self, cache_attr)
            if name not in cache:
                if index_arg is not None:
                    cache[name] = {}
                    index = args[index_arg - 1]
                    value = func(self, *args, **kwargs)
                    cache[name][index] = value
                    return value
                else:
                    value = func(self, *args, **kwargs)
                    cache[name] = value
                    return value
            elif index_arg is not None:
                index = args[index_arg - 1]
                if index not in cache[name]:
                    value = func(self, *args, **kwargs)
                    cache[name][index] = value
                    return value
                else:
                    return cache[name][index]
            else:
                return cache[name]

        return func_wrapper

    return cached_method_decorator


def row2string(row, sep=', '):
    """Converts a one-dimensional numpy.ndarray, list or tuple to string

    Args:
        row: one-dimensional list, tuple, numpy.ndarray or similar
        sep: string separator between elements

    Returns:
        string representation of a row
    """
    return sep.join("{0}".format(item) for item in row)


def matrix2string(matrix, col_sep=', ', row_sep='; '):
    """Converts a two-dimensional numpy.ndarray, list or tuple to string

    Args:
        matrix: two-dimensional list, tuple, numpy.ndarray or similar
        col_sep: string separator between columns
        row_sep: string separator between rows

    Returns:
        string representation of a matrix
    """
    return row_sep.join("{0}".format(
        row2string(row, col_sep)) for row in matrix)


def array2string(array, col_sep=', ', row_sep='; '):
    """Converts a 1- or 2-dimensional list, tuple or numpy.ndarray to string

    Args:
        array: a numpy.ndarray to stringify
        col_sep: string separator between columns
        row_sep: string separator between rows

    Returns:
        string representation of a matrix
    """
    array = np.asarray(array)
    if len(array.shape) == 1:
        return row2string(array, col_sep)
    elif len(array.shape) == 2:
        return matrix2string(array, col_sep, row_sep)
    else:
        raise ValueError('1-dim and 2-dim matrices supported')


# noinspection PyUnresolvedReferences
def parse_array(s, col_sep=',', row_sep=';', dtype=np.float64):
    rows = [x.strip() for x in s.split(row_sep)]
    if len(rows) == 1:
        columns = [x.strip() for x in s.split(col_sep)]
        if len(columns) == 1 and len(columns[0]) == 0:
            return np.array([])
        else:
            return np.array(list(dtype(item) for item in s.split(col_sep)),
                            dtype=dtype)
    else:
        return np.array(list(list(dtype(item) for item in row.split(col_sep))
                             for row in rows), dtype=dtype)


def is_square(matrix):
    """Checks that a given matrix has square shape.

    Args:
        matrix (numpy.ndarray, list or tuple): a matrix to test

    Returns:
        bool: True if the matrix as a square shape
    """
    matrix = np.asarray(matrix)
    return len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]


def is_vector(vector):
    """Checks that a given matrix represents a vector. One-dimensional
    lists, tuples or numpy arrays are vectors as well as two-dimensional,
    where one dimension equals 1 (e.g. row-vector or column-vector).

    Args:
        vector: a matrix to be tested

    Returns:
        True if the matrix represents a vector
    """
    vector = np.asarray(vector)
    return len(vector.shape) == 1 or (len(vector.shape) == 2 and (
            vector.shape[0] == 1 or vector.shape[1] == 1))


def order_of(matrix):
    """Returns an order of a square matrix or a vector - a number of rows
    (eq. columns) for square matrix and number of elements for a vector.

    Args:
        matrix: a square matrix or a vector

    Returns:
        A number of rows (eq. columns) for a square matrix;
        a number of elements for a vector
    """
    matrix = np.asarray(matrix)
    if is_square(matrix):
        return matrix.shape[0]
    elif is_vector(matrix):
        return len(matrix.flatten())
    else:
        raise ValueError("square matrix or vector expected")


def is_stochastic(matrix, rtol=1e-05, atol=1e-08):
    """Function checks whether a given matrix is stochastic, i.e. a square
    matrix of non-negative elements whose sum per each row is 1.0.

    All comparisons are performed "close-to", np.allclose() method
    is used to compare values.

    Args:
        matrix: a matrix to be tested
        rtol: relative tolerance, see numpy.allclose for reference
        atol: absolute tolerance, see numpy.allclose for reference

    Returns:
        True if the matrix is stochastic, False otherwise
    """
    matrix = np.asarray(matrix)
    return (is_square(matrix) and
            (matrix >= -atol).all() and
            (matrix <= 1.0 + atol).all() and
            np.allclose(matrix.sum(axis=1), np.ones(order_of(matrix)),
                        rtol=rtol, atol=atol))


def is_infinitesimal(matrix, rtol=1e-05, atol=1e-08):
    """Function checks whether a given matrix is infinitesimal, i.e. a square
    matrix of non-negative non-diagonal elements, sum per each row is zero.

    All comparisons are performed in "close-to" fashion, np.allclose() method
    is used to compare values.

    Args:
        matrix: a square matrix to test
        rtol: relative tolerance, see numpy.allclose for reference
        atol: absolute tolerance, see numpy.allclose for reference

    Returns:
        True if the matrix is infinitesimal, False otherwise
    """
    matrix = np.asarray(matrix)
    return (is_square(matrix) and
            ((matrix - np.diag(matrix.diagonal().flatten())) >= -atol).all() and
            np.allclose(np.zeros(order_of(matrix)), matrix.sum(axis=1),
                        atol=atol, rtol=rtol))


def is_subinfinitesimal(matrix, atol=1e-08):
    """Function checks whether a given matrix is sub-infinitesimal,
    i.e. a square matrix of non-negative non-diagonal elements,
    sum per each row is less or equal to zero and at least one sum is strictly
    less than zero.

    Args:
        matrix: a square matrix to test
        atol: absolute tolerance

    Returns:
        True if the matrix is sub-infinitesimal, False otherwise
    """
    matrix = np.asarray(matrix)
    rs = matrix.sum(axis=1)
    # noinspection PyUnresolvedReferences
    return (is_square(matrix) and
            ((matrix - np.diag(matrix.diagonal().flatten())) >= -atol).all() and
            (rs <= atol).all() and (rs < -atol).any())


def is_pmf(matrix, rtol=1e-05, atol=1e-08):
    """Checks whether each row of a given matrix represents a PMF
    (mass function). If a vector is given, it is treated as a
    single-row matrix.

    Args:
        matrix: a vector or a matrix where each row is checked to be a
            probability mass function
        rtol: relative tolerance, see numpy.allclose for reference
        atol: absolute tolerance, see numpy.allclose for reference

    Return:
        True if sum of all elements in each row is 1.0
    """
    matrix = np.asarray(matrix)
    if not (matrix >= -atol).all():
        return False
    if is_vector(matrix):
        return np.allclose(a=matrix.sum(), b=[1], atol=atol, rtol=rtol)
    elif len(matrix.shape) == 2:
        return np.allclose(a=np.ones(matrix.shape[0]), b=matrix.sum(axis=1),
                           atol=atol, rtol=rtol)
    else:
        raise ValueError("illegal shape {0}".format(matrix.shape))


def is_pdf(matrix, rtol=1e-05, atol=1e-08):
    """Checks whether each row of a given matrix represents a PDF
    (distribution), i.e. 0 <= matrix[i][j] <= matrix[i][j+1] and
    matrix[i][N] == 1 for each i. If a single vector is given
    it is treated as a row.

    Args:
        matrix: a vector or a matrix where each row is checked to be a
            cumulative probability distribution
        rtol: relative tolerance, see numpy.allclose for reference
        atol: absolute tolerance, see numpy.allclose for reference

    Returns:
        scalar bool if matrix is a vector, or a vector of bools of
        size equal to the number of rows of matrix
    """
    matrix = np.asarray(matrix)
    if is_vector(matrix):
        if isinstance(matrix, sparse.spmatrix):
            # noinspection PyUnresolvedReferences
            vector = matrix.toarray().flatten()
        else:
            vector = matrix.flatten()
        pmf = vector - np.hstack(([0], vector))[0:-1]
        return is_pmf(pmf, rtol, atol)
    else:
        rows_num = matrix.shape[0]
        if isinstance(matrix, sparse.spmatrix):
            matrix = matrix.tocsc()
            pmf = matrix - sparse.hstack((
                np.zeros((rows_num, 1)), matrix)
            ).tocsc()[:, :-1]
        else:
            pmf = matrix - np.hstack((
                np.zeros((rows_num, 1)), matrix))[:, :-1]
        return is_pmf(pmf, rtol, atol)


def pmf2pdf(pmf):
    """Converts a PMF matrix (each row is expected to be a PMF) into a matrix
    of the same size representing PDF. Please note, that no is_pmf() check
    is called. If needed, it should be called directly before this call.

    Args:
        pmf: a vector or a matrix where each row represents probability mass
            function

    Returns:
        a vector or matrix of the same size as the provided pmf, where
        each row corresponds to cumulative probability distribution
    """
    pmf = np.asarray(pmf)
    if is_vector(pmf):
        vector = pmf.toarray().flatten() if isinstance(
            pmf, sparse.spmatrix) else pmf.flatten()
        pdf = np.zeros(vector.size)
        pdf[0] = vector[0]
        for i in range(1, vector.size):
            pdf[i] = pdf[i - 1] + vector[i]
        return pdf
    else:
        rows_num, cols_num = pmf.shape
        if isinstance(pmf, sparse.spmatrix):
            matrix = pmf.tocsc().T
            pdf = matrix[0]
            for i in range(1, cols_num):
                pdf = sparse.vstack((pdf, pdf[i - 1] + matrix[i]))
        else:
            matrix = pmf.T
            pdf = matrix[0].reshape((1, rows_num))
            for i in range(1, cols_num):
                pdf = np.vstack((pdf, pdf[i - 1] + matrix[i]))
        return pdf.T


def pdf2pmf(pdf):
    """Converts a PDF matrix (each row is expected to be a PDF) into a matrix
    of the same size representing PMF. Please note, that no check ispdf()
    is called. If needed, it should be called directly before this call.

    Args:
        pdf: a vector or a matrix where each row represents cumulative
            probability distribution

    Return:
        a vector or matrix of the same size as the provided pdf, where
        each row corresponds to probability mass function
    """
    pdf = np.asarray(pdf)
    if is_vector(pdf):
        pdf = pdf.toarray().flatten() if isinstance(
            pdf, sparse.spmatrix) else pdf.flatten()
        pmf = np.zeros(pdf.size)
        pmf[0] = pdf[0]
        for i in range(pmf.size - 1, 0, -1):
            pmf[i] = pdf[i] - pdf[i - 1]
        return pmf
    else:
        rows_num, cols_num = pdf.shape
        if isinstance(pdf, sparse.spmatrix):
            matrix = pdf.tocsc().T
            pmf = matrix[0]
            for i in range(1, cols_num):
                pmf = sparse.vstack((pmf, matrix[i] - matrix[i - 1]))
        else:
            matrix = pdf.T
            pmf = matrix[0].reshape((1, rows_num))
            for i in range(1, cols_num):
                pmf = np.vstack((pmf, matrix[i] - matrix[i - 1]))
        return pmf.T


def almost_equal(m1, m2, atol=1e-6, rtol=1e-5):
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)
    return np.allclose(m1, m2, rtol=rtol, atol=atol)


def cbmat(blocks):
    blocks = [(row, col, np.asarray(block)) for row, col, block in blocks]
    shapes = [block.shape for row, col, block in blocks]
    if not all(shapes[0] == a_shape for a_shape in shapes):
        raise ValueError("block shapes mismatch")
    bshape = shapes[0]
    max_row = max(row for row, col, block in blocks)
    max_col = max(col for row, col, block in blocks)
    num_rows = (max_row + 1) * bshape[0]
    num_cols = (max_col + 1) * bshape[1]
    matrix = np.zeros((num_rows, num_cols))
    for row, col, block in blocks:
        row_0 = row * bshape[0]
        row_1 = (row + 1) * bshape[0]
        col_0 = col * bshape[1]
        col_1 = (col + 1) * bshape[1]
        matrix[row_0:row_1, col_0:col_1] = block
    return matrix


def cbdiag(size, blocks):
    blocks = [(i, np.asarray(b)) for i, b in blocks]
    shapes = [b.shape for i, b in blocks]
    if not all(shapes[0] == a_shape for a_shape in shapes):
        raise ValueError("block shapes mismatch")
    block_shape = shapes[0]
    matrix = np.zeros((size * block_shape[0], size * block_shape[1]))
    for i in range(size):
        row_0 = i * block_shape[0]
        row_1 = (i + 1) * block_shape[0]
        for b_col, block in blocks:
            b_col += i
            if 0 <= b_col < size:
                col_0 = b_col * block_shape[1]
                col_1 = (b_col + 1) * block_shape[1]
                matrix[row_0:row_1, col_0:col_1] = block
    return matrix
