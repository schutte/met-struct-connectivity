"""ctypes-based FFI for the glasso routine implemented in Fortran."""

import ctypes
import os
import numpy
import platform

__all__ = ["glasso"]

if platform.system() == "Windows":
    _libpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "native", "glasso.dll"))
else:
    _libpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "native", "glasso.so"))

_dll = ctypes.CDLL(_libpath)
_dll.glasso_.argtypes = [
        # matrix dimension
        numpy.ctypeslib.ndpointer(dtype=numpy.int32, ndim=1, shape=(1,)),
        # empirical covariance matrix
        numpy.ctypeslib.ndpointer(dtype=numpy.float64, ndim=2, flags="F_CONTIGUOUS"),
        # regularization matrix
        numpy.ctypeslib.ndpointer(dtype=numpy.float64, ndim=2, flags="F_CONTIGUOUS"),
        # approximation flag
        numpy.ctypeslib.ndpointer(dtype=numpy.int32, ndim=1, shape=(1,)),
        # initialization flag
        numpy.ctypeslib.ndpointer(dtype=numpy.int32, ndim=1, shape=(1,)),
        # debug output flag
        numpy.ctypeslib.ndpointer(dtype=numpy.int32, ndim=1, shape=(1,)),
        # diagonal penalty flag
        numpy.ctypeslib.ndpointer(dtype=numpy.int32, ndim=1, shape=(1,)),
        # convergence threshold
        numpy.ctypeslib.ndpointer(dtype=numpy.float64, ndim=1, shape=(1,)),
        # maximum number of iterations
        numpy.ctypeslib.ndpointer(dtype=numpy.int32, ndim=1, shape=(1,)),
        # covariance matrix estimate
        numpy.ctypeslib.ndpointer(dtype=numpy.float64, ndim=2, flags="F_CONTIGUOUS"),
        # precision matrix estimate
        numpy.ctypeslib.ndpointer(dtype=numpy.float64, ndim=2, flags="F_CONTIGUOUS"),
        # actual number of iterations
        numpy.ctypeslib.ndpointer(dtype=numpy.int32, ndim=1, shape=(1,)),
        # average absolute parameter change at termination
        numpy.ctypeslib.ndpointer(dtype=numpy.float64, ndim=1, shape=(1,)),
        # error flag
        numpy.ctypeslib.ndpointer(dtype=numpy.int32, ndim=1, shape=(1,)),
]

def glasso(empirical_covariance, regularization_matrix,
        convergence_threshold=1e-4, max_iterations=10000,
        warm_start=None):

    """Run the Graphical LASSO procedure implemented in Fortran with the
    given empirical covariance matrix and regularization parameter.

    Parameters
    ----------
    empirical_covariance : array
        Sample covariance matrix.
    regularization_matrix : array
        Matrix-valued regularization parameter, of the same dimension as
        the sample covariance matrix, or a scalar (to apply the same
        regularization at each element of the inverse covariance
        matrix).
    convergence_threshold : float, optional
        Stopping threshold.
    max_iterations : int, optional
        Iterations after which to stop the algorithm, even if the
        convergence threshold has not been reached yet.
    warm_start : pair of array, optional
        The inverse covariance and covariance matrices obtained
        from a previous run.

    Returns
    -------
    array
        The sparse inverse covariance matrix estimate.
    array
        The inverse of the first result value.

    """

    int_zero = numpy.array([0], dtype=numpy.int32)
    int_one = numpy.array([1], dtype=numpy.int32)

    n_parameters = numpy.array([empirical_covariance.shape[0]], dtype=numpy.int32)
    empirical_covariance = numpy.asfortranarray(empirical_covariance, dtype=numpy.float64)
    if type(regularization_matrix) is not numpy.ndarray or regularization_matrix.ndim < 2:
        regularization_matrix = regularization_matrix * numpy.ones_like(empirical_covariance)
    regularization_matrix = numpy.asfortranarray(regularization_matrix, dtype=numpy.float64)

    approx_flag = int_zero.copy()
    debug_flag = int_zero.copy()
    pen_diag_flag = int_one.copy()

    if warm_start:
        init_flag = int_one.copy()
        sigma = numpy.asfortranarray(warm_start[1].copy(), dtype=numpy.float64)
        theta = numpy.asfortranarray(warm_start[0].copy(), dtype=numpy.float64)
    else:
        init_flag = int_zero.copy()
        sigma = numpy.empty_like(empirical_covariance)
        theta = numpy.empty_like(empirical_covariance)

    convergence_threshold = numpy.array([convergence_threshold], dtype=numpy.float64)
    max_iterations = numpy.array([max_iterations], dtype=numpy.int32)

    n_iter = int_zero.copy()
    avg_change = numpy.empty(1, dtype=numpy.float64)
    error_flag = int_zero.copy()

    _dll.glasso_(n_parameters, empirical_covariance, regularization_matrix,
            approx_flag, init_flag, debug_flag, pen_diag_flag,
            convergence_threshold, max_iterations,
            sigma, theta, n_iter, avg_change, error_flag)

    if error_flag:
        raise ValueError("glasso failed")

    return theta, sigma
