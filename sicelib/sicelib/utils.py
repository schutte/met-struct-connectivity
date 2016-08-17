"""Miscellaneous utility functions associated with SICE."""

import numpy
import scipy.linalg
import re

def foerstner(cov1, cov2):
    """Compute the Förstner distance between two positive definite
    matrices.

    Parameters
    ----------
    cov1, cov2 : array_like
        Input matrices.

    Returns
    -------
    float
        The Förstner metric for the given pair of matrices.

    References
    ----------
    Förstner, Moonen.  A metric for covariance matrices.  Geodesy - The
    Challenge of the 3rd Millennium, pp. 299-309. Springer, 2003.

    """

    return numpy.sum(numpy.log(scipy.linalg.eigvals(cov1, cov2))**2)

def plot_detection_error(ground_truth, estimation, *args, **kwargs):
    """Compute the binary detection error (through
    `LassoEstimation.detection_error`) between a ground truth and a
    series of SICE estimates, and plot it in the style of an ROC curve.

    Parameters
    ----------
    ground_truth: array of float
        A ground-truth inverse covariance matrix (or sparsity pattern).
    estimation : LassoEstimation
        A series of SICE estimates.

    """

    import pylab

    fpr, tpr = estimation.detection_error(ground_truth)
    pylab.plot(
            numpy.concatenate([[1], fpr, [0]]),
            numpy.concatenate([[1], tpr, [0]]),
            *args, **kwargs)

def plot_foerstner_error(ground_truth, estimation, cs=False, *args, **kwargs):
    """Compute the Förstner error (through
    LassoEstimation.foerstner_error) between a ground truth and a series
    of SICE estimates, and plot it for each label of the regularisation
    parameter.

    Parameters
    ----------
    ground_truth : array of float
        A ground-truth covariance matrix.
    estimation : LassoEstimation
        A series of SICE estimates.

    """

    import pylab

    err = estimation.foerstner_error(ground_truth, cs=cs)
    pylab.plot(estimation.alpha_labels, err, *args, **kwargs)

def pow_alpha(base_matrix, degree):
    """Compute a power-weighted regularisation matrix.

    Parameters
    ----------
    base_matrix : array
        Underlying structural connectivity.
    degree : float
        Exponent applied at each element.

    Returns
    -------
    array of float
        The power-weighted matrix, normalized to maximum 1.

    """

    return to_max_one((1 - base_matrix) ** degree)

def exp_alpha(base_matrix, sigma):
    """Compute an exponentially weighted regularisation matrix.

    Parameters
    ----------
    base_matrix : array
        Underlying structural connectivity.
    sigma : float
        Scale parameter (denominator applied to the exponent).

    Returns
    -------
    array of float
        The exponentially weighted matrix, normalized to maximum 1.

    """

    return to_max_one(numpy.exp(-base_matrix / sigma))

def to_max_one(alpha):
    """Multiply a matrix with a scalar, such that it has maximum element
    1 or -1.

    Parameter
    ---------
    alpha : array_like
        The matrix to be scaled.

    Returns
    -------
    array_like
        The scaled matrix.

    """

    return alpha / numpy.max(numpy.abs(alpha))

NUM_PARAM_METHODS = re.compile("(exp|pow|randstruct|wildstruct|bootstrap)([0-9.]+)")

KNOWN_METHODS = {"bootstrap", "diff", "distance", "exp", "fa",
        "homavghammers", "invstruct", "normsc", "pow", "prodnormsc",
        "randstruct", "simhammers", "tfphammers", "unweighted",
        "wildstruct", "zero"}

def parse_method_string(method_string, extra_known_methods=set()):
    """Read a method string and turn it into a dictionary of options.

    Parameters
    ----------
    method_string : str
        A raw method string, with options separated by underscores.

    Returns
    -------
    dict
        A dictionary where each option name of the method string acts as
        a key, and the corresponding value is either a floating-point
        number (for options with a numeric argument) or True.

    Example
    -------
    The method string `exp0.1_randstruct9999_extended` will be processed
    into `{"exp": 0.1, "randstruct": 9999, "extended": True}`.
    """

    known_methods = KNOWN_METHODS | extra_known_methods

    method = {}
    for item in method_string.split("_"):
        mt = NUM_PARAM_METHODS.match(item)
        if mt:
            item = mt.group(1)
            value = mt.group(2)
            if "." in value:
                method[item] = float(value)
            else:
                method[item] = int(value)
        else:
            method[item] = True

        if item not in known_methods:
            print("WARNING: Unknown method parameter '{}'.".format(item))

    return method
