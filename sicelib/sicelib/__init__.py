"""Interface to the MATLAB package for the Friedman et al. Graphical
LASSO routine.  The necessary code is maintained by Hossein Karshenas
and can be downloaded from the web at
<http://statweb.stanford.edu/~tibs/glasso/Graphical-Lasso.zip>.

References
----------
Friedman, Hastie, Tibshirani.  Sparse inverse covariance estimation with
the graphical LASSO.  Biostatistics 9:432-441, 2008.

"""

import sklearn.metrics
import numpy

from .glasso import glasso
from .utils import foerstner

class LassoEstimation(object):
    """Interface to the Graphical LASSO routine through MATLAB.
    Represents a series of sparse inverse covariance estimates (for the
    same data, but a variable selection of regularisation parameter
    values).

    Parameters
    ----------
    data : array of float
        An n*m array of regional PET averages, where n is the number of
        subjects and m is the number of regions.
    alphas : array of float
        A three-dimensional array of matrix-valued regularization
        parameters.  Each parameter must be an m*m matrix.
    alpha_labels : array of float
        Array of the same length as `alphas`.  Can be used to provide
        human-readable descriptions of the individual regularization
        parameter values.

    Attributes
    ----------
    alphas : array of float
    alpha_labels : array of float
    precisions : array of float
        An array of the same dimension as `alphas`, containing the
        inverse covariance estimates for each parameter value.
    covariances : array of float
        Element-wise inverse of `precisions`.
    cs_covariances : array of float, or None
        Result of covariance selection, if `cs` is True; None otherwise.

    """

    def __init__(self, data, alphas, alpha_labels=None, warm_start=False):
        self.alphas = alphas
        if alpha_labels is None:
            self.alpha_labels = alphas
        else:
            self.alpha_labels = alpha_labels

        n_regions = data.shape[1]
        sample_covariance = numpy.cov(data.T)
        precisions, covariances = [], []
        for alpha in reversed(alphas):
            if precisions and warm_start:
                theta, sigma = glasso(sample_covariance, alpha, warm_start=(precisions[-1], covariances[-1]))
            else:
                theta, sigma = glasso(sample_covariance, alpha)
            precisions.append(theta)
            covariances.append(sigma)
        self.precisions = list(reversed(precisions))
        self.covariances = list(reversed(covariances))

    def detection_error(self, ground_truth):
        """Compute binary detection error scores with regard to a ground
        trouth inverse covariance matrix.

        Parameters
        ----------
        ground_truth : array_like
            The ground truth inverse covariance matrix against which to
            evaluate the sparse estimate.

        Returns
        -------
        fpr : array of float
            The false-positive rate of binary link detection for each
            value of the regularisation parameter.
        tpr : array of float
            The true-positive rate of binary link detection for each
            value of the regularisation parameter.

        """

        fpr, tpr = numpy.zeros((2, len(self.precisions)))
        for i, estimate in enumerate(self.precisions):
            (tn, fp), (fn, tp) = sklearn.metrics.confusion_matrix(
                    ground_truth.flatten() != 0,
                    estimate.flatten() != 0)
            fpr[i] = float(fp) / (fp + tn)
            tpr[i] = float(tp) / (tp + fn)
        return fpr, tpr

    def foerstner_error(self, ground_truth):
        """Compute the Förstner distance between the estimated
        covariance matrices and the given ground truth.

        Parameters
        ----------
        ground_truth : array_like
            The ground truth covariance matrix against which to compare
            the sparse estimate.

        Returns
        -------
        err : array of float
            Values of the Förstner distance for each regularisation
            parameter.

        """

        err = numpy.zeros(len(self.covariances))
        for i, estimate in enumerate(self.covariances):
            err[i] = foerstner(ground_truth, estimate)

        return err
