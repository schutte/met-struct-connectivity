"""Helper function for running graphical LASSO with different
regularization parameters."""

import collections
import numpy

from . import LassoEstimation

def estimate_fixed_alphas(scalar_alphas, pet_averages, train_indices_by_class, reg_matrix_by_class):
    """Run the graphical LASSO for sparse inverse covariance estimation
    on multiple subpopulations (classes) of subjects and for a range of
    scalar regularization parameters.

    Parameters
    ----------
    scalar_alphas : array
        Array of the desired scalar regularization parameters.
    pet_averages : array
        The n*m PET data matrix (where n is the number of subjects and m
        is the number of regions).
    train_indices_by_class : list of arrays
        For each class, an array of indices corresponding to the
        matching subjects in the training set.
    reg_matrix_by_class : list of arrays
        For each class, the corresponding regularization matrix.

    Returns
    -------
    dict
        A dictionary whose keys are the scalar regularization
        parameters, and whose values are dictionaries with the following
        entries: `prec`, the corresponding inverse covariance matrices
        (one per class); `cov`, the covariance matrices; `narcs`, the
        numbers of arcs, and `alpha`, the scalar regularization
        parameters (same as the key).

    """

    n_regions = pet_averages.shape[1]
    alphas = [scalar_alphas[:, None, None] * reg_matrix for reg_matrix in reg_matrix_by_class]
    result = []

    for i, indices in enumerate(train_indices_by_class):
        result.append(LassoEstimation(pet_averages[indices],
            alphas[i], alpha_labels=scalar_alphas))

    return {scalar_alpha: {"cov": [cls.covariances[j] for cls in result],
            "prec": [cls.precisions[j] for cls in result],
            "alpha": [cls.alpha_labels[j] for cls in result],
            "narcs": [numpy.count_nonzero(cls.precisions[j][numpy.triu_indices(n_regions, 1)]) for cls in result]}
        for j, scalar_alpha in enumerate(scalar_alphas)}

def estimate_fixed_narcs(fixed_narcs, pet_averages, train_indices_by_class, reg_matrix_by_class):
    """Apply something akin to a binary search to the scalar
    regularization parameter to find inverse covariance matrices with a
    desired sparsity.

    Parameters
    ----------
    fixed_narcs : list
        List of desired numbers of arcs (half of the number of
        off-diagonal elements in the resulting inverse covariance
        matrix).
    pet_averages : array
        The n*m PET data matrix (where n is the number of subjects and m
        is the number of regions).
    train_indices_by_class : list of arrays
        For each class, an array of indices corresponding to the
        matching subjects in the training set.
    reg_matrix_by_class : list of arrays
        For each class, the corresponding regularization matrix.

    Returns
    -------
    dict
        A dictionary whose keys are numbers of arcs, and whose values
        are dictionaries with the following entries: `prec`, the
        corresponding inverse covariance matrices (one per class);
        `cov`, the corresponding covariance matrices; `narcs`, the
        numbers of arcs (equal to the key), and `alpha`, the scalar
        regularization parameters at which the given sparsity was
        achieved.

    """

    n_regions = pet_averages.shape[1]
    result = collections.defaultdict(dict)

    for i, indices in enumerate(train_indices_by_class):
        reg_matrix = reg_matrix_by_class[i]
        alphas = numpy.array([0.1, 0])
        narcs = numpy.array([0, n_regions * (n_regions - 1) / 2])
        result[0][i] = LassoEstimation(pet_averages[indices],
                [alphas[0] * reg_matrix], alpha_labels=[alphas[0]])

        for n in reversed(fixed_narcs):
            print(n, end=" ", flush=True)

            iteration_count = 0
            while True:
                closest_higher_index = numpy.searchsorted(narcs, n)
                closest_lower_index = closest_higher_index - 1

                if narcs[closest_higher_index] == n:
                    break

                elif iteration_count >= 50:
                    higher_dist = numpy.abs(n - narcs[closest_higher_index])
                    lower_dist = numpy.abs(n - narcs[closest_lower_index])
                    if higher_dist < lower_dist:
                        result[n][i] = result[narcs[closest_higher_index]][i]
                    else:
                        result[n][i] = result[narcs[closest_lower_index]][i]
                    break

                a, b = alphas[closest_lower_index], alphas[closest_higher_index]
                a_n, b_n = narcs[closest_lower_index], narcs[closest_higher_index]
                interp = (n - b_n) / (a_n - b_n)
                c = (1 - interp) * b + interp * a
                estimate = LassoEstimation(pet_averages[indices],
                        [c * reg_matrix], alpha_labels=[c])
                c_n = numpy.count_nonzero(estimate.precisions[0][numpy.triu_indices(n_regions, 1)])
                result[c_n][i] = estimate
                c_index = len(alphas) - numpy.searchsorted(alphas[::-1], c)
                if c_n < narcs[c_index - 1]:
                    c_n = narcs[c_index - 1]
                narcs = numpy.insert(narcs, c_index, c_n)
                alphas = numpy.insert(alphas, c_index, c)

                iteration_count += 1

        print()

    return {n: {"cov": [result[n][i].covariances[0] for i in range(2)],
            "prec": [result[n][i].precisions[0] for i in range(2)],
            "alpha": [result[n][i].alpha_labels[0] for i in range(2)],
            "narcs": [numpy.count_nonzero(result[n][i].precisions[0]
                [numpy.triu_indices(n_regions, 1)]) for i in range(2)]}
        for n in fixed_narcs}

def estimate_all_narcs(pet_averages, reg_matrix, alpha_space=None):
    """Run the graphical LASSO for sparse inverse covariance estimation
    for a wide range of scalar regularization parameters, to cover as
    many sparsity levels as possible.

    Note that in contrast to the other `estimate_` functions, this one
    does not operate on multiple groups at once.  This is because
    occasionally, a certain sparsity will not be achieved, and the
    result would be confusing if this only happened to a subset of
    groups.  Additionally, for reasons of speed and storage efficiency,
    this function does not use the `LassoEstimation` class.

    Parameters
    ----------
    pet_averages : array
        The n*m PET data matrix (where n is the number of subjects and m
        is the number of regions).
    reg_matrix : array
        A regularization matrix for structure-weighted SICE.
    alpha_space : array, optional
        If specified, an ascending range of scalar regularization
        parameters.  The default is a sequence of values between 1e-4
        and 1e-2, equally spaced on a logarithmic scale.

    Returns
    -------
    dict
        A dictionary whose keys are numbers of arcs, and whose
        values are dictionaries with the following entries: `prec`, the
        corresponding inverse covariance matrix; `cov`, the
        corresponding covariance matrix; `narcs`, the number of arcs
        (equal to the key), and `alpha`, the scalar regularization
        parameter at which the given sparsity was achieved.

    """

    n_regions = pet_averages.shape[1]
    max_narcs = n_regions * (n_regions - 1) // 2

    if not alpha_space:
        alpha_space = numpy.logspace(-5, -2, 10000, base=10)
    precisions = numpy.zeros((max_narcs + 1, n_regions, n_regions))
    covariances = numpy.zeros((max_narcs + 1, n_regions, n_regions))
    alphas = numpy.zeros(max_narcs + 1)
    for i, alpha in enumerate(alpha_space):
        estimate = LassoEstimation(pet_averages, [alpha * reg_matrix])
        precision = estimate.precisions[0]
        narcs = numpy.count_nonzero(precision[numpy.triu_indices(n_regions, 1)])
        alphas[narcs] = alpha
        precisions[narcs] = precision
        if narcs == 0:
            break

    for prec, cov in zip(precisions, covariances):
        cov[:] = numpy.linalg.pinv(prec)

    return {narcs: {"cov": covariances[narcs],
            "prec": precisions[narcs],
            "alpha": alphas[narcs],
            "narcs": narcs}
        for narcs in numpy.where(alphas > 0)[0]}

def find_narc_alphas(target_narcs, pet_averages, train_indices_by_class,
        reg_matrix_by_class, return_estimates=False):

    """Use binary search to find the scalar regularization parameter
    corresponding to a desired sparsity.  If only a single sparsity is
    required, this is much faster than either `estimate_all_narcs` or
    `estimate_fixed_narcs`.

    Parameters
    ----------
    target_narcs : int
        Desired numbers of arcs (half of the number of off-diagonal
        elements in the resulting inverse covariance matrix).
    pet_averages : array
        The n*m PET data matrix (where n is the number of subjects and m
        is the number of regions).
    train_indices_by_class : list of arrays
        For each class, an array of indices corresponding to the
        matching subjects in the training set.
    reg_matrix_by_class : list of arrays
        For each class, the corresponding regularization matrix.
    return_estimate : bool, optional
        If specified and True, return the estimated covariance and
        inverse covariance matrices alongside the scalar regularization
        parameters.

    Returns
    -------
    list of floats
        The scalar regularization parameters corresponding to the
        desired sparsity (one per group).
    list of arrays (only if `return_estimate` is True)
        The covariance matrices (one per group).
    list of arrays (only if `return_estimate` is True)
        The precision matrices (one per group).

    """

    result = []
    covariances = []
    precisions = []
    n_regions = pet_averages.shape[1]

    for i, indices in enumerate(train_indices_by_class):
        reg_matrix = reg_matrix_by_class[i]

        a, b = 0, 1000
        narcs = 0
        j = 0
        while abs(narcs - target_narcs) >= 1 and j < 100:
            j += 1
            c = (a + b) / 2
            estimate = LassoEstimation(pet_averages[indices], [c * reg_matrix])
            narcs = numpy.count_nonzero(estimate.precisions[0][numpy.triu_indices(n_regions, 1)])
            if narcs < target_narcs:
                b = c
            else:
                a = c

        result.append(c)
        if return_estimates:
            covariances.append(estimate.covariances[0])
            precisions.append(estimate.precisions[0])

    if return_estimates:
        return result, covariances, precisions
    else:
        return result
