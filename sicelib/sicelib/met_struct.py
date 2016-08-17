"""Functions for structure-weighted SICE."""

import numpy
import networkx

def get_structural_matrices(method, structural_connectivities, train_indices_by_class):
    """Turn a list of per-subject structural connectivity matrices into
    one average structural connectivity matrix per group.  Optionally
    perform randomization.

    Parameters
    ----------
    method : dict
        Dictionary of options.  If `invstruct` is present, take the
        "inverse" structural connectivity matrix by subtracting every
        element from 1.  If `randstruct` is present, randomly but
        consistently reorder rows and columns of the connectivity matrix
        (to get an isomorphic network with shuffled vertex labels),
        using the parameter to `randstruct` as a seed for the random
        number generator.  If `wildstruct` is present, proceed
        analogously, but shuffle the elements in the connectivity matrix
        without regard for isomorphism.
    structural_connectivities : array
        An n*m*m array of structural connectivity matrices (where n is
        the number of subjects and m is the number of regions), as
        obtained by `sicelib.data.load_structural_connectivities`.
    train_indices_by_class : list of arrays
        For each group, an array of indices of training subjects within
        that group.

    Returns
    -------
    list of arrays
        For each group, an averaged structural connectivity matrix,
        scaled to have maximum value 1.

    """

    structural_connectivity_by_class = []

    for indices in train_indices_by_class:
        sc = numpy.mean(structural_connectivities[indices], axis=0)
        sc /= numpy.max(sc)

        if "invstruct" in method:
            sc = 1 - sc

        elif "randstruct" in method:
            seed = int(method["randstruct"])
            index_sequence = numpy.arange(sc.shape[0])
            numpy.random.RandomState(seed).shuffle(index_sequence)
            sc = sc[index_sequence, :][:, index_sequence]
            numpy.fill_diagonal(sc, 0)

        elif "wildstruct" in method:
            seed = int(method["wildstruct"])
            pairs = sc[numpy.triu_indices_from(sc, 1)]
            numpy.random.RandomState(seed).shuffle(pairs)
            sc[numpy.triu_indices_from(sc, 1)] = pairs
            sc -= numpy.tril(sc)
            sc += sc.T

        structural_connectivity_by_class.append(sc)

    return structural_connectivity_by_class

def get_regularization_matrices(method, structural_connectivity_by_class):
    """Derive SICE regularization matrices from groupwise average
    structural connectivity matrices.

    Parameters
    ----------
    method : dict
        Dictionary of options.  See the Schemes section for details.
    structural_connectivity_by_class : list of arrays
        For each group, a single structural connectivity matrix (as
        obtained by `get_structural_matrices`).
        
    Schemes
    -------
    A (generally) non-linear function is applied to each element in the
    structural connectivity matrix.  This function and its parameter can
    be selected through an element in the `method` list:

    `unweighted`
        Don't use the structural connectivity matrices at all.  Use
        scalar regularization parameters only, i.e. do standard SICE.
    `zero`
        Don't use the structural connectivity matrices at all.  Create
        regularization matrices which are 1 everywhere, but 0 on the
        diagonal, to obtain non-shrunk estimates of variance.
    `exp`
        Use an exponential scheme.  Requires a scale parameter.  For
        example, `exp0.5` would apply the function exp(-x/0.5) to every
        element x in the structural connectivity matrices.
    `pow`
        Use a power scheme.  Requires a degree parameter.  For example,
        `pow2` would apply the function x^2 to every element x.

    If `diff` is in `method`, take the relative difference between the
    two structural connectivity matrices instead of the matrices
    themselves.

    Returns
    -------
    list of arrays
        The calculated regularization matrix for each class.

    """

    n_classes = len(structural_connectivity_by_class)
    n_regions = structural_connectivity_by_class[0].shape[0]

    if "diff" in method:
        if n_classes != 2:
            raise ValueError("diff method meaningless for this number of classes")
        a, b = structural_connectivity_by_class
        sc_difference = numpy.empty_like(a)
        threshold = 0
        both_mask = (a > threshold) & (b > threshold)
        neither_mask = (a <= threshold) & (b <= threshold)
        one_mask = (a <= threshold) ^ (b <= threshold)
        sc_difference[both_mask] = 1 - numpy.minimum(a[both_mask] / b[both_mask], b[both_mask] / a[both_mask])
        sc_difference[neither_mask] = 0
        sc_difference[one_mask] = 1
        structural_connectivity_by_class = [sc_difference, sc_difference.copy()]

    if "unweighted" in method:
        return [numpy.ones((n_regions, n_regions)) for i in range(n_classes)]

    elif "zero" in method:
        return [1 - numpy.eye(n_regions) for i in range(n_classes)]

    elif "pow" in method:
        degree = method["pow"]
        result = []
        for sc in structural_connectivity_by_class:
            reg_matrix = (1 - sc) ** degree
            result.append(reg_matrix)

    elif "exp" in method:
        sigma = method["exp"]
        result = []
        for sc in structural_connectivity_by_class:
            reg_matrix = numpy.exp(-sc / sigma)
            result.append(reg_matrix)

    if "distance" in method:
        for reg_matrix in result:
            graph = networkx.from_numpy_matrix(reg_matrix)
            apsp = networkx.all_pairs_dijkstra_path_length(graph)
            for i, js in apsp.items():
                for j, dist in js.items():
                    reg_matrix[i, j] = dist

    for reg_matrix in result:
        reg_matrix /= numpy.mean(reg_matrix[numpy.triu_indices_from(reg_matrix, 1)])
        numpy.fill_diagonal(reg_matrix, 0)

    return result
