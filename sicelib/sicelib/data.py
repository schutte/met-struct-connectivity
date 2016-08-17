"""Functions for loading data for (standard and structure-weighted) SICE
experiments."""

import os
import numpy

from .hammers import TFP_SUBSET, TFP_HOMAVG_SUBSET, SIM_SUBSET, get_region_labels

def load_and_bootstrap(filename, method):
    """Load a numpy array (of regional PET averages or of structural
    connectivity matrices) from a file and, optionally, sample
    observations with replacement.

    Parameters
    ----------
    filename : str
        Path to the .npy file.
    method : dict
        Dictionary of options.  If a `bootstrap` key is present, its
        value will be used as a seed value for the random number
        generator for bootstrapping (sampling with replacement on the
        first axis of the loaded array).

    Returns
    -------
    array
        The numpy array loaded from the file, optionally bootstrapped.

    """

    result = numpy.load(filename)

    bootstrap_seed = method.get("bootstrap")
    if bootstrap_seed is not None:
        bootstrap_seed = int(bootstrap_seed)
        random_state = numpy.random.RandomState(bootstrap_seed)
        indices = random_state.choice(len(result), size=len(result), replace=True)
        result = result[indices]

    return result

def load_pet_averages(method, root_dir, classes, return_labels=False):
    """Load a numpy array of regional 18F-FDG PET averages.

    Parameters
    ----------
    method : dict
        Dictionary of options.  If `tfphammers` is set, the columns of
        the loaded numpy file are limited to a subset of 44 relevant
        regions of interest.  If `homavghammers` is set, the homologous
        regions within that set are averaged, giving 22 regions of
        interest.  For other loading options, see `load_and_bootstrap`.
    root_dir : str
        Base directory containing the class data directories.
    classes : list of str
        Subdirectories of `root_dir` from which `pet_averages.npy` files
        will be loaded.
    return_labels : bool, optional (default False)
        If True, an array of class labels is returned alongside the
        regional averages.

    Results
    -------
    array
        An n*m array of regional tracer uptake averages (concatenated
        from all classes), where n is the total number of subjects and m
        is the number of regions.  For other loading options, see
        `load_and_bootstrap`.
    array (optional, only if return_labels is set)
        An array of n integers, specifying the index of the class to
        which each subject belongs.

    """

    pet_averages = []
    labels = []
    for i, cls in enumerate(classes):
        cls_pet_averages = load_and_bootstrap(os.path.join(root_dir, cls, "pet_averages.npy"), method)

        if "tfphammers" in method:
            cls_pet_averages = cls_pet_averages[:, TFP_SUBSET]
        elif "homavghammers" in method:
            left = cls_pet_averages[:, TFP_HOMAVG_SUBSET[0]]
            right = cls_pet_averages[:, TFP_HOMAVG_SUBSET[1]]
            cls_pet_averages = numpy.mean([left, right], axis=0)
        elif "simhammers" in method:
            cls_pet_averages = cls_pet_averages[:, SIM_SUBSET]

        cls_pet_averages /= numpy.mean(cls_pet_averages, axis=1)[:, None]
        pet_averages.append(cls_pet_averages)
        labels.append([i] * len(cls_pet_averages))

    if return_labels:
        return numpy.concatenate(pet_averages), numpy.concatenate(labels)
    else:
        return numpy.concatenate(pet_averages)

def load_structural_connectivities(method, root_dir, classes):
    """Load a numpy array of structural connectivity matrices.

    Parameters
    ----------
    method : dict
        Dictionary of options.  If `tfphammers` is set, the columns of
        the loaded numpy file are limited to a subset of 44 relevant
        regions of interest.  If `homavghammers` is set, the homologous
        regions within that set are averaged, giving 22 regions of
        interest.  If `fa` is set, fractional anisotropy data is loaded;
        otherwise, a number-of-tracts matrix is loaded.  If `normsc` is
        set, each element in the matrix will be normalized by dividing
        the sum of elements of both regions.  For other loading options,
        see `load_and_bootstrap`.
    root_dir : str
        Base directory containing the class data directories.
    classes : list of str
        Subdirectories of `root_dir` from which
        `fractional_anisotropy.npy` (if `fa` is set) or
        `structural_connectivity.npy` (if it is not) files will be
        loaded.

    Results
    -------
    array
        An n*m*m array of structural connectivity matrices (concatenated
        from all classes), where n is the total number of subjects and m
        is the number of regions.

    """
    if "fa" in method:
        sc_filename = "fractional_anisotropy.npy"
    else:
        sc_filename = "structural_connectivity.npy"

    structural_connectivities = []
    for cls in classes:
        cls_structural_connectivities = load_and_bootstrap(os.path.join(root_dir, cls, sc_filename), method)

        if "normsc" in method:
            # normalize by sum of regional sums
            for individual_sc in cls_structural_connectivities:
                region_sums = numpy.sum(individual_sc, axis=0)
                norm_matrix = region_sums[:, None] + region_sums
                norm_matrix[norm_matrix == 0] = 1
                individual_sc /= norm_matrix
        elif "prodnormsc" in method:
            # normalize by product of regional sums
            for individual_sc in cls_structural_connectivities:
                region_sums = numpy.sum(individual_sc, axis=0)
                norm_matrix = numpy.outer(region_sums, region_sums)
                norm_matrix[norm_matrix == 0] = 1
                individual_sc /= norm_matrix

        if "tfphammers" in method:
            cls_structural_connectivities = cls_structural_connectivities[:, TFP_SUBSET, :][:, :, TFP_SUBSET]
        elif "homavghammers" in method:
            left = cls_structural_connectivities[:, TFP_HOMAVG_SUBSET[0], :][:, :, TFP_HOMAVG_SUBSET[0]]
            right = cls_structural_connectivities[:, TFP_HOMAVG_SUBSET[1], :][:, :, TFP_HOMAVG_SUBSET[1]]
            cls_structural_connectivities = numpy.mean([left, right], axis=0)
        elif "simhammers" in method:
            cls_structural_connectivities = cls_structural_connectivities[:, SIM_SUBSET, :][:, :, SIM_SUBSET]

        cls_structural_connectivities /= numpy.max(cls_structural_connectivities, axis=(1, 2))[:, None, None]
        structural_connectivities.append(cls_structural_connectivities)

    return numpy.concatenate(structural_connectivities)
