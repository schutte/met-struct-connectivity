#!/usr/bin/env python

"""Run a classification experiment: Load two classes and perform k-fold
or leave-one-out cross-validation on the following procedure:
    - Fit means and sparse inverse covariance matrices on the class 0
      and class 1 training data.
    - Use a multivariate Gaussian classifier to predict the labels on
      the training data.

The results are stored in either a CSV or a Python pickle file.

Usage: classify.py METHOD CLASS0 CLASS1 N_FOLDS SEED
Where:
    METHOD      is a string which specifies the exact regularization
                procedure (see sicelib for details); include the
                "extended" option if you want to get a detailed pickle
                dump instead of a compact CSV file;
    CLASS0      is the name of the directory which contains the
                pet_averages.npy, structural_connectivity.npy and
                fractional_anisotropy.npy files for one of the classes;
    CLASS1      is the name of the directory for the other class;
    N_FOLDS     is the number of folds to be used for k-fold cross;
                validation (if set to 0, leave-one-out is performed); and
    SEED        is the random seed for the fold split (may be omitted if
                N_FOLDS is 0)

"""

import sys
import os
import itertools
import pickle
import collections

import numpy
import scipy.stats

from sklearn.cross_validation import StratifiedKFold, LeaveOneOut
import sklearn.metrics

from sicelib.data import load_pet_averages, load_structural_connectivities
from sicelib.met_struct import get_structural_matrices, get_regularization_matrices
from sicelib.run import find_narc_alphas, estimate_fixed_alphas
from sicelib.utils import parse_method_string

# script locations
script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
classes_dir = os.path.join(script_dir, "../classes")

# input parameters
raw_method = sys.argv[1]
method = parse_method_string(raw_method,
        {"extended", "fixarcs", "lognarrow", "narrow", "prec", "scaled"})
classes = sys.argv[2:4]
n_folds = int(sys.argv[4])
seed = 0
if n_folds != 0:
    seed = int(sys.argv[5])

# load subject data
pet_averages, labels = load_pet_averages(method, classes_dir, classes, return_labels=True)
structural_connectivities = load_structural_connectivities(method, classes_dir, classes)

# set up cross-validation
if n_folds == 0:
    kfold = LeaveOneOut(len(labels))
else:
    kfold = StratifiedKFold(labels, n_folds=n_folds, shuffle=True, random_state=seed)
folds = []

for fold, (train_indices, test_indices) in enumerate(kfold):
    # find indices of training set subjects for each class
    train_indices_by_class = []
    for i in range(len(classes)):
        indices = train_indices[labels[train_indices] == i]
        train_indices_by_class.append(indices)

    # compute the mean parameters of the multivariate Gaussians
    means = [numpy.mean(pet_averages[index_subset], axis=0)
            for index_subset in train_indices_by_class]

    # average per-subject structural connectivities into one matrix per class
    structural_connectivity_by_class = get_structural_matrices(method,
            structural_connectivities, train_indices_by_class)
    # non-linearly map those into regularization matrices
    reg_matrix_by_class = get_regularization_matrices(method,
            structural_connectivity_by_class)

    print("... Estimating {}, fold {}".format(raw_method, fold))

    if "scaled" in method or "fixarcs" in method:
        # scale the regularization matrices such that at a scalar
        # regularization parameter value of 1, the sparse inverse
        # covariance matrix has a sparsity of 1/6.
        n_features = pet_averages.shape[1]
        narc_alphas = find_narc_alphas(n_features * (n_features - 1) / 6,
                pet_averages, train_indices_by_class, reg_matrix_by_class)
        for i in range(len(reg_matrix_by_class)):
            reg_matrix_by_class[i] *= narc_alphas[i]
        scalar_alphas = numpy.logspace(-2, 2, 300, base=10)

    elif "narrow" in method:
        scalar_alphas = numpy.linspace(0.00001, 0.005, 100)

    elif "lognarrow" in method:
        scalar_alphas = numpy.logspace(numpy.log10(0.00001), numpy.log10(0.005), 100, base=10)

    else:
        # select a range of regularization parameters which has been
        # empirically observed as useful
        scalar_alphas = numpy.linspace(0.00001, 0.03, 100)

    estimates = estimate_fixed_alphas(scalar_alphas, pet_averages,
            train_indices_by_class, reg_matrix_by_class)

    if "fixarcs" in method:
        # match up covariance matrices with the same number of nonzero
        # entries (instead of the same value of the regularization
        # parameter)

        estimates = list(estimates.values())

        n_features = pet_averages.shape[1]
        target_narcs = numpy.linspace(n_features, 0.9 * n_features ** 2, 100).round().astype(int)

        a_narcs, b_narcs = numpy.array([estimate["narcs"] for estimate in estimates]).T
        a_distances = numpy.abs(target_narcs[:, None] - a_narcs)
        b_distances = numpy.abs(target_narcs[:, None] - b_narcs)

        a_indices = numpy.argmin(a_distances, axis=1)
        b_indices = numpy.argmin(b_distances, axis=1)

        fixarcs_estimates = {}

        for target_narc, a_index, b_index in zip(target_narcs, a_indices, b_indices):
            fixarcs_estimates[target_narc] = \
                {k: [estimates[a_index][k][0], estimates[b_index][k][1]]
                    for k in estimates[a_index].keys()}

        estimates = fixarcs_estimates

    for estimate in estimates.values():
        # build multivariate Gaussian models for each class
        pdfs = [scipy.stats.multivariate_normal(mean=means[i], cov=estimate["cov"][i])
                for i in range(len(classes))]

        # check the probability of the test subjects' data under each of
        # those models
        scores = numpy.array([pdf.logpdf(pet_averages[test_indices]) for pdf in pdfs])

        if len(test_indices) == 1:
            scores = scores[:, None]

        estimate["scores"] = scores
        estimate["guesses"] = numpy.argmax(scores, axis=0)
        estimate["true_labels"] = labels[test_indices]
        estimate["method"] = raw_method

        # to save disk space in the pickle dump
        del estimate["cov"]
        if "prec" not in method:
            del estimate["prec"]

    folds.append(estimates)

print("    done.")

out_filename = "{}/results/{}_{}_{}_{}".format(script_dir, "".join(classes),
        seed, n_folds, raw_method)
if "extended" in method:
    out_filename += ".pkl"
    print("... Saving extended data to {}".format(out_filename))

    with open(out_filename, "wb") as fout:
        pickle.dump(folds, fout)

else:
    out_filename += ".csv"
    print("... Saving to {}".format(out_filename))

    with open(out_filename, "wt") as fout:
        print("fold,param,alpha0,alpha1,narcs0,narcs1,labels", file=fout)
        for i, fold in enumerate(folds):
            for j, (param, estimate) in enumerate(sorted(fold.items())):
                if j == 0:
                    print("{},0,0,0,0,0,{}".format(i,
                        "".join(map(str, estimate["true_labels"]))), file=fout)
                print("{},{},{},{},{}".format(i, param,
                    ",".join(map(str, estimate["alpha"])),
                    ",".join(map(str, estimate["narcs"])),
                    "".join(map(str, estimate["guesses"]))), file=fout)
