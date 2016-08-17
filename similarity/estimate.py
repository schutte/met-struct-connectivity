#!/usr/bin/env python

import sys
import os
import pickle
import numpy

from sicelib import LassoEstimation
from sicelib.data import load_pet_averages, load_structural_connectivities
from sicelib.met_struct import get_structural_matrices, get_regularization_matrices
from sicelib.run import estimate_all_narcs
from sicelib.utils import parse_method_string

raw_method = sys.argv[1]
method = parse_method_string(raw_method)
cls = sys.argv[2]

script_dir = os.path.dirname(sys.argv[0])
data_dir = os.path.join(script_dir, "../classes")

pet_averages = load_pet_averages(method, data_dir, [cls])
structural_connectivities = load_structural_connectivities(method, data_dir, [cls])

n_subjects = pet_averages.shape[0]

print("... Estimating {} on {}".format(raw_method, cls))

structural_connectivity = get_structural_matrices(method,
        structural_connectivities, [numpy.arange(n_subjects)])[0]
reg_matrix = get_regularization_matrices(method, [structural_connectivity])[0]

result = estimate_all_narcs(pet_averages, reg_matrix)
narcs = list(sorted(result.keys()))
print(narcs)
print("out of a possible", pet_averages.shape[1] * (pet_averages.shape[1] - 1) // 2)

print("    done.")

out_filename = "results/{}_{}.pkl".format(cls, raw_method)
print("... Saving to {}".format(out_filename))

with open(out_filename, "wb") as fout:
    pickle.dump({"struct": structural_connectivity,
        "reg_matrix": reg_matrix,
        "narc": narcs,
        "alpha": [result[k]["alpha"] for k in narcs],
        "cov": [result[k]["cov"] for k in narcs],
        "prec": [result[k]["prec"] for k in narcs]}, fout)
