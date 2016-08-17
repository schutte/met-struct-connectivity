#!/usr/bin/env python

import sys
import os
import collections
import pickle
import numpy
import pandas
from sklearn.metrics import accuracy_score

mean_accs = collections.defaultdict(dict)
peak_accs = collections.defaultdict(dict)
min_narcs0 = collections.defaultdict(dict)
min_narcs1 = collections.defaultdict(dict)

for filename in sys.argv[1:]:
    accs = []
    params = []

    df = pandas.read_csv(filename).sort_values("fold")
    true_labels = df[df["param"] == 0]["labels"].as_matrix()
    guess_df = df[df["param"] != 0]
    narcs0 = guess_df["narcs0"].min()
    narcs1 = guess_df["narcs1"].min()
    for param in guess_df["param"].unique():
        guesses = guess_df[guess_df["param"] == param]["labels"].as_matrix()
        params.append(param)
        accs.append(accuracy_score(true_labels, guesses))
    del df, guess_df

    sort_indices = numpy.argsort(params)
    accs = numpy.array(accs)[sort_indices]
    params = numpy.array(params)[sort_indices]

    basename, ext = os.path.splitext(os.path.basename(filename))
    problem, fold_seed, nfolds, raw_method = basename.split("_", 3)

    method = raw_method.split("_")
    rs_index, rs_item = next(
            ((i, item) for i, item in enumerate(method)
                if item.startswith("randstruct") or item.startswith("wildstruct")),
            (None, None))

    if rs_index is not None:
        rs_seed = int(rs_item[10:])
        del method[rs_index]
    else:
        rs_seed = 0

    mean_accs[(problem, "_".join(method), fold_seed)][rs_seed] = numpy.mean(accs)
    peak_accs[(problem, "_".join(method), fold_seed)][rs_seed] = numpy.max(accs)
    min_narcs0[(problem, "_".join(method), fold_seed)][rs_seed] = narcs0
    min_narcs1[(problem, "_".join(method), fold_seed)][rs_seed] = narcs1

ranks = {}

for key, method_results in mean_accs.items():
    rs_seeds, accs = zip(*method_results.items())
    rs_seeds = list(rs_seeds)
    accs = list(accs)

    try:
        zero_index = rs_seeds.index(0)
    except ValueError:
        print("No real structure found for {}/{}".format(key[0], key[1]), file=sys.stderr)
        sys.exit(1)

    rs_seeds.append(rs_seeds.pop(zero_index))
    accs.append(accs.pop(zero_index))

    method_ranks = numpy.argsort(numpy.argsort(accs, kind="mergesort"))
    ranks[key] = dict(zip(rs_seeds, method_ranks))

print("problem,method,fold_seed,rs_seed,min_narcs0,min_narcs1,peak_accuracy,mean_accuracy,mean_accuracy_rank")
for key, method_results in mean_accs.items():
    problem, method, fold_seed = key
    for rs_seed, acc in method_results.items():
        print(problem, method, fold_seed, rs_seed,
                min_narcs0[key][rs_seed], min_narcs1[key][rs_seed],
                peak_accs[key][rs_seed], acc, ranks[key][rs_seed], sep=",")
