Metabolic SICE-based classification with structural weighting
=============================================================
Michael Schutte <michael.schutte@uiae.at>
13 July 2016

Usage
-----

To perform a cross-validation experiment, make sure that the 'numpy',
'scipy', 'scikit-learn' and 'networkx' libraries are installed (they are
all present, for example, in the standard package list of the Anaconda
distribution) and that the 'sicelib' package is on the Python library
search path.  The latter can be achieved by setting the +PYTHONPATH+
environment variable, for example.  Then run a command following this
template:
[subs="quotes"]
-----
python classify.py 'method' 'class0' 'class1' 'folds' 'seed'
-----

The parameters have the following meanings:

'method'::
  Which type of structural weighting to apply, which subset of atlas
  regions to use, etc.  See below.
'class0', 'class1'::
  Names of the two classes to be discriminated.  These are the directory
  names found in the +../classes+ directory.  Originally, four choices
  are possible: +ad+, +ftld+, +hc+ and +mci+.
'folds'::
  Number of non-overlapping cross-validation folds.  The program will
  run 'folds' iterations, in which ('folds' - 1) folds are used as
  training data and the remaining fold for testing.  'folds' may be set
  to 0 to perform leave-one-out cross-validation.
'seed'::
  Seed integer for the random cross-validation split.  This provides for
  reproducibility of the results, as the same value for 'seed' will
  result in the same training–testing splits (as long as the
  pseudorandom number generator remains unchanged).  This parameter may
  be omitted if 'folds' is 0, since there is only one possible split for
  leave-one-out cross-validation.

Methods
-------

The 'method' argument is an underscore-separated string of options that
mostly influence the choice of regularization matrix.  The first part
should be the weighting scheme, one of:

+unweighted+::
  Scalar-valued regularization, i.e. “standard” SICE without any
  structural weighting.
+zero+::
  Like +unweighted+, but without penalization of diagonal elements of
  the estimate, such that column variances will always be computed
  without shrinkage.
+pow+'d'::
  Power-scheme weighting with degree 'd'.
+exp+'σ'::
  Exponential-scheme weighting with denominator 'σ'.

By default, one tract-count regularization matrix is calculated and used
per group.  Alternatively, the elementwise relative difference between
the two matrices may be used for both groups by passing the +diff+
parameter.  Alternatively, +distance+ causes the matrix to be treated as
a weighted graph and processed with Dijktra’s algorithm, in order to
take indirect structural connections into account.  In place of tract
counts, it is also possible to use fractional anisotropy; this requires
the addition of the `fa` switch.

By specifying `normsc`, each matrix element is divided by the sum of
fibres emanating from both regions.  This normalization method for
structural connectivity is only meaningful for tract-count matrices.

Unless told otherwise, +classify.py+ will use all regions of interest
implied by the columns of the PET data matrix (+pet_averages.npy+ in the
class directory), and by the rows and columns of the structural
connectivity matrices (+structural_connectivity.npy+ and
+fractional_anisotropy.npy+).  Since in many cases, the number of
regions is large compared to the number of subjects, it is frequently
desirable to limit this to a subset.  Two are built into the program and
may be selected through the 'method' parameter: `tfphammers` chooses 44
temporal, frontal and parietal lobe regions from the Hammers atlas,
whereas `homavghammers` selects the same regions but additionally
averages over homologous regions in both hemispheres (resulting in 22
regions).

Finally, random shuffling of the structural connectivity matrix, to
create a mismatch of regions between structure and metabolism, is done
by adding a `randstruct`'n' option (where 'n' is a seed).

Examples
--------

To perform classification based on standard SICE (without diagonal
penalties) on the healthy vs. Alzheimer’s problem, using five folds of
cross-validation with a random seed of 12345:
-----
python classify.py zero hc ad 5 12345
-----

To perform Alzheimer’s–vs.–FTLD classification based on
structure-weighted SICE, using an exponential function (σ=0.2) of the
difference between the shuffled groupwise fractional anisotropies as
regularization matrix, limiting the regions to a meaningful subset:
-----
python classify.py exp0.2_diff_fa_tfphammers_randstruct98765 ad ftld 0
-----
The latter command will perform leave-one-out cross-validation.

Evaluation
----------

The program will create a +class0class1_seed_folds_method.csv+ file in
the +results+ subdirectory with seven columns:

'fold'::
  The fold index.
'param'::
  The choice of regularization parameter from which the line was
  generated.  This will be +0+ for the ground truth.
'alpha0', 'alpha1'::
  Equal to 'param'.
'narcs0', 'narcs1'::
  The number of metabolic links (half of the number of non-diagonal,
  non-zero entries) contained in the sparse inverse covariance matrices
  estimated for each group.
'labels'::
  String of ++0++s and ++1++s; the class predictions, or the ground truth if
  'param' is +0+.

An example of how to read and interpret this file is shown in the
Jupyter notebook +evaluation.ipynb+.
