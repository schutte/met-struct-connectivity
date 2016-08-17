"""Definitions related to regions of the Hammers atlas.

References
----------
Hammers, Allom, Koepp et al.  Three-dimensional maximum probability
atlas of the human brain, with particular reference to the temporal
lobe.  Human Brain Mapping, 19:224-247, 2003.

Gousias, Rueckert, Heckemann et al.  Automatic segmentation of brain
MRIs of 2-year-olds into 83 regions of interest.  Neuroimage 40:672-684,
2008.

"""

import numpy

# temporal, frontal and parietal lobes only
TFP_SUBSET = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 29, 30, 81, 82, 27, 28, 49, 50, 53, 54, 55, 56, 57, 58, 67,
    68, 69, 70, 71, 72, 51, 52, 59, 60, 61, 62, 31, 32])

# the same subset, but grouped into homologous sets
TFP_HOMAVG_SUBSET = numpy.array([
        [1, 3, 5, 7, 9, 11, 13, 15, 27, 29, 31, 49, 51, 53, 55, 57, 59, 61, 67, 69, 71, 81],
        [0, 2, 4, 6, 8, 10, 12, 14, 28, 30, 32, 50, 52, 54, 56, 58, 60, 62, 68, 70, 72, 82]
])

# ROI subset for the similarity analysis
SIM_SUBSET = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 37, 38, 39, 40, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 81, 82
])

def get_region_labels(subset=None):
    """Obtain a list of region labels, where the 0th element corresponds
    to region 1 of the Hammers atlas, the 1st to region 2, etc.

    Parameters
    ----------
    subset : array, optional
        If given, the elements of the array will be used as indices for
        the full list of region labels.

    Returns
    -------
    array
        An array of region labels, optionally restricted to the given
        subset of regions.

    """

    if subset is None:
        return numpy.array([
            'Hippocampus_R', 'Hippocampus_L',
            'Amygdala_R', 'Amygdala_L',
            'Temporal_Lobe_Ant_Med_R', 'Temporal_Lobe_Ant_Med_L',
            'Temporal_Lobe_Ant_Lat_R', 'Temporal_Lobe_Ant_Lat_L',
            'ParaHippocampal_R', 'ParaHippocampal_L',
            'Temporal_Sup_Post_L', 'Temporal_Sup_Post_R',
            'Temporal_MidInf_R', 'Temporal_MidInf_L',
            'Fusiform_R', 'Fusiform_L',
            'Cerebelum_R', 'Cerebelum_L',
            'Brainstem',
            'Insula_L', 'Insula_R',
            'Occipital_Lat_Rem_L', 'Occipital_Lat_Rem_R',
            'Cingulum_Ant_L', 'Cingulum_Ant_R',
            'Cingulum_Post_L', 'Cingulum_Post_R',
            'Frontal_Mid_L', 'Frontal_Mid_R',
            'Temporal_Lobe_Post_L', 'Temporal_Lobe_Post_R',
            'Parietal_InfLat_L', 'Parietal_InfLat_R',
            'Caudate_L', 'Caudate_R',
            'Accumbens_L', 'Accumbens_R',
            'Putamen_L', 'Putamen_R',
            'Thalamus_L', 'Thalamus_R',
            'Pallidum_L', 'Pallidum_R',
            'Corpus_Callosum',
            'Ventricle_Lat_R', 'Ventricle_Lat_L',
            'Ventricle_Horn_R', 'Ventricle_Horn_L',
            'Third_Ventricle',
            'Precentral_L', 'Precentral_R',
            'Straight_L', 'Straight_R',
            'Frontal_Ant_Orb_L', 'Frontal_Ant_Orb_R',
            'Frontal_Inf_L', 'Frontal_Inf_R',
            'Frontal_Sup_L', 'Frontal_Sup_R',
            'Postcentral_L', 'Postcentral_R',
            'Parietal_Sup_L', 'Parietal_Sup_R',
            'Lingual_L', 'Lingual_R',
            'Cuneus_L', 'Cuneus_R',
            'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R',
            'Frontal_Lat_Orb_L', 'Frontal_Lat_Orb_R',
            'Frontal_Post_Orb_L', 'Frontal_Post_Orb_R',
            'Subs_Nigra_L', 'Subs_Nigra_R',
            'Frontal_Subgenual_L', 'Frontal_Subgenual_R',
            'Frontal_Subcallosal_L', 'Frontal_Subcallosal_R',
            'Frontal_Pre-subgenual_L', 'Frontal_Pre-subgenual_R',
            'Temporal_Sup_L', 'Temporal_Sup_R'])
    elif subset.ndim == 2:
        # set of homologous pairs:
        # cut off the _L/_R part of the region labels
        return [s[:-2] for s in get_region_labels()[subset[0]]]
    else:
        return get_region_labels()[subset]
