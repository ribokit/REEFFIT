from matplotlib.pylab import *
from map_analysis_utils import get_struct_types
"""
.. module:: likelihood
   :platform: Unix
   :synopsis: Log-likelihood functions for models used in the REEFFIT package

.. moduleauthor:: Pablo Cordero <dimenwarper@gmail.com>
"""


def factor_analysis_loglike(D_obs, D_fac, W, Psi, struct_types, lam_ridge=0.05):
    """Returns the likelihood of a factored model (predicted reactivities + weights + variances and the structure types)
    given the data.

        Args:
            D_obs (numpy array, nmeasurements x npositions): The observed data
            D_fac (numpy array, nstructs x npositions): The predicted reactivities for each structure in each position
            W (numpy array, nmeasurements x nstructs): The predicted structure weights for each measurement
            Psi (1d array of size npositions): The predicted variances per sequence position
            struct_types (2d array of characters, size npositions x nstructures): The structured types (paired or unpaired) per position, per structure. Use map_analysis_utils.get_struct_types(<dot-bracket-structures>) to get these easily.

        Returns:
            The log-likelihood of the model given the data
    """

    res = 0
    nmeas = D_obs.shape[0]
    npos = D_obs.shape[1]
    nstructs = W.shape[1]
    for i in xrange(npos):
        D_pred = dot(W, D_fac[:, i])
        diff = D_obs[:, i] - D_pred
        res += -0.5 * nmeas * Psi[i]
        res += -0.5 * dot(dot(diff, diag(array([1. / Psi[i]] * nmeas))), diff)
        for j in xrange(nstructs):
            if struct_types[i][j] == 'u':
                res += -2.0 * D_fac[j, i]
            else:
                res += -5.5 * D_fac[j, i]
    for j in xrange(nstructs):
        res += -lam_ridge*(W[j, :]**2).sum()

    return res


def test_factor_analysis_loglike():
    structs = ['....(((...)))', '(((.....)))..']
    struct_types = get_struct_types(structs)
    D_obs = abs(randn(13, 13)) * 2.0
    W = abs(randn(13, 2))
    D_fac = abs(randn(2, 13)) * 2.0
    Psi = [0.5] * 13
    print factor_analysis_loglike(D_obs, D_fac, W, Psi, struct_types)


if __name__ == '__main__':
    test_factor_analysis_loglike()
