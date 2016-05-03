from matplotlib.pylab import *

"""
.. module:: optimization
   :platform: Unix
   :synopsis: Classes and functions for optimizing factorization models

.. moduleauthor:: Pablo Cordero <dimenwarper@gmail.com>
"""


class BoltzmannFactorizationWeightSGD(object):
    """Stochastic gradient descent weight optimization routine of the Boltzmann Factorization model
    """

    def __init__(self, temperature=297.15, kB=0.0019872041, alpha=0.1):
        self.temperature = temperature
        self.kB = kB
        self.kBT = self.temperature * self.kB
        self.alpha = alpha


    def _get_factor_to_feature_dict(feature_to_factor_dict):
        res = defaultdict(list)
        for feature, factors in feature_to_factor_dict.iteritems():
            for f in factors:
                if feature not in res[f]:
                    res[f].append(feature)
        return res


    def _exp(self, fact_idx, energies, factor_to_feature_dict):
        return exp(energies[factor_to_feature_dict[fact_idx]].sum() / self.kBT)


    def _partition(self, factor_indices, energies, factor_to_feature_dict):
        res = 0
        for fact_idx in factor_indices:
            res += self._exp(fact_idx, energies, factor_to_feature_dict)
        return res


    def _gradient(self, D_obs, D_fac, Psi, energies, feat_idx, factors_in_measurement, feature_to_factor_dict, factor_to_feature_dict):
        res = 0
        for j in xrange(D_obs.shape[0]):
            Z = self._partition(factors_in_measurement[j], energies, factor_to_feature_dict)
            for i in xrange(D_obs.shape[1]):
                A = 0
                for fact_idx in xrange(factors_in_measurement[j]):
                    A += (self._exp(fact_idx, energies, factor_to_feature_dict) / Z) * D_fac[fact_idx, i]
                A -= D_obs[j, i]

                B = 0
                for fact_idx in feature_to_factor_dict[feat_idx]:
                    B += self._exp(fact_idx, energies, factor_to_feature_dict)

                C = 0
                for fact_idx in feature_to_factor_dict[feat_idx]:
                    tmp = -(self.kBT * self.exp(fact_idx, energies, factor_to_feature_dict)) / Z
                    tmp += B * self.exp(fact_idx, energies, factor_to_feature_dict) / (Z**2)
                    C = tmp * D_fac[fact_idx, i]

                res += (2. / Psi[i]) * A * C
        return res
        # return subtraction_part * quotient_part


    def _calculate_weights(energies, factors_in_measurement, factor_to_feature_dict):
        nmeas = len(factors_in_measurement)
        nfacs = len(factor_to_feature_dict)
        W = zeros([nmeas, nfacs])
        for j in xrange(nmeas):
            for fact_idx in xrange(nfacs):
                Z = self._partition(factors_in_measurement[j], energies, factor_to_feature_dict)
                W[fact_idx, j] = self._exp(fact_idx, energies, factor_to_feature_dict) / Z
        return W


    def fit(self, D_obs, D_fac, Psi, factors_in_measurement, energies_0=None, max_iter=100, feature_to_factor_dict=None):
        if feature_mappings is None:
            # Trivial feature mappings are one feature per factor
            feature_to_factor_dict = dict([(i, i) for i in xrange(D_fac.shape[0])])
        factor_to_feature_dict = self._get_factor_to_feature_dict(feature_to_factor_dict)
        # do the stochastic gradient descent
        finished = False
        feature_indices = feature_to_factor_dict.keys()
        if energies_0 is None:
            energies_0 = randn([len(feature_indices)])
        energies_prev = energies_0
        for t in xrange(max_iter):
            shuffle(feature_indices)
            for feat_idx in feature_indices:
                energies_curr[feat_idx] = energies_prev[feat_idx] - self.alpha * self._gradient(D_obs, D_fac, Psi, energies_prev, feat_idx, factors_in_measurement, feature_to_factor_dict, factor_to_feature_dict)
                if self._check_convergence(energies_prev, energies_curr):
                    finished = True
                if finished:
                    break
                energies_prev = energies_curr
        W = self._calculate_weights(energies_curr, factors_in_measurement, factor_to_feature_dict)
        return energies_curr, W

