#This file is part of the REEFFIT package.
#    Copyright (C) 2013 Pablo Cordero <tsuname@stanford.edu>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
from matplotlib.pylab import *
"""
.. module:: optimization 
   :platform: Unix
   :synopsis: Classes and functions for optimizing factorization models

.. moduleauthor:: Pablo Cordero <dimenwarper@gmail.com>
"""

class BoltzmannFactorizationEnergiesSGD(object):
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
        return exp(energies[factor_to_feature_dict[fact_idx]].sum()/self.kBT)

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
                    A += (self._exp(fact_idx, energies, factor_to_feature_dict)/Z)*D_fac[fact_idx,i]
                A -= D_obs[j,i]
                
                B = 0
                for fact_idx in feature_to_factor_dict[feat_idx]:
                    B += self._exp(fact_idx, energies, factor_to_feature_dict)
                
                C = 0
                for fact_idx in feature_to_factor_dict[feat_idx]:
                    tmp = -(self.kBT*self.exp(fact_idx, energies, factor_to_feature_dict))/Z
                    tmp += B*self.exp(fact_idx, energies, factor_to_feature_dict)/(Z**2)
                    C = tmp * D_fac[fact_idx,i]

                res += (2./Psi[i]) * A * C
        return res

            
        return subtraction_part * quotient_part

    def _calculate_weights(energies, factors_in_measurement, factor_to_feature_dict):
        nmeas = len(factors_in_measurement)
        nfacs = len(factor_to_feature_dict)
        W = zeros([nmeas, nfacs])
        for j in xrange(nmeas):
            for fact_idx in xrange(nfacs):
                Z = self._partition(factors_in_measurement[j], energies, factor_to_feature_dict)
                W[fact_idx,j] = self._exp(fact_idx, energies, factor_to_feature_dict)/Z
        return W

    def fit(self, D_obs, D_fac, Psi, factors_in_measurement, energies_0=None, max_iter=100, feature_to_factor_dict=None):
        if feature_mappings is None:
            # Trivial feature mappings are one feature per factor
            feature_to_factor_dict = dict[(i,i) for i in xrange(D_fac.shape[0])])
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
                energies_curr[feat_idx] = energies_prev[feat_idx] - self.alpha *  self._gradient(D_obs, D_fac, Psi, energies_prev, feat_idx, factors_in_measurement, feature_to_factor_dict, factor_to_feature_dict)
                if self._check_convergence(energies_prev, energies_curr):
                    finished = True
                if finished:
                    break
                energies_prev = energies_curr
        W = self._calculate_weights(energies_curr, factors_in_measurement, factor_to_feature_dict)
        return energies_curr, W

#TODO create parallelizable class, properly initialize factor_types, slack_vars, etc. when None
class BoltzmannFactorizationFactorsLinear(object)
    def __init__(self):
        pass


    def fit(self, D_obs, W, Psi, idx_feature_map, factor_types=None, slack_vars=None, slack_prior_factor=None, prior_factors=None, indices=None):
        if indices is None:
            indices = range(D_obs.shape[1])
        self.parallelize(self._fit_idx(self, idx, D_obs, W, Psi, idx_feature_map, factor_types=factor_types, slack_vars=slack_vars, slack_prior_factor=slack_prior_factor, prior_factors=prior_factors), indices)

    def _fit_idx(self, idx, D_obs, W, Psi, idx_feature_map, factor_types=None, slack_vars=None, slack_prior_factor=None, prior_factors=None):
        """Calculate factor/feature and slack variable values given factor weights
        
            Args:
                idx (int): The sequence position.
                W (2d numpy array): The structure weight matrix.
                Psi (1d numpy array): Variances per position
                D_obs (2d numpy array): The mapping D_obs matrix.
                factor_types (list): A list of structural types for each structure in this position.
                slack_vars (list): A list of matrices that contain the slack sites for each structure in this position.
                bp_dist (2d numpy array): A base-pair distance matrix between all structures.
                seq_indices (list): List of indices that have the sequence positions to be considered in the analysis.


            Returns:
                tuple. In order:
                * The column of reactivities inferred for this sequence position
                * The covariance matrix of the reactivities for this sequence position
                * The column of standard deviations for this position
                * The matrices of slack maps inferred for this position
        """
        # This basically boils down to a system of linear equations
        # Each entry is either a hidden factor value or a slack variable value
        # to be solved
        print 'Solving MAP feature/slack values for index %s' % idx
        nslack = 0
        nmeas = D_obs.shape[0]
        nfactors = len(factor_types[0])
        slack_idx_dict = {}
        nfeatures = 0
        feature_idx_dict = {}
        for feat_idx in xrange(len(idx_feature_map)):
            if (idx, feat_idx) in idx_feature_map:
                feature_idx_dict[nfeatures] = feat_idx
                nfeatures += 1
        # Some helper functions to make the code more compact

        def feature_idx(feat_idx):
            return idx_feature_map[(idx, feature_idx_dict[feat_idx])]

        def slack_feature_idx(feat_idx, j):
            return [m for m in feature_idx(feat_idx) if slack_vars[m][j, idx]]

        def check_for_slack_var(feat_idx, j):
            for m in feature_idx(feat_idx):
                if slack_vars[m][j, idx]:
                    return True
            return False
        sum_fun = lambda x: if isinstance(x, collections.Iterable): sum(x) else: x

        i = nfeatures
        for j in xrange(nmeas):
            for s in xrange(nfeatures):
                    if check_for_slack_var(s, j):
                        nslack += 1
                        slack_idx_dict[i] = (j, s)
                        slack_idx_dict[(j,s)] = i
                        i += 1

        dim = nfeatures + nslack
        A = zeros([dim, dim])
        b = zeros([dim])
        def fill_matrices(A, b, slack_prior):
            # We'll start by indexing the reactivity hidden variables by structure
            for p in xrange(A.shape[0]):
                for s in xrange(nfeatures):
                    if p < nfeatures:
                        for j in xrange(nmeas):
                            A[p,s] += sum_fun(W[j,feature_idx(p)])*sum_fun(W[j,feature_idx(s)])
                        """
                        # We can decide later if we add the lam_factors regularizer
                        if p == s:
                            for s1 in xrange(nfeatures):
                                if s != s1:
                                    A[p,s] += 4*self.lam_factors*1/feature_dist[feature_idx_dict[s1],feature_idx_dict[s]]
                        else:
                            A[p,s] -= 4*self.lam_factors*1/self.feature_dist[feature_idx_dict[p],feature_idx_dict[s]]
                        """

                    else:
                        j, s2 = slack_idx_dict[p]
                        A[p,s] = sum_fun(W[j,feature_idx(s)])*sum_fun(W[j,feature_idx(s2)])

            for s in xrange(nfeatures):
                for m in feature_idx(s):
                    if factor_types[idx][m] != factor_types[idx][feature_idx(s)[0]]:
                        raise ValueError('MOTIF DECOMPOSITION FAILED! STRUCTURES IN POSITION %s HAVE DIFFERENT STRUCTURE TYPES!!! %s' % (idx, factor_types[idx]))
                b[s] = -Psi[idx]/prior_factors[factor_types[idx][feature_idx(s)[0]]]
                for j in xrange(nmeas):
                    b[s] += sum_fun(W[j,feature_idx(s)])*D_obs[j,idx]
            # Then, the slack maps. No Lapacian priors here, we use Gaussian (i.e. 2-norm) priors
            # for easier calculations
            for p in xrange(A.shape[0]):
                for j in xrange(nmeas):
                    for s in xrange(nfeatures):
                        if check_for_slack_var(s, j):
                            if p < nfeatures:
                                A[p, slack_idx_dict[(j,s)]] = sum_fun(W[j,slack_feature_idx(p, j)])
                            else:
                                j2, s2 = slack_idx_dict[p]
                                if j == j2:
                                    #A[p,slack_idx_dict[(j,s)]] = W[j,s]
                                    if s == s2:
                                        A[p,slack_idx_dict[(j,s)]] = (sum_fun(W[j,slack_feature_idx(s, j)]) + slack_prior_loc)**2 + Psi[idx]/slack_prior
                                    else:
                                        A[p,slack_idx_dict[(j,s)]] = sum_fun(W[j2,slack_feature_idx(s2, j2)])*sum_fun(W[j,slack_feature_idx(s, j)])


            for j in xrange(nmeas):
                for s in xrange(nfeatures):
                    if check_for_slack_var(s, j):
                        b[slack_idx_dict[(j,s)]] = sum_fun(D_obs[j,idx]*W[j,feature_idx(s)])
            return A, b

        A, b = fill_matrices(A, b, slack_prior_factor)
        print 'Solving the linear equation system'
        D_hid = zeros([nfactors])
        C_hid = zeros([nmeas, nfactors])
        def f(x):
            return ((dot(A,x) - b)**2).sum()
        def fprime(x):
            return dot(dot(A,x) - b, A.T)
        solved = False
        tries = 0
        while not solved:
            solved = True
            bounds = [(0.001, D_obs.max())]*nfeatures + [(-10,10)]*(A.shape[0] - nfeatures)
            x0 = [0.002 if factor_types[idx][feature_idx(s)[0]] == 'p' else 1. for s in xrange(nfeatures)] + [0.0 for i in xrange(A.shape[0] - nfeatures)]
            Acvx = cvxmat(A)
            bcvx = cvxmat(b)
            n = Acvx.size[1]
            I = cvxmat(0.0, (n,n))
            I[::n+1] = 1.0
            G = cvxmat([-I])
            h = cvxmat(n*[0.001])
            dims = {'l':n, 'q':[], 's':[]}
            try:
                #x = array(solvers.coneqp(Acvx.T*Acvx, -Acvx.T*bcvx, G, h, dims, None, None, {'x':cvxmat(x0)})['x'])
                #x = optimize.fmin_slsqp(f, x0, fprime=fprime, bounds=bounds, iter=2000)
                x = optimize.fmin_l_bfgs_b(f, x0, fprime=fprime, bounds=bounds)[0]
                #x = linalg.solve(A, b)
            except ValueError:
                solved=False
            for s in xrange(nfeatures):
                if x[s] <= 0.001:
                    D_hid[feature_idx(s)] = 0.001
                else:
                    if factor_types[idx][feature_idx(s)[0]] == 'p' and x[s] > 0.2:
                        D_hid[feature_idx(s)] = 0.1
                    else:
                        D_hid[feature_idx(s)] = x[s]

            DDT_hid = dot(mat(D_hid).T, mat(D_hid))
            if (DDT_hid < 0).ravel().sum() > 1:
                solved = False

            if not solved:
                # This means that we need to rescale the slack prior
                tries += 1
                if tries > 100:
                    new_prior_factor = slack_prior_factor/(tries - 100)
                else:
                    new_prior_factor = slack_prior_factor + 0.1*tries
                if tries > 0:
                    print 'Could not figure out a good slack prior'
                    print 'Blaming DDT_hid singularities on D_obs similarity'
                    print 'Adding a bit of white noise to alleviate'
                    A, b = fill_matrices(A, b, slack_prior_factor)

                   
                print 'MAP system was not solved properly retrying with different slack priors'
                print 'Number of slacks is %s' % nslack
                print 'Changing prior factor to %s' % (new_prior_factor)
                A, b = fill_matrices(A, b, new_prior_factor)

        for s in xrange(nfeatures):
            for j in xrange(nmeas):
                if isinstance(feature_idx, collections.Iterable)
                    if slack_vars[i][j,idx]:
                        C_hid[j,i] = x[s]
                    else:
                        C_hid[j,i] = nan
                else:
                    if slack_vars[feature_idx(s)][j,idx]:
                        C_hid[j,i] = x[s]
                    else:
                        C_hid[j,i] = nan

            for s in xrange(nfeatures):
                for j in xrange(nmeas):
                    if isinstance(feature_idx, collections.Iterable)
                        for i in feature_idx(s):
                            if slack_vars[i][j,idx]:
                                C_hid[j,i] += x[slack_idx_dict[(j,s)]]
                    else:
                        if slack_vars[feature_idx(s)][j,idx]:
                            C_hid[j,i] += x[slack_idx_dict[(j,s)]]

        sigma_D_hid = mat(zeros([nfactors]))
        for s in xrange(nfeatures):
            sigma_D_hid[0,feature_idx(s)] = sqrt(Psi[idx]/(W[:,feature_idx(s)]**2).sum() + 1e-10))

        return D_hid, DDT_hid, sigma_D_hid, C_hid




