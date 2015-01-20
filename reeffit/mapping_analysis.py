#This file is part of the REEFFIT package.
#    Copyright (C) 2013 Pablo Cordero <tsuname@stanford.edu>
import pdb
import inspect
import pymc
import joblib
import sys
import map_analysis_utils as utils
import rdatkit.secondary_structure as ss
from rdatkit import mapping
import scipy.cluster.hierarchy as sphclust
from scipy.sparse import csr_matrix
import itertools
from random import choice, sample
from matplotlib.pylab import *
from numpy import lib
from reactivity_distributions import *
from collections import defaultdict, Counter
from scipy.stats import gamma
from scipy.stats.kde import gaussian_kde
import scipy.spatial.distance
from cvxopt import solvers
from scipy import optimize
from cvxopt import matrix as cvxmat
"""
.. module:: mapping_analysis 
   :platform: Unix
   :synopsis: Classes and functions for analyzing mapping data

.. moduleauthor:: Pablo Cordero <dimenwarper@gmail.com>
"""


class MappingAnalysisMethod(object):
    """Base class for methods that model mapping data. 
    In general, methods that model mapping data have a set of 
    sequences, a set of structures, and the mapping data reactivities.
    """

    # TODO Should probably ma ke wt_indices, mutpos, c_size, and seqpos_range optional arguments
    def __init__(self, data, structures, sequences, energies, wt_indices, mutpos, c_size, seqpos_range):
        """Constructor that take a mapping reactiviy matrix, a list of structures, sequences, and energies.
        For mutate-and-map data, this constructor also accepts wild type indices (wt_indices), mutation positions
        (mutpos), and contact size (c_size) to build the structure-wise contact maps.
        A range of sequence positions to focus on can also be specificed using seqpos_range (positions outside this
        range will be ignored in the analysis)
        
        Args:

           data (2d numpy array): Mapping reactivity data (rows are experiments, columns are sequence positions)
           structures (list): List of structures that will be used to model the data
           sequences (list): List of sequences for the constructs in each experimemt. Must be the same length as the number of rows in data.
           energies (2d numpy array): Energies of the structures, in the context of each experiment. Number of rows equals the number of rows in data and number of columns is the number of structures.
           wt_indices (list): List of wild-type indices (for mutate-and-map experiments)
           mutpos (list): List of lists, each list containing the positions that are mutated in each experiment (for mutate-and-map experiments).
           c_size (int): Size of 'contact neighborhoods' induced by the mutations (for mutate-and-map experiments)
           seqpos_range (list): List of two elements that specifies the starting and ending sequence positions to take into account when modeling.

        """
        self.seqpos_range = seqpos_range
        if self.seqpos_range:
            self.data = matrix(data)[:,self.seqpos_range[0]:self.seqpos_range[1]]
            self._origdata = data
            self.sequences = [s[self.seqpos_range[0]:self.seqpos_range[1]] for s in sequences]
        else:
            self.data = matrix(data)
            self._origdata = self.data
            self.sequences = sequences
        self.wt = self.sequences[wt_indices[0]]
        self.wt_indices = wt_indices
        self.energies = energies
        self.mutpos = mutpos
        self.c_size = c_size
        self.set_structures(structures)
        self._data = self.data


        self._options = {}

    def set_option(self, name, value):
        self._options[name] = value

    def get_option(self, name):
        return self._options[name]

    def set_structures(self, structures):
        """Sets the structures and adjusts them to self.seqpos_range as well as calculates 'contact neighborhood' matrices

        Args:
           structures (list): A list of structures
        """
        # Get contact sites
        self._origstructures = structures
        self._origstruct_types = utils.get_struct_types(structures)
        nmeas = self._origdata.shape[0]
        npos = self._origdata.shape[1]
        nstructs = len(structures)
        if self.seqpos_range:
            self.contact_sites = utils.get_contact_sites(structures, self.mutpos, nmeas, npos, self.c_size, restrict_range=self.seqpos_range)
            self.structures = [s[self.seqpos_range[0]:self.seqpos_range[1]] for s in structures]
            #self.contact_sites = utils.get_contact_sites(structures, self.mutpos, nmeas, npos, self.c_size)
            #self.structures = structures
            """
            new_structures = []
            for struct in self.structures:
                stack = []
                new_struct = ['']*len(struct)
                for i, s in enumerate(struct):
                    if s == ')':
                        if len(stack) == 0:
                            new_struct[i] = '.'
                        else:
                            new_struct[i] = s
                            stack.pop()
                    elif s == '(':
                        stack.append(i)
                        new_struct[i] = s
                    else:
                        new_struct[i] = s
                for i in stack:
                    new_struct[i] = '.'
                new_structures.append(''.join(new_struct))
            self.structures = new_structures
            """
        else:
            self.structures = structures
            self.contact_sites = utils.get_contact_sites(structures, self.mutpos, nmeas, npos, self.c_size)
        self.struct_types = utils.get_struct_types(self.structures)

    def _cluster_data(self, cluster_factor):
        """Clusters the reactivity data matrix. Useful to reduce complexity/redundancy in dataset.

        Args:
            cluster_factor (int): The clustering factor. The higher clustering factor, the more clusters, the lower, less clusters. Controls the dendogram cutting threshold of the hierarchical clustering used in this function.

        Returns:
            tuple. The first element is a list of indices that were chosen as the cluster medoids. The second element is a list that maps data row indices to the chosen medoids of the first element.
        """
        print 'Clustering data'
        distance_mat = scipy.spatial.distance.pdist(self.data, metric='seuclidean')
        Z = sphclust.linkage(distance_mat)
        maxdist = distance_mat.max()
        mincrit = inf
        bestclusts = []
        for t in arange(0, maxdist, maxdist/10.):
            clusters = sphclust.fcluster(Z, t, criterion='distance')
            # Heuristic: number of clusters should be at the closer to cluster_factor times the number of structures
            numclusts = float(len(set(clusters)))
            diff = abs(numclusts - cluster_factor*len(self.structures))
            #cluster_counts = Counter(clusters).values()
            #diff = max(cluster_counts) - min(cluster_counts) + numclusts
            if mincrit > diff:
                mincrit = diff
                bestclusts = clusters
        visitedclusts = []
        chosenindices = {}
        for i, c in enumerate(bestclusts):
            if c not in visitedclusts:
                chosenindices[c] = i
                visitedclusts.append(c)
        measindices = chosenindices.values()
        print 'Chose %s indices, which were: %s' % (len(chosenindices), chosenindices)
        return chosenindices, bestclusts

class FAMappingAnalysis(MappingAnalysisMethod):
    """Factor analysis method for modeling mapping data.
    """

    FACTOR_ESTIMATION_METHODS = ['fanalysis', 'hclust']
    MODEL_SELECT_METHODS = ['heuristic', 'bruteforce']
    #TODO The njobs argument shouldn't be used this way. Instead, it should be an annotation that wraps for loops in parallel jobs using joblib
    def __init__(self, data, structures, sequences, wt_indices=[0], mutpos=[], concentrations=[], kds=[], energies=[], c_size=3, seqpos_range=None, lam_reacts=0, lam_weights=0, lam_mut=0, lam_ridge=0.01, njobs=None):
        """Constructor that takes data, structures, and sequences, as well as other optional arguments

        Args:
            data (2d numpy array): The mapping reactivity data. Rows are experiments, columns are sequence positions.
            structures (list): A list of structures to model the data with.
            sequences (list): A list of sequences, one corresponding to each experiment.

        Kwargs:
            wt_indices (list): A list of wild type indices (for mutate-and-map data). Default is [0] (the first row of the data corresponds to the wild type).
            mutpos (list): A list of lists of positions of the mutations in each experiment (for mutate-and-map data). Default is an empty list (no mutations).
            concentrations (list): A list of concentrations of chemical (e.g. a small molecule that an RNA binds). Used in conjuntion with kds argument.
            kds (list):  A list of dissociation constants of the chemical (e..g a small molecule that an RNA binds). Used in conjunction with concentrations argument.
            c_size (int): The size of the neighborhood of the 'contact map' induced by the mutations (for mutate-and-map experiments).
            seqpos_range (list): A list of two elements: first is the starting position and last is the ending position of the data to be analyzed.
            lam_reacts (float): Regularization parameter for coallescing reactivities of structures based on structural similarities.
            lam_weights (float): Regularization parameter for an L2 penalty that forces equality between inferred weights and those calculated by RNAstructure.
            lam_ridge (float): Regularization parameter for an L2 (ridge) penalty enforcing smooth sparsity on the weights. Default is 0.
            lam_mut (float): Regularization parameter for an L2 penalty that forces equality between inferred mutant $\Delta \Delta G$ values and those given by RNAstructure/ViennaRNA.
            """
        MappingAnalysisMethod.__init__(self, data, structures, sequences, energies, wt_indices, mutpos, c_size, seqpos_range)
        self.njobs = njobs
        self.concentrations = concentrations
        self.kds = kds
        self.unpaired_pdf = SHAPE_unpaired_pdf
        self.paired_pdf = SHAPE_paired_pdf

        self.lam_reacts = lam_reacts
        self.lam_weights = lam_weights
        self.lam_mut = lam_mut
        self.lam_ridge = lam_ridge
        self.logpriors = 0
        
        self.motif_decomposition = 'none'
    

    def perform_motif_decomposition(self, type='motif'):
        """Perform motif decomposition. Type can be "element" or "motif"
           
           Kwargs:
               type (str): Possible values are 'motif' for decomposing by motif, 'element' for decomposing by secondary structure element (i.e. all helices at position X will have the same reactivity at X, or 'none'/None for no motif decomposition.
           
        """
        if type == 'none' or type is None:
            print 'Skipping motif decomposition'
            return
        print 'Starting motif decomposition'
        if self.seqpos_range != None:
            offset = -self.seqpos_range[0]
        else:
            offset = 0
        if type == 'motif':
            bytype = False
        else:
            bytype = True
        self.pos_motif_map, self.motif_ids, self.motif_dist = utils.get_minimal_overlapping_motif_decomposition(self._origstructures, bytype=bytype, offset=offset, sequences=self.sequences)
        self.nmotpos = []
        self.posmap = defaultdict(list)
        for i in xrange(self.data.shape[1]):
            nmotifs = 0
            for midx in xrange(len(self.motif_ids)):
                for seq_idx in xrange(len(self.sequences)):
                    if (i, midx, seq_idx) in self.pos_motif_map:
                        self.posmap[(i, seq_idx)].append(self.motif_ids[midx])
                        nmotifs += 1
                        break
            self.nmotpos.append(nmotifs)

        self.structures_by_motif = {}
        self.motifs_by_structure = {}
        for pos__motif_idx, struct_indices in self.pos_motif_map.iteritems():
            pos, motif_idx, meas_idx = pos__motif_idx
            for struct_idx in struct_indices:
                if (struct_idx, meas_idx) not in self.motifs_by_structure:
                    self.motifs_by_structure[struct_idx, meas_idx] = set()
                if (motif_idx, meas_idx) not in self.structures_by_motif:
                   self.structures_by_motif[motif_idx, meas_idx] = set()
                self.structures_by_motif[motif_idx, meas_idx].add(struct_idx)
                self.motifs_by_structure[struct_idx, meas_idx].add(motif_idx)

        # We want to convert sets to lists for easy indexing
        for k, v in self.structures_by_motif.iteritems():
            self.structures_by_motif[k] = list(v)
        for k, v in self.motifs_by_structure.iteritems():
            self.motifs_by_structure[k] = list(v)
        for k, v in self.motifs_by_structure.iteritems():
            self.motifs_by_structure[k] = list(v)

        self.motif_decomposition = type
        print 'Number of motifs per position: %s' % self.nmotpos


    def set_priors_by_rvs(self, unpaired_rvs, paired_rvs):
        """Sets prior distributions for paired/unpaired reactivities using the data to be modeled. This is done as follows:
           1. Sets paired data as anything that is below mean of data matrix and unpaired data as the rest.
           2. Further seeds paired and unpaired data with random variables as sampled by the unpaired_rvs and paired_rvs.
           3. Uses paired and unpaired data to create an empirical distribution using kernel density estimation with gaussian kernels.
              This is set as the self.unpaired_pdf and self.paired_pdf functions.

           Returns:
               Two lists, one with paired and the other with unpaired data that were used to create the prior probability distributions.
        """
        unpaired_data = self.data[self.data > self.data.mean()].tolist()[0]
        paired_data = self.data[self.data <= self.data.mean()].tolist()[0]
        unpaired_pdf_data = [unpaired_rvs() for i in xrange(len(unpaired_data))]
        paired_pdf_data = [paired_rvs() for i in xrange(len(paired_data))]
        unpaired_data += unpaired_pdf_data
        paired_data += paired_pdf_data
        self.unpaired_pdf = gaussian_kde(unpaired_data)
        self.paired_pdf = gaussian_kde(paired_data)
        return unpaired_data, paired_data


    def set_structure_clusters(self, structure_clusters, struct_medoid_indices, struct_weights_by_clust=[]):
        """Cluster structures together to get a more coarse-grained estimate
        """
        self.structure_clusters = structure_clusters
        # It is more convenient internally to have struct_mediod_indices as a list
        # with the same order as self.structure_clusters.key()
        self.struct_medoid_indices = [struct_medoid_indices[c] for c in self.structure_clusters]
        cluster_indices = {}
        # TODO this is too inefficient, maybe there's a smarter way to get
        # the indices, or should we ask the function caller to provide them?
        for c, structs in structure_clusters.iteritems():
            cluster_indices[c] = [self.structures.index(s) for s in structs]
        self.cluster_struct_types = []
        for i in xrange(len(self.struct_types)):
            self.cluster_struct_types.append([])
            for j, structs in enumerate(structure_clusters.values()):
                self.cluster_struct_types[i].append([])
                for struct in structs:
                    if struct[i] == '.':
                        self.cluster_struct_types[i][j].append('u')
                    else:
                        self.cluster_struct_types[i][j].append('p')
        if len(struct_weights_by_clust) > 0:
            self.W = zeros([self.data.shape[0], len(structure_clusters)])
            for i, c in enumerate(struct_weights_by_clust):
                self.W[:,i] = struct_weights_by_clust[c].sum(axis=1)
            self.struct_weights_by_clust = struct_weights_by_clust
        else:
            self.W, self.struct_weights_by_clust = utils.calculate_weights(self.energies, clusters=cluster_indices)
    
    # TODO this optimization is useful for other purposes, should extract and wrap it in an external function/class!
    def hard_EM_vars(self, idx, W, Psi_inv, data, struct_types, contact_sites, seq_indices):
        """Calculate expectation maximization variables from the E-step (getting the structure reactivities and variances). This is the hard EM version, returning a MAP estimate for the reactivities.
        
            Args:
                idx (int): The sequence position.
                W (2d numpy array): The structure weight matrix.
                Psi_inv (2d numpy array): The inverse of the a priori covariance matrix.
                data (2d numpy array): The mapping data matrix.
                struct_types (list): A list of structural types for each structure in this position.
                contact_sites (list): A list of matrices that contain the contact sites for each structure in this position.
                seq_indices (list): List of indices that have the sequence positions to be considered in the analysis.


            Returns:
                tuple. In order:
                * The column of reactivities inferred for this sequence position
                * The covariance matrix of the reactivities for this sequence position
                * The column of standard deviations for this position
                * The matrices of contact maps inferred for this position
        """
        # This basically boils down to a system of linear equations
        # Each entry is either a hidden reactivity or a hidden contact
        # to be solved
        ncontacts = 0
        # No Lapacian priors here, we use exponential (for hidden reactivities) and Gaussian (i.e. 2-norm) for contacts for easier calculations
        prior_factors = {}
        # TODO These should be exposed as option to choose from rather than hard coded
        # "Naive" priors
        prior_factors['u'] = 0.5
        prior_factors['p'] = 7.5
        # These are the true priors from the RMDB
        prior_factors['u'] = 2.0
        prior_factors['p'] = 5.5
        contact_prior_factor = 0.5
        contact_prior_loc = 0.5
        # STRONG priors
        #prior_factors['u'] = 0.25
        #prior_factors['p'] = 50.5
        nmeas = W.shape[0]
        nstructs = W.shape[1]
        contact_idx_dict = {}
        if self.motif_decomposition != 'none':
            nmotifs = 0
            motif_idx_dict = {}
            for midx in xrange(len(self.motif_ids)):
                meas_list = []
                for meas_idx in xrange(nmeas):
                    if (seq_indices[idx], midx, meas_idx) in self.pos_motif_map:
                        meas_list.append(meas_idx)
                        motif_idx_dict[nmotifs, meas_idx] = midx
                if len(meas_list) > 0:
                    nmotifs += 1
            # Some helper functions to make the code more compact
            if self.motif_decomposition == 'element' or self.motif_decomposition == 'motif':

                def motifidx(midx, meas_idx):
                    if (midx, meas_idx) not in motif_idx_dict:
                        return []
                    motif_idx = motif_idx_dict[midx, meas_idx]
                    if (seq_indices[idx], motif_idx, meas_idx) not in self.pos_motif_map:
                        return []
                    else:
                        return self.pos_motif_map[(seq_indices[idx], motif_idx, meas_idx)]

                def contact_motifidx(midx, j):
                    return [m for m in motifidx(midx, j) if contact_sites[m][j, idx]]

                def motif_struct_type(idx, midx):
                    for meas_idx in xrange(nmeas):
                        if motifidx(midx, meas_idx):
                            return struct_types[idx][motifidx(midx, meas_idx)[0]]

                def check_contact_site(midx, j):
                    try:
                        for m in motifidx(midx, j):
                            if contact_sites[m][j, idx]:
                                return True
                        return False
                    except KeyError:
                        #If the motif is not in measurement j, then return fals
                        return False
                sum_fun = lambda x: x.sum()

            i = nmotifs
            for j in xrange(nmeas):
                for s in xrange(nmotifs):
                    if check_contact_site(s, j):
                        ncontacts += 1
                        contact_idx_dict[i] = (j, s)
                        contact_idx_dict[(j,s)] = i
                        i += 1

            dim = nmotifs 
        else:
            i = nstructs
            for j in xrange(nmeas):
                for s in xrange(nstructs):
                        if contact_sites[s][j, idx]:
                            ncontacts += 1
                            contact_idx_dict[i] = (j, s)
                            contact_idx_dict[(j,s)] = i
                            i += 1
            dim = nstructs + ncontacts
        A = zeros([dim, dim])
        b = zeros([dim])
        def fill_matrices(A, b, contact_prior):
            if self.motif_decomposition != 'none':
                nstruct_elems = nmotifs
            else:
                nstruct_elems = nstructs
            # We'll start by indexing the reactivity hidden variables by structure/motif
            for p in xrange(A.shape[0]):
                for s in xrange(nstruct_elems):
                    if p < nstruct_elems:
                        if self.motif_decomposition != 'none':
                            for j in xrange(nmeas):
                                A[p,s] += sum_fun(W[j,motifidx(p, j)])*sum_fun(W[j,motifidx(s, j)])
                            """
                            if p == s:
                                #TODO This is buggy!! and regularizing by motif structural distance doesn't help anyway Remove?
                                for s1 in xrange(nstruct_elems):
                                    if s != s1:
                                        A[p,s] += 4*self.lam_reacts*1/self.motif_dist[motif_idx_dict[s1],motif_idx_dict[s]]
                            else:
                                A[p,s] -= 4*self.lam_reacts*1/self.motif_dist[motif_idx_dict[p],motif_idx_dict[s]]

                        else:
                            A[p,s] = dot(W[:,s], W[:,p])
                            if p == s:
                                for s1 in xrange(nstruct_elems):
                                    if s != s1:
                                        A[p,s] += 4*self.lam_reacts
                            else:
                                A[p,s] -= 4*self.lam_reacts

                            """
                    else:
                        j, s2 = contact_idx_dict[p]
                        if self.motif_decomposition != 'none':
                            A[p,s] = sum_fun(W[j,motifidx(s, j)])*sum_fun(W[j,motifidx(s2, j)])
                        else:
                            A[p,s] = W[j,s]*W[j,s2]
            for s in xrange(nstruct_elems):
                if self.motif_decomposition != 'none':
                    for m in motifidx(s, 0):
                        if struct_types[idx][m] != struct_types[idx][motifidx(s, 0)[0]]:
                            raise ValueError('MOTIF DECOMPOSITION FAILED! STRUCTURES IN POSITION %s HAVE DIFFERENT STRUCTURE TYPES!!! %s' % (idx, struct_types[idx]))
                    b[s] = 0.5*prior_factors[motif_struct_type(idx, s)]/Psi_inv
                    for j in xrange(nmeas):
                        b[s] += sum_fun(W[j,motifidx(s, j)])*data[j,idx]
                else:
                    b[s] = prior_factors[struct_types[idx][s]]/Psi_inv + dot(W[:,s], data[:,idx])
            # Then, the contact maps. No Lapacian priors here, we use Gaussian (i.e. 2-norm) priors
            # for easier calculations
            for p in xrange(nstruct_elems, A.shape[0]):
                jp, sp = contact_idx_dict[p]
                for s in xrange(nstruct_elems):
                    if self.motif_decomposition != 'none':
                        A[p,s] = sum_fun(W[jp, contact_motifidx(sp, jp)])
                        if check_contact_site(s, jp):
                            A[p, contact_idx_dict[(jp,s)]] = sum_fun(W[jp, contact_motifidx(sp, jp)])
                        if s == sp:
                                A[p,contact_idx_dict[(jp,sp)]] += -(contact_prior)/Psi_inv

                    else:
                        A[p, s] = W[j,sp]
                        if contact_sites[s][j, idx]:
                            A[p, contact_idx_dict[(j,s)]] = W[j,sp]
                        if s == sp:
                                A[p,contact_idx_dict[(jp,sp)]] += -(contact_prior)/Psi_inv
                if self.motif_decomposition != 'none':
                    b[p] = sum_fun(data[j,idx]*W[j,contact_motifidx(sp, jp)])
                else:
                    b[p] = data[j,idx]*W[j,sp]

            """
            for p in xrange(A.shape[0]):
                for j in xrange(nmeas):
                    for s in xrange(nstruct_elems):
                        if self.motif_decomposition != 'none':
                            if check_contact_site(s, j):
                                if p < nstruct_elems:
                                    A[p, contact_idx_dict[(j,s)]] = sum_fun(W[j,contact_motifidx(p, j)])
                                else:
                                    j2, s2 = contact_idx_dict[p]
                                    if j == j2:
                                        #A[p,contact_idx_dict[(j,s)]] = W[j,s]
                                        if s == s2:
                                            A[p,contact_idx_dict[(j,s)]] = (sum_fun(W[j,contact_motifidx(s, j)]) + contact_prior_loc)**2 + (contact_prior)/Psi_inv
                                        else:
                                            A[p,contact_idx_dict[(j,s)]] = sum_fun(W[j2,contact_motifidx(s2, j2)])*sum_fun(W[j,contact_motifidx(s, j)])

                        else:
                            if contact_sites[s][j, idx]:
                                if p < nstruct_elems:
                                    A[p, contact_idx_dict[(j,s)]] = W[j,p]
                                else:
                                    j2, s2 = contact_idx_dict[p]
                                    if j == j2:
                                        if s == s2:
                                            A[p,contact_idx_dict[(j,s)]] = W[j,s]**2 + (contact_prior)/Psi_inv
                                        else:
                                            A[p,contact_idx_dict[(j,s)]] = W[j2,s2]*W[j,s]
            for j in xrange(nmeas):
                for s in xrange(nstruct_elems):
                    if self.motif_decomposition != 'none':
                        if check_contact_site(s, j):
                            b[contact_idx_dict[(j,s)]] = sum_fun(data[j,idx]*W[j,motifidx(s, j)])
                    else:
                        if contact_sites[s][j, idx]:
                            #b[contact_idx_dict[(j,s)]] = -contact_prior_factor/(Psi_inv*W[j,s]) + data[j,idx]
                            #b[contact_idx_dict[(j,s)]] = -contact_prior_factor/(Psi_inv) + data[j,idx]
                            b[contact_idx_dict[(j,s)]] = data[j,idx]*W[j,s]
                """
            return A, b

        A, b = fill_matrices(A, b, contact_prior_factor)
        print 'Solving MAP for reactivities for position %s, system with dimensions %s' % (idx, A.shape)
        E_d__obs = zeros([nstructs])
        E_c__obs = zeros([nmeas, nstructs])
        def f(x):
            return ((dot(A,x) - b)**2).sum()
        def fprime(x):
            return dot(dot(A,x) - b, A.T)
        solved = False
        tries = 0
        while not solved:
            solved = True
            if self.motif_decomposition != 'none':
                bounds = [(0.001, data.max())]*nmotifs
                bounds = [(bound[0], 0.5) if motif_struct_type(idx, m) == 'p' else bound for m, bound in enumerate(bounds)]
                bounds += [(-1, 1)]*(A.shape[0] - nmotifs)
                #bounds += [(-data.mean(),data.mean())]*(A.shape[0] - nmotifs)
                x0 = [0.002 if motif_struct_type(idx, s) == 'p' else 1. for s in xrange(nmotifs)] + [0.0 for i in xrange(A.shape[0] - nmotifs)]
            else:
                bounds = [(0.001, data.max())]*nstructs
                bounds = [(low, 0.5) if struct_types[idx][s] == 'p' else (low, up) for low, up in bounds]
                bounds += [(-data.mean(),data.mean())]*(A.shape[0] - nstructs)
                x0 = [0.002 if struct_types[idx][s] == 'p' else 1. for s in xrange(nstructs)] + [0.0 for i in xrange(A.shape[0] - nstructs)]
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
                print 'System not solved!'
                solved=False
            if self.motif_decomposition == 'element':
                for s in xrange(nmotifs):
                    if x[s] <= 0.001:
                        E_d__obs[motifidx(s, j)] = 0.001
                    else:
                        if motif_struct_type(idx, s) == 'p' and x[s] > 0.5:
                            E_d__obs[motifidx(s, 0)] = 0.5
                        else:
                            E_d__obs[motifidx(s, 0)] = x[s]
            elif self.motif_decomposition == 'motif':
                for s in xrange(nmotifs):
                    if motifidx(s, self.wt_indices[0]):
                        st = struct_types[idx][motifidx(s, self.wt_indices[0])[0]]
                        if x[s] <= 0.001:
                            E_d__obs[motifidx(s, self.wt_indices[0])] = 0.001
                        else:
                            if st == 'p' and x[s] > 0.5:
                                E_d__obs[motifidx(s, self.wt_indices[0])] = 0.5
                            else:
                                E_d__obs[motifidx(s, self.wt_indices[0])] = x[s]
                for j in xrange(nmeas):
                    for s in xrange(nmotifs):
                        E_c__obs[j, motifidx(s, j)] = -E_d__obs[motifidx(s, j)] + x[s]
                        if check_contact_site(s, j):
                            pass
                            #E_c__obs[j, motifidx(s, j)] = x[s]


            else:
                for s in xrange(nstructs):
                    if x[s] <= 0.001:
                        E_d__obs[s] = 0.001
                    else:
                        E_d__obs[s] = x[s]

            E_ddT__obs = dot(mat(E_d__obs).T, mat(E_d__obs))
            if (E_ddT__obs < 0).ravel().sum() > 1:
                solved = False

            if not solved:
                # This means that we need to rescale the contact prior
                tries += 1
                if tries > 100:
                    new_prior_factor = contact_prior_factor/(tries - 100)
                else:
                    new_prior_factor = contact_prior_factor + 0.1*tries
                if tries > 0:
                    print 'Could not figure out a good contact prior'
                    print 'Blaming E_ddT__obs singularities on data similarity'
                    print 'Adding a bit of white noise to alleviate'
                    A, b = fill_matrices(A, b, contact_prior_factor)

                   
                print 'MAP system was not solved properly retrying with different contact priors'
                print 'Number of contacts is %s' % ncontacts
                print 'Changing prior factor to %s' % (new_prior_factor)
                A, b = fill_matrices(A, b, new_prior_factor)

        sigma_d__obs = mat(zeros([nstructs]))
        if self.motif_decomposition != 'none':
            for s in xrange(nmotifs):
                sigma_d__obs[0,motifidx(s, 0)] = sqrt(1/(Psi_inv*(W[:,motifidx(s, 0)]**2).sum() + 1e-10))
        else:
            for s in xrange(nstructs):
                sigma_d__obs[0,s] = sqrt(1/(Psi_inv*(W[:,s]**2).sum()))


        return mat(E_d__obs), sigma_d__obs, E_c__obs



    def calculate_data_pred(self, no_contacts=False):
        self.sigma_pred = zeros([self.W.shape[0], self.E_d.shape[1]])
        if no_contacts:
            self.data_pred = dot(self.W, self.E_d)
        else:
            self.data_pred = zeros([self.W.shape[0], self.E_d.shape[1]])
            for i in xrange(self.data_pred.shape[1]):
                self.data_pred[:,i] = self._dot_E_d_i(self.W, self.E_d[:,i], self.E_c[:,:,i], i, self._restricted_contact_sites).T

                #sigma_ddT = sqrt(diag(self.E_ddT[:,:,i]))
                #self.sigma_pred[:,i] = sqrt(dot(self.W_std / self.W, ones([self.E_d.shape[0]]))**2 + asarray(self._dot_E_d_i(ones(self.W_std.shape), sigma_ddT / self.E_d[:,i], self.E_c[:,:,i], i)).T**2)
                self.sigma_pred[:,i] = sqrt(self.Psi[i])

        return self.data_pred, self.sigma_pred

    def _calculate_chi_sq(self):
        """Calculates the chi squared statistic given the fit to the data
        """
        data_pred, sigma_pred = self.calculate_data_pred()
        chi_sq = ((asarray(self.data) - asarray(data_pred))**2/asarray(sigma_pred)**2).sum()
        return chi_sq


    def calculate_fit_statistics(self, data_pred=None, sigma_pred=None):
        """Calculate all fit statistics given the fit to the data.
            Returns:
                tuple. In order, chi-squared/Deg. Freedom, AIC, RMSEA
        """
        if data_pred == None or sigma_pred == None:
            data_pred, sigma_pred = self.calculate_data_pred()
        chi_sq = ((asarray(self.data) - asarray(data_pred))**2/asarray(sigma_pred)**2).sum()
        df = self.data.size - self.data.shape[1]
        if self.motif_decomposition != 'none':
            df += -2*sum(self.nmotpos)
        else:
            df += -self.E_d.size - self.W.size
            df += -self.E_c[logical_not(self.E_c == 0)].size
        k = -df - self.data.size + 1
        rmsea = sqrt(max((chi_sq/df - 1)/(self.data.shape[1] - 1), 0.0))
        aic = asscalar(chi_sq + 2*k - self.data.shape[0]*self.logpriors)

        return chi_sq/df, rmsea, aic

    def correct_scale(self, stype='linear'):
        """Corrects the scale of the fitted data. Sometimes the fitted model will have 'higher-than-average' reactivities. This procedure
        corrects this artifact, without affecting the inferred weights.
            Returns:
                tuple. In order, the list of correction factors inferred, the new data predicted by the model, and the new errors of the predicted data.
        """
        data_pred, sigma_pred = self.calculate_data_pred()
        corr_facs = [1]*data_pred.shape[1]
        if stype == 'none':
            return corr_facs, self.data_pred, self.sigma_pred

        if stype == 'linear':
            for i in xrange(len(corr_facs)):
                corr_facs[i] = asscalar(dot(data_pred[:,i], self.data[:,i])/dot(data_pred[:,i], data_pred[:,i]))
                self.E_d[:,i] *= corr_facs[i]
                data_pred[:,i] *= corr_facs[i]
                print 'Linear correction scale factor for position %s is %s' % (i, corr_facs[i])
        if stype == 'exponential':
            def errorf(idx, x):
                return sum([(exp(x*data_pred[i,idx])*data_pred[i,idx] - self.data[i,idx])**2 for i in xrange(data_pred.shape[0])])
            for i in xrange(len(corr_facs)):
                f = lambda x: errorf(i, x)
                opt_grid = (-20, 20, 0.01)
                corr_facs[i] = asscalar(optimize.brute(f, (opt_grid,) ))
                for j in xrange(data_pred.shape[0]):
                    data_pred[j,i] *= exp(corr_facs[i]*data_pred[j,i])
                print 'Exponential correction scale factor for position %s is exp(%s * x)' % (i, corr_facs[i])
        return corr_facs, self.data_pred, self.sigma_pred

    def _assign_EM_structure_variables(self, select_struct_indices, use_struct_clusters):
        """Given selected structure indices and clusters, initialize and assign all variables to use in the analysis
        """
        structures = [self.structures[i] for i in select_struct_indices]
        contact_sites = {}
        for s in select_struct_indices:
            contact_sites[s] = self.contact_sites[s].copy()
        if use_struct_clusters:
            energies = self.energies
            struct_types = [[s[i] for i in select_struct_indices] for s in self.cluster_struct_types]
        else:
            if len(self.energies) > 0:
                energies = self.energies[:,select_struct_indices].copy()
            else:
                energies = []
            struct_types = [[s[i] for i in select_struct_indices] for s in self.struct_types]
        if len(self.concentrations) > 0:
            concentrations = [self.concentrations[i] for i in select_struct_indices]
        else:
            concentrations = []
        if len(self.kds) > 0:
            kds = self.kds[:, select_struct_indices].copy()
        else:
            kds = []
        return structures, energies, struct_types, contact_sites, concentrations, kds

    def _assign_W_by_energies(self, data, E_d, E_c, Psi, contact_sites, energies):
        model = MappingModel(D_fac=E_d, Psi=Psi, D_contacts=E_c, contact_sites=contact_sites)
        if self.motif_decomposition == 'motif':
            energy_fitter = EnergiesByMotifFitter()

            W_new = energy_fitter.fit(data, model, self.motifs_by_structure, self.structures_by_motif, self.energies)
        else:
            energy_fitter = EnergiesByStructureFitter(njobs=self.njobs)
            W_new = energy_fitter.fit(data, model, energies_0=energies)
        return W_new

    def _assign_W(self, data, E_d, E_c, data_E_d, Psi, contact_sites, energies, concentrations, kds, G_constraint, nmeas, nstructs, use_struct_clusters, Wupper, Wlower, W_initvals):
        """Assign the weight matrices given inferred reactivities for the structures and contacts (the M-step). This is currently done in a quadratic program with convex combination constraints solved by CVXOPT.

            Args:
                data (2d numpy array): The mapping reactivity matrix.
                E_d (2d numpy array): The matrix containing the structure reactivities.
                E_c (3d numpy array): The matrix containing the 'contact map' for each structure.
                Psi (3d numpy array): The matrix containing the position-wise error covariance matrices.
                contact_sites (dict): The dictionary of contact site matrices, per structure.
                energies (2d numpy array): The matrix of initial energies for each structure.
                concentrations (list): The list of chemical concentrations for each experiment.
                kds (list): The list of dissociation constants for each experiment.
                G_constraint (float): The hard energetic constraint not allowing weights to float beyond certain kcals/mol of initial energies.
                nmeas (int): The number of measurements (experiments).
                nstructs (int): The number of structures.
                use_struct_clusters (bool): Indicates if we should use structure clusters or not.
                Wupper (2d numpy array): The upper bound weight matrix. Used only if G_constraint is not None.
                Wlower (2d numpy array): The lower bound weight matrix. Used only if G_constraint is not None.
                W_initvals (2d numpy array): The initial weight matrix
            
            Returns:
                The inferred weight matrix.

        """
        # Constrain the weights to be positive
        # If we were specified energies, we need to apply the lagrange multipliers
        # to constarint weights to be convex combinations and then
        # back-calculate the perturbations.
        # Furthermore, we constrain the weights by G_factor, if available, which was calculated from G_constraint
        npos = len(Psi)
        Psi_data_E_d_T = zeros([nmeas, nstructs])
        W = zeros([nmeas, nstructs])
        Psi_inv = 1./Psi
        Psi_inv_sum = 1./(Psi.sum()) 

        for i in xrange(npos):
            Psi_data_E_d_T += dot(Psi_inv[i], self._dot_E_d_i(data[:,i], E_d[:,i], E_c[:,:,i], i, contact_sites, T=True))

        if len(energies) > 0:
            print 'Solving weights'
            # ========= Using CVXOPT ====================
            # Converting to quadratic program
            frame = inspect.currentframe()
            args, _, _, argvals = inspect.getargvalues(frame)
            def par_fun(j, *args):
                E_d_proj = mat(zeros([nstructs,1]))
                E_ddT_Psi_inv = zeros([nstructs, nstructs])
                E_d_c_j = self._E_d_c_j(E_d, E_c, j, contact_sites)
                E_ddT_Psi_inv += self.lam_ridge*eye(W.shape[1])
                for i in xrange(E_d_c_j.shape[1]):
                    #E_d_proj += data[j,i]*E_d[:,i]
                    E_d_proj += -Psi_inv[i]*data[j,i]*E_d_c_j[:,i]
                    E_ddT_Psi_inv += Psi_inv[i]*dot(E_d[:,i].T, E_d[:,i])

                if self.lam_weights != 0:
                    E_ddT_Psi_inv += self.lam_weights*diag(1/(W_initvals[j,:]**2 + 1e-10))
                    E_d_proj += self.lam_weights*mat(1./(W_initvals[j,:] + 1e-10)).T

                if self.lam_mut != 0 and j in self.wt_indices:
                    E_ddT_Psi_inv += self.lam_mut*eye(W.shape[1])
                    for m in xrange(W.shape[0]):
                        if m not in self.wt_indices:
                            E_d_proj += -2*self.lam_mut*mat(-W[m,:] - (W_initvals[j,:] - W_initvals[m,:])).T

                'Solving weights for measurement %s' % j
                #P = cvxmat(E_ddT_Psi_inv - (2*eye(nstructs) - ones([nstructs, nstructs])))
                P = cvxmat(E_ddT_Psi_inv)
                Gc = cvxmat(-eye(nstructs))
                h = cvxmat(0.0, (nstructs,1))
                A = cvxmat(1.0, (1, nstructs))
                b = cvxmat(1.0)
                p = cvxmat(E_d_proj)
                def quadfun(x):
                    return asscalar(0.5*dot(x.T, dot(E_ddT_Psi_inv, x)) + dot(E_d_proj.T,x))
                def quadfunprime(x):
                    return 0.5*asarray(dot(E_ddT_Psi_inv, x) + asarray(E_d_proj).ravel())
                bounds = [(1e-10, None)]*nstructs
                #sol = optimize.fmin_l_bfgs_b(quadfun, W_initvals[j,:], fprime=quadfunprime, bounds=bounds)[0]
                sol = array(solvers.qp(P, p, Gc, h, A, b, None, {'x':cvxmat(W_initvals[j,:])})['x'])
                #sol = array(solvers.qp(P, p, Gc, h, A, b, None, {'x':cvxmat(0.5, (nstructs, 1))})['x'])
                #sol = array(solvers.qp(P, p, Gc, h, A, b)['x'])
                
                sol[sol <= 0] = 1e-10
                sol = sol/sol.sum()
                return sol.T

            if self.njobs != None:
                sys.modules[__name__].par_fun = par_fun
                resvecs = joblib.Parallel(n_jobs=self.njobs)(joblib.delayed(par_fun)(j, *argvals) for j in xrange(nmeas))
                for j, sol in enumerate(resvecs):
                    W[j,:] = sol
            else:
                for j in xrange(nmeas):
                    W[j,:] = par_fun(j, *argvals)
            # ========= End using CVXOPT ================
            if G_constraint != None:
                # Constrain by using G_constraint
                for m in xrange(nmeas):
                    for p in xrange(nstructs):
                        if W[m,p] > Wupper[m,p]:
                            W[m,p] = Wupper[m,p]
                        if W[m,p] < Wlower[m,p]:
                            W[m,p] = Wlower[m,p]
            # Sometimes there are small numerical errors that amplify a bit and they need to set straight
            # by forcing the sum of weights to be one -- maybe there is a better way to do the math
            # here to bypass these numerical instabilities?
            for j in xrange(nmeas):
                W[j,:] /= W[j,:].sum()

        # If we were given concentrations and kds, we need to set the weights accordingly
        if len(concentrations) > 0:
            indices_to_adjust = defaultdict(list)
            cum_weights = defaultdict(int)
            for i in xrange(nmeas):
                for j in xrange(nstructs):
                    if not isnan(kds[i,j]):
                        # If a kd is zero, means that the weight should be zero
                        if kds[i,j] == 0:
                            W[i,j] = 0
                        else:
                            # I'm using a standard Hill equation here with n=1; no cooperativity or second order reactions
                            W[i,j] = concentrations[i]/(concentrations[i] + kds[i,j])
                            cum_weights[i] += W[i,j]
                    else:
                        indices_to_adjust[i].append(j)
            # If we have energies, then the weights must sum to 1, but we might have disrupted this when enforcing weights by kd
            # so re-enforce the sum to 1 constraint in the indices that are allowed to float.
            if len(energies) > 0:
                for i, indices in indices_to_adjust.iteritems():
                    sum_indices = (1-cum_weights[i])/W[i,indices].sum()
                    for idx in indices:
                        W[i,idx] *= sum_indices
        return W

    def _calculate_MLE_std(self, W, Psi_inv, E_d):
        """Calculates maximum likelihood estimate of weight error matrix using a Fisher information matrix approach.
        """
        I_W = ones(W.shape)
        """
        Psi_inv_vec = array([[Psi_inv[i] for i in xrange(len(Psi_inv))]])
        for j in xrange(I_W.shape[0]):
            for s in xrange(I_W.shape[1]):
                I_W[j,s] = dot(Psi_inv_vec, dot(E_d, E_d.T)[s,:,:]).sum()
            for s in xrange(I_W.shape[1]):
                for sp in xrange(I_W.shape[1]):
                    I_W[j,s] += dot(Psi_inv_vec, E_ddT[s,sp,:])
            """
        I_W[I_W == 0] = 1e-100
        return sqrt(1/I_W)

    def _E_d_c_j(self, E_d, E_c, j, contact_sites):
        """Helper function to multiply E_d and E_c matrices at structure j given the contact sites
        """
        E_d_c_j = mat(zeros(E_d.shape))
        for s in xrange(E_d.shape[0]):
            for i in xrange(E_d.shape[1]):
                if contact_sites[s][j,i]:
                    E_d_c_j[s,i] = E_d[s,i] + E_c[j,s,i]
                else:
                    E_d_c_j[s,i] = E_d[s,i]
        return E_d_c_j

    def _dot_E_d_i(self, M, E_d_i, E_c_i, i, contact_sites, T=False):
        """Helper function to calculate the dot product of the E_d matrix at position i with an nmeas x nstructs matrix M given the 
        contact matrix E_c_i (at position i) and the contact sites
        """
        # i is the sequence position index
        nmeas = M.shape[0]
        nstructs = E_d_i.shape[0]
        if T:
            res = mat(zeros([nmeas, nstructs]))
            for j in xrange(nmeas):
                for s in xrange(nstructs):
                    if E_c_i[j,s] != 0 or contact_sites[s][j,i]:
                        res[j,s] = asscalar(M[j]) * (E_d_i[s] + E_c_i[j,s])
                    else:
                        res[j,s] = asscalar(M[j]) * asscalar(E_d_i[s])
        else:
            res = mat(zeros([nmeas, 1]))
            for j in xrange(nmeas):
                d_c = zeros([nstructs])
                for s in xrange(nstructs):
                    if E_c_i[j,s] != 0 or contact_sites[s][j,i]:
                        d_c[s] = E_c_i[j,s] + E_d_i[s]
                    else:
                        d_c[s] = E_d_i[s]
                res[j] = dot(M[j,:], d_c)

        return res

    # TODO Better documentation! Also, we should deprecate the following options: 
    # sigma_d0, G_constraint, and post_model -- and wrap them up in other functions
    # We should also break up this method, it's too big!!!
    def analyze(self, max_iterations=100, tol=0.01, select_struct_indices=[], W0=None, Psi0=None, E_d0=None, E_c0=None, sigma_d0=None, cluster_data_factor=None, G_constraint=None, use_struct_clusters=False, seq_indices=None, return_loglikes=False, post_model=False):
        """The main method that models the data in terms of the structures.
            Kargs:
                max_iterations (int): Maximum number of EM iterations.
                tol (float): The tolerance difference between the last and current base pair probability matrix of the inferred structure weights before the EM cycle is deemed convergent and terminates.
                select_struct_indices (list): A list of structure indices to consider for the analysis.
                W0 (2d numpy array): An initial estimate of the structure weight matrix.
                Psi0 (3d numpy array): An inital estimate of the error covariance matrices.
                E_d0 (3d numpy array): An initial estimate of the reactivity matrices.
                E_c0 (3d numpy array): An initial estimate of the contact matrices.
                sigma_d0 (2d numpy array): An initial estimate for the reactivity standard deviations.
                cluster_data_factor (float): Factor used to cluster the data. Useful for data redundancy elimination.
                G_constraint (float): Hard energetic constraint to prevent the inferred weights to be beyond a +/- energetic difference of the initial weights.
                use_struct_clusters (bool): Indicates if we use structure clusters or not.
                seq_indices (list): Indicates which sequence positions we use for the analysis.
                return_loglikes (bool): Chooses whether to return the trace of log-likelihoods [TODO: THIS CURRENTLY RETURNS CONVERGENCE VALUES]
                post_model (bool): Indicates whether to use mapping-directed MFE modeling using the infer reactivities to refine the initial structures.

            
            Returns:
                tuple. In order:
                * lhood: Final log-likelihood of the fit.
                * W: Inferred structure weight matrix.
                * W_std: Inferred structure weight matrix errors.
                * Psi: Inferred matrix of error covariance matrices.
                * E_d: Inferred matrix of structure reactivities.
                * E_c: Inferred matrix of contact sites reactivity values matrices.
                * sigma_d: Inferred standard deviation of reactivity matrix.
                * (Optional) post_structures: The structures resulting from posterior mapping directed MFE modeling.
        """
        # Sometimes, e.g. model selection, we want just to test a subset of structures, we do that via select_struct_indices
        # to indicate the structure indices that we want to try
        if len(select_struct_indices) > 0:
            structures, energies, struct_types, contact_sites, concentrations, kds = self._assign_EM_structure_variables(select_struct_indices, use_struct_clusters)
        else:
            structures = self.structures
            energies = self.energies
            if use_struct_clusters:
                struct_types = self.cluster_struct_types
            else:
                struct_types = self.struct_types
            kds = self.kds
            concentrations = self.concentrations

            select_struct_indices = range(len(self.structures))

        # The data may have a lot of mutants that are minimally perturbed, i.e. they basically look like the WT with some local perturbations.
        # To stabilize the covariance matrix, clustering may be used.
        if cluster_data_factor:
            allmeas  = self.data.shape[0]
            allenergies = energies
            allconcentrations = concentrations
            allkds = kds
            chosenindices, bestclusts = self._cluster_data(cluster_data_factor)
            measindices = chosenindices.values()
            data = self.data[measindices, :].copy()
            energies = energies[measindices, :]
            kds = [kds[i] for i in measindices if i < len(kds)]
            concentrations = [concentrations[i] for i in measindices if i < len(concentrations)]
            for s in xrange(len(structures)):
                contact_sites[s] = contact_sites[s][measindices, :]
        else:
            data = self.data.copy()
            contact_sites = self.contact_sites.copy()
            measindices = range(self.data.shape[0])
        if use_struct_clusters:
            nstructs = len(self.structure_clusters)
        else:
            nstructs = len(structures)

        if seq_indices != None:
            data = data[:,seq_indices]
            struct_types = [struct_types[i] for i in seq_indices]
            for s in xrange(len(structures)):
                contact_sites[s] = contact_sites[s][:, seq_indices]
        else:
            seq_indices = range(data.shape[1])
        self._restricted_contact_sites = contact_sites

        npos = data.shape[1]
        nmeas =data.shape[0]
        """
        Initialize the variables
        For clarity: the model is
        data_i = W*d_i + epsilon
        epsilon ~ Normal(0, Psi)
        d_i ~ multivariate gamma mixture depending on struct_types
        """
        print 'Initializing weights W and covariances Psi'
        min_var = 1e-100**(1./nmeas)
        max_var = 2
        if Psi0 != None:
            Psi = Psi0
        else:
            Psi = zeros([npos])
            for i in xrange(npos):
                indices = where(data[:,i] < mean(data[:,i]))[0]
                if len(indices) < 10:
                    indices = range(nmeas)
                Psi[i] = min(max((data[indices,i].std())**2, min_var), max_var)
        if W0 != None:
            W = W0
        else:
            scale = Psi.sum()*(1./npos)
            if len(energies) > 0:
                if use_struct_clusters:
                    if len(self.struct_weights_by_clust) == 0:
                        # Means we haven't calculated the structure weights by cluster for some reason
                        W, struct_weights_by_clust = utils.calculate_weights(energies, clusters=self.structure_clusters)
                    else:
                        # When we have, just retrieve pre-calculated weights
                        if len(select_struct_indices) > 0:
                            W = self.W[:, select_struct_indices]
                        else:
                            W = self.W
                else:
                    W = utils.calculate_weights(energies)
            else:
                W = normal(0, sqrt(scale/(nstructs)), size=(nmeas, nstructs))

        Wupper = zeros(W.shape)
        Wlower = zeros(W.shape)

        if G_constraint != None:
            for m in xrange(nmeas):
                for p in xrange(nstructs):
                    Wupper[m,p] = W[m,p]*exp(G_constraint/(utils.k*utils.T))
                    Wlower[m,p] = W[m,p]*exp(-G_constraint/(utils.k*utils.T))
                    if Wupper[m,p] < Wlower[m,p]:
                        tmp = Wupper[m,p]
                        Wupper[m,p] = Wlower[m,p]
                        Wlower[m,p] = tmp

        W_initvals = W
        post_structures = structures
        print 'Finished initializing'
        # For brevity E_d = E[d | data], E_d =[E[c | data], and E_ddT = E[d*dT | data], and so on...

        Psi_inv = 1./Psi

        E_d = mat(zeros([nstructs, npos]))
        E_c = zeros([nmeas, nstructs, npos])
        data_E_d = zeros([nmeas, nstructs, npos])
        sigma_d= mat(zeros([nstructs, npos]))

        data_dataT = zeros([nmeas, nmeas, npos])

        old_loglike = -inf
        base_loglike = None
        Psi_opt = Psi
        W_opt = W
        E_d_opt = E_d
        E_c_opt = E_c
        sigma_d_opt = sigma_d
        max_loglike = old_loglike
        loglikes = []
        Psi_reinits = []
        bppm_prev = utils.bpp_matrix_from_structures(self._origstructures, W[self.wt_indices[0],:])
        adaptive_factor = 1


        t = 0
        for i in xrange(npos):
            data_dataT[:,:,i] = dot(data[:,i], data[:,i].T)
        while t < max_iterations:
            # E-step
            loglike = -npos*nmeas/2.*log(2.*pi)

            if self.njobs != None:
                def par_fun(i):
                    return self.hard_EM_vars(i, W, Psi_inv[i], data, struct_types, contact_sites, seq_indices)
                sys.modules[__name__].par_fun = par_fun
                restuples = joblib.Parallel(n_jobs=self.njobs)(joblib.delayed(par_fun)(i) for i in xrange(npos))

            else:
                restuples = []
                for i in xrange(npos):
                    restuples.append(self.hard_EM_vars(i, W, Psi_inv[i], data, struct_types, contact_sites, seq_indices))

            for i, tup in enumerate(restuples):
                E_d_i, sigma_d_i, E_c_i = tup
                E_d[:,i] = E_d_i.T
                sigma_d[:,i] = sigma_d_i.T
                E_c[:,:,i] = E_c_i
            
            if E_d0 is not None and t == 0:
                E_d, E_c, sigma_d = E_d0, E_c0, sigma_d0

            if self.njobs != None:
                def par_fun(i):
                    B = self._dot_E_d_i(data[:,i], E_d[:,i], E_c[:,:,i], i, contact_sites, T=True)
                    return B
                sys.modules[__name__].par_fun = par_fun
                results = joblib.Parallel(n_jobs=self.njobs)(joblib.delayed(par_fun)(i) for i in xrange(npos))
                for i, B in enumerate(results):
                    data_E_d[:,:,i] = B
            else:
                for i in xrange(npos):
                    data_E_d[:,:,i] = self._dot_E_d_i(data[:,i], E_d[:,i], E_c[:,:,i], i, contact_sites, T=True)
            # M-step

            # Given our expected reactivities, now assign the weights


            weight_optimization = self.get_option('weight_optimization')


            if weight_optimization == 'weights':
                W_new = self._assign_W(data, E_d, E_c, data_E_d, Psi, contact_sites, energies, concentrations, kds, G_constraint, nmeas, nstructs, use_struct_clusters, Wupper, Wlower, W_initvals)
            elif weight_optimization == 'energies':
                W_new = self._assign_W_by_energies(data, E_d, E_c, Psi, contact_sites, energies)
            elif weight_optimization =='none'
                W_new = W
            else:
                raise ValueError('%s is not a valid weight optimization strategy!' % weight_optimization)

                W_new[W_new < 0] = 1e-10
            for j in xrange(nmeas):
                W_new[j,:] /= W_new[j,:].sum()
            
            # Update stopping criterion
            bppm_new = utils.bpp_matrix_from_structures(self._origstructures, W_new[self.wt_indices[0],:])
            currdiff = abs(bppm_prev[bppm_new != 0] - bppm_new[bppm_new != 0]).max()
            bppm_prev = bppm_new
            W = W_new

            # Now get covariance matrix
            data_pred = zeros([W.shape[0], E_d.shape[1]])

            if self.njobs != None:
                def par_fun(i):
                    return self._dot_E_d_i(W, E_d[:,i], E_c[:,:,i], i, contact_sites).T
                sys.modules[__name__].par_fun = par_fun
                restuples = joblib.Parallel(n_jobs=self.njobs)(joblib.delayed(par_fun)(i) for i in xrange(npos))
                for i, D in enumerate(restuples):
                    data_pred[:,i] = D
            else:
                for i in xrange(data_pred.shape[1]):
                    data_pred[:,i] = self._dot_E_d_i(W, E_d[:,i], E_c[:,:,i], i, contact_sites).T
            for i in xrange(npos):
                P = sum(array(data[:,i].ravel() - data_pred[:,i].ravel())**2)
                Psi[i] = P/nmeas
                if isnan(Psi[i]) or Psi[i] < min_var:
                    Psi[i] = min_var
                if Psi[i] > max_var:
                    Psi[i] = max_var

            # Do post-facto SHAPE-directed modeling, if asked
            didpostmodel = False
            if post_model:
                for sidx, s in enumerate(select_struct_indices):
                    domodeling = False
                    # Be conservative, for each structure, only do post-facto modeling if E_d is totally off for at least one position
                    for iidx in xrange(len(seq_indices)):
                        if struct_types[iidx][sidx] == 'p' and self.paired_pdf(E_d[sidx,iidx]) < 0.05:
                            print self.paired_pdf(E_d[sidx,iidx])
                            domodeling = True
                    if domodeling:
                        md = mapping.MappingData(data=asarray(E_d[sidx,:]*1.5).ravel(), seqpos=seq_indices)
                        structure = ss.fold(self.wt, mapping_data=md)[0].dbn
                        new_struct_types = utils.get_struct_types([structure])
                        new_contact_sites = utils.get_contact_sites([structure], self.mutpos, self.data.shape[0], self.data.shape[1], self.c_size, restrict_range=self.seqpos_range)
                        if (contact_sites[sidx] != new_contact_sites[0]).sum() > 0:
                            didpostmodel = True
                        for i in xrange(self.data.shape[1]):
                            struct_types[i][sidx] = new_struct_types[i][0]
                        contact_sites[sidx] = new_contact_sites[0][:, seq_indices]
                        self._restricted_contact_sites[s] = new_contact_sites[0][:, seq_indices]
                        post_structures[sidx] = structure

                        for i in xrange(self.data.shape[1]):
                            self.struct_types[i][s] = new_struct_types[i][0]
                        self.contact_sites[s] = new_contact_sites[0] 
                        self.structures[s] = structure
            
            # Add prior likelihoods
            logpriors = 0
            MIN_LOGLIKE = -1e100
            MIN_LOGLIKE = 0
            """
            for sidx in xrange(len(select_struct_indices)):
                for sidx2 in xrange(len(select_struct_indices)):
                    try:
                        logpriors += -self.lam_reacts*(asarray(E_d[sidx,:] - E_d[sidx2,:])**2).sum()
                    except FloatingPointError:
                        logpriors += MIN_LOGLIKE
                try:
                    logpriors += -(self.lam_weights*(W*(1./W_initvals))).sum()
                except FloatingPointError:
                    logpriors += MIN_LOGLIKE
                for iidx in xrange(len(seq_indices)):
                    try:
                        if struct_types[iidx][sidx] == 'p':
                            logpriors += log(self.paired_pdf(E_d[sidx,iidx]))
                        else:
                            logpriors += log(self.unpaired_pdf(E_d[sidx,iidx]))
                    except FloatingPointError:
                        logpriors += MIN_LOGLIKE
            loglike += logpriors
            """
            # Check if we are done
            print 'Finished iteration %s with log-likelihood %s' % (t, loglike)
            if not didpostmodel:
                print t
                t += 1
            #loglikes.append(asscalar(loglike))
            loglikes.append(currdiff)
            #loglikes.append(asscalar(chi_sq))
            max_loglike = loglike
            self.logpriors = logpriors
            W_opt = W.copy()
            E_d_opt = E_d.copy()
            E_c_opt = E_c.copy()
            sigma_d_opt = sigma_d.copy()
            Psi_opt = Psi.copy()

            if currdiff <= tol or t+1 == max_iterations:
                break
        print 'Finished'
        if return_loglikes:
            lhood = [loglikes, Psi_reinits]
        else:
            lhood = loglikes[-1]
        self.W, self.Psi, self.E_d, self.E_c, self.sigma_d = asarray(W_opt), asarray(Psi_opt), asarray(E_d_opt), asarray(E_c_opt), asarray(sigma_d_opt)


        print 'W_std calculation started'
        # Calculate the standard deviation of the MLE
        self.W_std = self._calculate_MLE_std(W_opt, Psi_inv, E_d)
        print 'W_std calculation finished'

        # Map final values for weights, reactivities and errors to structure space if we did motif decomposition
        #if self.motif_decomposition == 'motif':
        #    self._map_variables_to_structures()
        #    # Need to also "restore" the restricted_contact_sites variables to structure, rather than motif space
        #    self._restricted_contact_sites = saved_contact_sites


        if len(self.energies) > 0 or use_struct_clusters:
            return lhood, self.W, self.W_std, self.Psi, self.E_d, self.E_c, self.sigma_d, post_structures
        return lhood, self.W, self.W_std, self.Psi, self.E_d, self.E_c, self.sigma_d, post_structures

#============== Models =============================


class MappingModel(object):
    def __init__(self, D_fac=None, W=None, Psi=None, D_contacts=None, contact_sites=None):
        self.Psi = Psi
        self.D_fac = D_fac
        self.W = W
        self.D_contacts = D_contacts
        self.contact_sites = contact_sites

        if W is not None:
            self.nmeas, self.nstructs = W.shape
        if D_fac is not None:
            self.nstructs, self.npos = D_fac.shape
        if D_contacts is not None:
            self.nmeas = D_contacts.shape[0]
            self.nstructs = D_contacts.shape[1]


    def get_D_fac_with_contacts(self, meas):
        D_fac_w_contacts = self.D_fac.copy()
        for pos in xrange(self.D_fac.shape[1]):
            for struct_idx in xrange(self.D_fac.shape[0]):
                D_fac_w_contacts[struct_idx, pos] = self.D_fac[struct_idx, pos]
                if self.D_contacts[meas, struct_idx, pos] != 0:
                    D_fac_w_contacts[struct_idx, pos] += self.D_contacts[meas, struct_idx,  pos] 
        return D_fac_w_contacts
#===================================================


class EnergyFitter(object):

    def __init__(self):
        pass

    def fit(self, data, model):
        pass


class EnergyFitter(object):

    def __init__(self):
        self.k = 0.0019872041 # Boltzmann constant
        self.T = 310.15 # temperature
        self.kBT = self.k * self.T

    def get_bfactors(self, energies):
        return exp(-energies/self.kBT)


class EnergiesByStructureFitter(EnergyFitter):

    def __init__(self, njobs=None):
        super(EnergiesByStructureFitter, self).__init__()
        self.njobs = njobs


    def fit(self, data, model, energies_0=None):
        nmeas = data.shape[0]
        def _fit_function(D_fac, *energies):
            bfactors = self.get_bfactors(array(energies))
            partition = bfactors.sum()
            data_pred = dot(bfactors/partition, D_fac)
            return asarray(data_pred).ravel()

        def par_fun(meas):
            print 'Solving weights using structure energies for measurement %s' % meas
            D_fac_w_contacts = model.get_D_fac_with_contacts(meas)
            energies_opt, _ = optimize.curve_fit(_fit_function, D_fac_w_contacts, asarray(data[meas,:]).ravel(), p0=energies_0[meas,:], sigma=sqrt(model.Psi))
            bfactors_opt = self.get_bfactors(energies_opt)
            return bfactors_opt/bfactors_opt.sum()
        
        if self.njobs is None:
            resvecs = []
            for meas in xrange(nmeas):
                resvecs.append(par_fun(meas))
        else:
            sys.modules[__name__].par_fun = par_fun
            resvecs = joblib.Parallel(n_jobs=self.njobs)(joblib.delayed(par_fun)(meas) for meas in xrange(nmeas))

        W_new = zeros([nmeas, model.D_fac.shape[0]])
        for meas in xrange(nmeas):
            W_new[meas,:] = resvecs[meas]
        return W_new

class EnergiesByMotifFitter(EnergyFitter):

    def __init__(self):
        super(EnergiesByMotifFitter, self).__init__()
    
    def _get_initial_energies(self, model, motifs_by_structure, struct_energies, nmotifs):
        A = zeros([model.nstructs*model.nmeas, nmotifs])
        b = zeros([model.nstructs*model.nmeas])
        for meas_idx in xrange(model.nmeas):
            for struct_idx in xrange(model.nstructs):
                idx = meas_idx*model.nstructs + struct_idx
                A[idx, motifs_by_structure[struct_idx, meas_idx]] = 1
                b[idx] = struct_energies[meas_idx, struct_idx]

        def _f(x):
            return ((dot(A,x) -b)**2).sum()

        def _fprime(x):
            return dot(dot(A,x) - b, A)

        x0 = ones([nmotifs])
        bounds = [(-10, 10)]*nmotifs
        x = optimize.fmin_l_bfgs_b(_f, x0, fprime=_fprime, bounds=bounds)[0]
        return x



    def _get_weights_from_energies(self, model, energies, motifs_by_structure, nmotifs):
        W = zeros([model.nmeas, model.nstructs])
        partitions = zeros([model.nmeas])
        bfactors = zeros([model.nmeas, model.nstructs])
        all_indices = []
        for meas_idx in xrange(model.nmeas):
            meas_energies = []
            for struct_idx in xrange(model.nstructs):
                motif_indices = motifs_by_structure[struct_idx, meas_idx]
                all_indices += motif_indices
                meas_energies.append(energies[motif_indices].sum())
            meas_energies = array(meas_energies)
            bfactors[meas_idx, :] = self.get_bfactors(meas_energies)
            partitions[meas_idx] = bfactors[meas_idx, :].sum()
            W[meas_idx,:] = bfactors[meas_idx, :]/partitions[meas_idx]
        return W, bfactors, partitions

    def _get_weights_and_data_pred(self, model, energies, D_fac_extended, motifs_by_structure, nmotifs):
        W, bfactors, partitions = self._get_weights_from_energies(model, array(energies), motifs_by_structure, nmotifs)
        data_pred = zeros([model.nmeas, model.npos])
        for meas_idx in xrange(model.nmeas):
            data_pred[meas_idx, :] = asarray(dot(W[meas_idx,:], D_fac_extended[:, meas_idx*(model.npos):model.npos*(meas_idx + 1)]))
        return W, bfactors, partitions, data_pred



    def fit(self, data, model, motifs_by_structure, structures_by_motif, struct_energies):
        nmeas = data.shape[0]
        motif_set = set()
        for motif_indices in motifs_by_structure.values():
            for motif in motif_indices:
                motif_set.add(motif)
        nmotifs = len(motif_set)
        print 'Getting initial energies'
        energies_0 = self._get_initial_energies(model, motifs_by_structure, struct_energies, nmotifs)

        def _fit_function(energies):
            W, bfactors, partitions, data_pred = self._get_weights_and_data_pred(model, energies, D_fac_extended, motifs_by_structure, nmotifs)
            return asarray(data_pred).ravel()

        def _min_function(energies):
            return (((asarray(data).ravel() - _fit_function(energies))/sigma)**2).sum()

        def _min_function_grad(energies):
            W, bfactors, partitions, data_pred = self._get_weights_and_data_pred(model, energies, D_fac_extended, motifs_by_structure, nmotifs)

            res = zeros([len(energies)])
            C = zeros([model.nstructs])
            for motif_idx in xrange(len(energies)):
                for meas_idx in xrange(model.nmeas):
                    if (motif_idx, meas_idx) in structures_by_motif:
                        mindices = structures_by_motif[motif_idx, meas_idx]
                        dZ = -1/self.kBT * bfactors[meas_idx, mindices].sum()
                        A = 2./model.Psi*asarray(data_pred[meas_idx, :] - data[meas_idx, :]).ravel()
                        C = (-dZ/partitions[meas_idx]) * W[meas_idx, :]
                        C[mindices] += -1/self.kBT * W[meas_idx, mindices]

                        B = dot(C, asarray(D_fac_extended[:, model.npos*meas_idx:model.npos*(meas_idx + 1)]))
                        res[motif_idx] += (A*B).sum()

            return res

        print 'Solving weights using motif energies for all measurements'
        D_fac_extended = model.get_D_fac_with_contacts(0)
        sigma = sqrt(model.Psi)
        for meas_idx in xrange(1, nmeas):
            D_fac_extended = append(D_fac_extended, model.get_D_fac_with_contacts(meas_idx), axis=1)
            sigma = append(sigma, sqrt(model.Psi))
        #energies_opt, _ = optimize.curve_fit(_fit_function, D_fac_extended, asarray(data).ravel(), sigma=sigma, p0=energies_0)
        bounds = [(-10, 10)]*len(energies_0)
        energies_opt = optimize.fmin_l_bfgs_b(_min_function, energies_0, fprime=_min_function_grad, disp=1, bounds=bounds)[0]
        

        W_opt, _, _ = self._get_weights_from_energies(model, energies_opt, motifs_by_structure, nmotifs)

        return W_opt

# TODO Better logging!
class MockPrint(object):
    """For silencing prints
    """
    def write(self, s):
        pass


        
