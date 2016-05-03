from collections import defaultdict, Counter
from cvxopt import solvers
from cvxopt import matrix as cvxmat
import inspect
import itertools
import joblib
import pdb
import pymc
import pymc.graph
import pymc.Matplot
from pymc import MAP, Model, MvNormal, stochastic, Deterministic, deterministic, MCMC, Wishart, AdaptiveMetropolis, Uniform, Cauchy, distributions, Laplace
from random import choice, sample
import sys

from matplotlib.pylab import *
from numpy import lib
import scipy.spatial.distance
import scipy.cluster.hierarchy as sphclust
from scipy.stats import gamma
from scipy.stats.kde import gaussian_kde
from scipy import optimize

import rdatkit.secstr as ss
from rdatkit import mapping

import map_analysis_utils as utils
from reactivity_distributions import *

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
            self.data = matrix(data)[:, self.seqpos_range[0]:self.seqpos_range[1]]
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
        for t in arange(0, maxdist, maxdist / 10.):
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
        self.bp_dist = utils.get_structure_distance_matrix(self._origstructures, self._origstruct_types, distance='basepair')
        self.bp_dist /= float(len(self._origstructures[0]))

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

        offset = -self.seqpos_range[0] if self.seqpos_range is not None else 0
        bytype = True if type == 'motif' else False
        self.pos_motif_map, self.motif_ids, self.motif_dist = utils.get_minimal_overlapping_motif_decomposition(self._origstructures, bytype=bytype, offset=offset)
        self.nmotpos = []
        self.posmap = defaultdict(list)
        for i in xrange(self.data.shape[1]):
            nmotifs = 0
            for midx in xrange(len(self.pos_motif_map)):
                if (i, midx) in self.pos_motif_map:
                    nmotifs += 1
                    self.posmap[i].append(self.motif_ids[midx])
            self.nmotpos.append(nmotifs)

        self.structs_by_motif = {}
        for pos__motif_idx, struct_indices in self.pos_motif_map.iteritems():
            pos, motif_idx = pos__motif_idx
            for idx in struct_indices:
                if idx not in self.structs_by_motif.iteritems():
                    self.structs_by_motif[idx] = set()
                self.structs_by_motif[idx].add(motif_idx)

        # We want to convert sets to lists for easy indexing
        for k, v in self.structs_by_motif.iteritems():
            self.structs_by_motif[k] = list(v)

        print 'Number of motifs per position: %s' % self.nmotpos


    def _initialize_motif_reactivities(self, npos):
        nmotifs = len(self.motif_ids)
        E_d = mat(zeros([nmotifs, npos]))
        sigma_d = mat(zeros([nmotifs, npos]))
        E_ddT = zeros([nmotifs, nmotifs, npos])
        E_c = zeros([nmeas, nmotifs, npos])
        return E_d, E_c, sigma_d, E_ddT, nmotifs


    def _initialize_motif_weights(self, W):
        nmotifs = len(self.motif_ids)
        W_motif = mat(zeros([W.shape[0], nmotifs]))
        for motif_idx, motif_id in enumerate(self.motif_ids):
            W_motif[:, motif_idx] = W[:, list(self.structs_by_motif[motif_idx])].sum(axis=1)
        return W_motif


    def _get_motif_contact_sites(self, contact_sites):
        contact_sites_motif = {}
        nmeas, npos = contact_sites.values()[0].shape
        for motif_idx, struct_indices in self.structs_by_motif.iteritems():
            contact_sites_motif[motif_idx] = zeros([nmeas, npos])
            struct_idx = struct_indices[0]
            for i in xrange(contact_sites[struct_idx].shape[1]):
                if self.motif_ids[motif_idx] in self.posmap[i]:
                    contact_sites_motif[motif_idx][:, i] = contact_sites[struct_idx][:, i]
        return contact_sites_motif


    def _map_motif_variables_to_structures(self):
        self.W_motif = self.W
        self.W_std_motif = self.W_std
        self.E_d_motif = self.E_d
        self.E_ddT_motif = self.E_ddT
        self.E_c_motif = self.E_c
        self.sigma_d_motif = self.sigma_d


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
                self.W[:, i] = struct_weights_by_clust[c].sum(axis=1)
            self.struct_weights_by_clust = struct_weights_by_clust
        else:
            self.W, self.struct_weights_by_clust = utils.calculate_weights(self.energies, clusters=cluster_indices)


    # TODO This is deprecated, should extract useful functionality to an external class/function and delete the rest...
    def model_select(self, expected_structures=0, greedy_iter=0, expstruct_estimation='hclust', tol=1e-4, max_iterations=10, prior_swap=True, G_constraint=None, apply_pseudoenergies=True, algorithm='rnastructure', soft_em=False, method='heuristic', post_model=False):
        """Selects a subset of structures as a model of the data  using an exploratory AICc selection approach.
        """
        print 'Starting model selection'
        if method not in FAMappingAnalysis.MODEL_SELECT_METHODS:
            raise ValueError('Unrecognized model selection method %s' % method)
        if expstruct_estimation not in FAMappingAnalysis.FACTOR_ESTIMATION_METHODS:
            raise ValueError('Unrecognized cluster method %s' % expstruct_estimation)

        nstructs = len(self.structures)
        npos = self.data.shape[1]
        all_struct_indices = range(nstructs)
        if expected_structures <= 0:
            print 'No expected number of structures given'
        if method == 'heuristic':
            if expstruct_estimation == 'fanalysis':
                maxmedoids, assignments = utils.cluster_structures(self.struct_types, expected_medoids=expected_structures)
            if expstruct_estimation == 'hclust':
                maxmedoids, assignments = utils.cluster_structures(self.struct_types)
            # Our starting estimate of the number of structures in the ensemble -- right now equal to expected_structures, but
            # that may change as we try to add/substract. This is mainly to calculate how many iterations we have to do in the
            # simulation step of the EM algorithm for Factor analysis using our custom priors.
            est_nstructs = len(maxmedoids)


            # Now, we will start to explore the model space by switching structures WITHIN A STRUCTURE CLUSTER using two iterations
            # of the EM Factor analysis. These two iterations will give us an estimate of the rate of EM convergence per model
            # and, using our current likelihood value, we can use that as an estimate of the likelihood value that we may be
            # able to reach. Thus, we just "waste" two EM iterations per model, accelerating the model selection procedure.
            data_structs, data_energies, mapping_datas = [], [], []
            struct_data_energies = zeros([len(self.structures), self.data.shape[0]])
            for j in xrange(self.data.shape[0]):
                #md = mapping.MappingData(data=array(self.data[j, :]).ravel(), enforce_positives=True)
                md = mapping.MappingData(data=array(self.data[j, :]).ravel())
                mapping_datas.append(md)
                fold_structures = ss.fold(self.sequences[j], mapping_data=mapping_datas[j], algorithm=algorithm)
                data_structs.append(fold_structures[0].dbn)
                energies = ss.get_structure_energies(self.sequences[j], fold_structures, mapping_data=mapping_datas[j], algorithm=algorithm)
                data_energies.append(energies[0])
            data_struct_types = utils.get_struct_types(data_structs)

            print 'Getting energies of each structure guided by the chemical mapping data'
            all_struct_obj = [ss.SecondaryStructure(dbn=s) for s in self.structures]
            cannonical_bp = [('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G'), ('A', 'U'), ('U', 'A')]
            for s in xrange(len(self.structures)):
                bp_dict = all_struct_obj[s].base_pair_dict()
                for j in xrange(self.data.shape[0]):
                    skip_struct = False
                    for n1, n2 in bp_dict.iteritems():
                        if (self.sequences[j][n1], self.sequences[j][n2]) not in cannonical_bp:
                            energy = 200
                            skip_struct = True
                    if not skip_struct:
                        if apply_pseudoenergies:
                            energy = ss.get_structure_energies(self.sequences[j], all_struct_obj[s], mapping_data=mapping_datas[j], algorithm=algorithm)[0]
                        else:
                            energy = ss.get_structure_energies(self.sequences[j], all_struct_obj[s], algorithm=algorithm)[0]
                        struct_data_energies[s, j] = max(energy, 200)

            print 'Getting probabilities for each structure given energies'
            struct_data_probs = utils.calculate_weights(struct_data_energies.T).T

            chosenindices, bestclusts = self._cluster_data(float(est_nstructs) / len(self.structures))
            data_red = self.data[chosenindices.keys(), :]
            nmeas = data_red.shape[0]
            print 'Scoring structures for wild type -- for prior swapping'
            struct_scores = [-struct_data_energies[idx][self.wt_indices[0]] for idx in all_struct_indices]

            #struct_scores = [array(structure_likelihood(self.struct_types, idx, data_energies, data_struct_types, struct_data_probs)).max() for idx in all_struct_indices]
            #struct_scores = [struct_data_probs[idx, :].max() for idx in all_struct_indices]
            struct_scores = [struct_data_probs[idx, self.wt_indices[0]] for idx in all_struct_indices]
            sorted_structs = sorted(range(len(struct_scores)), key=lambda x:struct_scores[x], reverse=True)
            sorted_dbns = [self.structures[s] for s in sorted_structs]


            if prior_swap:
                print 'Looking for best structure within each cluster'
                opt_swapped_structs = {}
                for k in assignments:
                    print 'Doing Structure cluster %s...' % k
                    medoid = maxmedoids[k]
                    currscore = struct_scores[medoid]
                    opt_swapped_structs[k] = medoid
                    for swapstruct in assignments[k]:
                        print '...structure %s' % swapstruct
                        if swapstruct != medoid:
                            score = struct_scores[swapstruct]
                            if currscore < score:
                                currscore = score
                                opt_swapped_structs[k] = swapstruct
                selected_structs = opt_swapped_structs.values()
                print 'Done finding best structures per cluster'
            else:
                selected_structs = maxmedoids.values()
            if greedy_iter == 0:
                return selected_structs, assignments
            print 'Scoring structures for all data -- for greedy search'
            #struct_scores = [array(structure_likelihood(self.struct_types, idx, data_energies, data_struct_types, struct_data_energies)).max() for idx in all_struct_indices]
            struct_scores = [struct_data_probs[idx, :].max() for idx in all_struct_indices]
            sorted_structs = sorted(range(len(struct_scores)), key=lambda x:struct_scores[x], reverse=True)
            sorted_dbns = [self.structures[s] for s in sorted_structs]
            sorted_structs = sorted_structs[:greedy_iter]
            sorted_assignments = defaultdict(list)
            for k, struct_indices in assignments.iteritems():
                sorted_assignments[k] = sorted(struct_indices, key=lambda x:struct_scores[x], reverse=True)
            print 'Doing a greedy search for more structures'
            # Ok, now we have the best structure for each cluster, let's see if adding structures, prioritized by score, for greedy_iter steps helps,
            # we score each run with AICc and use a greedy strategy
            print 'Initial EM iteration'
            print [self.structures[i] for i in selected_structs]
            lhood_opt, W_opt, W_std_opt, Psi_opt, E_d_opt, E_c_opt, sigma_d_opt, E_ddT_opt, perturbs_opt, post_structures_opt = self.analyze(max_iterations=2, tol=tol, nsim=5000, select_struct_indices=maxmedoids.values(), G_constraint=G_constraint, soft_em=soft_em, post_model=post_model)
            minAICc = utils.AICc(lhood_opt, self.data, W_opt, E_d_opt, E_c_opt, Psi_opt)


            def structure_similarity(s1, s2):
                simcounts = 0.
                for i in xrange(len(s1)):
                    if s1[i] == s2[i]:
                        simcounts += 1.
                return simcounts / len(s1)


            curr_idx = -1
            first_clust = sorted_assignments.keys()[0]
            AICc_decreased = True
            iteration = 0
            for curr_cluster in itertools.cycle(sorted_assignments.keys()):
                if iteration > greedy_iter:
                    break
                if curr_cluster == first_clust:
                    if not AICc_decreased:
                        break
                    else:
                        AICc_decreased = False
                    curr_idx += 1
                if curr_idx >= len(sorted_assignments[curr_cluster]):
                    continue
                new_struct = sorted_assignments[curr_cluster][curr_idx]
                if new_struct in selected_structs:
                    continue
                skip = False
                for s in selected_structs:
                    if structure_similarity(self.structures[new_struct], self.structures[s]) >= 0.9:
                        skip = True
                if skip:
                    print 'skipping %s' % self.structures[new_struct]
                    print 'given %s' % [self.structures[s] for s in selected_structs]
                    continue
                lhood, W, W_std, Psi, E_d, E_c, sigma_d, E_ddT, perturbs, post_structures = self.analyze(max_iterations=max_iterations, tol=tol,  nsim=5000, select_struct_indices=selected_structs + [new_struct], G_constraint=G_constraint, soft_em=soft_em, post_model=post_model)
                currAICc = utils.AICc(lhood, self.data, W, E_d, E_c, Psi)
                if currAICc < minAICc:
                    AICc_decreased = True
                    minAICc = currAICc
                    lhood_opt = lhood
                    W_opt = W
                    W_std_opt = W_std
                    Psi_opt = Psi
                    E_d_opt = E_d
                    E_ddT_opt = E_ddT
                    perturbs_opt = perturbs
                    post_structures_opt = post_structures
                    selected_structs.append(new_struct)
                iteration += 1

            # If we didn't add any structure, then maybe removing it will do
            if len(assignments) == len(selected_structs) and len(selected_structs) > 2:
                for i in xrange(len(selected_structs)):
                    reduced_structs = [selected_structs[j] for j in xrange(len(selected_structs)) if j != i]
                    lhood, W, W_std, Psi, E_d, E_c, sigma_d, E_ddT, perturbs, post_structures = self.analyze(max_iterations=max_iterations, tol=tol, nsim=5000, select_struct_indices=reduced_structs, G_constraint=G_constraint, soft_em=soft_em, post_model=post_model)
                    currAICc = utils.AICc(lhood, self.data, W, E_d, E_c, Psi)
                    if currAICc < minAICc:
                        AICc_decreased = True
                        minAICc = currAICc
                        lhood_opt = lhood
                        W_opt = W
                        W_std_opt = W_std
                        Psi_opt = Psi
                        E_d_opt = E_d
                        E_ddT_opt = E_ddT
                        perturbs_opt = perturbs
                        post_structures_opt = post_structures
                        selected_structs = reduced_structs

            if len(selected_structs) != len(assignments):
                # The number of selected structures has changed
                # We have to recluster using selected_structures as medoids
                assignments = defaultdict(list)
                selected_struct_types = dict([(j, [x[j] for x in self.struct_types]) for j in selected_structs])
                for i in xrange(len(self.structures)):
                    currmed = 0
                    currscore = inf
                    st1 = [x[i] for x in self.struct_types]
                    for j, st2 in selected_struct_types.iteritems():
                        score = utils._mutinf(st1, st2)
                        if currscore > score:
                            currscore = score
                            currmed = j
                    assignments[currmed].append(i)

            for sidx, s in enumerate(selected_structs):
                self.structures[s] = post_structures_opt[sidx]

            return selected_structs, assignments

        if method == 'bruteforce':
            print 'Using brute force to estimate model, this is going to take quite some time!'
            # Try all combinations of structures, brute force
            minAICc = inf
            starting_structures = sample(all_struct_indices, expected_structures)
            lhood_opt, W_opt, W_std_opt, Psi_opt, E_d_opt, E_c_opt, sigma_d_opt, E_ddT_opt, perturbs, post_structures = self.analyze(max_iterations=max_iteration, tol=tol, nsim=est_nstructs * 100, select_struct_indices=starting_structures, G_constraint=G_constraint, soft_em=soft_em, post_model=post_model)
            minAICc = utils.AICc(lhood_opt, self.data, W_opt, E_d_opt, E_c_opt, Psi_opt)
            for subset in itertools.combinations(all_struct_indices, expected_structures):
                if subset != starting_structures:
                    lhood, W, W_std, Psi, E_d, E_d, sigma_d, E_ddT, perturbs, post_structures = self.analyze(max_iterations=max_iteration, tol=tol, nsim=est_nstructs * 100, select_struct_indices=subset, G_constraint=G_constraint, soft_em=soft_em, post_model=post_model)
                    currAICc = utils.AICc(lhood, self.data, W, E_d, E_c, Psi)
                    if currAICc < minAICc:
                        minAICc = currAICc
                        lhood_opt = lhood
                        W_opt = W
                        W_std_opt = W_std
                        Psi_opt = Psi
                        E_d_opt = E_d
                        E_ddT_opt = E_ddT
                        perturbs_opt = perturbs
                        post_structures_opt = post_structures
                        selected_structs = subset
            return selected_structs


    # TODO this optimization is useful for other purposes, should extract and wrap it in an external function/class!
    def hard_EM_vars(self, idx, W, Psi_inv, data, struct_types, contact_sites, bp_dist, seq_indices):
        """Calculate expectation maximization variables from the E-step (getting the structure reactivities and variances). This is the hard EM version, returning a MAP estimate for the reactivities.
        
            Args:
                idx (int): The sequence position.
                W (2d numpy array): The structure weight matrix.
                Psi_inv (2d numpy array): The inverse of the a priori covariance matrix.
                data (2d numpy array): The mapping data matrix.
                struct_types (list): A list of structural types for each structure in this position.
                contact_sites (list): A list of matrices that contain the contact sites for each structure in this position.
                bp_dist (2d numpy array): A base-pair distance matrix between all structures.
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
        print 'Solving MAP for E-step (hard EM) for sequence position %s' % idx

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
        nstructs = len(struct_types[0])
        contact_idx_dict = {}
        if self.motif_decomposition != 'none':
            nmotifs = 0
            motif_idx_dict = {}
            for midx in xrange(len(self.pos_motif_map)):
                if (seq_indices[idx], midx) in self.pos_motif_map:
                    motif_idx_dict[nmotifs] = midx
                    nmotifs += 1
            # Some helper functions to make the code more compact
            if self.motif_decomposition == 'element':

                def motifidx(midx):
                    return self.pos_motif_map[(seq_indices[idx], motif_idx_dict[midx])]

                def contact_motifidx(midx, j):
                    return [m for m in motifidx(midx) if contact_sites[m][j, idx]]

                def check_contact_site(midx, j):
                    for m in motifidx(midx):
                        if contact_sites[m][j, idx]:
                            return True
                    return False
                sum_fun = lambda x: x.sum()

            if self.motif_decomposition == 'motif':

                def motifidx(midx):
                    return motif_idx_dict[midx]

                def contact_motifidx(midx, j):
                    return motifidx(midx)

                def check_contact_site(midx, j):
                    contact_sites[midx][j, idx]

                sum_fun = lambda x: x

            i = nmotifs
            for j in xrange(nmeas):
                for s in xrange(nmotifs):
                        if check_contact_site(s, j):
                            ncontacts += 1
                            contact_idx_dict[i] = (j, s)
                            contact_idx_dict[(j, s)] = i
                            i += 1

            dim = nmotifs + ncontacts
        else:
            i = nstructs
            for j in xrange(nmeas):
                for s in xrange(nstructs):
                        if contact_sites[s][j, idx]:
                            ncontacts += 1
                            contact_idx_dict[i] = (j, s)
                            contact_idx_dict[(j, s)] = i
                            i += 1
            dim = nstructs + ncontacts
        A = zeros([dim, dim])
        b = zeros([dim])


        def fill_matrices(A, b, contact_prior):
            if self.motif_decomposition != 'none':
                nstruct_elems = nmotifs
            else:
                nstruct_elems = nstructs
            # We'll start by indexing the reactivity hidden variables by structure
            for p in xrange(A.shape[0]):
                for s in xrange(nstruct_elems):
                    if p < nstruct_elems:
                        if self.motif_decomposition != 'none':
                            for j in xrange(nmeas):
                                A[p, s] += sum_fun(W[j, motifidx(p)]) * sum_fun(W[j, motifidx(s)])
                            if p == s:
                                for s1 in xrange(nstruct_elems):
                                    if s != s1:
                                        A[p, s] += 4 * self.lam_reacts / self.motif_dist[motif_idx_dict[s1], motif_idx_dict[s]]
                            else:
                                A[p, s] -= 4 * self.lam_reacts / self.motif_dist[motif_idx_dict[p], motif_idx_dict[s]]

                        else:
                            A[p, s] = dot(W[:, s], W[:, p])
                            if p == s:
                                for s1 in xrange(nstruct_elems):
                                    if s != s1:
                                        A[p, s] += 4 * self.lam_reacts / bp_dist[s1, s]
                            else:
                                A[p, s] -= 4 * self.lam_reacts / bp_dist[p, s]

                    else:
                        j, s2 = contact_idx_dict[p]
                        if self.motif_decomposition != 'none':
                            A[p, s] = sum_fun(W[j, motifidx(s)]) * sum_fun(W[j, motifidx(s2)])
                        else:
                            A[p, s] = W[j, s]*W[j, s2]

            for s in xrange(nstruct_elems):
                if self.motif_decomposition != 'none':
                    for m in motifidx(s):
                        if struct_types[idx][m] != struct_types[idx][motifidx(s)[0]]:
                            raise ValueError('MOTIF DECOMPOSITION FAILED! STRUCTURES IN POSITION %s HAVE DIFFERENT STRUCTURE TYPES!!! %s' % (idx, struct_types[idx]))
                    b[s] = -prior_factors[struct_types[idx][motifidx(s)[0]]] / Psi_inv[0, 0]
                    for j in xrange(nmeas):
                        b[s] += sum_fun(W[j, motifidx(s)])*data[j, idx]
                else:
                    b[s] = -prior_factors[struct_types[idx][s]] / Psi_inv[0, 0] + dot(W[:, s], data[:, idx])

            # Then, the contact maps. No Lapacian priors here, we use Gaussian (i.e. 2-norm) priors
            # for easier calculations
            for p in xrange(A.shape[0]):
                for j in xrange(nmeas):
                    for s in xrange(nstruct_elems):
                        if self.motif_decomposition != 'none':
                            if check_contact_site(s, j):
                                if p < nstruct_elems:
                                    A[p, contact_idx_dict[(j, s)]] = sum_fun(W[j, contact_motifidx(p, j)])
                                else:
                                    j2, s2 = contact_idx_dict[p]
                                    if j == j2:
                                        #A[p,contact_idx_dict[(j, s)]] = W[j, s]
                                        if s == s2:
                                            A[p, contact_idx_dict[(j, s)]] = (sum_fun(W[j, contact_motifidx(s, j)]) + contact_prior_loc)**2 + (contact_prior) / Psi_inv[0, 0]
                                        else:
                                            A[p, contact_idx_dict[(j, s)]] = sum_fun(W[j2, contact_motifidx(s2, j2)]) * sum_fun(W[j, contact_motifidx(s, j)])

                        else:
                            if contact_sites[s][j, idx]:
                                if p < nstruct_elems:
                                    A[p, contact_idx_dict[(j, s)]] = W[j, p]
                                else:
                                    j2, s2 = contact_idx_dict[p]
                                    if j == j2:
                                        if s == s2:
                                            A[p, contact_idx_dict[(j, s)]] = W[j, s]**2 + (contact_prior) / Psi_inv[0, 0]
                                        else:
                                            A[p, contact_idx_dict[(j, s)]] = W[j2, s2] * W[j, s]

            for j in xrange(nmeas):
                for s in xrange(nstruct_elems):
                    if self.motif_decomposition != 'none':
                        if check_contact_site(s, j):
                            b[contact_idx_dict[(j, s)]] = sum_fun(data[j, idx] * W[j, motifidx(s)])
                    else:
                        if contact_sites[s][j, idx]:
                            #b[contact_idx_dict[(j, s)]] = -contact_prior_factor/(Psi_inv[0, 0]*W[j, s]) + data[j, idx]
                            #b[contact_idx_dict[(j, s)]] = -contact_prior_factor/(Psi_inv[0, 0]) + data[j, idx]
                            b[contact_idx_dict[(j, s)]] = data[j, idx] * W[j, s]
            return A, b


        A, b = fill_matrices(A, b, contact_prior_factor)
        print 'Solving the linear equation system'
        # same variable names as in soft_EM_vars for consistency
        E_d__obs = zeros([nstructs])
        E_c__obs = zeros([nmeas, nstructs])

        def f(x):
            return ((dot(A, x) - b)**2).sum()

        def fprime(x):
            return dot(dot(A, x) - b, A.T)

        solved = False
        tries = 0
        while not solved:
            solved = True
            if self.motif_decomposition != 'none':
                bounds = [(0.001, data.max())]*nmotifs + [(-10, 10)]*(A.shape[0] - nmotifs)
                x0 = [0.002 if struct_types[idx][motifidx(s)[0]] == 'p' else 1. for s in xrange(nmotifs)] + [0.0 for i in xrange(A.shape[0] - nmotifs)]
            else:
                bounds = [(0.001, data.max())]*nstructs + [(-10, 10)]*(A.shape[0] - nstructs)
                x0 = [0.002 if struct_types[idx][s] == 'p' else 1. for s in xrange(nstructs)] + [0.0 for i in xrange(A.shape[0] - nstructs)]

            Acvx = cvxmat(A)
            bcvx = cvxmat(b)
            n = Acvx.size[1]
            I = cvxmat(0.0, (n, n))
            I[::n+1] = 1.0
            G = cvxmat([-I])
            h = cvxmat(n * [0.001])
            dims = {'l': n, 'q': [], 's': []}

            try:
                #x = array(solvers.coneqp(Acvx.T*Acvx, -Acvx.T*bcvx, G, h, dims, None, None, {'x':cvxmat(x0)})['x'])
                #x = optimize.fmin_slsqp(f, x0, fprime=fprime, bounds=bounds, iter=2000)
                x = optimize.fmin_l_bfgs_b(f, x0, fprime=fprime, bounds=bounds)[0]
                #x = linalg.solve(A, b)
            except ValueError:
                solved = False

            if self.motif_decomposition != 'none':
                for s in xrange(nmotifs):
                    if x[s] <= 0.001:
                        E_d__obs[motifidx(s)] = 0.001
                    else:
                        if struct_types[idx][motifidx(s)[0]] == 'p' and x[s] > 0.2:
                            E_d__obs[motifidx(s)] = 0.1
                        else:
                            E_d__obs[motifidx(s)] = x[s]
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
                    new_prior_factor = contact_prior_factor / (tries - 100)
                else:
                    new_prior_factor = contact_prior_factor + 0.1 * tries
                if tries > 0:
                    print 'Could not figure out a good contact prior'
                    print 'Blaming E_ddT__obs singularities on data similarity'
                    print 'Adding a bit of white noise to alleviate'
                    A, b = fill_matrices(A, b, contact_prior_factor)

                print 'MAP system was not solved properly retrying with different contact priors'
                print 'Number of contacts is %s' % ncontacts
                print 'Changing prior factor to %s' % (new_prior_factor)
                A, b = fill_matrices(A, b, new_prior_factor)

        if self.motif_decomposition == 'element':
            for s in xrange(nmotifs):
                for j in xrange(nmeas):
                    for i in motifidx(s):
                        if contact_sites[i][j, idx]:
                            E_c__obs[j, i] = x[s]
                        else:
                            E_c__obs[j, i] = nan

            for s in xrange(nmotifs):
                for j in xrange(nmeas):
                    for i in motifidx(s):
                        if contact_sites[i][j, idx]:
                            E_c__obs[j, i] += x[contact_idx_dict[(j, s)]]

        elif self.motif_decomposition == 'motif':
            for s in xrange(nmotifs):
                for j in xrange(nmeas):
                    if contact_sites[motifidx(s)][j, idx]:
                        E_c__obs[j, i] = x[s]
                    else:
                        E_c__obs[j, i] = nan

            for s in xrange(nmotifs):
                for j in xrange(nmeas):
                    if contact_sites[motifidx(s)][j, idx]:
                        E_c__obs[j, i] += x[contact_idx_dict[(j, s)]]

        else:
            for s in xrange(nstructs):
                for j in xrange(nmeas):
                    if contact_sites[s][j, idx]:
                        E_c__obs[j, s] = x[s]
                    else:
                        E_c__obs[j, s] = nan

            for s in xrange(nstructs):
                for j in xrange(nmeas):
                    if contact_sites[s][j, idx]:
                        E_c__obs[j, s] += x[contact_idx_dict[(j, s)]]

        sigma_d__obs = mat(zeros([nstructs]))
        if self.motif_decomposition != 'none':
            for s in xrange(nmotifs):
                sigma_d__obs[0, motifidx(s)] = sqrt(1/(Psi_inv[0, 0]*(W[:, motifidx(s)]**2).sum() + 1e-10))
        else:
            for s in xrange(nstructs):
                sigma_d__obs[0, s] = sqrt(1 / (Psi_inv[0, 0] * (W[:, s]**2).sum()))

        return mat(E_d__obs), E_ddT__obs, sigma_d__obs, E_c__obs


    # TODO This is deprecated, basically as well as the hard EM alternative, which is much faster
    def soft_EM_vars(self, idx, W, Psi_inv, data, struct_types, contact_sites, n, burn, chosenindices, use_struct_clusters):
        """Calculate expectation maximization variables from the E-step (getting the structure reactivities and variances). This is the soft EM version, returning a MCMC-obtained estimate for the reactivities.

            Args:
                idx (int): The sequence position.
                W (2d numpy array): The structure weight matrix.
                Psi_inv (2d numpy array): The inverse of the a priori covariance matrix.
                data (2d numpy array): The mapping data matrix.
                struct_types (list): A list of structural types for each structure in this position.
                contact_sites (list): A list of matrices that contain the contact sites for each structure in this position.
                n (int): Number of MCMC samples
                burn (int): Number of samples to burn
                chosenindices (list): Experiment indices to take into consideration.
                use_structure_clusters (bool): Use or not structure clusters


            Returns:
                tuple. In order:
                * The column of reactivities inferred for this sequence position
                * The covariance matrix of the reactivities for this sequence position
                * The column of standard deviations for this position
        """
        # idx stands for the index of position
        # If struture weights by cluster were specified, then we do
        # a different prior function
        print '\nSimulating for position %s' % idx
        nstructs = len(struct_types[idx])
        nmeas = W.shape[0]
        if use_struct_clusters:
            def d_prior_loglike(value):
                loglike = 0
                for s in xrange(len(value)):
                    if value[s] < 0:
                        return -inf
                    logprob = 0
                    state = struct_types[idx][s][self.struct_medoid_indices[s]]
                    if state == 'u':
                        logprob += log(self.unpaired_pdf(value[s]))
                        # To increase sampling in low probability regions for unpaired nucleotides quickly, we add a bonus
                        #logprob += log(normpdf(value[s], 1.0, 1))
                    if state == 'p':
                        logprob += log(self.paired_pdf(value[s]))
                        #logprob += log(normpdf(value[s], 0.1, 1))
                    loglike += logprob
                return loglike
        else:

            def d_prior_loglike(value):
                loglike = 0
                for s in xrange(len(value)):
                    if value[s] < 0 or self.unpaired_pdf(value[s]) <= 0 or self.paired_pdf(value[s]) <= 0:
                        return -inf
                    if struct_types[idx][s] == 'u':
                        loglike += log(self.unpaired_pdf(value[s]))
                        #loglike += log(normpdf(value[s], 1.0, 0.2))
                    else:
                        loglike += log(self.paired_pdf(value[s]))
                        #loglike += log(normpdf(value[s], 0.1, 1))
                return loglike

        print 'Choosing good d_0'
        while True:
            d_i_0 = rand(nstructs)
            logr = log([rand() for i in xrange(nstructs)]).sum()
            if d_prior_loglike(d_i_0) >= logr:
                break

        print 'Building contact site stochastic objects'
        contact_stochastics = {}
        for j in xrange(nmeas):
            for s in xrange(nstructs):
                if contact_sites[s][j, idx]:
                    contact_stochastics['c_%s_%s' % (j, s)] = Laplace('c_%s_%s' % (j, s), mu=contact_diff_params[0], tau=contact_diff_params[1])
                    #contact_stochastics['c_%s_%s' % (j, s)] = Cauchy('c_%s_%s' % (j, s), alpha=0.00039655433662119373, beta=0.022686230868563868)


        @stochastic
        def d_i(value=d_i_0):
            return d_prior_loglike(value)

        def d_calc_i_eval(**kwargs):
            L = kwargs['W']
            d = kwargs['d_i']
            M = zeros([nmeas, 1])
            for j in xrange(nmeas):
                d_c = zeros([nstructs])
                for s in xrange(nstructs):
                    if contact_sites[s][j, idx]:
                        d_c[s] = d[s] + kwargs['c_%s_%s' % (j, s)]
                    else:
                        d_c[s] = d[s]
                M[j] = dot(asarray(L[j, :]), d_c.T)
            return M


        # We have to construct the d_calc_i variable manually...
        d_calc_i_parents = {'W':W, 'd_i':d_i}
        d_calc_i_parents.update(contact_stochastics)
        d_calc_i = Deterministic(eval=d_calc_i_eval, name='d_calc_i', parents=d_calc_i_parents, doc='', trace=True, verbose=0, plot=False)

        data_obs_i = MvNormal('data_obs_i', value=[data[chosenindices, idx]], mu=d_calc_i, tau=Psi_inv, observed=True)

        mc = MCMC([d_i, d_calc_i, data_obs_i] + contact_stochastics.values())
        mc.use_step_method(pymc.AdaptiveMetropolis, [d_i] + contact_stochastics.values())
        mc.sample(iter=n, burn=burn)
        d_samples = mat(d_i.trace())

        E_d__obs = d_samples[0, :]
        E_c__obs = zeros([nmeas, nstructs])
        # This is actually samples[0, :] * samples[0, :].T in 'real math' language
        E_ddT__obs = dot(d_samples[0, :].T, d_samples[0, :])
        sigma_d__obs = mat(zeros(d_samples[0, :].shape))
        for k in range(1, d_samples.shape[0]):
            E_d__obs += d_samples[k, :]
            E_ddT__obs += dot(d_samples[k, :].T, d_samples[k, :])
        E_d__obs, E_ddT__obs = E_d__obs * (1. / n), E_ddT__obs * (1. / n)
        for k in range(d_samples.shape[0]):
            sigma_d__obs += np.power(d_samples[k, :] - E_d__obs, 2)

        for j in xrange(nmeas):
            for s in xrange(nstructs):
                if contact_sites[s][j, idx]:
                    E_c__obs[j, s] = E_d__obs[:, s] + contact_stochastics['c_%s_%s' % (j, s)].trace().mean()
                else:
                    E_c__obs[j, s] = nan

        return E_d__obs, E_ddT__obs, sqrt(sigma_d__obs * (1. / n)), E_c__obs


    def calculate_data_pred(self, no_contacts=False):
        self.sigma_pred = zeros([self.W.shape[0], self.E_d.shape[1]])
        if no_contacts:
            self.data_pred = dot(self.W, self.E_d)
        else:
            self.data_pred = zeros([self.W.shape[0], self.E_d.shape[1]])
            for i in xrange(self.data_pred.shape[1]):
                self.data_pred[:, i] = self._dot_E_d_i(self.W, self.E_d[:, i], self.E_c[:, :, i], i, self._restricted_contact_sites).T

                #sigma_ddT = sqrt(diag(self.E_ddT[:, :, i]))
                #self.sigma_pred[:, i] = sqrt(dot(self.W_std / self.W, ones([self.E_d.shape[0]]))**2 + asarray(self._dot_E_d_i(ones(self.W_std.shape), sigma_ddT / self.E_d[:, i], self.E_c[:, :, i], i)).T**2)
                self.sigma_pred[:, i] = sqrt(self.Psi[i, 0, 0])

        return self.data_pred, self.sigma_pred


    def _calculate_chi_sq(self):
        """Calculates the chi squared statistic given the fit to the data
        """
        data_pred, sigma_pred = self.calculate_data_pred()
        chi_sq = ((asarray(self.data) - asarray(data_pred))**2 / asarray(sigma_pred)**2).sum()
        return chi_sq


    def calculate_fit_statistics(self, data_pred=None, sigma_pred=None):
        """Calculate all fit statistics given the fit to the data.
            Returns:
                tuple. In order, chi-squared/Deg. Freedom, AIC, RMSEA
        """
        if data_pred is None or sigma_pred is None:
            data_pred, sigma_pred = self.calculate_data_pred()
        chi_sq = ((asarray(self.data) - asarray(data_pred))**2 / asarray(sigma_pred)**2).sum()
        df = self.data.size - self.data.shape[1] - 1
        if self.motif_decomposition != 'none':
            df += -2 * sum(self.nmotpos)
        else:
            df += -self.E_d.size - self.W.size
            df += - self.E_c[logical_not(isnan(self.E_c))].size
        k = -df - self.data.size + 1
        rmsea = sqrt(max((chi_sq / df - 1) / (self.data.shape[1] - 1), 0.0))
        aic = asscalar(chi_sq + 2 * k - self.data.shape[0] * self.logpriors)

        return chi_sq/df, rmsea, aic


    def correct_scale(self, stype='linear'):
        """Corrects the scale of the fitted data. Sometimes the fitted model will have 'higher-than-average' reactivities. This procedure
        corrects this artifact, without affecting the inferred weights.
            Returns:
                tuple. In order, the list of correction factors inferred, the new data predicted by the model, and the new errors of the predicted data.
        """
        data_pred, sigma_pred = self.calculate_data_pred()
        corr_facs = [1] * data_pred.shape[1]
        if stype == 'none':
            return corr_facs, self.data_pred, self.sigma_pred

        if stype == 'linear':
            for i in xrange(len(corr_facs)):
                corr_facs[i] = asscalar(dot(data_pred[:, i], self.data[:, i])/dot(data_pred[:, i], data_pred[:, i]))
                self.E_d[:, i] *= corr_facs[i]
                data_pred[:, i] *= corr_facs[i]
                print 'Linear correction scale factor for position %s is %s' % (i, corr_facs[i])

        if stype == 'exponential':
            def errorf(idx, x):
                return sum([(exp(x * data_pred[i, idx]) * data_pred[i, idx] - self.data[i, idx])**2 for i in xrange(data_pred.shape[0])])

            for i in xrange(len(corr_facs)):
                f = lambda x: errorf(i, x)
                opt_grid = (-20, 20, 0.01)
                corr_facs[i] = asscalar(optimize.brute(f, (opt_grid,)))
                for j in xrange(data_pred.shape[0]):
                    data_pred[j, i] *= exp(corr_facs[i] * data_pred[j, i])
                print 'Exponential correction scale factor for position %s is exp(%s * x)' % (i, corr_facs[i])

        return corr_facs, self.data_pred, self.sigma_pred


    def calculate_missed_predictions(self, sigthresh=0.05, data_pred=None, sigma_pred=None):
        """Calculates where the data is significantly NOT predicted by the model (i.e. where the model gives less than 0.05 probability
        to the data). This is useful to find special 'contacts' that were not in the specified variables (e.g. a hidden pseudoknot in
        mutate-and-map data).
            Returns:
               tuple. In order, the indices of the 'missed' predictions, and the values of these missed predictions.
        """
        missed_vals = zeros(self.data.shape)
        nmeas = self.data.shape[0]
        npos = self.data.shape[1]
        missed_indices = []
        if data_pred is None or sigma_pred is None:
            data_pred, sigma_pred = self.calculate_data_pred()
        for i in xrange(nmeas):
            for j in xrange(npos):
                if normpdf(self.data[i, j], data_pred[i, j], sigma_pred[i, j]) <= sigthresh:
                    # interpolate to our predicted data dynamic range
                    val = (self.data[i, j] / self.data.max()) * data_pred.max() - data_pred[i, j]
                    missed_vals[i, j] = val
                    missed_indices.append((i, j))
        return missed_indices, missed_vals

    
    # TODO Should not pass perturbs as by-reference argument, just return it.
    def _back_calculate_perturbations(self, W, energies, perturbs):
        """Given inferred weights and initial energies, calculates the perturbations needed for the energies to yield the inferred weights.
            Args:
                W (2d numpy array): The structure weight matrix.
                energies (2d numpy array): The initial structure energies (same dimensions as W).
                perturbs (2d numpy array): The perturbations (passed by reference)

            Returns:
                The perturbation matrix
        """
        for j in xrange(W.shape[0]):
            # Index por "pivot" energy, which will be the lowest energy
            # This is totally heuristic, we are assuming that the minimum energy
            # is the most "confident" and therefore needs no perturbation
            idx = where(energies[j, :] == energies[j, :].min())[0][0]
            perturbs[j, idx] = 0
            for s in xrange(W.shape[1]):
                if s != idx:
                    perturbs[j, s] = energies[j, idx] - energies[j, s] + utils.k * utils.T * log(abs((W[j, s] + 0.0001) / W[j, idx]))
        return perturbs


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
            energies = self.energies[:, select_struct_indices].copy() if len(self.energies) > 0 else []
            struct_types = [[s[i] for i in select_struct_indices] for s in self.struct_types]

        concentrations = [self.concentrations[i] for i in select_struct_indices] if len(self.concentrations) > 0 else []
        kds = self.kds[:, select_struct_indices].copy() if len(self.kds) > 0 else []
        return structures, energies, struct_types, contact_sites, concentrations, kds


    def _assign_W(self, data, E_d, E_c, E_ddT, E_ddT_inv, data_E_d, Psi, contact_sites, energies, concentrations, kds, G_constraint, nmeas, nstructs, use_struct_clusters, Wupper, Wlower, W_initvals):
        """Assign the weight matrices given inferred reactivities for the structures and contacts (the M-step). This is currently done in a quadratic program with convex combination constraints solved by CVXOPT.

            Args:
                data (2d numpy array): The mapping reactivity matrix.
                E_d (2d numpy array): The matrix containing the structure reactivities.
                E_c (3d numpy array): The matrix containing the 'contact map' for each structure.
                E_ddT (3d numpy array): The matrix containing the position-wise covariance reactivity matrices.
                E_ddT_inv (3d numpy array): Inverted matrices of E_ddT.
                data_E_d (3d numpy array): The matrix containing the position-wise products of E_d and data matrices.
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
        npos = Psi.shape[0]
        Psi_data_E_d_T = zeros([nmeas, nstructs])
        W = zeros([nmeas, nstructs])
        Psi_inv = zeros(Psi.shape)
        Psi_inv_sum = zeros([nmeas, nmeas])

        for i in xrange(npos):
            Psi_inv[i, :, :] = linalg.inv(Psi[i, :, :])
            Psi_inv_sum += linalg.inv(Psi[i, :, :])

        Psi_inv_sum = linalg.inv(Psi_inv_sum)

        for i in xrange(npos):
            Psi_data_E_d_T += dot(Psi_inv[i, :, :], data_E_d[:, :, i])

        if len(energies) > 0:
            print 'Solving weights'
            # ========= Using CVXOPT ====================
            # Converting to quadratic program
            frame = inspect.currentframe()
            args, _, _, argvals = inspect.getargvalues(frame)

            def par_fun(j, *args):
                E_d_proj = mat(zeros([nstructs, 1]))
                E_ddT_Psi_inv = zeros(E_ddT[:, :, 0].shape)
                E_d_c_j = self._E_d_c_j(E_d, E_c, j, contact_sites)
                E_ddT_Psi_inv += self.lam_ridge * eye(W.shape[1])
                for i in xrange(E_d_c_j.shape[1]):
                    #E_d_proj += data[j, i]*E_d[:, i]
                    E_d_proj += -Psi_inv[i, j, j] * data[j, i] * E_d_c_j[:, i]
                    E_ddT_Psi_inv += Psi_inv[i, j, j] * E_ddT[:, :, i]

                if self.lam_weights != 0:
                    E_ddT_Psi_inv += self.lam_weights * diag(1 / (W_initvals[j, :]**2 + 1e-10))
                    E_d_proj += self.lam_weights * mat(1. / (W_initvals[j, :] + 1e-10)).T

                if self.lam_mut != 0 and j in self.wt_indices:
                    E_ddT_Psi_inv += self.lam_mut*eye(W.shape[1])
                    for m in xrange(W.shape[0]):
                        if m not in self.wt_indices:
                            E_d_proj += -2 * self.lam_mut*mat(-W[m, :] - (W_initvals[j, :] - W_initvals[m, :])).T

                'Solving weights for measurement %s' % j
                #P = cvxmat(E_ddT_Psi_inv - (2*eye(nstructs) - ones([nstructs, nstructs])))
                P = cvxmat(E_ddT_Psi_inv)
                Gc = cvxmat(-eye(nstructs))
                h = cvxmat(0.0, (nstructs, 1))
                A = cvxmat(1.0, (1, nstructs))
                b = cvxmat(1.0)
                p = cvxmat(E_d_proj)

                def quadfun(x):
                    return asscalar(0.5 * dot(x.T, dot(E_ddT_Psi_inv, x)) + dot(E_d_proj.T, x))

                def quadfunprime(x):
                    return 0.5 * asarray(dot(E_ddT_Psi_inv, x) + asarray(E_d_proj).ravel())

                bounds = [(1e-10, None)] * nstructs
                #sol = optimize.fmin_l_bfgs_b(quadfun, W_initvals[j, :], fprime=quadfunprime, bounds=bounds)[0]
                sol = array(solvers.qp(P, p, Gc, h, A, b, None, {'x': cvxmat(W_initvals[j, :])})['x'])
                #sol = array(solvers.qp(P, p, Gc, h, A, b, None, {'x':cvxmat(0.5, (nstructs, 1))})['x'])
                #sol = array(solvers.qp(P, p, Gc, h, A, b)['x'])

                sol[sol <= 0] = 1e-10
                sol = sol / sol.sum()
                return sol.T

            if self.njobs is not None:
                sys.modules[__name__].par_fun = par_fun
                resvecs = joblib.Parallel(n_jobs=self.njobs)(joblib.delayed(par_fun)(j, *argvals) for j in xrange(nmeas))
                for j, sol in enumerate(resvecs):
                    W[j, :] = sol
            else:
                for j in xrange(nmeas):
                    W[j, :] = par_fun(j, *argvals)
            # ========= End using CVXOPT ================
            if G_constraint is not None:
                # Constrain by using G_constraint
                for m in xrange(nmeas):
                    for p in xrange(nstructs):
                        if W[m, p] > Wupper[m, p]:
                            W[m, p] = Wupper[m, p]
                        if W[m, p] > Wlower[m, p]:
                            W[m, p] = Wlower[m, p]
            # Sometimes there are small numerical errors that amplify a bit and they need to set straight
            # by forcing the sum of weights to be one -- maybe there is a better way to do the math
            # here to bypass these numerical instabilities?
            for j in xrange(nmeas):
                W[j, :] /= W[j, :].sum()
            if not use_struct_clusters:
                self.perturbs = self._back_calculate_perturbations(W, energies, self.perturbs)

        # If we were given concentrations and kds, we need to set the weights accordingly
        if len(concentrations) > 0:
            indices_to_adjust = defaultdict(list)
            cum_weights = defaultdict(int)
            for i in xrange(nmeas):
                for j in xrange(nstructs):
                    if not isnan(kds[i, j]):
                        # If a kd is zero, means that the weight should be zero
                        if kds[i, j] == 0:
                            W[i, j] = 0
                        else:
                            # I'm using a standard Hill equation here with n=1; no cooperativity or second order reactions
                            W[i, j] = concentrations[i]/(concentrations[i] + kds[i, j])
                            cum_weights[i] += W[i, j]
                    else:
                        indices_to_adjust[i].append(j)
            # If we have energies, then the weights must sum to 1, but we might have disrupted this when enforcing weights by kd
            # so re-enforce the sum to 1 constraint in the indices that are allowed to float.
            if len(energies) > 0:
                for i, indices in indices_to_adjust.iteritems():
                    sum_indices = (1 - cum_weights[i]) / W[i, indices].sum()
                    for idx in indices:
                        W[i, idx] *= sum_indices
        return W


    def _calculate_MLE_std(self, W, Psi_inv, E_ddT):
        """Calculates maximum likelihood estimate of weight error matrix using a Fisher information matrix approach.
        """
        I_W = zeros(W.shape)
        Psi_inv_vec = array([[Psi_inv[i, 0, 0] for i in xrange(Psi_inv.shape[0])]])
        for j in xrange(I_W.shape[0]):
            for s in xrange(I_W.shape[1]):
                I_W[j, s] = dot(Psi_inv_vec, E_ddT.T[:, :, s]).sum()
            """
            for s in xrange(I_W.shape[1]):
                for sp in xrange(I_W.shape[1]):
                    I_W[j, s] += dot(Psi_inv_vec, E_ddT[s, sp, :])
            """
        I_W[I_W == 0] = 1e-100
        return sqrt(1 / I_W)


    def _E_d_c_j(self, E_d, E_c, j, contact_sites):
        """Helper function to multiply E_d and E_c matrices at structure j given the contact sites
        """
        E_d_c_j = mat(zeros(E_d.shape))
        for s in xrange(E_d.shape[0]):
            for i in xrange(E_d.shape[1]):
                if contact_sites[s][j, i]:
                    E_d_c_j[s, i] = E_c[j, s, i]
                else:
                    E_d_c_j[s, i] = E_d[s, i]
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
                    if contact_sites[s][j, i]:
                        res[j, s] = asscalar(M[j]) * E_c_i[j, s]
                    else:
                        res[j, s] = asscalar(M[j]) * asscalar(E_d_i[s])
        else:
            res = mat(zeros([nmeas, 1]))
            for j in xrange(nmeas):
                d_c = zeros([nstructs])
                for s in xrange(nstructs):
                    if contact_sites[s][j, i]:
                        d_c[s] = E_c_i[j, s]
                    else:
                        d_c[s] = E_d_i[s]
                res[j] = dot(M[j, :], d_c)

        return res


    # TODO remove, this is just for testing
    def _get_determinant(self, Psi):
        if linalg.det(Psi) == 0:
            return 1e-100
        else:
            return linalg.det(Psi)

    # TODO Better documentation! Also, we should deprecate the following options: 
    # sigma_d0, G_constraint, soft_em, and post_model -- and wrap them up in other functions
    # We should also break up this method, it's too big!!!
    def analyze(self, max_iterations=100, tol=0.05, nsim=1000, select_struct_indices=[], W0=None, Psi0=None, E_d0=None, E_ddT0=None, E_c0=None, sigma_d0=None, cluster_data_factor=None, G_constraint=None, use_struct_clusters=False, seq_indices=None, return_loglikes=False, soft_em=False, post_model=False):
        """The main method that models the data in terms of the structures.
            Kargs:
                max_iterations (int): Maximum number of EM iterations.
                tol (float): The tolerance difference between the last and current base pair probability matrix of the inferred structure weights before the EM cycle is deemed convergent and terminates.
                nsim (int): For the soft EM version, number of samples for MCMC.
                select_struct_indices (list): A list of structure indices to consider for the analysis.
                W0 (2d numpy array): An initial estimate of the structure weight matrix.
                Psi0 (3d numpy array): An inital estimate of the error covariance matrices.
                E_d0 (3d numpy array): An initial estimate of the reactivity matrices.
                E_ddT0 (3d numpy array): An initial estimate of the covariance reactivity matrices.
                E_c0 (3d numpy array): An initial estimate of the contact matrices.
                sigma_d0 (2d numpy array): An initial estimate for the reactivity standard deviations.
                cluster_data_factor (float): Factor used to cluster the data. Useful for data redundancy elimination.
                G_constraint (float): Hard energetic constraint to prevent the inferred weights to be beyond a +/- energetic difference of the initial weights.
                use_struct_clusters (bool): Indicates if we use structure clusters or not.
                seq_indices (list): Indicates which sequence positions we use for the analysis.
                return_loglikes (bool): Chooses whether to return the trace of log-likelihoods [TODO: THIS CURRENTLY RETURNS CONVERGENCE VALUES]
                soft_em (bool): Indicates whether to use soft EM instead of hard EM
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
                * E_ddT: Inferred matrix of reactivity covariance matrices.
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
            allmeas = self.data.shape[0]
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
        nstructs = len(self.structure_clusters) if use_struct_clusters else len(structures)

        if self.motif_decomposition == 'motif':
            contact_sites = self._get_motif_contact_sites(contact_sites)

        if seq_indices is not None:
            data = data[:, seq_indices]
            struct_types = [struct_types[i] for i in seq_indices]
            for s in xrange(len(structures)):
                contact_sites[s] = contact_sites[s][:, seq_indices]
        else:
            seq_indices = range(data.shape[1])
        self._restricted_contact_sites = contact_sites

        npos = data.shape[1]
        nmeas = data.shape[0]
        """
        Initialize the variables
        For clarity: the model is
        data_i = W*d_i + epsilon
        epsilon ~ Normal(0, Psi)
        d_i ~ multivariate gamma mixture depending on struct_types
        """
        print 'Initializing weights W and covariance matrix Psi'
        min_var = 1e-100**(1. / nmeas)
        if Psi0 is not None:
            Psi = Psi0
        else:
            Psi = zeros([npos, nmeas, nmeas])
            for i in xrange(npos):
                indices = where(data[:, i] < mean(data[:, i]))[0]
                if len(indices) < 10:
                    indices = range(nmeas)
                Psi[i, :, :] = diag([max((data[indices, i].std())**2, min_var)] * nmeas)
        if W0 is not None:
            W = W0
            self.perturbs = zeros([nmeas, nstructs])
        else:
            scale = sum([self._get_determinant(Psi[i, :, :])**(1. / nmeas) for i in xrange(npos)]) * (1. / npos)
            # Starting point is the weights given by the RNAstructure energies plus a small gaussian-dist perturbation
            self.perturbs = normal(0, sqrt(scale/nstructs), size=(nmeas, nstructs))
            if len(energies):
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

        if G_constraint is not None:
            for m in xrange(nmeas):
                for p in xrange(nstructs):
                    Wupper[m, p] = W[m, p] * exp(G_constraint / (utils.k * utils.T))
                    Wlower[m, p] = W[m, p] * exp(-G_constraint / (utils.k * utils.T))
                    if Wupper[m, p] < Wlower[m, p]:
                        tmp = Wupper[m, p]
                        Wupper[m, p] = Wlower[m, p]
                        Wlower[m, p] = tmp

        W_initvals = W
        post_structures = structures
        bp_dist = self.bp_dist[select_struct_indices, :][:, select_struct_indices]
        print 'Finished initializing'
        # For brevity E_d = E[d | data], E_d =[E[c | data], and E_ddT = E[d*dT | data], and so on...

        Psi_inv = zeros(Psi.shape)
        for i in xrange(npos):
            Psi_inv[i, :, :] = linalg.inv(Psi[i, :, :])

        if self.motif_decomposition in ['none', 'element']:
            E_d = mat(zeros([nstructs, npos]))
            E_c = zeros([nmeas, nstructs, npos])
            sigma_d = mat(zeros([nstructs, npos]))
            E_ddT = zeros([nstructs, nstructs, npos])

        else:
            # We have motif decomposition, so adjust variables accordingly
            E_d, E_c, sigma_d, E_ddT, nmotifs = self._initialize_motif_reactivities(npos)
            W = self._initialize_motif_weights(W)
            nstructs = len(self.motif_ids)

        E_ddT_i = mat(zeros(E_ddT.shape[:2]))
        E_ddT_inv = zeros(E_ddT.shape)
        data_E_d = zeros(E_c.shape)

        data_dataT = zeros([nmeas, nmeas, npos])

        old_loglike = -inf
        base_loglike = None
        Psi_opt = Psi
        W_opt = W
        E_d_opt = E_d
        E_c_opt = E_c
        sigma_d_opt = sigma_d
        E_ddT_opt = E_ddT
        max_loglike = old_loglike
        loglikes = []
        Psi_reinits = []
        bppm_prev = utils.bpp_matrix_from_structures(self._origstructures, W[self.wt_indices[0], :])
        adaptive_factor = 1

        t = 0
        for i in xrange(npos):
            data_dataT[:, :, i] = dot(data[:, i], data[:, i].T)
        while t < max_iterations:
            # E-step
            loglike = -npos*nmeas / 2. * log(2. * pi)
            if not soft_em:
                if self.njobs is not None:
                    def par_fun(i):
                        return self.hard_EM_vars(i, W, Psi_inv[i, :, :], data, struct_types, contact_sites, bp_dist, seq_indices)
                    sys.modules[__name__].par_fun = par_fun
                    restuples = joblib.Parallel(n_jobs=self.njobs)(joblib.delayed(par_fun)(i) for i in xrange(npos))

                else:
                    restuples = []
                    for i in xrange(npos):
                        restuples.append(self.hard_EM_vars(i, W, Psi_inv[i, :, :], data, struct_types, contact_sites, bp_dist, seq_indices))

            else:
                if self.njobs is not None:
                    def par_fun(i):
                        return self.soft_EM_vars(i, W, Psi_inv[i, :, :], data, struct_types, contact_sites, nsim, nsim / 5, measindices, use_struct_clusters)
                    sys.modules[__name__].par_fun = par_fun
                    restuples = joblib.Parallel(n_jobs=self.njobs)(joblib.delayed(par_fun)(i) for i in xrange(npos))
                else:
                    restuples = []
                    for i in xrange(npos):
                        restuples.append(self.soft_EM_vars(i, W, Psi_inv[i, :, :], data, struct_types, contact_sites, nsim, nsim / 5, measindices, use_struct_clusters))

            for i, tup in enumerate(restuples):
                E_d_i, E_ddT_i, sigma_d_i, E_c_i = tup
                E_d[:, i] = E_d_i.T
                sigma_d[:, i] = sigma_d_i.T
                E_c[:, :, i] = E_c_i
                E_ddT[:, :, i] = E_ddT_i

            if E_d0 is not None and t == 0:
                E_d, E_ddT, E_c, sigma_d = E_d0, E_ddT0, E_c0, sigma_d0
                for i in xrange(npos):
                    E_ddT[:, :, i] = dot(E_d[:, i].T, E_d[:, i])

            #imshow(E_d, vmax=E_d.mean(), vmin=0, cmap=get_cmap('Greys'))


            if self.njobs is not None:
                def par_fun(i):
                    B = self._dot_E_d_i(data[:, i], E_d[:, i], E_c[:, :, i], i, contact_sites, T=True)
                    C = -dot(dot(data[:, i].T, Psi_inv[i, :, :]), data[:, i])
                    C += 2. * self._dot_E_d_i(dot(dot(data[:, i].T, Psi_inv[i, :, :]), W), E_d[:, i], E_c[:, :, i], i, contact_sites)
                    C += -(dot(dot(dot(W.T, Psi_inv[i, :, :]), W), E_ddT[:, :, i])).diagonal().sum()
                    l = 0.5 * (C - npos * log(self._get_determinant(Psi[i, :, :])))
                    return B, l
                sys.modules[__name__].par_fun = par_fun
                restuples = joblib.Parallel(n_jobs=self.njobs)(joblib.delayed(par_fun)(i) for i in xrange(npos))
                for i, tup in enumerate(restuples):
                    B, l = tup
                    data_E_d[:, :, i] = B
                    loglike += l
            else:
                for i in xrange(npos):
                    data_E_d[:, :, i] = self._dot_E_d_i(data[:, i], E_d[:, i], E_c[:, :, i], i, contact_sites, T=True)
                    #data_E_d += dot(data[:, i], E_d[:, i].T)
                    # Don't forget to calculate the log-likelihood to track progress!
                    C = -dot(dot(data[:, i].T, Psi_inv[i, :, :]), data[:, i])
                    C += 2. * self._dot_E_d_i(dot(dot(data[:, i].T, Psi_inv[i, :, :]), W), E_d[:, i], E_c[:, :, i], i, contact_sites)
                    #C += -2.*dot(dot(dot(data[:, i].T, Psi_inv), W), E_d[:, i])
                    C += -(dot(dot(dot(W.T, Psi_inv[i, :, :]), W), E_ddT[:, :, i])).diagonal().sum()
                    loglike += 0.5 * (C - npos * log(self._get_determinant(Psi[i, :, :])))
            # M-step
            """
            # This is no longer necessary since we are not using the closed-form solution of W
            try:
                for i in xrange(npos):
                    E_ddT_inv[:, :, i] = linalg.inv(E_ddT[:, :, i])
            except LinAlgError:
                print 'An error occured on inverting E[d*dT | data]'
                print 'Position %s' % i
                print 'E[d*dT | data]'
                print E_ddT[:, :, i]
                print 'E[d*dT | data][:, i]'
                print E_d[:, i]
                raise ValueError('E[d*dT | data] matrix is singular!')
            """

            # Given our expected reactivities, now assign the weights
            W_new = self._assign_W(data, E_d, E_c, E_ddT, E_ddT_inv, data_E_d, Psi, contact_sites, energies, concentrations, kds, G_constraint, nmeas, nstructs, use_struct_clusters, Wupper, Wlower, W_initvals)

            #W_new = W + adaptive_factor*(W_new - W)
            W_new[W_new < 0] = 1e-10
            for j in xrange(nmeas):
                W_new[j, :] /= W_new[j, :].sum()

            # Update stopping criterion
            bppm_new = utils.bpp_matrix_from_structures(self._origstructures, W_new[self.wt_indices[0], :])
            currdiff = abs(bppm_prev[bppm_new != 0] - bppm_new[bppm_new != 0]).mean()
            bppm_prev = bppm_new
            W = W_new

            # Now get covariance matrix
            data_pred = zeros([W.shape[0], E_d.shape[1]])

            if self.njobs is not None:

                def par_fun(i):
                    return self._dot_E_d_i(W, E_d[:, i], E_c[:, :, i], i, contact_sites).T

                sys.modules[__name__].par_fun = par_fun
                restuples = joblib.Parallel(n_jobs=self.njobs)(joblib.delayed(par_fun)(i) for i in xrange(npos))
                for i, D in enumerate(restuples):
                    data_pred[:, i] = D
            else:
                for i in xrange(data_pred.shape[1]):
                    data_pred[:, i] = self._dot_E_d_i(W, E_d[:, i], E_c[:, :, i], i, contact_sites).T
            for i in xrange(npos):
                P = sum(array(data[:, i].ravel() - data_pred[:, i].ravel())**2)
                Pd = array([P/nmeas]*nmeas)
                Pd[Pd == 0] = min_var
                Pd[isnan(Pd)] = min_var
                Psi[i, :, :] = diag(Pd)
                # Sometimes we have a weird case where the estimated Psi is negative, mostly when the noise is very very low
                if sum(Psi[i, :, :] < 0) > 0:
                    print 'Psi is out of bounds (%s values were below zero!), setting to minimum variance' % sum(Psi[i, :, :] < 0)
                    Psi[i, :, :][Psi[i, :, :] < 0] = min_var
                    if t not in Psi_reinits:
                        Psi_reinits.append(t)
                try:
                    Psi_inv[i, :, :] = linalg.inv(Psi[i, :, :])
                except LinAlgError:
                    pdb.set_trace()

            # Do post-facto SHAPE-directed modeling, if asked
            didpostmodel = False
            if post_model:
                for sidx, s in enumerate(select_struct_indices):
                    domodeling = False
                    # Be conservative, for each structure, only do post-facto modeling if E_d is totally off for at least one position
                    for iidx in xrange(len(seq_indices)):
                        if struct_types[iidx][sidx] == 'p' and self.paired_pdf(E_d[sidx, iidx]) < 0.05:
                            print self.paired_pdf(E_d[sidx, iidx])
                            domodeling = True
                    if domodeling:
                        md = mapping.MappingData(data=asarray(E_d[sidx, :] * 1.5).ravel(), seqpos=seq_indices)
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
                        logpriors += -self.lam_reacts*1/self.bp_dist[sidx, sidx2]*(asarray(E_d[sidx, :] - E_d[sidx2, :])**2).sum()
                    except FloatingPointError:
                        logpriors += MIN_LOGLIKE
                try:
                    logpriors += -(self.lam_weights*(W*(1./W_initvals))).sum()
                except FloatingPointError:
                    logpriors += MIN_LOGLIKE
                for iidx in xrange(len(seq_indices)):
                    try:
                        if struct_types[iidx][sidx] == 'p':
                            logpriors += log(self.paired_pdf(E_d[sidx, iidx]))
                        else:
                            logpriors += log(self.unpaired_pdf(E_d[sidx, iidx]))
                    except FloatingPointError:
                        logpriors += MIN_LOGLIKE
            """
            loglike += logpriors
            # Check if we are done
            print 'Finished iteration %s with log-likelihood %s' % (t, loglike)
            if not didpostmodel:
                print t
                t += 1
            #loglikes.append(asscalar(loglike))
            loglikes.append(currdiff)
            #loglikes.append(asscalar(chi_sq))
            if base_loglike is None and max_iterations != 1:
                base_loglike = loglike
            else:
                if True: # loglike > max_loglike or max_iterations == 1:
                    max_loglike = loglike
                    self.logpriors = logpriors
                    W_opt = W.copy()
                    E_d_opt = E_d.copy()
                    E_c_opt = E_c.copy()
                    sigma_d_opt = sigma_d.copy()
                    Psi_opt = Psi.copy()
                    E_ddT_opt = E_ddT.copy()
                    adaptive_factor *= 2

                if currdiff <= tol:
                    break

                if t + 1 == max_iterations:
                    break
                if loglike < old_loglike:
                    print 'Warning: Likelihood value is decreasing. This is probably due to:'
                    print '* Covariance or weights matrices were out of bounds due to constraints and were reinitizialized (likely)'
                    print '* Not enough sampling in the E-step if simulating from it (less likely)'
                    print '* Numerical error due to extremely low noise (unlikely)'
                    adaptive_factor = 1

                # Reinitialize the relevant variables
                E_d[:] = 0
                sigma_d[:] = 0
                E_c[:] = 0
                E_ddT[:] = 0
                E_ddT_i[:] = 0
                data_E_d[:] = 0
            old_loglike = loglike
        print ' Finished -- log-likelihood value %s' % max_loglike
        lhood = [loglikes, Psi_reinits] if return_loglikes else max_loglike
        self.W, self.Psi, self.E_d, self.E_c, self.sigma_d, self.E_ddT = asarray(W_opt), asarray(Psi_opt), asarray(E_d_opt), asarray(E_c_opt), asarray(sigma_d_opt), asarray(E_ddT_opt)

        print 'W_std calculation started'
        # Calculate the standard deviation of the MLE
        self.W_std = self._calculate_MLE_std(W_opt, Psi_inv, E_ddT_opt)
        print 'W_std calculation finished'

        # Interpolate for the rest of the data if we clustered
        if cluster_data_factor:
            self.cluster_data_pred = dot(W, E_d)
            W_std = zeros([allmeas, nstructs])
            W = zeros([allmeas, nstructs])
            Psi = zeros([allmeas, allmeas])
            perturbs = zeros([allmeas, nstructs])
            alldata_E_d = mat(zeros([allmeas, nstructs]))
            for i in xrange(npos):
                alldata_E_d += dot(self.data[:, i], E_d[:, i].T)

            for i, c in enumerate(bestclusts):
                cidx = measindices.index(chosenindices[c])
                if len(self.energies) > 0:
                    perturbs[i, :] = self.perturbs[cidx, :]
                Psi[i, i] = self.Psi[cidx, cidx]

            Psi_inv = linalg.inv(Psi)
            self.perturbs = perturbs
            self.W = asarray(self._assign_W(data, E_d, E_ddT, E_ddT_inv, alldata_E_d, Psi, contact_sites, allenergies, allconcentrations, allkds, G_constraint, allmeas, nstructs, use_struct_clusters, Wupper, Wlower, W_initvals))
            self.W_std = asarray(self._calculate_MLE_std(self.W, Psi_inv, E_ddT))
            self.Psi = asarray(Psi)
            self.perturbs = asarray(perturbs)
            self.data_clusters = bestclusts
            self.data_cluster_medoids = chosenindices

        # Map final values for weights, reactivities and errors to structure space if we did motif decomposition
        if self.motif_decomposition == 'motif':
            self._map_variables_to_structures()
            # Need to also "restore" the restricted_contact_sites variables to structure, rather than motif space
            self._restricted_contact_sites = saved_contact_sites


        if len(self.energies) > 0 or use_struct_clusters:
            self.perturbs = zeros(self.W.shape)
            return lhood, self.W, self.W_std, self.Psi, self.E_d, self.E_c, self.sigma_d, self.E_ddT, self.perturbs, post_structures
        return lhood, self.W, self.W_std, self.Psi, self.E_d, self.E_c, self.sigma_d, self.E_ddT, post_structures


class FullBayesAnalysis(MappingAnalysisMethod):
    """This method performs a full Bayesian analysis using MCMC over the energies, reactivities, and covariances.
    """

    def __init__(self, data, structures, backend='pickle', dbname='ba_traces.pickle', mutpos=[], concentrations=[], kds=[], energies=[], c_size=3, cutoff=0, njobs=None):
        MappingAnalysisMethod.__init__(self, data, structures, energies, mutpos, c_size, cutoff)
        self.concentrations = concentrations
        self.kds = kds
        self.backend = backend
        self.dbname = dbname
        self.njobs = njobs


    def _get_FAobj(self):
        if 'fa' not in self.__dict__:
            self.fa = FAMappingAnalysis(self.data, self.structures, mutpos=self.mutpos, concentrations=self.concentrations, kds=self.kds, energies=self.energies, c_size=self.c_size)
        return self.fa


    def _initialize_mcmc_variables(self, method, data_obs, D, M, Psi_diag, C, inititer):
        if not method:
            return [data_obs, D, M, Psi_diag, C]

        if method == 'map':
            mp = MAP([data_obs, D, M])
            mp.fit(iterlim=inititer, verbose=True)
            Psi_diag.value = diag(cov(data))
            return list(mp.variables) + [Psi_diag, C]

        if method == 'fa':
            W, Psi_opt, D_opt, E_ddT_opt, M_opt = self._get_FAobj().analyze(nsim=10000, max_iterations=inititer)
            Psi_diag.value = diag(asarray(linalg.inv(Psi_opt)))
            for i in xrange(len(D)):
                D[i].value = D_opt[:, i]
            for j in xrange(len(M)):
                M[j].value = M_opt[j, :]
            return [data_obs, D, M, Psi_diag]


    def analyze(self, n=1000, burn=100, thin=1, verbose=0, inititer=5, initmethod=None, print_stats=False):
        # This is the full model
        nmeas = self.data.shape[0]
        npos = self.data.shape[1]
        nstructs = len(self.structures)
        # Likelihood value for M (the energy perturbations)

        def m_prior_loglike(value):
            loglike = 0
            for s in xrange(len(value)):
                loglike += distributions.uniform_like(value[s], -200, 200)
            return loglike

        def psidiag_loglike(value):
            loglike = 0
            for s in xrange(len(value)):
                loglike += distributions.uniform_like(value[s], -1, 500)
            return loglike

        # Likelihood value for D (the hidden reactivities)
        def d_prior_loglike(value, i):
            loglike = 0
            for s in xrange(len(value)):
                if value[s] < 0:
                    return -inf
                if self.struct_types[i][s] == 'u':
                    loglike += log(SHAPE_unpaired_pdf(value[s]))
                else:
                    loglike += log(SHAPE_paired_pdf(value[s]))
            return loglike

        # Diagonal and off-diagonal contacts and mutation effects
        def c_prior_loglike(value, i):
            loglike = 0
            for j in xrange(len(value)):
                if self.contact_sites[j, i] == 1:
                    loglike += log(SHAPE_contacts_diff_pdf(value[j]))
                else:
                    if value[j] != 0:
                        return -inf
            return loglike

        d0 = zeros([nstructs, npos])
        m0 = zeros([nmeas, nstructs])
        c0 = zeros([nmeas, npos])
        psidiag0 = rand(nmeas)
        print 'Choosing good d_0s, c_0s, and m_0s'
        for j in xrange(nmeas):
            for s in xrange(nstructs):
                m0[j, s] = rand()
        for i in xrange(npos):
            for j in xrange(nmeas):
                if self.contact_sites[j, i]:
                    c0[j, i] = SHAPE_contacts_diff_sample()
            while True:
                d0[:, i] = rand(len(self.struct_types[i]))
                logr = log([rand() for x in xrange(len(self.struct_types[i]))]).sum()
                if d_prior_loglike(d0[:, i], i) >= logr:
                    break

        def d_calc_eval(Weights, d):
            return dot(Weights, d)

        def Weights_eval(free_energies_perturbed=[]):
            Z = sum(exp(free_energies_perturbed / (utils.k * utils.T)))
            return exp(free_energies_perturbed / (utils.k * utils.T)) / Z

        @stochastic
        def Psi_diag(value=psidiag0):
            return psidiag_loglike(value)

        @deterministic
        def Psi_mat(d=Psi_diag):
            return diag(d)

        D = [stochastic(lambda value=d0[:, i]: d_prior_loglike(value, i), name='d_%s' % i) for i in xrange(npos)]
        C = [stochastic(lambda value=c0[:, i]: c_prior_loglike(value, i), name='c_%s' % i) for i in xrange(npos)]
        #M = [MvNormal('m_%s' % j, mu=zeros(nstructs) + 5, tau=eye(nstructs)) for j in xrange(nmeas)]
        M = [stochastic(lambda value=m0[j, :]: m_prior_loglike(value), name='m_%s' % j) for j in xrange(nmeas)]
        free_energies_perturbed = [deterministic(lambda x=self.energies[j, :], y=M[j]: x + y, name='free_energies_preturbed_%s' % j) for j in xrange(nmeas)]
        Weights = [Deterministic(eval=Weights_eval, name='Weights_%s' % j, parents={'free_energies_perturbed': free_energies_perturbed[j]}, doc='weights', trace=True, verbose=1, dtype=ndarray, plot=False, cache_depth=2) for j in xrange(nmeas)]
        d_calc = [Deterministic(eval=lambda **kwargs: d_calc_eval(kwargs['Weights'], kwargs['d']), name='d_calc_%s' % i, parents={'Weights': Weights, 'd': D[i]}, doc='d_calc', trace=True, verbose=1, dtype=ndarray, plot=False, cache_depth=2) for i in xrange(npos)]
        data_obs = [MvNormal('data_obs_%s' % i, value=[self.data[:, i]], mu=d_calc[i], tau=Psi_mat, observed=True) for i in xrange(npos)]
        print 'Finished declaring'
        mc = MCMC(self._initialize_mcmc_variables(initmethod, data_obs, D, M, Psi_diag, C, inititer) + [d_calc, Weights, free_energies_perturbed], db=self.backend, dbname=self.dbname)
        mc.use_step_method(AdaptiveMetropolis, M)
        mc.use_step_method(AdaptiveMetropolis, D)
        mc.use_step_method(AdaptiveMetropolis, Psi_diag)
        #mc.use_step_method(AdaptiveMetropolis, C)

        print 'Starting to sample'
        mc.sample(iter=n, burn=burn, thin=thin, verbose=verbose)
        print '\nDone sampling, recovering traces'
        nsamples = len(D[0].trace())
        self.Psi_trace = Psi_diag.trace()
        self.Weights_trace = zeros([nsamples, nmeas, nstructs])
        self.d_calc_trace = zeros([nsamples, nmeas, npos])
        self.D_trace = zeros([nsamples, nstructs, npos])
        self.free_energies_perturbed_trace = zeros([nsamples, nmeas, nstructs])
        self.M_trace = zeros([nsamples, nmeas, nstructs])
        for j, w in enumerate(Weights):
            for s in xrange(nstructs):
                self.Weights_trace[:, j, s] = w.trace()[:, s]
        for i, d in enumerate(d_calc):
            for j in xrange(nmeas):
                self.d_calc_trace[:, j, i] = d.trace()[:, j].T
        for j, f in enumerate(free_energies_perturbed):
            for s in xrange(nstructs):
                self.free_energies_perturbed_trace[:, j, s] = f.trace()[:, s]
        for j, m in enumerate(M):
            for s in xrange(nstructs):
                self.M_trace[:, j, s] = m.trace()[:, s]
        for i, d in enumerate(D):
            for s in xrange(nstructs):
                self.D_trace[:, s, i] = d.trace()[:, s]

        print 'Done recovering traces'
        if print_stats:
            print 'Finished, statistics are:'
            print mc.stats()
            pymc.Matplot.plot(mc, path='.')
        mc.db.close()
        return self.Psi_trace, self.d_calc_trace, self.free_energies_perturbed_trace, self.Weights_trace, self.M_trace, self.D_trace


class MCMappingAnalysis(FAMappingAnalysis):
    """Experimental method, wrap REEFFIT in MCMC procedure?
    """

    def __init__(self, *args, **kwargs):
        FAMappingAnalysis.__init__(self, *args, **kwargs)
        self.bppms = []
        for seq in self.sequences:
            self.bppms.append(zeros([len(seq), len(seq)]))


    def simulate(self, nsamples, **kwargs):
        nmeas = self.data.shape[0]
        counts = [0.] * nmeas
        nfactors = 2
        all_structures = []
        for i in xrange(nsamples):
            print 'Sampling %s' % i
            # Choose sequence to sample from
            sequence = choice(self.sequences)
            struct = ss.sample(sequence, nstructs=1)[0][0]
            all_structures.append(struct.dbn)
            print 'Structure %s' % struct.dbn
            bps = struct.base_pairs()
            self.set_structures([struct.dbn] + ['a' * len(struct) for x in xrange(nfactors)])
            self.energies = ones([nmeas, nfactors + 1])
            sys.stdout = MockPrint()
            sys.stderr = MockPrint()
            _, W, _, _, _, _, _, _, _, _ = self.analyze(**kwargs)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            # Accept/reject structure depending its weight
            for j in xrange(nmeas):
                if rand() < W[j, 0]:
                    if j == 0:
                        print 'Accept! (weight %s)' % W[j, 0]
                    for b1, b2 in bps:
                        self.bppms[j][b1, b2] += 1.
                    counts[j] += 1.
                else:
                    if j == 0:
                        print 'Reject! (weight %s)' % W[j, 0]
        for j in xrange(nmeas):
            self.bppms[j] /= counts[j]
        print 'Done, getting initial bppms'
        energies = utils.get_free_energy_matrix(all_structures, self.sequences)
        W0 = utils.calculate_weights(energies)
        for j in xrange(nmeas):
            bppm = utils.bpp_matrix_from_structures(all_structures, W0[j, :])
            for k in xrange(bppm.shape[0]):
                for l in xrange(bppm.shape[1]):
                    if bppm[k, l] != 0:
                        self.bppms[j][l, k] = bppm[k, l]


# TODO Better logging!
class MockPrint(object):
    """For silencing prints
    """
    def write(self, s):
        pass


        
