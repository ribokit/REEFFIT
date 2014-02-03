import pdb
import inspect
#import mdp
import pymc
import joblib
import map_analysis_utils as utils
import rdatkit.secondary_structure as ss
from rdatkit import mapping
import scipy.cluster.hierarchy as sphclust
import pymc.graph
import pymc.Matplot
import itertools
from random import choice, sample
from matplotlib.pylab import *
from numpy import lib
from reactivity_distributions import *
from collections import defaultdict, Counter
from scipy.stats import gamma
from scipy.stats.kde import gaussian_kde
import scipy.spatial.distance
from pymc import MAP, Model, MvNormal, stochastic, Deterministic, deterministic, MCMC, Wishart, AdaptiveMetropolis, Uniform, Cauchy, distributions, Laplace
from cvxopt import solvers
from scipy import optimize
from cvxopt import matrix as cvxmat

# Homebrew implementation of k-medoids using a pairwise distance matrix.
# I know this should be imported from some package, but I want to reduce
# dependencies and would not want to require pycluster just for this
def _cost(assigned_elems, medoid, distmat):
    res = 0.
    for e in assigned_elems:
        res += distmat[e, medoid]
    return res/len(assigned_elems)

def _assign(elems, medoids):
    assignments = defaultdict(list)
    for e in elems:
        mindist = inf
        for m in medoids:
            if distmat[e, m] < mindist:
                mindist = distmat[e,m]
                currmed = m
            assignments[med].append(e)
    return assignments

def _kmedoids(distmat, k):
    elems = range(distmat.shape[0])
    medoids = sample(elems, k)
    assignments = _assign(elems, medoids)
    stopcond = False
    while not stopcond:
        stopcond = True
        newmedoids = [m for m in medoids]
        currassignments = assignments.copy()
        for m in medoids:
            currcost = _cost(assignments[m], m, distmat)
            currmed = m
            for e in elems:
                if e not in medoids:
                    c = _cost(currassignments[m], e, distmat)
                    if c < currcost:
                        currcost = c
                        currmed = e
            if currmed != m:
                newmedoids.remove(m)
                newmedoids.append(currmed)
                stopcond = False
                currassignments = _assign(elems, newmedoids)
        medoids = newmedoids
        assignments = currasignments
    return medoids, assignments

# Here, the data can be one or two-dimensional
def _structure_likelihood(struct_types, idx, data, Psi, prior_weights=None):
    if prior_weights == None:
        prior_weights = ones([data.shape[0]])
    res = zeros([data.shape[0]])
    for i in xrange(len(struct_types)):
        if struct_types[i][idx] == 'u':
            for j in xrange(data.shape[0]):
                if data[j,i] > 0:
                    res[j] += log(SHAPE_unpaired_pdf(data[j,i])) - log(Psi[i,0,0]) + log(prior_weights[j])
        elif struct_types[i][idx] == 'p':
            for j in xrange(data.shape[0]):
                if data[j,i] > 0:
                    res[j] += log(SHAPE_paired_pdf(data[j,i])) - log(Psi[i,0,0]) + log(prior_weights[j])
        else:
            print 'WARNING: Structure type %s not recognized' % struct_type[i][idx]
            res = None
    return res

def structure_likelihood(struct_types, idx, data_energies, data_struct_types, struct_data_energies):
    res = zeros([len(data_energies)])
    """data_min = min([-d for d in data_energies])
    for j in xrange(len(data_energies)):
        res[j] = -data_energies[j]
        for i in xrange(len(data_struct_types)):
            if data_struct_types[i][j] != struct_types[i][idx]:
                res[j] -=  data_min"""
    for j in xrange(len(data_energies)):
        res[j] = struct_data_energies[idx][j]
    return res



# MappingAnalysisMethod class for mapping analysis
class MappingAnalysisMethod():
    def __init__(self, data, structures, sequences, energies, wt_indices, mutpos, c_size, seqpos_range):
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

    def set_structures(self, structures):
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

    FACTOR_ESTIMATION_METHODS = ['fanalysis', 'hclust']
    MODEL_SELECT_METHODS = ['heuristic', 'bruteforce']
    def __init__(self, data, structures, sequences, wt_indices=[0], mutpos=[], concentrations=[], kds=[], energies=[], c_size=3, seqpos_range=None, lam_reacts=0, lam_weights=0, lam_mut=0, njobs=None):
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
        self.lam_mut = lam_weights
        self.use_motif_decomposition = False

    def perform_motif_decomposition(self):
        print 'Starting motif decomposition'
        self.pos_motif_map, self.motif_ids, self.motif_dist = utils.get_minimal_overlapping_motif_decomposition(self._origstructures, bytype=True)
        self.nmotpos = []
        self.posmap = defaultdict(list)
        for i in xrange(self.data.shape[1]):
            nmotifs = 0
            for midx in xrange(len(self.pos_motif_map)):
                if (i, midx) in self.pos_motif_map:
                    nmotifs += 1
                    self.posmap[i].append(self.motif_ids[midx])
            self.nmotpos.append(nmotifs)
        print 'Number of motifs per position: %s' % self.nmotpos
        self.use_motif_decomposition = True

    # To perform a standard factor analysis, with normal priors
    # We use this mainly to have a first estimate of the number
    # structures needed before performing the real factor analysis
    def _standard_factor_analysis(self, nstructs):
        fnode = mdp.nodes.FANode(output_dim=nstructs)
        fnode.train(asarray(self.data.T))
        weights = fnode.execute(self.data.T)
        factors = fnode.A.T
        data_pred = dot(weights, factors).T
        return fnode.lhood[-1], data_pred, weights, factors

    def set_priors_by_rvs(self, unpaired_rvs, paired_rvs):
        unpaired_data = self.data[self.data > self.data.mean()].tolist()[0]
        paired_data = self.data[self.data <= self.data.mean()].tolist()[0]
        unpaired_pdf_data = [unpaired_rvs() for i in xrange(len(unpaired_data))]
        paired_pdf_data = [paired_rvs() for i in xrange(len(paired_data))]
        unpaired_data += unpaired_pdf_data
        paired_data += paired_pdf_data
        self.unpaired_pdf = gaussian_kde(unpaired_data)
        self.paired_pdf = gaussian_kde(paired_data)
        return unpaired_data, paired_data


    # Cluster structures together to get a more coarse-grained estimate
    def set_structure_clusters(self, structure_clusters, struct_medoid_indices, struct_weights_by_clust=[]):
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

    def model_select(self, expected_structures=0, greedy_iter=0, expstruct_estimation='hclust', tol=1e-4, max_iterations=10, prior_swap=True, G_constraint=None, apply_pseudoenergies=True, algorithm='rnastructure', hard_em=False, method='heuristic', post_model=False):
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
            if expstruct_estimation == 'fanalysis':
                print 'Using standard factor analysis to estimate this'
                aics = [inf]*(nstructs+1)
                for ns in range(2, nstructs+1):
                    lhood, data_pred, weights, factors = self._standard_factor_analysis(ns)
                    aics[ns] = utils._AICc(lhood, size(weights), size(self.data))
                    print 'For %s structures, AICc is %s' % (ns, aics[ns])
                expected_structures = aics.index(min(aics))
                print 'Estimated %s number of structures for model' % expected_structures
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
            data_structs = []
            data_energies = []
            struct_data_energies = zeros([len(self.structures), self.data.shape[0]])
            mapping_datas = []
            for j in xrange(self.data.shape[0]):
                #md = mapping.MappingData(data=array(self.data[j,:]).ravel(), enforce_positives=True)
                md = mapping.MappingData(data=array(self.data[j,:]).ravel())
                mapping_datas.append(md)
                fold_structures = ss.fold(self.sequences[j], mapping_data=mapping_datas[j], algorithm=algorithm)
                data_structs.append(fold_structures[0].dbn)
                energies = ss.get_structure_energies(self.sequences[j], fold_structures, mapping_data=mapping_datas[j], algorithm=algorithm)
                data_energies.append(energies[0])
            data_struct_types = utils.get_struct_types(data_structs)

            print 'Getting energies of each structure guided by the chemical mapping data'
            all_struct_obj = [ss.SecondaryStructure(dbn=s) for s in self.structures]
            cannonical_bp = [('G','C'), ('C','G'), ('G','U'), ('U','G'), ('A','U'), ('U','A')]
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
                        if energy > 200:
                            struct_data_energies[s,j] = 200
                        else:
                            struct_data_energies[s,j] = energy

            print 'Getting probabilities for each structure given energies'
            struct_data_probs = utils.calculate_weights(struct_data_energies.T).T

            chosenindices, bestclusts = self._cluster_data(float(est_nstructs)/len(self.structures))
            data_red = self.data[chosenindices.keys(),:]
            nmeas = data_red.shape[0]
            print 'Scoring structures for wild type -- for prior swapping'
            struct_scores = [-struct_data_energies[idx][self.wt_indices[0]] for idx in all_struct_indices]

            #struct_scores = [array(structure_likelihood(self.struct_types, idx, data_energies, data_struct_types, struct_data_probs)).max() for idx in all_struct_indices]
            #struct_scores = [struct_data_probs[idx,:].max() for idx in all_struct_indices]
            struct_scores = [struct_data_probs[idx,self.wt_indices[0]] for idx in all_struct_indices]
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
            struct_scores = [struct_data_probs[idx,:].max() for idx in all_struct_indices]
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
            lhood_opt, W_opt, W_std_opt, Psi_opt, E_d_opt, E_c_opt, sigma_d_opt, E_ddT_opt, perturbs_opt, post_structures_opt = self.analyze(max_iterations=2, tol=tol, \
                                                                         nsim=5000, select_struct_indices=maxmedoids.values(), G_constraint=G_constraint,\
                                                                         hard_em=hard_em, post_model=post_model)
            minAICc = utils.AICc(lhood_opt, self.data, W_opt, E_d_opt, E_c_opt, Psi_opt)
            def structure_similarity(s1, s2):
                simcounts = 0.
                for i in xrange(len(s1)):
                    if s1[i] == s2[i]:
                        simcounts += 1.
                return simcounts/len(s1)
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
                lhood, W, W_std, Psi, E_d, E_c, sigma_d, E_ddT, perturbs, post_structures = self.analyze(max_iterations=max_iterations, tol=tol, \
                                                                   nsim=5000, \
                                                                   select_struct_indices=selected_structs + [new_struct], G_constraint=G_constraint, hard_em=hard_em, post_model=post_model)
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
                    lhood, W, W_std, Psi, E_d, E_c, sigma_d, E_ddT, perturbs, post_structures = self.analyze(max_iterations=max_iterations, tol=tol, \
                                                                       nsim=5000, \
                                                                       select_struct_indices=reduced_structs, G_constraint=G_constraint, hard_em=hard_em, post_model=post_model)
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
            lhood_opt, W_opt, W_std_opt, Psi_opt, E_d_opt, E_c_opt, sigma_d_opt, E_ddT_opt, perturbs, post_structures = self.analyze(max_iterations=max_iteration, tol=tol, \
                                                                               nsim=est_nstructs*100, \
                                                                               select_struct_indices=starting_structures, G_constraint=G_constraint, hard_em=hard_em, post_model=post_model)
            minAICc = utils.AICc(lhood_opt, self.data, W_opt, E_d_opt, E_c_opt, Psi_opt)
            for subset in itertools.combinations(all_struct_indices, expected_structures):
                if subset != starting_structures:
                    lhood, W, W_std, Psi, E_d, E_d, sigma_d, E_ddT, perturbs, post_structures = self.analyze(max_iterations=max_iteration, tol=tol, \
                                                                                       nsim=est_nstructs*100, \
                                                                                       select_struct_indices=subset, G_constraint=G_constraint, hard_em=hard_em, post_model=post_model)
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

    def hard_EM_vars(self, idx, W, Psi_inv, data, struct_types, contact_sites, bp_dist, seq_indices):
        # This basically solves into a system of linear equations
        # Each entry is either a hidden reactivity or a hidden contact
        # to be solved
        print 'Solving MAP for E-step (hard EM) for sequence position %s' % idx
        ncontacts = 0
        # No Lapacian priors here, we use exponential (for hidden reactivities) and Gaussian (i.e. 2-norm) for contacts for easier calculations
        contact_prior_factor = 1/1.5
        prior_factors = {}
        prior_factors['u'] = 0.5
        prior_factors['p'] = 7.5
        nmeas = W.shape[0]
        nstructs = len(struct_types[0])
        contact_idx_dict = {}
        if self.use_motif_decomposition:
            nmotifs = 0
            motif_idx_dict = {}
            for midx in xrange(len(self.pos_motif_map)):
                if (seq_indices[idx], midx) in self.pos_motif_map:
                    motif_idx_dict[nmotifs] = midx
                    nmotifs += 1
            # Some helper functions to make the code more compact
            def motifidx(midx):
                return self.pos_motif_map[(seq_indices[idx], motif_idx_dict[midx])]

            def check_contact_site(midx, j):
                for m in motifidx(midx):
                    if contact_sites[m][j, idx]:
                        return True
                return False

            i = nmotifs
            for j in xrange(nmeas):
                for s in xrange(nmotifs):
                        if check_contact_site(s, j):
                            ncontacts += 1
                            contact_idx_dict[i] = (j, s)
                            contact_idx_dict[(j,s)] = i
                            i += 1

            dim = nmotifs + ncontacts
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
            if self.use_motif_decomposition:
                nstruct_elems = nmotifs
            else:
                nstruct_elems = nstructs
            # We'll start by indexing the reactivity hidden variables by structure
            for p in xrange(A.shape[0]):
                for s in xrange(nstruct_elems):
                    if p < nstruct_elems:
                        if self.use_motif_decomposition:
                            for j in xrange(nmeas):
                                A[p,s] += W[j,motifidx(p)].sum()*W[j,motifidx(s)].sum()
                            if p == s:
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
                                        A[p,s] += 4*self.lam_reacts*1/bp_dist[s1,s]
                            else:
                                A[p,s] -= 4*self.lam_reacts*1/bp_dist[p,s]

                    else:
                        j, s2 = contact_idx_dict[p]
                        if self.use_motif_decomposition:
                            A[p,s] = W[j,motifidx(s)].sum()*W[j,motifidx(s2)].sum()
                        else:
                            A[p,s] = W[j,s]*W[j,s2]
            for s in xrange(nstruct_elems):
                for m in motifidx(s):
                    if struct_types[idx][m] != struct_types[idx][motifidx(s)[0]]:
                        raise ValueError('MOTIF DECOMPOSITION FAILED! STRUCTURES IN POSITION %s HAVE DIFFERENT STATES!!! %s' % (idx))
                if self.use_motif_decomposition:
                    b[s] = -prior_factors[struct_types[idx][motifidx(s)[0]]]/Psi_inv[0,0]
                    for j in xrange(nmeas):
                        b[s] += W[j,motifidx(s)].sum()*data[j,idx]
                else:
                    b[s] = -prior_factors[struct_types[idx][s]]/Psi_inv[0,0] + dot(W[:,s], data[:,idx])
            # Then, the contact maps. No Lapacian priors here, we use Gaussian (i.e. 2-norm) priors
            # for easier calculations
            for p in xrange(A.shape[0]):
                for j in xrange(nmeas):
                    for s in xrange(nstruct_elems):
                        if self.use_motif_decomposition:
                            if check_contact_site(s, j):
                                if p < nstruct_elems:
                                    A[p, contact_idx_dict[(j,s)]] = W[j,motifidx(p)].sum()
                                else:
                                    j2, s2 = contact_idx_dict[p]
                                    if j == j2:
                                        #A[p,contact_idx_dict[(j,s)]] = W[j,s]
                                        if s == s2:
                                            A[p,contact_idx_dict[(j,s)]] = W[j,motifidx(s)].sum()**2 + (contact_prior)/Psi_inv[0,0]
                                        else:
                                            A[p,contact_idx_dict[(j,s)]] = W[j2,motifidx(s2)].sum()*W[j,motifidx(s)].sum()

                        else:
                            if contact_sites[s][j, idx]:
                                if p < nstruct_elems:
                                    A[p, contact_idx_dict[(j,s)]] = W[j,p]
                                else:
                                    j2, s2 = contact_idx_dict[p]
                                    if j == j2:
                                        if s == s2:
                                            A[p,contact_idx_dict[(j,s)]] = W[j,s]**2 + (contact_prior)/Psi_inv[0,0]
                                        else:
                                            A[p,contact_idx_dict[(j,s)]] = W[j2,s2]*W[j,s]
            for j in xrange(nmeas):
                for s in xrange(nstruct_elems):
                    if self.use_motif_decomposition:
                        if check_contact_site(s, j):
                            b[contact_idx_dict[(j,s)]] = data[j,idx]*W[j,motifidx(s)].sum()
                    else:
                        if contact_sites[s][j, idx]:
                            #b[contact_idx_dict[(j,s)]] = -contact_prior_factor/(Psi_inv[0,0]*W[j,s]) + data[j,idx]
                            #b[contact_idx_dict[(j,s)]] = -contact_prior_factor/(Psi_inv[0,0]) + data[j,idx]
                            b[contact_idx_dict[(j,s)]] = data[j,idx]*W[j,s]
            return A, b

        A, b = fill_matrices(A, b, contact_prior_factor)
        print 'Solving the linear equation system'
        # same variable names as in soft_EM_vars for consistency
        E_d__obs = zeros([nstructs])
        E_c__obs = zeros([nmeas, nstructs])
        def f(x):
            return ((dot(A,x) - b)**2).sum()
        def fprime(x):
            return dot(dot(A,x) - b, A.T)
        solved = False
        add_rand = False
        tries = 0
        while not solved:
            solved = True
            if self.use_motif_decomposition:
                bounds = [(0.001, data.max())]*nmotifs + [(-10,10)]*(A.shape[0] - nmotifs)
                x0 = [0.002 if struct_types[idx][motifidx(s)[0]] == 'p' else 1. for s in xrange(nmotifs)] + [0.0 for i in xrange(A.shape[0] - nmotifs)]
            else:
                bounds = [(0.001, data.max())]*nstructs + [(-10,10)]*(A.shape[0] - nstructs)
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
                solved=False
            if self.use_motif_decomposition:
                for s in xrange(nmotifs):
                    if x[s] <= 0.001:
                        E_d__obs[motifidx(s)] = 0.001
                    else:
                        E_d__obs[motifidx(s)] = x[s]
                    if add_rand:
                        E_d__obs[motifidx(s)] += rand(len(motifidx(s)))*0.0001 
            else:
                for s in xrange(nstructs):
                    if x[s] <= 0.001:
                        E_d__obs[s] = 0.001
                    else:
                        E_d__obs[s] = x[s]
                    if add_rand:
                        E_d__obs[s] += rand()*0.0001

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
                    #add_rand = True

                   
                print 'MAP system was not solved properly retrying with different contact priors'
                print 'Number of contacts is %s' % ncontacts
                print 'Changing prior factor to %s' % (new_prior_factor)
                A, b = fill_matrices(A, b, new_prior_factor)

        if self.use_motif_decomposition:
            for s in xrange(nmotifs):
                for j in xrange(nmeas):
                    for i in motifidx(s):
                        if contact_sites[i][j,idx]:
                            E_c__obs[j,i] = x[s]
                        else:
                            E_c__obs[j,i] = nan

            for s in xrange(nmotifs):
                for j in xrange(nmeas):
                    for i in motifidx(s):
                        if contact_sites[i][j,idx]:
                            E_c__obs[j,i] += x[contact_idx_dict[(j,s)]]
        else:
            for s in xrange(nstructs):
                for j in xrange(nmeas):
                    if contact_sites[s][j,idx]:
                        E_c__obs[j,s] = x[s]
                    else:
                        E_c__obs[j,s] = nan

            for s in xrange(nstructs):
                for j in xrange(nmeas):
                    if contact_sites[s][j,idx]:
                        E_c__obs[j,s] += x[contact_idx_dict[(j,s)]]

        sigma_d__obs = mat(zeros([nstructs]))
        if self.use_motif_decomposition:
            for s in xrange(nmotifs):
                sigma_d__obs[0,motifidx(s)] = sqrt(1/(Psi_inv[0,0]*(W[:,motifidx(s)]**2).sum()))
        else:
            for s in xrange(nstructs):
                sigma_d__obs[0,s] = sqrt(1/(Psi_inv[0,0]*(W[:,s]**2).sum()))


        return mat(E_d__obs), E_ddT__obs, sigma_d__obs, E_c__obs




    def soft_EM_vars(self, idx, W, Psi_inv, data, struct_types, contact_sites, n, burn, chosenindices, use_struct_clusters):
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
                M[j] = dot(asarray(L[j,:]), d_c.T)
            return M


        # We have to construct the d_calc_i variable manually...
        d_calc_i_parents = {'W':W, 'd_i':d_i}
        d_calc_i_parents.update(contact_stochastics)
        d_calc_i = Deterministic(eval=d_calc_i_eval,
                            name='d_calc_i',
                            parents=d_calc_i_parents,
                            doc='',
                            trace=True,
                            verbose=0,
                            plot=False)

        data_obs_i = MvNormal('data_obs_i', value=[data[chosenindices,idx]], mu=d_calc_i, tau=Psi_inv, observed=True)

        mc = MCMC([d_i, d_calc_i, data_obs_i] + contact_stochastics.values())
        mc.use_step_method(pymc.AdaptiveMetropolis, [d_i] + contact_stochastics.values())
        mc.sample(iter=n, burn=burn)
        d_samples = mat(d_i.trace())

        E_d__obs = d_samples[0,:]
        E_c__obs = zeros([nmeas, nstructs])
        # This is actually samples[0,:] * samples[0,:].T in 'real math' language
        E_ddT__obs = dot(d_samples[0,:].T, d_samples[0,:])
        sigma_d__obs = mat(zeros(d_samples[0,:].shape))
        for k in range(1, d_samples.shape[0]):
            E_d__obs += d_samples[k,:]
            E_ddT__obs += dot(d_samples[k,:].T, d_samples[k,:])
        E_d__obs, E_ddT__obs = E_d__obs*(1./n), E_ddT__obs*(1./n)
        for k in range(d_samples.shape[0]):
            sigma_d__obs += np.power(d_samples[k,:] - E_d__obs, 2)

        for j in xrange(nmeas):
            for s in xrange(nstructs):
                if contact_sites[s][j,idx]:
                    E_c__obs[j,s] = E_d__obs[:,s] + contact_stochastics['c_%s_%s' % (j, s)].trace().mean()
                else:
                    E_c__obs[j,s] = nan

        return E_d__obs, E_ddT__obs, sqrt(sigma_d__obs*(1./n)), E_c__obs


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
                self.sigma_pred[:,i] = sqrt(self.Psi[i,0,0])

        return self.data_pred, self.sigma_pred

    def calculate_fit_statistics(self, data_pred=None, sigma_pred=None):
        if data_pred == None or sigma_pred == None:
            data_pred, sigma_pred = self.calculate_data_pred()
        chi_sq = ((asarray(self.data) - asarray(data_pred))**2/asarray(sigma_pred)**2).sum()
        df = self.data.size - self.data.shape[1] - 1
        if self.use_motif_decomposition:
            df += -2*sum(self.nmotpos)
        else:
            df += -self.E_d.size - self.W.size
            df += - self.E_c[logical_not(isnan(self.E_c))].size
        k = -df - self.data.size + 1
        rmsea = sqrt(max((chi_sq/df - 1)/(self.data.shape[1] - 1), 0.0))
        aic = asscalar(chi_sq + 2*k - self.data.shape[0]*self.logpriors)

        return chi_sq/df, rmsea, aic

    def correct_scale(self, stype='linear'):
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

    def calculate_missed_predictions(self, sigthresh=0.05, data_pred=None, sigma_pred=None):
        missed_vals = zeros(self.data.shape)
        nmeas = self.data.shape[0]
        npos = self.data.shape[1]
        missed_indices = []
        if data_pred == None or sigma_pred == None:
            data_pred, sigma_pred = self.calculate_data_pred()
        for i in xrange(nmeas):
            for j in xrange(npos):
                if normpdf(self.data[i,j], data_pred[i,j], sigma_pred[i,j]) <= sigthresh:
                    # interpolate to our predicted data dynamic range
                    val = (self.data[i,j]/self.data.max())*data_pred.max() - data_pred[i,j]
                    missed_vals[i,j] = val
                    missed_indices.append((i,j))
        return missed_indices, missed_vals


    def _back_calculate_perturbations(self, W, energies, perturbs):
        for j in xrange(W.shape[0]):
            # Index por "pivot" energy, which will be the lowest energy
            # This is totally heuristic, we are assuming that the minimum energy
            # is the most "confident" and therefore needs no perturbation
            idx = where(energies[j,:] == energies[j,:].min())[0][0]
            perturbs[j,idx] = 0
            for s in xrange(W.shape[1]):
                if s != idx:
                    perturbs[j,s] = energies[j,idx] - energies[j,s] + utils.k*utils.T*log(abs((W[j,s] + 0.0001)/W[j,idx]))
        return perturbs


    def _assign_EM_structure_variables(self, select_struct_indices, use_struct_clusters):
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

    def _assign_W(self, data, E_d, E_c, E_ddT, E_ddT_inv, data_E_d, Psi, contact_sites, energies, concentrations, kds, G_constraint, nmeas, nstructs, use_struct_clusters, Wupper, Wlower, W_initvals):
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
            Psi_inv[i,:,:] = linalg.inv(Psi[i,:,:])
            Psi_inv_sum += linalg.inv(Psi[i,:,:])

        Psi_inv_sum = linalg.inv(Psi_inv_sum)

        for i in xrange(npos):
            Psi_data_E_d_T += dot(Psi_inv[i,:,:], data_E_d[:,:,i])

        if len(energies) > 0:
            print 'Solving weights'
            # ========= Using CVXOPT ====================
            frame = inspect.currentframe()
            args, _, _, argvals = inspect.getargvalues(frame)
            def par_fun(j, *args):
                E_d_proj = mat(zeros([nstructs,1]))
                E_ddT_Psi_inv = zeros(E_ddT[:,:,0].shape)
                E_d_c_j = self._E_d_c_j(E_d, E_c, j, contact_sites)
                E_ddT_Psi_inv += 0.01*eye(W.shape[1])
                for i in xrange(E_d_c_j.shape[1]):
                    #E_d_proj += data[j,i]*E_d[:,i]
                    E_d_proj += -Psi_inv[i,j,j]*data[j,i]*E_d_c_j[:,i]
                    E_ddT_Psi_inv += Psi_inv[i,j,j]*E_ddT[:,:,i]

                if self.lam_weights != 0:
                    E_ddT_Psi_inv += self.lam_weights*diag(1/(W_initvals[j,:]**2))
                    E_d_proj += self.lam_weights*mat(1./W_initvals[j,:]).T

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
                sol = array(solvers.qp(P, p, Gc, h, A, b, None, {'x':cvxmat(0.5, (nstructs, 1))})['x'])
                #sol = array(solvers.qp(P, p, Gc, h, A, b)['x'])
                
                sol[sol <= 0] = 1e-10
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
                        if W[m,p] > Wlower[m,p]:
                            W[m,p] = Wlower[m,p]
            # Sometimes there are small numerical errors that amplify a bit and they need to set straight
            # by forcing the sum of weights to be one -- maybe there is a better way to do the math
            # here to bypass these numerical instabilities?
            for j in xrange(nmeas):
                W[j,:] /= W[j,:].sum()
            if not use_struct_clusters:
                self.perturbs = self._back_calculate_perturbations(W, energies, self.perturbs)

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

    def _calculate_MLE_std(self, W, Psi_inv, E_ddT):
        I_W = zeros(W.shape)
        Psi_inv_vec = array([[Psi_inv[i,0,0] for i in xrange(Psi_inv.shape[0])]])
        for j in xrange(I_W.shape[0]):
            for s in xrange(I_W.shape[1]):
                I_W[j,s] = dot(Psi_inv_vec, E_ddT.T[:,:,s]).sum()
            """
            for s in xrange(I_W.shape[1]):
                for sp in xrange(I_W.shape[1]):
                    I_W[j,s] += dot(Psi_inv_vec, E_ddT[s,sp,:])
            """
        I_W[I_W == 0] = 1e-100
        return sqrt(1/I_W)

    def _E_d_c_j(self, E_d, E_c, j, contact_sites):
        E_d_c_j = mat(zeros(E_d.shape))
        for s in xrange(E_d.shape[0]):
            for i in xrange(E_d.shape[1]):
                if contact_sites[s][j,i]:
                    E_d_c_j[s,i] = E_c[j,s,i]
                else:
                    E_d_c_j[s,i] = E_d[s,i]
        return E_d_c_j

    def _dot_E_d_i(self, M, E_d_i, E_c_i, i, contact_sites, T=False):
        # i is the sequence position index
        nmeas = M.shape[0]
        nstructs = E_d_i.shape[0]
        if T:
            res = mat(zeros([nmeas, nstructs]))
            for j in xrange(nmeas):
                for s in xrange(nstructs):
                    if contact_sites[s][j,i]:
                        res[j,s] = asscalar(M[j]) * E_c_i[j,s]
                    else:
                        res[j,s] = asscalar(M[j]) * asscalar(E_d_i[s])
        else:
            res = mat(zeros([nmeas, 1]))
            for j in xrange(nmeas):
                d_c = zeros([nstructs])
                for s in xrange(nstructs):
                    if contact_sites[s][j,i]:
                        d_c[s] = E_c_i[j,s]
                    else:
                        d_c[s] = E_d_i[s]
                res[j] = dot(M[j,:], d_c)

        return res


    def _get_determinant(self, Psi):
        if linalg.det(Psi) == 0:
            return 1e-100
        else:
            return linalg.det(Psi)

    def analyze(self, max_iterations=100, tol=1e-4, nsim=1000, select_struct_indices=[], W0=None, Psi0=None, cluster_data_factor=None, G_constraint=None, use_struct_clusters=False, seq_indices=None, return_loglikes=False, hard_em=False, post_model=False):
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
        print 'Initializing weights W and covariance matrix Psi'
        if Psi0 != None:
            Psi = Psi0
        else:
            Psi = zeros([npos, nmeas, nmeas])
            min_var = 1e-100**(1./nmeas)
            for i in xrange(npos):
                indices = where(data[:,i] < mean(data[:,i]))[0]
                if len(indices) < 10:
                    indices = range(nmeas)
                Psi[i,:,:] = diag([max((data[indices,i].std())**2, min_var)]*nmeas)
        if W0 != None:
            W = W0
            self.perturbs = zeros([nmeas, nstructs])
        else:
            scale = sum([self._get_determinant(Psi[i,:,:])**(1./nmeas) for i in xrange(npos)])*(1./npos)
            # Starting point is the weights given by the RNAstructure energies plus a small gaussian-dist perturbation
            self.perturbs = normal(0, sqrt(scale/nstructs), size=(nmeas, nstructs))
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
        bp_dist = self.bp_dist[select_struct_indices,:][:,select_struct_indices]
        print 'Finished initializing'
        # For brevity E_d = E[d | data], E_d =[E[c | data], and E_ddT = E[d*dT | data], and so on...

        Psi_inv = zeros(Psi.shape)
        for i in xrange(npos):
            Psi_inv[i,:,:] = linalg.inv(Psi[i,:,:])
        E_d = mat(zeros([nstructs, npos]))
        E_c = zeros([nmeas, nstructs, npos])
        sigma_d= mat(zeros([nstructs, npos]))
        E_ddT_i = mat(zeros([nstructs, nstructs]))
        data_dataT = zeros([nmeas, nmeas, npos])
        E_ddT = zeros([nstructs, nstructs, npos])
        E_ddT_inv = zeros([nstructs, nstructs, npos])
        data_E_d = zeros([nmeas, nstructs, npos])

        old_loglike = -inf
        base_loglike = None
        Psi_opt = Psi
        W_opt = W
        E_d_opt = E_d
        sigma_d_opt = sigma_d
        E_ddT_opt = E_ddT
        max_loglike = old_loglike
        loglikes = []
        Psi_reinits = []
        adaptive_factor = 1

        t = 0
        for i in xrange(npos):
            data_dataT[:,:,i] = dot(data[:,i], data[:,i].T)
        while t < max_iterations:
            # E-step
            loglike = -npos*nmeas/2.*log(2.*pi)
            if hard_em:
                if self.njobs != None:
                    def par_fun(i):
                        return self.hard_EM_vars(i, W, Psi_inv[i,:,:], data, struct_types, contact_sites, bp_dist, seq_indices)
                    sys.modules[__name__].par_fun = par_fun
                    restuples = joblib.Parallel(n_jobs=self.njobs)(joblib.delayed(par_fun)(i) for i in xrange(npos))

                else:
                    restuples = []
                    for i in xrange(npos):
                        restuples.append(self.hard_EM_vars(i, W, Psi_inv[i,:,:], data, struct_types, contact_sites, bp_dist, seq_indices))

            else:
                if self.njobs != None:
                    def par_fun(i):
                        return self.soft_EM_vars(i, W, Psi_inv[i,:,:], data, struct_types, contact_sites, nsim, nsim/5, measindices, use_struct_clusters)
                    sys.modules[__name__].par_fun = par_fun
                    restuples = joblib.Parallel(n_jobs=self.njobs)(joblib.delayed(par_fun)(i) for i in xrange(npos))
                else:
                    restuples = []
                    for i in xrange(npos):
                        restuples.append(self.soft_EM_vars(i, W, Psi_inv[i,:,:], data, struct_types, contact_sites, nsim, nsim/5, measindices, use_struct_clusters))

            for i, tup in enumerate(restuples):
                E_d_i, E_ddT_i, sigma_d_i, E_c_i = tup
                E_d[:,i] = E_d_i.T
                sigma_d[:,i] = sigma_d_i.T
                E_c[:,:,i] = E_c_i
                E_ddT[:,:,i] = E_ddT_i

            #imshow(E_d, vmax=E_d.mean(), vmin=0, cmap=get_cmap('Greys'))


            if self.njobs != None:
                def par_fun(i):
                    B = self._dot_E_d_i(data[:,i], E_d[:,i], E_c[:,:,i], i, contact_sites, T=True)
                    C = -dot(dot(data[:,i].T, Psi_inv[i,:,:]), data[:,i])
                    C += 2.*self._dot_E_d_i(dot(dot(data[:,i].T, Psi_inv[i,:,:]), W), E_d[:,i], E_c[:,:,i], i, contact_sites)
                    C += -(dot(dot(dot(W.T, Psi_inv[i,:,:]), W), E_ddT[:,:,i])).diagonal().sum()
                    l = 0.5 * (C - npos*log(self._get_determinant(Psi[i,:,:])))
                    return B, l
                sys.modules[__name__].par_fun = par_fun
                restuples = joblib.Parallel(n_jobs=self.njobs)(joblib.delayed(par_fun)(i) for i in xrange(npos))
                for i, tup in enumerate(restuples):
                    B, l = tup
                    data_E_d[:,:,i] = B
                    loglike += l
            else:
                for i in xrange(npos):
                    data_E_d[:,:,i] = self._dot_E_d_i(data[:,i], E_d[:,i], E_c[:,:,i], i, contact_sites, T=True)
                    #data_E_d += dot(data[:,i], E_d[:,i].T)
                    # Don't forget to calculate the log-likelihood to track progress!
                    C = -dot(dot(data[:,i].T, Psi_inv[i,:,:]), data[:,i])
                    C += 2.*self._dot_E_d_i(dot(dot(data[:,i].T, Psi_inv[i,:,:]), W), E_d[:,i], E_c[:,:,i], i, contact_sites)
                    #C += -2.*dot(dot(dot(data[:,i].T, Psi_inv), W), E_d[:,i])
                    C += -(dot(dot(dot(W.T, Psi_inv[i,:,:]), W), E_ddT[:,:,i])).diagonal().sum()
                    loglike += 0.5 * (C - npos*log(self._get_determinant(Psi[i,:,:])))
            # M-step
            """
            # This is no longer necessary since we are not using the closed-form solution of W
            try:
                for i in xrange(npos):
                    E_ddT_inv[:,:,i] = linalg.inv(E_ddT[:,:,i])
            except LinAlgError:
                print 'An error occured on inverting E[d*dT | data]'
                print 'Position %s' % i
                print 'E[d*dT | data]'
                print E_ddT[:,:,i]
                print 'E[d*dT | data][:,i]'
                print E_d[:,i]
                raise ValueError('E[d*dT | data] matrix is singular!')
            """

            # Given our expected reactivities, now assign the weights
            W_new = self._assign_W(data, E_d, E_c, E_ddT, E_ddT_inv, data_E_d, Psi, contact_sites, energies, concentrations, kds, G_constraint, nmeas, nstructs, use_struct_clusters, Wupper, Wlower, W_initvals)

            #W_new = W + adaptive_factor*(W_new - W)
            W_new[W_new < 0] = 1e-10
            for j in xrange(nmeas):
                W_new[j,:] /= W_new[j,:].sum()
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
                Pd = array([P/nmeas]*nmeas)
                Pd[Pd == 0] = min_var
                Pd[isnan(Pd)] = min_var
                Psi[i,:,:] = diag(Pd)
                # Sometimes we have a weird case where the estimated Psi is negative, mostly when the noise is very very low
                if sum(Psi[i,:,:] < 0) > 0:
                    print 'Psi is out of bounds (%s values were below zero!), setting to minimum variance' % sum(Psi[i,:,:] < 0)
                    Psi[i,:,:][Psi[i,:,:] < 0] = min_var
                    if t not in Psi_reinits:
                        Psi_reinits.append(t)
                try:
                    Psi_inv[i,:,:] = linalg.inv(Psi[i,:,:])
                except LinAlgError:
                    pdb.set_trace()
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
                        logpriors += -self.lam_reacts*1/self.bp_dist[sidx,sidx2]*(asarray(E_d[sidx,:] - E_d[sidx2,:])**2).sum()
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
            """
            loglike += logpriors
            # Check if we are done
            print 'Finished iteration %s with log-likelihood %s' % (t, loglike)
            if not didpostmodel:
                print t
                t += 1
            loglikes.append(asscalar(loglike))
            if base_loglike is None and max_iterations != 1:
                base_loglike = loglike
            else:
                if True and loglike > max_loglike or max_iterations == 1:
                    max_loglike = loglike
                    self.logpriors = logpriors
                    W_opt = W.copy()
                    E_d_opt = E_d.copy()
                    E_c_opt = E_c.copy()
                    sigma_d_opt = sigma_d.copy()
                    Psi_opt = Psi.copy()
                    E_ddT_opt = E_ddT.copy()
                    adaptive_factor *= 2

                if abs(loglike - old_loglike) < tol or t+1 == max_iterations:
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
        if return_loglikes:
            lhood = [loglikes, Psi_reinits]
        else:
            lhood = max_loglike
        self.W, self.Psi, self.E_d, self.E_c, self.sigma_d, self.E_ddT= asarray(W_opt), asarray(Psi_opt), asarray(E_d_opt), asarray(E_c_opt), asarray(sigma_d_opt), asarray(E_ddT_opt)


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
                alldata_E_d += dot(self.data[:,i], E_d[:,i].T)

            for i, c in enumerate(bestclusts):
                cidx = measindices.index(chosenindices[c])
                if len(self.energies) > 0:
                    perturbs[i,:] = self.perturbs[cidx,:]
                Psi[i,i] = self.Psi[cidx, cidx]

            Psi_inv = linalg.inv(Psi)
            self.perturbs = perturbs
            self.W = asarray(self._assign_W(data, E_d, E_ddT, E_ddT_inv, alldata_E_d, Psi, contact_sites, allenergies, allconcentrations, allkds, G_constraint, allmeas, nstructs, use_struct_clusters, Wupper, Wlower, W_initvals))
            self.W_std = asarray(self._calculate_MLE_std(self.W, Psi_inv, E_ddT))
            self.Psi = asarray(Psi)
            self.perturbs = asarray(perturbs)
            self.data_clusters = bestclusts
            self.data_cluster_medoids = chosenindices

        if len(self.energies) > 0 or use_struct_clusters:
            self.perturbs = zeros(self.W.shape)
            return lhood, self.W, self.W_std, self.Psi, self.E_d, self.E_c, self.sigma_d, self.E_ddT, self.perturbs, post_structures
        return lhood, self.W, self.W_std, self.Psi, self.E_d, self.E_c, self.sigma_d, self.E_ddT, post_structures

class FullBayesAnalysis(MappingAnalysisMethod):

    def __init__(self, data, structures, backend='pickle', dbname='ba_traces.pickle', mutpos=[], concentrations=[], kds=[], energies=[], c_size=3, cutoff=0, njobs=None):
        MappingAnalysisMethod.__init__(self, data, structures, energies, mutpos, c_size, cutoff)
        self.concentrations = concentrations
        self.kds = kds
        self.backend=backend
        self.dbname=dbname
        self.njobs = njobs

    def _get_FAobj(self):
        if 'fa' not in self.__dict__:
            self.fa = FAMappingAnalysis(self.data, self.structures, \
                                        mutpos=self.mutpos, concentrations=self.concentrations, \
                                        kds=self.kds, energies=self.energies, c_size=self.c_size)
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
                D[i].value = D_opt[:,i]
            for j in xrange(len(M)):
                M[j].value = M_opt[j,:]
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
                if self.contact_sites[j,i] == 1:
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
                m0[j,s] = rand()
        for i in xrange(npos):
            for j in xrange(nmeas):
                if self.contact_sites[j,i]:
                    c0[j,i] = SHAPE_contacts_diff_sample()
            while True:
                d0[:,i] = rand(len(self.struct_types[i]))
                logr = log([rand() for x in xrange(len(self.struct_types[i]))]).sum()
                if d_prior_loglike(d0[:,i], i) >= logr:
                    break

        def d_calc_eval(Weights, d):
            return dot(Weights, d)

        def Weights_eval(free_energies_perturbed=[]):
            Z = sum(exp(free_energies_perturbed/(utils.k*utils.T)))
            return exp(free_energies_perturbed/(utils.k*utils.T)) / Z

        @stochastic
        def Psi_diag(value=psidiag0):
            return psidiag_loglike(value)

        @deterministic
        def Psi_mat(d=Psi_diag):
            return diag(d)

        D = [stochastic(lambda value=d0[:,i]: d_prior_loglike(value, i), name='d_%s' % i) for i in xrange(npos)]
        C = [stochastic(lambda value=c0[:,i]: c_prior_loglike(value, i), name='c_%s' % i) for i in xrange(npos)]
        #M = [MvNormal('m_%s' % j, mu=zeros(nstructs) + 5, tau=eye(nstructs)) for j in xrange(nmeas)]
        M = [stochastic(lambda value=m0[j,:]: m_prior_loglike(value), name='m_%s' % j) for j in xrange(nmeas)]
        free_energies_perturbed = [deterministic(lambda x=self.energies[j,:], y=M[j]: x + y, name='free_energies_preturbed_%s' % j) for j in xrange(nmeas)]
        Weights = [Deterministic(eval=Weights_eval, name='Weights_%s' % j, parents={'free_energies_perturbed':free_energies_perturbed[j]}, doc='weights', trace=True, verbose=1, dtype=ndarray, plot=False, cache_depth=2) for j in xrange(nmeas)]
        d_calc = [Deterministic(eval=lambda **kwargs: d_calc_eval(kwargs['Weights'], kwargs['d']), name='d_calc_%s' % i, parents={'Weights':Weights, 'd':D[i]}, doc='d_calc', trace=True, verbose=1, dtype=ndarray, plot=False, cache_depth=2) for i in xrange(npos)]
        data_obs = [MvNormal('data_obs_%s' % i, value=[self.data[:,i]], mu=d_calc[i], tau=Psi_mat, observed=True) for i in xrange(npos)]
        print 'Finished declaring'
        mc = MCMC(self._initialize_mcmc_variables(initmethod, data_obs, D, M, Psi_diag, C, inititer) + [d_calc, Weights, free_energies_perturbed], \
                   db=self.backend, dbname=self.dbname)
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
                self.Weights_trace[:,j,s] = w.trace()[:,s]
        for i, d in enumerate(d_calc):
            for j in xrange(nmeas):
                self.d_calc_trace[:,j,i] = d.trace()[:,j].T
        for j, f in enumerate(free_energies_perturbed):
            for s in xrange(nstructs):
                self.free_energies_perturbed_trace[:,j,s] = f.trace()[:,s]
        for j, m in enumerate(M):
            for s in xrange(nstructs):
                self.M_trace[:,j,s] = m.trace()[:,s]
        for i, d in enumerate(D):
            for s in xrange(nstructs):
                self.D_trace[:,s,i] = d.trace()[:,s]
        print 'Done recovering traces'
        if print_stats:
            print 'Finished, statistics are:'
            print mc.stats()
            pymc.Matplot.plot(mc, path='.')
        mc.db.close()
        return self.Psi_trace, self.d_calc_trace, self.free_energies_perturbed_trace, self.Weights_trace, self.M_trace, self.D_trace
