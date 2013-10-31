import pdb
from matplotlib.pylab import *
from reactivity_distributions import *
from scipy.stats import stats
from scipy.spatial.distance import squareform
from collections import defaultdict
import rdatkit.secondary_structure
import scipy.cluster.hierarchy as hcluster

k = 0.0019872041
T = 310.15 # temperature

# Akaike information criterion (corrected form) for model selection
def _AICc(lhood, nparams, ndata):
    return 2*nparams - 2*lhood + (2*nparams*(nparams + 1))/(ndata - nparams -1)

def AICc(lhood, data, W, E_d, E_c, Psi):
    return _AICc(lhood, W.size + E_d.size + Psi.shape[0] + E_c[logical_not(isnan(E_c))].size, data.size)

# Calculate mutual information between types
def _mutinf(typ1, typ2):
    #TODO: Right now (just using paired and unpaired distributions) 
    # we just need to check if the two types are the same or not
    # in the future, when we have confident distributions between all structure subtypes
    # (bulges, internal pairs, etc.) we will replace this with the true mutual information
    # between types -- given that these structure subtypes are fixed, we could even cache them!
    res = 0
    for i in xrange(len(typ1)):
        if typ1[i] != typ2[i]:
            res += 1
    return res

def _bpdist(sbp1, sbp2):
    res = 0
    for bp1_1, bp1_2 in sbp1:
        for bp2_1, bp2_2 in sbp2:
            if (bp1_1 == bp2_1 and bp1_2 == bp2_2) or (bp1_2 == bp2_1 and bp1_1 == bp2_2):
                res += 1
    return res
    

def get_struct_types(structures, cutoff=-1):
    struct_types = []
    for i in xrange(len(structures[0])):
        # Just mark paried as p and unpaired as u, later we may refine the model
        struct_types.append(['u' if s[i] == '.' else 'p' for s in structures])
    return struct_types


def cluster_structures(struct_types, structures=[], distance='mutinf', expected_medoids=-1):
    nstructs = len(struct_types[0])
    print 'Clustering structures'
    # Ok, find a subset of structures that are mostly "orthogonal", clustering
    # with mutual information (mutinf option) or base pair distance (basepair option)
    # Looks ugly, but seems that numpy's symmetric matrix support is limited (i.e. none)
    if distance == 'basepair':
        structures_bp = [rdatkit.secondary_structure.SecondaryStructure(dbn=struct).base_pairs() for struct in structures]
    D = zeros([nstructs, nstructs])
    for i in xrange(nstructs):
        st1 = [s[i] for s in struct_types]
        D[i,i] = 0
        for j in xrange(i+1, nstructs):
            if distance == 'mutinf':
                D[i,j] = _mutinf(st1, [s[j] for s in struct_types])  
            elif distance == 'basepair':
                D[i,j] = _bpdist(structures_bp[i], structures_bp[j])
            else:
                raise ValueError('Distance %s not recognized: options are "mutinf" and "basepair"' % distance)
            D[j,i] = D[i,j]
    # Proxy for ensemble centroid (EC)
    minsumdist = inf
    for i in xrange(nstructs):
        sumdist = D[i,:].sum()
        if sumdist < minsumdist:
            minsumdist = sumdist
            EC = i
    Z = hcluster.linkage(squareform(D), method='complete')
    prevnumclusts = nstructs
    maxCH =-inf
    assignments = [(ns, ns) for ns in xrange(nstructs)]
    for t in arange(0, D.max(), D.max()/10.):
        flatclusters = hcluster.fcluster(Z,t, criterion='distance')
        clusters = defaultdict(list)
        for elem, cidx in enumerate(flatclusters):
            clusters[cidx].append(elem)
        numclusts = len(clusters)
        if numclusts == prevnumclusts:
            continue
        else:
            prevnumclusts = numclusts
        if numclusts == 1:
            break
        # Choose cluster medoids
        medoids = {}
        mindist = 0
        for cidx, elems in clusters.iteritems():
            for j, elem in enumerate(elems):
                sumdist = D[elem,elems].sum()/2.
                if j == 0 or sumdist < mindist:
                    mindist = sumdist
                    medoids[cidx] = elem
        if expected_medoids > 0:
            # If we already estimated the expected structures,
            # then just retrieve the cluster assignments and best medoids
            maxmedoids = medoids
            assignments = clusters
            break
        # Calculate Calinki-Harabasz index to choose clusters
        # Using between-cluster distances (BC) and within-cluster (WC) distances
        BC = sum([len(elems)*D[medoids[cidx],EC] for cidx, elems in clusters.iteritems() if len(elems) > 1])
        WC = sum([D[medoids[cidx],elems].sum()/2. for cidx, elems in clusters.iteritems() if len(elems) > 1])
        if BC == 0 or WC == 0:
            CH = -inf
        else:
            CH = (BC/(numclusts - 1))/(WC/(D.shape[0] - numclusts))
        if CH > maxCH:
            maxCH = CH
            maxmedoids = medoids
            assignments = clusters
        print 'For %s threshold we have %s clusters with %s CH' % (t, numclusts, CH)
    print 'Done clustering structures, with %s clusters' % numclusts
    return maxmedoids, assignments

# To perform a standard factor analysis, with normal priors
# We use this mainly to have a first estimate of the number 
# structures needed before performing the real factor analysis
def standard_factor_analysis(data, nstructs):
    fnode = mdp.nodes.FANode(output_dim=nstructs)
    fnode.train(asarray(data.T))
    weights = fnode.execute(data.T)
    factors = fnode.A.T
    data_pred = dot(weights, factors).T
    return fnode.lhood[-1]*fnode.tlen, data_pred, weights, factors

def factor_index(data, nstructs):
    aics = [inf]*(nstructs+1)
    for ns in range(2, nstructs+1):
        lhood, data_pred, weights, factors = standard_factor_analysis(data, ns)
        aics[ns] = _AICc(lhood, 1.5*size(factors), size(data))
        print 'For %s structures, AICc is %s' % (ns, aics[ns])
    return aics.index(min(aics))


def normalize(bonuses):
    l = len(bonuses)
    wtdata = array(bonuses)
    if wtdata.min() < 0:
	wtdata -= wtdata.min()
    interquart = stats.scoreatpercentile(wtdata, 75) - stats.scoreatpercentile(wtdata, 25)
    tenperc = stats.scoreatpercentile(wtdata, 90)
    maxcount = 0
    maxav = 0.
    for i in range(l):
	if wtdata[i] >= tenperc:
	    maxav += wtdata[i]
	    maxcount += 1
    maxav /= maxcount
    wtdata = wtdata/maxav
    return wtdata

# Energies is the energies used to calculate the weights,
# clusters is a dictionary with lists of structure indices
def calculate_weights(energies, clusters=None):
    if clusters != None:
        W = zeros([energies.shape[0], len(clusters)])
        struct_weights_by_clust = {}
        part_func = zeros([energies.shape[0]])
        for c, struct_indices in clusters.iteritems():
            struct_weights_by_clust[c] = zeros([energies.shape[0], len(struct_indices)])
            for i, j in enumerate(struct_indices):
                struct_weights_by_clust[c][:,i] = exp(-energies[:,j]/(k*T))
                part_func[:] += struct_weights_by_clust[c][:,i]
        for i, c in enumerate(struct_weights_by_clust):
            for j in xrange(energies.shape[0]):
                struct_weights_by_clust[c][j,:] /= part_func[j]
            W[:,i] = struct_weights_by_clust[c].sum(axis=1)
        return W, struct_weights_by_clust
    else:
        W = zeros(energies.shape)
        for j in xrange(energies.shape[0]):
            W[j,:] = exp(-energies[j,:]/(k*T)) / exp(-energies[j,:]/(k*T)).sum()
        return W

def mock_data(sequences, structures=None, energy_mu=0.5, energy_sigma=0.5, obs_sigma=0.01,
        paired_sampling=lambda : SHAPE_paired_sample(), unpaired_sampling= lambda : SHAPE_unpaired_sample(),\
        contact_sampling=lambda : SHAPE_contacts_diff_sample(), mutpos=None, c_size=3, return_steps=False, correlate_regions=False):
    if structures != None:
        print 'Generating mock data'
        print 'Getting "true" free energies (from RNAstructure)'
        true_energies = get_free_energy_matrix(structures, sequences)

        print 'Energies'
        print true_energies

        # Add some noise to the energies
        noise = energy_sigma*randn(true_energies.shape[0], true_energies.shape[1]) + energy_mu
        #noise = rand(true_energies.shape[0], true_energies.shape[1])
        noised_energies = true_energies + noise
        weights_noised =  calculate_weights(noised_energies)
        weights =  calculate_weights(true_energies)
        print 'Weights'
        print weights
        print 'Noised Weights'
        print weights_noised

        print 'Generating mock reactivities'
        # Mock reactivities
        reacts = zeros([len(structures), len(sequences[0])])
        prev_s = ''
        MIN_REACT = 1e-5
        P_CONTACT = 1
        if correlate_regions:
            for j, s in enumerate(structures):
                for i in xrange(len(sequences[0])):
                    if s[i] != prev_s:
                        if s[i] == '.':
                            curr_val = unpaired_sampling()
                        else:
                            curr_val = max(MIN_REACT, paired_sampling())
                            curr_val = paired_sampling()
                    reacts[j,i] = max(MIN_REACT, curr_val + 0.01*SHAPE_contacts_diff_sample())
                    #reacts[j,i] = curr_val
                    prev_s = s[i]
        else:
            prevstate = '.'
            for j, s in enumerate(structures):
                for i in xrange(len(sequences[0])):
                    if s[i] == '.':
                        reacts[j,i] = unpaired_sampling()
                        prevstate = '.'
                    else:
                        if prevstate == '.' or (i < len(sequences[0]) - 1 and s[i+1] == '.'):
                            reacts[j,i] = paired_sampling()*1.5
                        else:
                            reacts[j,i] = paired_sampling()*0.3
                        prevstate = '('

        data = dot(weights_noised, reacts)
        data_orig = dot(weights_noised, reacts)
        if mutpos:
            """
            for i, pos in enumerate(mutpos):
                if i >=0:
                    for k in xrange(-c_size/2, c_size/2+1):
                        if pos+k < data.shape[1] and rand() > P_CONTACT:
                            data[i,pos+k] = contact_sampling()
            """
            print 'Simulate diagonal and off-diagonal contact sites'
            max_tries = 1000
            def add_local_perturb(reactivity, weight):
                dd = weight*contact_sampling()
                tries = 0 
                while reactivity + dd < MIN_REACT or reactivity + dd > 4.5:
                    dd = weight*contact_sampling()*0.1
                    tries += 1
                    if tries > max_tries:
                        if reactivity + dd > 4.5:
                            return 4.5
                        if reactivity + dd < MIN_REACT:
                            return MIN_REACT - reactivity
                return dd

            bp_dicts = []
            for s, struct in enumerate(structures):
                bp_dicts.append(rdatkit.secondary_structure.SecondaryStructure(dbn=struct).base_pair_dict())
            for j in xrange(data.shape[0]):
                for k in xrange(-(c_size-1)/2, (c_size-1)/2+1):
                    for s in xrange(len(structures)):
                        #if weights_noised[j,s] < 0.1:
                        #    continue
                        if mutpos[j] + k < data.shape[1] and rand() < P_CONTACT:
                            if structures[s][mutpos[j]+k] == '.':
                                dd = 0.1*add_local_perturb(data_orig[j,mutpos[j]+k], weights_noised[j,s])
                            else:
                                dd = 0.3*add_local_perturb(data_orig[j,mutpos[j]+k], weights_noised[j,s])
                            if k != 0:
                                dd *= 0.2
                            data[j,mutpos[j]+k] += dd
                        if mutpos[j] in bp_dicts[s] and bp_dicts[s][mutpos[j]] + k < data.shape[1] and rand() < P_CONTACT:
                            data[j,bp_dicts[s][mutpos[j]] + k] += dd

            
        print 'Adding observational noise'
        data_noised = zeros(data.shape)
        obs_noise_sigmas = []
        params = (0.10524313598815455, 0.034741986764665007)
        for i in xrange(data.shape[1]):
            sigma = rand()*0.2 + obs_sigma
            sigma = max(stats.distributions.cauchy(loc=params[0], scale=params[1]).rvs(), 0.001)*0.2
            data_noised[:,i] = data[:,i] + randn(data.shape[0])*sigma
            obs_noise_sigmas.append(sigma)
        data_noised = data + randn(data.shape[0], data.shape[1])*obs_sigma
        if return_steps:
            return dot(weights, reacts), data, data_noised, true_energies, weights_noised, reacts, obs_noise_sigmas
        else:
            return data_noised
    else:
        data = zeros(len(sequences), len(sequences[0]))
        for j, seq in enumerate(sequences):
            bppm = rdatkit.secondary_structure.partition(sequence)
            unpaired_probs = 1 - bppm.sum(axis=0)
            for i, up in enumerate(unpaired_probs):
                data[j,i] = up*unpaired_sampling() + (1-up)*paired_sampling()
        data_noised = data + obs_sigma*randn(data.shape[0], data.shape[1])
        return data_noised

def get_free_energy_matrix(structures, sequences, algorithm='rnastructure'):
    energies = zeros([len(sequences), len(structures)])
    for j, seq in enumerate(sequences):
        print 'Calculating structure energies for sequence %s: %s' % (j, seq)
        energies[j,:] = array(rdatkit.secondary_structure.get_structure_energies(seq, [rdatkit.secondary_structure.SecondaryStructure(dbn=s) for s in structures], algorithm=algorithm))
	minenergy = energies[j,:].min()
	#energies[j,:] -= minenergy
	for i in xrange(len(structures)):
	    if energies[j,i] > 200:
		energies[j,i] = 200
    return energies

def get_contact_sites(structures, mutpos, nmeas, npos, c_size, restrict_range=None):
    bp_dicts = []
    nstructs = len(structures)
    if restrict_range:
        mutpos_cutoff = [m + restrict_range[0] if m > 0 else m for m in mutpos]
    else:
        mutpos_cutoff = mutpos
    for s, struct in enumerate(structures):
        bp_dicts.append(rdatkit.secondary_structure.SecondaryStructure(dbn=struct).base_pair_dict())
    contact_sites = {}
    for s in xrange(nstructs):
        contact_sites[s] = zeros([nmeas, npos])
    nstructs = len(structures)
    for j in xrange(nmeas):
        if mutpos_cutoff[j] > 0:
            for s in xrange(nstructs):
                for k in xrange(-(c_size-1)/2, (c_size-1)/2+1):
                    if mutpos_cutoff[j] + k >= 0 and mutpos[j] + k < npos:
                        contact_sites[s][j,mutpos_cutoff[j]+k] = 1
                    if mutpos_cutoff[j] in bp_dicts[s] and bp_dicts[s][mutpos_cutoff[j]] + k < npos and bp_dicts[s][mutpos_cutoff[j]] + k >= 0:
                        contact_sites[s][j,bp_dicts[s][mutpos_cutoff[j]]+k] = 1
    if restrict_range != None:
        for s in xrange(nstructs):
            contact_sites[s] = contact_sites[s][:,restrict_range[0]:restrict_range[1]]
    return contact_sites

