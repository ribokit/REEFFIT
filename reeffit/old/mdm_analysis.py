import rdatkit.secondary_structure
import pdb
from matplotlib.pylab import *
import mdp
from scipy.optimize  import fmin_bfgs
from scipy.optimize  import fmin_l_bfgs_b


# Note: lo = linear objective; fd refers to the "d-step", fa refers to the "\alpha-step"; prime is the derivative
# l2 and l1 are the norms used


def conc_weight(conc, kd):
    return conc/kd

"""
Linear way of calculating the reactivities (d) given alphas as weights and at
certain concentrations
"""
def linear_dcalc_conc(i, j, c, d, alphas, partition, num_pos, num_mut, num_struct, concentrations, kds):
    dcalc_ijc = conc_weight(conc, kds[j]) * d[i,0]
    for s in xrange(num_struct):
	dcalc_ijc += alphas[j,s] * d[i,s]/partition
    return dcalc_ijc

"""
Same as above but without concentrations
"""
def linear_dcalc(i, j, d, alphas, partition, num_pos, num_mut, num_struct):
    dcalc_ij = 0
    for s in xrange(num_struct):
	dcalc_ij += alphas[j,s] * d[i,s]/partition
    return dcalc_ij


"""
Linear objective, d(i,s) are the reactivities and alphas(j,s) are the weights indexed by:
i = position
j = mutant
s = structures

IN ORDER: pos=1,...,n; THEN mut=1,...,m; THEN struct=1,...,s
When available, we can scpecify a set of concentrations and Kd's (one per mutant), in which case the FIRST strucature will have a weight according to the concentration and Kd specified.
"""


def linear_obj(d_pre, data, alphas_pre, num_pos, num_mut, num_struct, concentrations=[], kds=[], func=lambda x:x**2):
    d = reshape(d_pre, [num_pos, num_struct])
    alphas = reshape(alphas_pre, [num_mut, num_struct])
    res = 0
    if len(concentrations) > 0:
	for j in xrange(num_mut):
	    partition = alphas[j,:].sum()
	    for i in xrange(num_pos):
		for c, conc in enumerate(concentrations):
		    partition_conc = partition + conc_weight(conc, kds[j])
		    dcalc_ijc = linear_dcalc_conc(i, j, c, d, alphas, partition_conc, num_pos, num_mut, num_struct, concentrations, kds)
	            res += func(dcalc_ijc - data[i,j,c])
    else:
	for j in xrange(num_mut):
	    partition = alphas[j,:].sum()
	    for i in xrange(num_pos):
		dcalc_ij = linear_dcalc(i, j, d, alphas, partition, num_pos, num_mut, num_struct)
		res += func(dcalc_ij - data[i,j])
    return res

def linear_obj_grad_dstep(d_pre, data, alphas_pre, num_pos, num_mut, num_struct, concentrations=[], kds=[], func=lambda x, y:2*(x - y)):
    d = reshape(d_pre, [num_pos, num_struct])
    alphas = reshape(alphas_pre, [num_mut, num_struct])
    partitions = zeros(num_mut)
    res = zeros([num_pos, num_struct])
    for j in xrange(num_mut):
	partitions[j] = alphas[j, :].sum()
    if len(concentrations) > 0:
	for i in xrange(num_pos):
	    for s in xrange(num_struct):
		res[i,s] = 0
		for j in xrange(num_mut):
		    for c, conc in enumerate(concentrations):
			partition_conc = partitions[j] + conc_weight(conc, kds[j])
			dcalc_ijc = linear_dcalc_conc(i, j, c, d, alphas, partition_conc, num_pos, num_mut, num_struct, concentrations, kds)
			res[i,s] += func(dcalc_ijc, data[i,j,c])
			if s == 0:
			    res[i,s] *= ((alphas[j,s]/partition_conc) + conc_weight(conc, kds[j])/partition_conc)
			else:
			    res[i,s] *= (alphas[j,s]/partition_conc)
    else:
	for i in xrange(num_pos):
	    for s in xrange(num_struct):
		res[i,s] = 0
		for j in xrange(num_mut):
		    dcalc_ij = linear_dcalc(i, j, d, alphas, partitions[j], num_pos, num_mut, num_struct)
		    res[i,s] += func(dcalc_ij, data[i,j])
		    res[i,s] *= (alphas[j,s]/partitions[j])
    return reshape(res, num_pos*num_struct)

	    
def linear_obj_grad_astep(d_pre, data, alphas_pre, num_pos, num_mut, num_struct, concentrations=[], kds=[], func=lambda x, y:2*(x - y)):
    d = reshape(d_pre, [num_pos, num_struct])
    alphas = reshape(alphas_pre, [num_mut, num_struct])
    partitions = zeros(num_mut)
    d_sums = zeros(num_pos)
    res = zeros([num_mut, num_struct])
    for j in xrange(num_mut):
	partitions[j] = alphas[j, :].sum()
    for i in xrange(num_pos):
	d_sums[i] = d[i, :].sum()
    if len(concentrations) > 0:
	for j in xrange(num_mut):
	    for s in xrange(num_struct):
		res[j,s] = 0
		for i in xrange(num_pos):
		    for c, conc in enumerate(concentrations):
			dcalc_ijc = linear_dcalc_conc(i, j, c, d, alphas, partitions[j], num_pos, num_mut, num_struct, concentrations, kds)
			res[j,s] += func(dcalc_ijc, data[i,j,c])
			res[i,s] *= (d[i,s] - d_sums[i])/partitions[j]
    else:
	for j in xrange(num_mut):
	    for s in xrange(num_struct):
		res[j,s] = 0
		for i in xrange(num_pos):
		    dcalc_ij = linear_dcalc(i, j, d, alphas, partitions[j], num_pos, num_mut, num_struct)
		    res[j,s] += func(dcalc_ij, data[i,j])
		    res[j,s] *= (d[i,s] - d_sums[i])/partitions[j]
    return reshape(res, num_mut*num_struct)

def solve_by_linalg(alphas_pre, data, num_pos, num_mut, num_struct):
    alphas = reshape(alphas_pre, [num_mut, num_struct])
    d = zeros([num_pos, num_struct])
    for i in xrange(num_pos):
	res = linalg.lstsq(alphas, data[i, :])
	d[i,:] = res[0]
    return reshape(d, num_pos*num_struct)

def linear_reactivity_model(sequences, structures, data_pre, objtype='l2', concentrations=[], kds=[], max_iter=5, tol=0.001, cutoff=0):
    k = 0.0019872041
    T = 310.15 # temperature
    num_pos = len(sequences[0]) - 2*cutoff
    num_mut = len(sequences)
    num_struct = len(structures)
    alphas_current = []
    if cutoff > 0:
	data = data_pre[cutoff:-cutoff,:]
    else:
	data = data_pre
    d_current = randn(num_pos, num_struct)
    if objtype == 'l2':
	func_for_obj = lambda x: x**2
	func_for_grad = lambda x,y: 2*(x - y)
    else:
	func_for_obj = lambda x: abs(x)
	func_for_grad = lambda x,y: 1 if (x > y) else -1
    for i, seq in enumerate(sequences):
	energies = rdatkit.secondary_structure.get_structure_energies(seq, structures)
        # make relative to minimum structure
	minenergy = min(energies)
	energies = [e - minenergy for e in energies]
	if i == 0:
	    print 'Energies for first sequence'
	    print energies
	alphas_current.append([exp(E/(k*T)) for E in energies])
    alphas_current = array(alphas_current)
    print 'Weights for first sequence'
    print alphas_current[0, :]
    
    i = 1
    curr_val = float('inf')
    past_val = 0
    alpha_bounds = [(0,None) for q in range(num_mut*num_struct)]
    while (i <= max_iter) and (abs(past_val - curr_val) > tol): 
	print 'Iteration %s:' % i
	print 'Performing d-step'
	lo_dstep = lambda d_pre: linear_obj(d_pre, data, alphas_current, num_pos, num_mut, num_struct, concentrations=concentrations, kds=kds, func=func_for_obj)
	lo_dstep_grad = lambda d_pre: linear_obj_grad_dstep(d_pre, data, alphas_current, num_pos, num_mut, num_struct, concentrations=concentrations, kds=kds, func=func_for_grad)
	#d_current = fmin_bfgs(lo_dstep, d_current, fprime=lo_dstep_grad)
	d_current = solve_by_linalg(alphas_current, data, num_pos, num_mut, num_struct)
	print 'Performing a-step'
	lo_astep = lambda alphas_pre: linear_obj(d_current, data, alphas_pre, num_pos, num_mut, num_struct, concentrations=concentrations, kds=kds, func=func_for_obj)
	lo_astep_grad = lambda alphas_pre: linear_obj_grad_astep(d_current, data, alphas_pre, num_pos, num_mut, num_struct, concentrations=concentrations, kds=kds, func=func_for_grad)
	results = fmin_l_bfgs_b(lo_astep, alphas_current, fprime=lo_astep_grad, bounds=alpha_bounds)
	alphas_current = results[0]
	past_val = curr_val
	curr_val = lo_astep(alphas_current)
	i += 1
    d_current = reshape(d_current, [num_pos, num_struct])
    alphas_current = reshape(alphas_current, [num_mut, num_struct])
    print 'Finished'
    print 'Number of iterations %s' % i
    print 'Final objective function value: %s' % curr_val
    print 'Structure\tWeight'
    for i, s in enumerate(structures):
	print '%s\t%s' % (s.dbn, alphas_current[0,i])
    data_pred = zeros(data.shape)
    if len(concentrations) > 0:
	pass
    else:
	for i in range(num_pos):
	    for j in range(num_mut):
		partition = alphas_current[j,:].sum()
		data_pred[i,j] = linear_dcalc(i, j, d_current, alphas_current, partition, num_pos, num_mut, num_struct)

    return data_pred, d_current, alphas_current 

def factor_analysis(data_pre, nstructs, cutoff=1):
    if cutoff > 0:
	data = data_pre[cutoff:-cutoff,:]
    else:
	data = data_pre
    fnode = mdp.nodes.FANode(output_dim=nstructs)
    fnode.train(data)
    weights = fnode.execute(data)
    factors = fnode.A.T
    data_pred = dot(weights, factors)
    return data_pred, weights, factors
