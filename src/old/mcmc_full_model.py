from pymc import MAP, Model, MvNormal, stochastic, Deterministic, deterministic, MCMC, Wishart, AdaptiveMetropolis, distributions
import mcmc_factor_analysis as factor_analysis
import pymc.graph
from matplotlib.pylab import *
import pymc.Matplot
import rdatkit.secondary_structure
from reactivity_distributions import *
from map_analysis_utils import *
import pdb


k = 0.0019872041
T = 310.15 # temperature

def _initialize_mcmc_variables(method, data, data_obs, D, M, energies, Psi, C, contact_sites, structures, inititer, mutpos):
    if not method:
        return [data_obs, D, M, Psi, C]
    if method == 'map':
        mp = MAP([data_obs, D, M])
        mp.fit(iterlim=inititer, verbose=True)
        Psi.value = diag(diag(cov(data)))
        return list(mp.variables) + [Psi, C]
    if method == 'fa':
        W, Psi.value, D_opt, E_ddT_opt, M_opt = factor_analysis.analyze(data, structures, mutpos, energies=energies, nsim=100, max_iterations=inititer)
        data_pred = dot(W, D_opt)
        Psi.value = linalg.inv(asarray(Psi.value))
        for i in xrange(len(D)):
            D[i].value = D_opt[:,i]
        for j in xrange(len(M)):
            M[j].value = M_opt[j,:]
        return [data_obs, D, M, Psi]
        

def simulate(data, struct_types, structs, energies, mutpos, n=1000, burn=100, c_size=3, inititer=5, initmethod=None):
    # This is the full model
    nmeas = data.shape[0]
    npos = data.shape[1]
    nstructs = len(structs)
    # Get contact sites
    contact_sites = get_contact_sites(structs, mutpos, nmeas, npos, c_size)
    # Likelihood value for M (the energy perturbations)
    def m_prior_loglike(value):
        loglike = 0
        for s in xrange(len(value)):
            loglike += distributions.uniform_like(value[s], -200, 200)
        return loglike

    # Likelihood value for D (the hidden reactivities)
    def d_prior_loglike(value, i):
        loglike = 0
	for s in xrange(len(value)):
	    if value[s] < 0:
		return -inf
	    if struct_types[i][s] == 'u':
		loglike += log(SHAPE_unpaired_pdf(value[s]))
	    else:
		loglike += log(SHAPE_paired_pdf(value[s]))
	return loglike
    # Diagonal and off-diagonal contacts and mutation effects
    def c_prior_loglike(value, i):
        loglike = 0
        for j in xrange(len(value)):
            if contact_sites[j,i] == 1:
                loglike += log(SHAPE_contacts_diff_pdf(value[j]))
            else:
                if value[j] != 0:
                    return -inf
        return loglike

    d0 = zeros([nstructs, npos])
    c0 = zeros([nmeas, npos])
    m0 = zeros([nmeas, nstructs])
    print 'Choosing good d_0s, c_0s, and m_0s' 
    for j in xrange(nmeas):
        for s in xrange(nstructs):
            m0[j,s] = rand()
    for i in xrange(npos):
        for j in xrange(nmeas):
            if contact_sites[j,i]:
                c0[j,i] = SHAPE_contacts_diff_sample()
        while True: 
            d0[:,i] = rand(len(struct_types[i]))
            logr = log([rand() for x in xrange(len(struct_types[i]))]).sum()
            if d_prior_loglike(d0[:,i], i) >= logr:
                break

    def d_calc_eval(Weights, d):
	return dot(Weights, d) 


    def Weights_eval(free_energies_perturbed=[]):
        Z = sum(exp(free_energies_perturbed/(k*T)))
	return exp(free_energies_perturbed/(k*T)) / Z

    Psi = Wishart('Psi', n=nmeas, Tau=eye(nmeas))
    Sigma = Wishart('Sigma', n=nstructs, Tau=eye(nstructs))
    D = [stochastic(lambda value=d0[:,i]: d_prior_loglike(value, i), name='d_%s' % i) for i in xrange(npos)]
    C = [stochastic(lambda value=c0[:,i]: c_prior_loglike(value, i), name='c_%s' % i) for i in xrange(npos)]
    #M = [MvNormal('m_%s' % j, mu=zeros(nstructs) + 5, tau=eye(nstructs)) for j in xrange(nmeas)]
    M = [stochastic(lambda value=m0[j,:]: m_prior_loglike(value), name='m_%s' % j) for j in xrange(nmeas)]
    free_energies_perturbed = [deterministic(lambda x=energies[j,:], y=M[j]: x + y, name='free_energies_preturbed_%s' % j) for j in xrange(nmeas)]
    Weights = [Deterministic(eval=Weights_eval, name='Weights_%s' % j, parents={'free_energies_perturbed':free_energies_perturbed[j]}, doc='weights', trace=True, verbose=1, dtype=ndarray, plot=False, cache_depth=2) for j in xrange(nmeas)]
    d_calc = [Deterministic(eval=lambda **kwargs: d_calc_eval(kwargs['Weights'], kwargs['d']), name='d_calc_%s' % i, parents={'Weights':Weights, 'd':D[i]}, doc='d_calc', trace=True, verbose=1, dtype=ndarray, plot=False, cache_depth=2) for i in xrange(npos)]
    data_obs = [MvNormal('data_obs_%s' % i, value=[data[:,i]], mu=d_calc[i], tau=Psi, observed=True) for i in xrange(npos)]
    print 'Finished declaring' 
    mc = MCMC([Psi, Sigma, data_obs, d_calc, Weights, D, M, C, free_energies_perturbed])
    mc = MCMC(_initialize_mcmc_variables(initmethod, data, data_obs, D, M, energies, Psi, C, contact_sites, structs, inititer, mutpos) + [Sigma, d_calc, Weights, free_energies_perturbed], db='pickle', dbname='full_model_traces.pickle') 
    mc.use_step_method(AdaptiveMetropolis, M)
    mc.use_step_method(AdaptiveMetropolis, D)
    mc.use_step_method(AdaptiveMetropolis, Psi)
    mc.use_step_method(AdaptiveMetropolis, Sigma)
    #mc.use_step_method(AdaptiveMetropolis, C)
    print 'Starting to sample'
    mc.sample(iter=n, burn=burn, thin=4, verbose=2)
    print '\nDone sampling, recovering traces'
    nsamples = len(D[0].trace())
    Weights_trace = zeros([nsamples, nmeas, nstructs])
    d_calc_trace = zeros([nsamples, nmeas, npos])
    D_trace = zeros([nsamples, nstructs, npos])
    free_energies_perturbed_trace = zeros([nsamples, nmeas, nstructs])
    M_trace = zeros([nsamples, nmeas, nstructs])
    for j, w in enumerate(Weights):
	for s in xrange(nstructs):
	    Weights_trace[:,j,s] = w.trace()[:,s]
    for i, d in enumerate(d_calc):
	for j in xrange(nmeas):
	    d_calc_trace[:,j,i] = d.trace()[:,j].T
    for j, f in enumerate(free_energies_perturbed):
	for s in xrange(nstructs):
	    free_energies_perturbed_trace[:,j,s] = f.trace()[:,s]
    for j, m in enumerate(M):
	for s in xrange(nstructs):
	    M_trace[:,j,s] = m.trace()[:,s]
    for i, d in enumerate(D):
        for s in xrange(nstructs):
            D_trace[:,s,i] = d.trace()[:,s]
    print 'Done recovering traces'
    #print 'Finished, statistics are:'
    #print mc.stats()
    #pymc.Matplot.plot(mc, path='full_model_plots/')
    mc.db.close()
    return Psi.trace(), Sigma.trace(), d_calc_trace, free_energies_perturbed_trace, Weights_trace, M_trace, D_trace


