import map_analysis_utils as utils
import mcmc_factor_analysis as mcmcfa
import scipy.optimize
import pickle
import os
import pdb
from rdatkit.datahandlers import RDATFile
from rdatkit import secondary_structure as ss
from matplotlib.pylab import *
from numpy import linalg
from scipy import stats

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


print 'Parsing RDAT'
rdat = RDATFile()
rdat.load(open('../rdat/TEBOWNED3D.rdat'))
construct = rdat.constructs.values()[0]

seqpos = construct.seqpos
sorted_seqpos = sorted(seqpos)
sequence = construct.sequence[min(seqpos) - construct.offset - 1:max(seqpos) - construct.offset].upper()


print 'Parsing mutants and data'
data = []

mut_labels = []
mutants = []
mutpos = []
cutoff = 1
for d in construct.data:
    label = d.annotations['mutation'][0]
    if label == 'WT':
        mutant = sequence
        pos = -1
    else:
        pos = int(label[1:len(label)-1])
        mutant = sequence[:pos-2] + label[-1] + sequence[pos-1:]
    mutpos.append(pos)
    mut_labels.append(label)
    mutants.append(mutant)
    nd = normalize([d.values[seqpos.index(i)] for i in sorted_seqpos])[cutoff:-cutoff]
    data.append(nd)

data = array(data)
mut_labels = mut_labels[:48]
mutants = mutants[:48]


# The structures that we are considering
# Note that I'm marking the aptamer region as a's, since they will be considered "paired"
structures = ['..........................(((((((............)))))))....................',\
'.....((((aaaaaa((((....))))aaaaa))))....................................',\
'.....((((......((((....)))).....))))....................................',\
'.....((((......((((....)))).....)))).(((((....))))).....................']
nstructs = len(structures)
npos = data.shape[1]
nmeas = 48 #we've got 48 mutants

# Kds of each mutant for each structure (if it is affected by ligand, else we put nan's), calculated using Rhiju's Lifft
kds = zeros([nmeas, nstructs])
kds[:,1] = [8.888777e+00,1.353322e+01,1.681395e+01,3.801457e-01,8.044252e+01,0,0,2.550548e+02,1.291630e+02,1.695319e+02,4.153400e+02,1.224333e+01,0,0,0,3.575409e+02,1.518050e+01,1.615539e+01,1.662590e+01,6.107143e+01,8.680208e+00,2.003769e+01,1.446766e+01,8.181826e+01,1.268425e+02,0,0,3.515185e+02,1.856386e+02,1.143099e+02,2.209287e+02,0,0,8.455294e+01,1.427258e+02,0,1.577406e+01,1.312597e+01,1.220498e+01,1.052557e+01,0,9.400513e+00,0,2.390639e+01,6.783398e+01,1.472217e+00,1.689676e+00,5.159439e+00]
kds[:,0] = [nan]*nmeas
kds[:,2] = [nan]*nmeas
kds[:,3] = [nan]*nmeas

print 'Considering the following structures:'
structures = [s[cutoff:-cutoff-20] for s in structures]

for s in structures:
    print s


# We have sets = [0,0.5,1,5,10,20,50,100,200,500,1000,2000] for uM concentrations of FMN
sets = [0,0.5,1,5,10,20,50,100,200,500,1000,2000]
E_d_all = zeros([nstructs, npos, len(sets)])
E_ddT_all = zeros([nstructs, nstructs, len(sets)])
W_all = zeros([nmeas*len(sets), nstructs])

energies = utils.get_free_energy_matrix(structures, mutants)
for k, r in enumerate(sets):
    print 'Creating directory'
    if not os.path.exists('%suM' % r):
	os.mkdir('%suM' %r)
    data_r = array(data[48*k:48*k+48])

    nboot = 1
    
    concentrations = [r]*nmeas

    Wboot = zeros([nmeas, nstructs, nboot])
    Psiboot = zeros([nmeas, nmeas, nboot])
    E_dboot = zeros([nstructs, npos, nboot])
    E_ddTboot = zeros([nstructs, nstructs, nboot])
    print 'Starting bootstrap...'
    for b in xrange(nboot):
	print 'Calling factor analysis module'
	W, Psi, E_d, E_ddT, perturbs = mcmcfa.analyze(data_r, structures, max_iterations=10, concentrations=concentrations, kds=kds, energies=energies, nsim=100)
        """
        # We find the scale factors that force
	# the weights to sum up to one and normalize the expected reactivities
        # at the same time
	scale_factors = linalg.lstsq(W[:,:nstructs], ones(nmeas))[0]
	for s in xrange(nstructs):
	    scale_factors[s] = norm(E_d[s,:],ord=2)/4

	for s in xrange(nstructs):
	    W[:,s] *= scale_factors[s]
	    E_d[s,:] /= scale_factors[s]
	    E_ddT[s,:] /= scale_factors[s]
        """
	data_pred = array(dot(W, E_d))
	Wboot[:,:,b] = W
	E_dboot[:,:,b] = E_d
	E_ddTboot[:,:,b] = E_ddT
	Psiboot[:,:,b] = Psi

	if b == 0:
	    print 'Plotting data'
	    figure(1)
	    clf()
	    imshow(data_r, cmap=get_cmap('Greys'), vmin=0, vmax=data_r.mean(), aspect='auto', interpolation='nearest')
	    xticks(range(len(sequence)), ['%s%s' % (i, x) for i, x in enumerate(sequence)], fontsize='x-small', rotation=90)
	    yticks(range(len(mut_labels)), mut_labels, fontsize='x-small')
	    savefig('%suM/realdata.png' % r, dpi=300)
	    clf()
	    imshow(data_pred, cmap=get_cmap('Greys'), vmin=0, vmax=data_pred.mean(), aspect='auto', interpolation='nearest')
	    xticks(range(len(sequence)), ['%s%s' % (i, x) for i, x in enumerate(sequence)], fontsize='x-small', rotation=90)
	    yticks(range(len(mut_labels)), mut_labels, fontsize='x-small')
	    savefig('%suM/predicteddata.png' % r, dpi=300)

    E_d = E_dboot.mean(axis=2)
    W = Wboot.mean(axis=2)
    Psi = Psiboot.mean(axis=2)
    E_ddT = E_ddTboot.mean(axis=2)
    
    if r == 0:
	W_all = W
    else:
	W_all = append(W_all, W, axis=0)
    
    E_d_all[:,:,k] = E_d
    E_ddT_all[:,:,k] = E_ddT

    pickle.dump(W, open('%suM/W.pickle' % r, 'w'))
    pickle.dump(E_d, open('%suM/E_d.pickle' % r, 'w'))
    pickle.dump(E_ddT, open('%suM/E_ddT.pickle' % r, 'w'))

    E_dstd = E_dboot.std(axis=2)
    Wstd = Wboot.std(axis=2)
    Psistd = Psiboot.std(axis=2)

    for i, s in enumerate(structures):
	binarized_react = [1 if x == '.' else 0 for x in s]
	f = figure(2)
	f.set_size_inches(15, 5)
	clf()
	title('Structure %s: %s' % (i, s))
	exp_react = array(E_d[i,:])
	plot(1+arange(len(s)), binarized_react, linewidth=2, c='r')
	bar(0.5+arange(len(s)), exp_react, linewidth=0, width=1, color='gray')
	errorbar(1+arange(len(s)), exp_react, yerr=E_dstd[i,:], fmt='black')
	xticks(1 + arange(len(s)), ['%s%s' % (j+1, x) for j, x in enumerate(sequence[cutoff:-cutoff])], fontsize='x-small', rotation=90)
	xlim(1,len(s))
	ylim(0,2)
	savefig('%suM/exp_react_struct_%s.png' % (r, i), dpi=300)

    f = figure(3)
    f.set_size_inches(15, 5)
    clf()
    for i in xrange(len(structures)):
	#plot(W[:,i], label='structure %s' % i, linewidth=2)
	errorbar(range(nmeas), W[:,i], yerr=Wstd[:,i], linewidth=2, label='structure %s' % i)
    legend()
    xticks(xrange(len(mut_labels)), mut_labels, fontsize='x-small', rotation=90)
    savefig('%suM/weights_by_mutant.png' % r, dpi=300)

print 'Analysis finished, plotting and saving final results'
pickle.dump(W_all, open('W_all.pickle', 'w'))
pickle.dump(E_d_all, open('E_d_all.pickle', 'w'))
pickle.dump(E_ddT_all, open('E_ddT_all.pickle', 'w'))
pickle.dump([sets, mut_labels, nmeas], open('factor_analysis_variables.pickle','w'))

E_d_all_mean = E_d_all.mean(axis=2)
E_d_all_std = E_d_all.std(axis=2)

for i, s in enumerate(structures):
    binarized_react = [1 if x == '.' else 0 for x in s]
    figure(5)
    f.set_size_inches(15, 5)
    clf()
    title('Structure %s: %s' % (i, s))
    exp_react = E_d_all_mean[i,:]
    plot(1+arange(len(s)), binarized_react, linewidth=2, c='r')
    bar(0.5+arange(len(s)), exp_react, linewidth=0, width=1, color='gray')
    errorbar(1+arange(len(s)), exp_react, yerr=E_d_all_std[i,:], fmt='black')
    xticks(1 + arange(len(s)), ['%s%s' % (j+1, x) for j, x in enumerate(sequence[cutoff:-cutoff])], fontsize='x-small', rotation=90)
    xlim(1,len(s))
    ylim(0,1.5)
    savefig('exp_react_struct_%s.png' %  i, dpi=300)


