import map_analysis_utils as utils
import mcmc_full_model as mcmcfull
import scipy.optimize
import pickle
import os
import pdb
from rdatkit.datahandlers import RDATFile
from rdatkit import secondary_structure as ss
from matplotlib.pylab import *
from numpy import linalg
from scipy import stats

def simulate_mm_partition(mutants):
    res = []
    for mutant in mutants:
	bppm = ss.partition(mutant)
        res.append((1 - bppm.sum(axis=0)).tolist())
    return array(res)

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
cutoff = 1

seqpos = construct.seqpos
sorted_seqpos = sorted(seqpos)
sequence_original = construct.sequence[min(seqpos) - construct.offset - 1:max(seqpos) - construct.offset].upper()
sequence = sequence_original[cutoff:-cutoff]


print 'Parsing mutants and data'
data = []

mut_labels = []
mutpos = []
mutants = []
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
mutpos = mutpos[:48]
mutants = mutants[:48]

# The structures that we are considering
# Note that I'm marking the aptamer region as a's, since they will be considered "paired"
structures = ['..........................(((((((............)))))))....................',\
#'.....((((aaaaaa((((....))))aaaaa))))....................................',\
'.....((((......((((....)))).....))))....................................',\
'.....((((......((((....)))).....)))).(((((....))))).....................']
nstructs = len(structures)
npos = data.shape[1]
nmeas = 48 #we've got 48 mutants

print 'Considering the following structures:'
structures = [s[cutoff:-cutoff-20] for s in structures]

for s in structures:
    print s

"""
print 'Generating some tebowned mock data with considered structures'
data_mock  = utils.mock_data(mutants, structures)
figure(1)
clf()
imshow(data_mock, cmap=get_cmap('Greys'), vmin=0, vmax=data_mock.mean(), aspect='auto', interpolation='nearest')
xticks(range(len(sequence)), ['%s%s' % (i, x) for i, x in enumerate(sequence)], fontsize='x-small', rotation=90)
yticks(range(len(mut_labels)), mut_labels, fontsize='x-small')
savefig('full_model_analysis/mockdata.png', dpi=300)

print 'Simulating mutate and map using full partition calculation'
data_sim = simulate_mm_partition(mutants)
imshow(data_sim, cmap=get_cmap('Greys'), vmin=0, vmax=data_sim.mean(), aspect='auto', interpolation='nearest')
xticks(range(len(sequence)), ['%s%s' % (i, x) for i, x in enumerate(sequence)], fontsize='x-small', rotation=90)
yticks(range(len(mut_labels)), mut_labels, fontsize='x-small')
savefig('full_model_analysis/simdata.png', dpi=300)
exit()
"""
"""
# Kds of each mutant for each structure (if it is affected by ligand, else we put nan's), calculated using Rhiju's Lifft
kds = zeros([nmeas, nstructs])
kds[:,1] = [8.888777e+00,1.353322e+01,1.681395e+01,3.801457e-01,8.044252e+01,0,0,2.550548e+02,1.291630e+02,1.695319e+02,4.153400e+02,1.224333e+01,0,0,0,3.575409e+02,1.518050e+01,1.615539e+01,1.662590e+01,6.107143e+01,8.680208e+00,2.003769e+01,1.446766e+01,8.181826e+01,1.268425e+02,0,0,3.515185e+02,1.856386e+02,1.143099e+02,2.209287e+02,0,0,8.455294e+01,1.427258e+02,0,1.577406e+01,1.312597e+01,1.220498e+01,1.052557e+01,0,9.400513e+00,0,2.390639e+01,6.783398e+01,1.472217e+00,1.689676e+00,5.159439e+00]
kds[:,0] = [nan]*nmeas
kds[:,2] = [nan]*nmeas
kds[:,3] = [nan]*nmeas
"""
print 'Calculating structure energies by mutant'
energies = utils.get_free_energy_matrix(structures, mutants)
print 'Done, defining struct types'
struct_types = []
for i in xrange(len(sequence)):
    struct_types.append(['u' if structures[j][i] == '.' else 'p' for j in xrange(len(structures))])

print 'Beginning analysis'
# We have sets = [0,0.5,1,5,10,20,50,100,200,500,1000,2000] for uM concentrations of FMN
sets = [0,0.5,1,5,10,20,50,100,200,500,1000,2000]
# Just for testing, remove later
sets = [sets[0]]
for k, r in enumerate(sets):
    print 'Creating directory'
    if not os.path.exists('full_model_analysis/%suM' % r):
	os.mkdir('full_model_analysis/%suM' %r)
    data_r = array(data[48*k:48*k+48])
    concentrations = [r]*nmeas
    Psi_t, Sigma_t, d_calc_t, free_energies_perturbed_t, Weights_t, M_t, D_t = mcmcfull.simulate(data_r, struct_types, structures, energies, mutpos, n=5000, burn=100, initmethod='fa', inititer=5)

    Psi = Psi_t.mean(axis=0)
    Sigma = Sigma_t.mean(axis=0)
    d_calc = d_calc_t.mean(axis=0)
    D = D_t.mean(axis=0)
    M = M_t.mean(axis=0)
    free_energies_perturbed = free_energies_perturbed_t.mean(axis=0)
    Weights = Weights_t.mean(axis=0)

    print 'Plotting data'
    figure(1)
    clf()
    imshow(data_r, cmap=get_cmap('Greys'), vmin=0, vmax=data_r.mean(), aspect='auto', interpolation='nearest')
    xticks(range(len(sequence)), ['%s%s' % (i, x) for i, x in enumerate(sequence)], fontsize='x-small', rotation=90)
    yticks(range(len(mut_labels)), mut_labels, fontsize='x-small')
    savefig('full_model_analysis/%suM/realdata.png' % r, dpi=300)
    clf()
    imshow(d_calc, cmap=get_cmap('Greys'), vmin=0, vmax=d_calc.mean(), aspect='auto', interpolation='nearest')
    xticks(range(len(sequence)), ['%s%s' % (i, x) for i, x in enumerate(sequence)], fontsize='x-small', rotation=90)
    yticks(range(len(mut_labels)), mut_labels, fontsize='x-small')
    savefig('full_model_analysis/%suM/predicteddata.png' % r, dpi=300)
    clf()
    imshow(d_calc, cmap=get_cmap('Greys'), aspect='auto', interpolation='nearest')
    xticks(range(len(sequence)), ['%s%s' % (i, x) for i, x in enumerate(sequence)], fontsize='x-small', rotation=90)
    yticks(range(len(mut_labels)), mut_labels, fontsize='x-small')
    savefig('full_model_analysis/%suM/predicteddata_RAW.png' % r, dpi=300)


    for i, s in enumerate(structures):
	binarized_react = [1 if x == '.' else 0 for x in s]
	f = figure(2)
	f.set_size_inches(15, 5)
	clf()
	title('Structure %s: %s' % (i, s))
	exp_react = array(D[i,:])
	plot(1+arange(len(s)), binarized_react, linewidth=2, c='r')
	bar(0.5+arange(len(s)), exp_react, linewidth=0, width=1, color='gray')
	errorbar(1+arange(len(s)), exp_react, yerr=D_t.std(axis=0)[i,:], fmt='black')
	xticks(1 + arange(len(s)), ['%s%s' % (j+1, x) for j, x in enumerate(sequence[cutoff:-cutoff])], fontsize='x-small', rotation=90)
	xlim(1,len(s))
	ylim(0,2)
	savefig('full_model_analysis/%suM/exp_react_struct_%s.png' % (r, i), dpi=300)

    f = figure(3)
    f.set_size_inches(15, 5)
    clf()
    for i in xrange(len(structures)):
	#plot(W[:,i], label='structure %s' % i, linewidth=2)
	errorbar(range(nmeas), Weights[:,i], yerr=Weights_t.std(axis=0)[:,i], linewidth=2, label='structure %s' % i)
    legend()
    xticks(xrange(len(mut_labels)), mut_labels, fontsize='x-small', rotation=90)
    savefig('full_model_analysis/%suM/weights_by_mutant.png' % r, dpi=300)

print 'Finished'
