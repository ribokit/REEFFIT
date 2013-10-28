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
rdat.load(open('../rdat/MDLOOP_SHP_0002.rdat'))
construct = rdat.constructs.values()[0]
offset = construct.offset
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
nmuts = 36
for d in construct.data:
    label = d.annotations['mutation'][0]
    if label == 'WT':
        mutant = sequence
        pos = -1
    else:
        pos = int(label[1:len(label)-1]) - 1 - offset
        mutant = sequence[:pos] + label[-1] + sequence[pos+1:]
    mutpos.append(pos)
    mut_labels.append(label)
    mutants.append(mutant)
    nd = normalize([d.values[seqpos.index(i)] for i in sorted_seqpos])[cutoff:-cutoff]
    data.append(nd)

data = array(data)
mut_labels = mut_labels[:nmuts]
mutpos = mutpos[:nmuts]
mutants = mutants[:nmuts]

# The structures that we are considering
structures = ['..........((((((((((...............))))))))))...................................',\
'...................................(((((((((.......)))))))))....................']
nstructs = len(structures)
npos = data.shape[1]

print 'Considering the following structures:'
structures = [s[cutoff:-cutoff-20] for s in structures]

for s in structures:
    print s
"""
print 'Generating some medloop mock data with considered structures'
data_mock  = utils.mock_data(mutants, structures)
figure(1)
clf()
imshow(data_mock, cmap=get_cmap('Greys'), vmin=0, vmax=data_mock.mean(), aspect='auto', interpolation='nearest')
xticks(range(len(sequence)), ['%s%s' % (i, x) for i, x in enumerate(sequence)], fontsize='x-small', rotation=90)
yticks(range(len(mut_labels)), mut_labels, fontsize='x-small')
savefig('medloop_full_model_analysis/mockdata.png', dpi=300)

print 'Simulating mutate and map using full partition calculation'
data_sim = simulate_mm_partition(mutants)
imshow(data_sim, cmap=get_cmap('Greys'), vmin=0, vmax=data_sim.mean(), aspect='auto', interpolation='nearest')
xticks(range(len(sequence)), ['%s%s' % (i, x) for i, x in enumerate(sequence)], fontsize='x-small', rotation=90)
yticks(range(len(mut_labels)), mut_labels, fontsize='x-small')
savefig('medloop_full_model_analysis/simdata.png', dpi=300)
"""

print 'Calculating structure energies by mutant'
energies = utils.get_free_energy_matrix(structures, mutants)
print 'Energies are'
print energies
print 'Done, defining struct types'
struct_types = []
for i in xrange(len(sequence)):
    struct_types.append(['u' if structures[j][i] == '.' else 'p' for j in xrange(len(structures))])

print 'Beginning analysis'
sets = [0,1,2]
# Just for testing, remove later
sets = [sets[0]]
for k, r in enumerate(sets):
    print 'Creating directory'
    if not os.path.exists('medloop_full_model_analysis/%suM' % r):
	os.mkdir('medloop_full_model_analysis/%suM' %r)
    data_r = array(data[nmuts*k:nmuts*k+nmuts])
    Psi_t, Sigma_t, d_calc_t, free_energies_perturbed_t, Weights_t, M_t, D_t = mcmcfull.simulate(data_r, struct_types, structures, energies, mutpos, n=5000, burn=2500, initmethod='fa', inititer=2)

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
    savefig('medloop_full_model_analysis/%suM/realdata.png' % r, dpi=300)
    clf()
    imshow(d_calc, cmap=get_cmap('Greys'), vmin=0, vmax=d_calc.mean(), aspect='auto', interpolation='nearest')
    xticks(range(len(sequence)), ['%s%s' % (i, x) for i, x in enumerate(sequence)], fontsize='x-small', rotation=90)
    yticks(range(len(mut_labels)), mut_labels, fontsize='x-small')
    savefig('medloop_full_model_analysis/%suM/predicteddata.png' % r, dpi=300)
    savetxt(open('d_calc.txt', 'w'), d_calc)

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
	savefig('medloop_full_model_analysis/%suM/exp_react_struct_%s.png' % (r, i), dpi=300)

    f = figure(3)
    f.set_size_inches(15, 5)
    clf()
    for i in xrange(len(structures)):
	errorbar(range(nmuts), Weights[:,i], yerr=Weights_t.std(axis=0)[:,i], linewidth=2, label='structure %s' % i)
    legend()
    xticks(xrange(len(mut_labels)), mut_labels, fontsize='x-small', rotation=90)
    savefig('medloop_full_model_analysis/%suM/weights_by_mutant.png' % r, dpi=300)
