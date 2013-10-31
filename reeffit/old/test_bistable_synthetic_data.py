import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *
import mapping_analysis
from map_analysis_utils import *
from rdatkit.datahandlers import *
from collections import defaultdict

def plot_mutxpos_image(d):
    imshow(d, cmap=get_cmap('Greys'), vmin=0, vmax=d.mean(), aspect='auto', interpolation='nearest')
    #imshow(d, cmap=cmap, aspect='auto', interpolation='nearest')
    xticks(range(len(sequence)), ['%s%s' % (i+1, x) for i, x in enumerate(sequence)], fontsize='x-small', rotation=90)
    yticks(range(len(mut_labels)), mut_labels, fontsize='x-small')


# Let's do a quick test, with a bistable simulated RNA!:

sequence = 'GGGGAAAAAAGGGGGUUUUUGGGGGAAAAAA'
structures = ['.....(((((.....)))))...........', '...............(((((.....))))).']

complement = {'A':'U', 'C':'G', 'U':'A', 'G':'C'}
mutants = [sequence] + [sequence[:i] + complement[sequence[i]] + sequence[i+1:] for i in xrange(len(sequence))]
struct_types = []
for i in xrange(len(sequence)):
    struct_types.append(['u' if structures[j][i] == '.' else 'p' for j in xrange(len(structures))])

mutpos = [-1] + range(len(sequence))
mut_labels = ['%s%s%s' % (sequence[pos], pos+1, complement[sequence[pos]]) if pos >= 0 else 'WT' for pos in mutpos]
true_data, perturbed_data, data_noised, true_energies, weights_noised = mock_data(mutants, structures, energy_mu=10, energy_sigma=5, mutpos=mutpos, return_steps=True)

figure(1)
clf()
plot_mutxpos_image(true_data)
savefig('bistable_rna_test/full_model_test_simulated_no_noise.png', dpi=300)
plot_mutxpos_image(perturbed_data)
savefig('bistable_rna_test/full_model_test_simulated_noised_weights.png', dpi=300)
plot_mutxpos_image(data_noised)
savefig('bistable_rna_test/full_model_test_simulated_fully_noise.png', dpi=300)

print 'Saving to RDAT'

rdat = RDATFile()
rdat.version = 0.32
construct = 'Synthetic bistable RNA'
rdat.constructs[construct] = RDATSection()
rdat.constructs[construct].name = construct
rdat.constructs[construct].data = []
rdat.constructs[construct].seqpos = range(len(sequence))
rdat.constructs[construct].sequence = sequence
rdat.constructs[construct].sequences = defaultdict(int)
rdat.constructs[construct].structures = defaultdict(int)
rdat.constructs[construct].structure = structures[0]
rdat.constructs[construct].offset = 0
rdat.constructs[construct].annotations = {}
rdat.constructs[construct].xsel = []
rdat.constructs[construct].mutpos = mutpos
rdat.mutpos[construct] = mutpos
rdat.values[construct] = data_noised.tolist()
rdat.comments = 'A bistable RNA simulated data'
for i in xrange(data_noised.shape[0]):
    rdat.append_a_new_data_section(construct)
    if mutpos[i] >= 0:
        print mutpos[i]
        rdat.constructs[construct].data[i].annotations = {'mutation':['%s%s%s' % (sequence[mutpos[i]], mutpos[i]+1, complement[sequence[mutpos[i]]])]}
    else:
        rdat.constructs[construct].data[i].annotations = {'mutation':['WT']}
    rdat.constructs[construct].data[i].values = data_noised[i,:].tolist()
rdat.loaded = True

rdat.save('bistable_rna_test/bistable.rdat')

print 'Factor analysis'
fa = mapping_analysis.FAMappingAnalysis(data_noised, structures, mutpos=mutpos, energies=true_energies)
lhood, W_fa, W_fa_std, Psi_fa, E_d_fa, E_c_fa, sigma_d_fa, E_ddT_fa, M_fa = fa.analyze(nsim=1000, max_iterations=10, n_jobs=15)
corr_facs, data_pred, sigma_data_pred = fa.correct_scale()

for i, s in enumerate(structures):
    binarized_react = [1 if x == '.' else 0 for x in s]
    f = figure(2)
    f.set_size_inches(15, 5)
    clf()
    title('Structure %s: %s' % (i, s))
    exp_react = E_d_fa[i,:]
    plot(1+arange(len(s)), binarized_react, linewidth=2, c='r')
    bar(0.5+arange(len(s)), exp_react, linewidth=0, width=1, color='gray')
    xlim(1,len(s))
    ylim(0,2)
    savefig('bistable_rna_test/exp_react_struct_%s.png' % i, dpi=300)
figure(1)
clf()
imshow(data_pred, cmap=get_cmap('Greys'), vmin=0, vmax=data_pred.mean(), aspect='auto', interpolation='nearest')
savefig('bistable_rna_test/reeffit_data_pred.png', dpi=300)

figure(3)
clf()
cm = get_cmap('Paired')
for i in xrange(len(structures)):
    errorbar(range(W_fa.shape[0]), W_fa[:,i], yerr=W_fa_std[:,i], linewidth=2, label='FA structure %s ' % i, color=cm(50*i))
    plot(range(weights_noised.shape[0]), weights_noised[:,i], '--', linewidth=4,  label='Sim. data structure %s ' % i, color=cm(50*i))

ylim(0,1)
xticks(range(len(mut_labels)), mut_labels, fontsize='x-small', rotation=90)
xlim(-1, len(mut_labels))
#legend()
savefig('bistable_rna_test/fa_weights_by_mutant.png', dpi=300)

"""
fba = mapping_analysis.FullBayesAnalysis(data_noised, structures, energies=true_energies, mutpos=mutpos, dbname='bistable_rna_test/full_bayesian_analysis_bistable_test.pickle')
Psi_t, d_calc_t, free_energies_perturbed_t, Weights_t, M_t, D_t = fba.analyze(n=50000, burn=20000, thin=4, initmethod='fa', inititer=10)
Psi = Psi_t.mean(axis=0)
d_calc = d_calc_t.mean(axis=0)
D = D_t.mean(axis=0)
M = M_t.mean(axis=0)
free_energies_perturbed = free_energies_perturbed_t.mean(axis=0)
Weights = Weights_t.mean(axis=0)
Weights_std = Weights_t.std(axis=0)
figure(1)
clf()
plot_mutxpos_image(d_calc)
savefig('bistable_rna_test/full_bayesian_test_predicted.png', dpi=300)
figure(1)
clf()
for i in xrange(len(structures)):
    errorbar(range(Weights.shape[0]), Weights[:,i], yerr=Weights_std[:,i], linewidth=2, label='structure %s' % i)
legend()
savefig('bistable_rna_test/full_bayesian_weights_by_mutant.png', dpi=300)

for i in xrange(len(structures)):
    plot(range(weights_noised.shape[0]), weights_noised[:,i], linewidth=2, label='structure %s' % i)
legend()
savefig('bistable_rna_test/noised_weights_by_mutant.png', dpi=300)
"""
