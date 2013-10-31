import pdb
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
    xticks(range(len(sequence)), ['%s%s' % (i+1, x) for i, x in enumerate(sequence)], fontsize='xx-small', rotation=90)
    yticks(range(len(mut_labels)), mut_labels, fontsize='x-small')

# This is the RF00005;tRNA from Rfam, scrambled...lots of structures from this random RNA!

sequence = 'UUGUGUGACGGACAACCCCCGUAGUCGCCAUAACAGUGGCUAAGCGUGUUGCCGGGGGGUACCUAGUUUCGCA'
all_structures = ['.((((..((......(((((((((((((.......))))))).((.....)))))))).......))..))))',\
'.((((..((......(((((((((((((.......))))))).(((...))))))))).......))..))))',\
'.((((..((......((((((.....(((((....)))))...((.....)))))))).......))..))))',\
'.((((..((......((((((.....(((((....)))))...(((...))))))))).......))..))))',\
'.((((..((.....((((((.(((((((.......))))))).((.....))..)))))).....))..))))',\
'.((((..((((....(((((((((((((.......))))))).((.....))))))))...))..))..))))',\
'.((((..((((....(((((((((((((.......))))))).(((...)))))))))...))..))..))))',\
'.((((..((((....((((((.....(((((....)))))...((.....))))))))...))..))..))))',\
'.((((..((((....((((((.....(((((....)))))...(((...)))))))))...))..))..))))',\
'.((((..((((((..(((((((((((((.......))))))).((.....)))))))))).))..))..))))',\
'.((((..((((....(((((((((((((.......))))))).((.....))))))))....)).))..))))',\
'.((((.(((......(((((((((((((.......))))))).((.....)))))))).......))).))))',\
'.((((.(((......(((((((((((((.......))))))).(((...))))))))).......))).))))',\
'.((((.(((......((((((.....(((((....)))))...((.....)))))))).......))).))))',\
'.((((.(((......((((((.....(((((....)))))...(((...))))))))).......))).))))',\
'.((((.(((.....((((((.(((((((.......)))))))..((......)))))))).....))).))))',\
'.((((.(((.....((((((.(((((((.......))))))).((.....))..)))))).....))).))))',\
'.((((.(((.....((((((.(((((((.......))))))).(((...)))..)))))).....))).))))',\
'.((((.(((.....((((((......(((((....)))))...((.....))..)))))).....))).))))',\
'.((((.(((((....(((((((((((((.......))))))).((.....))))))))...))..))).))))',\
'.((((.(((((....(((((((((((((.......))))))).(((...)))))))))...))..))).))))',\
'.((((.(((((....((((((.....(((((....)))))...((.....))))))))...))..))).))))',\
'.((((.(((((....((((((.....(((((....)))))...(((...)))))))))...))..))).))))',\
'.((((.(((((((..(((((((((((((.......))))))).((.....)))))))))).))..))).))))',\
'.((((.(((((((..(((((((((((((.......))))))).(((...))))))))))).))..))).))))',\
'.((((.(((((((..((((((.....(((((....)))))...((.....)))))))))).))..))).))))',\
'.((((.(((((....(((((((((((((.......))))))).((.....))))))))....)).))).))))',\
'.((((.(((((....(((((((((((((.......))))))).(((...)))))))))....)).))).))))',\
'.((((.(((((....((((((.....(((((....)))))...((.....))))))))....)).))).))))',\
'..(((..((......(((((((((((((.......))))))).((.....)))))))).......))..))).',\
'..(((..((......(((((((((((((.......))))))).(((...))))))))).......))..))).',\
'..(((..((......((((((.....(((((....)))))...((.....)))))))).......))..))).']

sim_structures = all_structures[:10]
structures = sim_structures

complement = {'A':'U', 'C':'G', 'U':'A', 'G':'C'}
mutants = [sequence] + [sequence[:i] + complement[sequence[i]] + sequence[i+1:] for i in xrange(len(sequence))]
mutpos = [-1] + range(len(sequence))
mut_labels = ['%s%s%s' % (sequence[pos], pos+1, complement[sequence[pos]]) if pos >= 0 else 'WT' for pos in mutpos]
all_energies = get_free_energy_matrix(all_structures, mutants)
true_data, perturbed_data, data_noised, true_energies, weights_noised = mock_data(mutants, sim_structures, energy_mu=10, energy_sigma=10, obs_sigma=0.005, mutpos=mutpos, return_steps=True)
# Let's save these data
savetxt('random_rna_test/true_data.txt', true_data)
savetxt('random_rna_test/perturbed_data.txt', perturbed_data)
savetxt('random_rna_test/data_noised.txt', data_noised)
savetxt('random_rna_test/weights_noised.txt', weights_noised)


figure(1)
clf()
plot(weights_noised.max(axis=0), color='magenta', linewidth=3)
ylim(0,1)
savefig('random_rna_test/wt_weights_noised.png', dpi=300)

figure(1)
clf()
plot_mutxpos_image(true_data)
savefig('random_rna_test/full_model_test_simulated_no_noise.png', dpi=300)
plot_mutxpos_image(perturbed_data)
savefig('random_rna_test/full_model_test_simulated_noised_weights.png', dpi=300)
plot_mutxpos_image(data_noised)
savefig('random_rna_test/full_model_test_simulated_fully_noise.png', dpi=300)

rdat = RDATFile()
rdat.version = 0.32
construct = 'A scrambled tRNA'
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
rdat.comments = 'Simulated data for a scrambled version of the tRNA sequence from Rfam RF00005'
for i in xrange(data_noised.shape[0]):
    rdat.append_a_new_data_section(construct)
    if mutpos[i] >= 0:
        print mutpos[i]
        rdat.constructs[construct].data[i].annotations = {'mutation':['%s%s%s' % (sequence[mutpos[i]], mutpos[i]+1, complement[sequence[mutpos[i]]])]}
    else:
        rdat.constructs[construct].data[i].annotations = {'mutation':['WT']}
    rdat.constructs[construct].data[i].values = data_noised[i,:].tolist()
rdat.loaded = True

rdat.save('random_rna_test/RF00005sc.rdat')


print 'Model selection with factor analysis'
fa = mapping_analysis.FAMappingAnalysis(data_noised, all_structures, mutpos=mutpos, energies=all_energies)
selected_structures, assignments  = fa.model_select(greedy_iter=1, max_iterations=2)

print 'Plotting structure clusters'

# convert structures to binary, paired/unpaired
Mstruct = zeros([len(all_structures[0]), len(all_structures)])
for i, struct in enumerate(all_structures):
    for j, s in enumerate(struct):
        if s == '.':
            Mstruct[j,i] = 1
        else:
            Mstruct[j,i] = 0
# PCA on Mstruct
cm = get_cmap('Paired')
U, S, V = svd(Mstruct)
proj = dot(U[[0,1],:], Mstruct)
figure(5)
clf()
for i, s in enumerate(selected_structures):
    v = assignments[s]
    scatter(proj[0,v], proj[1,v], color=(cm(50*i)), alpha=0.8)
savefig('random_rna_test/structure_PCA.png', dpi=300)

lhood, W_fa, W_fa_std, Psi_fa, D_fa, E_c_fa, sigma_d_fa, E_ddT_fa, M_fa = fa.analyze(max_iterations=4, nsim=2000, select_struct_indices=selected_structures, n_jobs=30)
print 'Selected structures were:'
for i in selected_structures:
    print '%s %s' % (i, all_structures[i])
print 'Original structures were:'
for s in sim_structures:
    print s
data_pred = dot(W_fa, D_fa)
for i, s in enumerate(selected_structures):
    struct = fa.structures[s]
    binarized_react = [1 if x == '.' else 0 for x in struct]
    f = figure(2)
    f.set_size_inches(15, 5)
    clf()
    title('Structure %s: %s' % (i, s))
    exp_react = D_fa[i,:]
    plot(1+arange(len(struct)), binarized_react, linewidth=2, c='r')
    bar(0.5+arange(len(struct)), exp_react, linewidth=0, width=1, color='gray')
    xlim(1,len(struct))
    ylim(0,2)
    savefig('random_rna_test/exp_react_struct_%s.png' % i, dpi=300)
figure(1)
clf()
imshow(data_pred, cmap=get_cmap('Greys'), vmin=0, vmax=data_pred.mean(), aspect='auto', interpolation='nearest')
savefig('random_rna_test/reeffit_data_pred.png', dpi=300)

cluster_weights = zeros([weights_noised.shape[0], len(assignments)])
sim_structs_indices = range(len(sim_structures))
for i, structidx in enumerate(selected_structures):
    for k, elems in assignments.iteritems():
        if structidx in elems:
            break
    for j in sim_structs_indices:
        if j in elems:
            cluster_weights[:,i] += weights_noised[:,j]
figure(3)
clf()
cm = get_cmap('Paired')
for i in xrange(cluster_weights.shape[1]):
    errorbar(range(W_fa.shape[0]), W_fa[:,i], yerr=W_fa_std[:,i], linewidth=2, label='FA structure %s ' % i, color=cm(50*i))
    #plot(range(W_fa.shape[0]), W_fa[:,i], linewidth=2, label='FA structure %s ' % i, color=cm(50*i))
    plot(range(cluster_weights.shape[0]), cluster_weights[:,i], '--', linewidth=4, label='structure cluster %s' % i, color=cm(50*i))
#legend()
xticks(range(len(mut_labels)), mut_labels, fontsize='xx-small', rotation=90)
xlim(-1, len(mut_labels))
ylim(0,1)
savefig('random_rna_test/weights_by_mutant.png', dpi=300)


for i in xrange(cluster_weights.shape[1]):
    figure(3)
    clf()
    errorbar(range(W_fa.shape[0]), W_fa[:,i], yerr=W_fa_std[:,i], linewidth=2, label='FA structure %s ' % i, color=cm(50*i))
    #plot(range(W_fa.shape[0]), W_fa[:,i], linewidth=2, label='FA structure %s ' % i, color=cm(50*i))
    plot(range(cluster_weights.shape[0]), cluster_weights[:,i], '--', linewidth=4, label='structure cluster %s' % i, color=cm(50*i))
    xticks(range(len(mut_labels)), mut_labels, fontsize='xx-small', rotation=90)
    xlim(-1, len(mut_labels))
    ylim(0,1)
    savefig('random_rna_test/weights_by_mutant_struct%s.png' % i, dpi=300)

