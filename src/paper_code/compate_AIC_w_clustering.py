from Bio import SeqIO
from matplotlib.pylab import *
import pickle
import rdatkit.secondary_structure as ss
import map_analysis_utils as utils
import pdb
import random
import os

rfam = SeqIO.parse('../../../databases/Rfam_rep.fasta', format='fasta')
fa_indices = {}
nclusts = {}
complement = {'A':'U', 'C':'G', 'U':'A', 'G':'C'}
l = 0
for record in rfam:
    sequence = str(record.seq)
    if len(sequence) > 500:
        continue
    print 'Doing %s' % record.id
    print 'Getting structures'
    mutants = [sequence[:i] + complement[sequence[i]] + sequence[i+1:] for i in xrange(len(sequence))]
    mutpos = range(len(sequence))
    structures, deltas = ss.subopt(sequence, nstructs=20)
    structures = [s.dbn for s in structures]
    print 'Found %s structures' % len(structures)
    if len(structures) < 4:
        continue
    print 'Simulating data'
    true_data, perturbed_data, data_noised, true_energies, weights_noised = utils.mock_data(mutants, structures, energy_mu=10, energy_sigma=5, mutpos=mutpos, return_steps=True)
    print 'Getting factor index'
    struct_types = []
    for i in xrange(len(structures[0])):
        struct_types.append(['u' if s[i] == '.' else 'p' for s in structures])
    data_indices = random.sample(range(data_noised.shape[0]), min(data_noised.shape[0]-1, 50))
    fa_indices[record.id] = utils.factor_index(data_noised[data_indices,:], len(structures))
    print fa_indices[record.id]
    print 'Getting cluster number'
    medoids, assignments = utils.cluster_structures(struct_types)
    nclusts[record.id] = len(medoids)
    print nclusts[record.id]
    os.popen('rm /tmp/tmp*')
    l += 1
    if l >70:
        break

print 'Finished, writing...'
pickle.dump(nclusts, open('rfam_nclusts.pickle', 'w'))
pickle.dump(fa_indices, open('rfam_fa_indices.pickle', 'w'))
figure(1)
clf()
scatter(nclusts.values(), [fa_indices[k] for k in nclusts])
savefig('nclusts_vs_factor_index.png')
xlabel('Number of structure clusters')
ylabel('Number of factors explaining the data')
