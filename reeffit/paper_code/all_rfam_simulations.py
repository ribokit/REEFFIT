import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *
import mapping_analysis
from map_analysis_utils import *
from rdatkit.datahandlers import *
from collections import defaultdict
import os
from plot_utils import plot_mutxpos_image
import pickle
from shutil import copy2
from rdatkit.view import VARNA


prefix = 'unperturbed_simulations/'
benchmarkfile = open('Rfam_rep_all_benchmark_200_subopt.txt')
nseqs = inf
complement = {'A':'U', 'C':'G', 'U':'A', 'G':'C'}

BENCHMARKDIR = 'insilico_simulations/'
def simulate_dataset(sequence, structures, name, pathname, outname):
    print 'Simulating dataset'
    mutants = [sequence] + [sequence[:i] + complement[sequence[i]] + sequence[i+1:] for i in xrange(len(sequence))]
    struct_types = []
    for i in xrange(len(sequence)):
        struct_types.append(['u' if structures[j][i] == '.' else 'p' for j in xrange(len(structures))])

    mutpos = [-1] + range(len(sequence))
    seqpos = range(len(sequence))
    mut_labels = ['%s%s%s' % (sequence[pos], pos+1, complement[sequence[pos]]) if pos >= 0 else 'WT' for pos in mutpos]
    true_data, perturbed_data, data_noised, true_energies, weights_noised, reacts, obs_noise_sigmas = mock_data(mutants, structures, energy_mu=0, energy_sigma=0, obs_sigma=0.02, mutpos=mutpos, c_size=3, correlate_regions=False, return_steps=True)

    figure(1)
    clf()
    plot_mutxpos_image(true_data, sequence, seqpos, 0, mut_labels)
    savefig('%s/simulated_no_noise.png' % pathname, dpi=100)
    plot_mutxpos_image(perturbed_data, sequence, seqpos, 0, mut_labels)
    savefig('%s/simulated_noised_weights.png' % pathname, dpi=100)
    plot_mutxpos_image(data_noised, sequence, seqpos, 0, mut_labels)
    savefig('%s/simulated_fully_noised.png' % pathname, dpi=100)
    plot_mutxpos_image(data_noised, sequence, seqpos, 0, mut_labels)
    savefig('%s/%s_results/%s.png' % (BENCHMARKDIR, prefix, outname), dpi=100)

    print 'Saving to RDAT'

    rdat = RDATFile()
    rdat.version = 0.32
    construct = name
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
    rdat.comments = 'This is simulated data'
    for i in xrange(data_noised.shape[0]):
        rdat.append_a_new_data_section(construct)
        if mutpos[i] >= 0:
            print mutpos[i]
            rdat.constructs[construct].data[i].annotations = {'mutation':['%s%s%s' % (sequence[mutpos[i]], mutpos[i]+1, complement[sequence[mutpos[i]]])]}
        else:
            rdat.constructs[construct].data[i].annotations = {'mutation':['WT']}
        rdat.constructs[construct].data[i].values = data_noised[i,:].tolist()
    rdat.loaded = True

    rdat.save('%s/%s.rdat' % (pathname, outname))
    return weights_noised, mut_labels, obs_noise_sigmas, reacts

count = 1 
ids = []
for line in benchmarkfile.readlines():
    if nseqs and count > nseqs:
        break
    count += 1
    if line.strip()[0] == '#':
        continue
    else:
        name, sequence, structures = line.strip().split('\t')
        print 'Doing %s' % name
        structures = structures.split(',')
        outname = name.lower().replace(' ','_').replace(';','_').replace('/','_')
        pathname = BENCHMARKDIR + prefix + '/'  + outname
        ids.append(pathname)
        if not os.path.exists(pathname):
            os.mkdir(pathname)
        if not os.path.exists('%s/%s_results/' % (BENCHMARKDIR, prefix)):
            os.mkdir('%s/%s_results/' % (BENCHMARKDIR, prefix))

        weights_noised, mut_labels, obs_noise_sigmas, reacts = simulate_dataset(sequence, structures, name, pathname, outname)

        pickle.dump(weights_noised, open('%s/weights_noised.pickle' % pathname, 'w'))
        pickle.dump(mut_labels, open('%s/mut_labels.pickle' % pathname, 'w'))
        pickle.dump(obs_noise_sigmas, open('%s/obs_noise_sigmas.pickle' % pathname, 'w'))
        pickle.dump(reacts, open('%s/reacts.pickle' % pathname, 'w'))


