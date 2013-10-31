import rdatkit.secondary_structure as ss
import map_analysis_utils as utils
import pickle
import os

rfamfile = open('Rfam_rep_all_benchmark_200_subopt.txt')
outdir = 'insilico_simulations'
energies = {}
weights = {}
for l in rfamfile.readlines():
    os.system('rm /tmp/tmp*')
    name, sequence, structures = l.strip().split('\t')
    print 'Doing %s' % name
    structures = structures.split(',')
    keyname = name.lower().replace(' ','_').replace(';','_').replace('/','_')
    energies[keyname] = utils.get_free_energy_matrix(structures, [sequence])
    weights[keyname] = utils.calculate_weights(energies[keyname])

pickle.dump(energies, open(outdir + '/all_energies_dict.pickle', 'w'))
pickle.dump(weights, open(outdir + '/all_weights_dict.pickle', 'w'))
