import argparse
import pdb
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import rgb2hex
import os
import rdatkit.secondary_structure as ss
from rdatkit.view import VARNA
import pickle
from matplotlib.pylab import *
from rdatkit.datahandlers import RDATFile
from rdatkit import mapping
from plot_utils import *
from map_analysis_utils import *
from random import choice, sample
import mapping_analysis

parser = argparse.ArgumentParser()
parser.prog = 'reeffit'
parser.add_argument('rdatfile', type=argparse.FileType('r'), help='The RDAT file that has the multi-dimensional chemical mapping data')
parser.add_argument('outprefix', type=str, help='Prefix (e.g. directory) that all resulting reports, plot files, etc. will have')
parser.add_argument('--structfile', default=None, type=argparse.FileType('r'), help='Text files with structures to analyze, one per line, in dot-bracket notation')
parser.add_argument('--clusterfile', default=None, type=argparse.FileType('r'), help='File with clusters of structures: one structure per line, with tab-delimited fields. Format is cluster_id, dot-bracket structure, comma-separated energies per cluster' )
parser.add_argument('--medoidfile', default=None, type=argparse.FileType('r'), help='File specifying the medoid structures of each cluster. Must also specify a clusterfile.')
parser.add_argument('--structset', default=None, type=str, help='Subset of structures in the specified structfile to use in the analysis. Each subset is identified by a "header" specified after a hash (#) preceding the set of structures. This option will search all headers for structset and analyze the data with those structures. Used in worker mode.')
parser.add_argument('--nsubopt', default=100, type=int, help='For model selection. Number of maximum suboptimal structures to take from each sequence\'s structural ensemble in the RDAT file')
parser.add_argument('--nsim', default=1000, type=int, help='Number of simulations used for each E-step when performing soft EM.')
parser.add_argument('--cutoff', default=0, type=int, help='Number of data points and nucleotides to cut off from the beginning and end of the data and sequences: these will not be used in the analysis. Useful to avoid saturated "outliers" in the data.')
parser.add_argument('--start', default=None, type=int, help='Seqpos starting position of the data and sequences from where to analyze. Useful to focus the analysis on a particular location. Must specify both start and end options.')
parser.add_argument('--end', default=None, type=int, help='Seqpos ending position of the data and sequences to analyze. Useful to focus the analysis on a particular location. Must specify both start and end options.')
parser.add_argument('--greedyiter', default=10, type=int, help='For heuristic model selection. Number of greedy iterations in which REEFFIT tries to add more structures to make the model better.')
parser.add_argument('--nopriorswap', action='store_true', help='For heuristic model selection. Do not swap structure medoids in each cluster, take the default medoids found by centrality maximization')
parser.add_argument('--nopseudoenergies', action='store_true', help='For heuristic model selection. Do not use pseudoenergies (i.e. SHAPE-directed modeling) to score the structures. Structures will be scored using regular RNAstructure energies, with no data.')
parser.add_argument('--modelselect', type=str, default=None, help='Model selection mode. Can be one of "heuristic" or "mc" (Monte Carlo, including MCMC)')
parser.add_argument('--hardem', default=False, action='store_true', help='Perform hard EM instead of soft EM, finding a MAP of the hidden reactivities, rather than simulating from the posterior. This makes REEFFIT run considerably faster, but does not yield a rigorous, posterior distribution estimation of the hidden reactivities of each structure.')
parser.add_argument('--energydelta', default=None, type=float, help='Kcal/mol free energy limit that the structure weights are allowed to deviate from the initial weights.')
parser.add_argument('--refineiter', default=10, type=int, help='Maximum number of EM iterations to perform')
parser.add_argument('--structest', default='hclust', type=str, help='For heuristic model selection. Method for estimating the number of structures underlying the data, and later used for clustering. Available methods are "hclust" (hierarchical clustering) and "fa" (standard, gaussian factor analysis)')
parser.add_argument('--clusterdatafactor', type=int, default=None, help='For clustering the data to reduce dimensionality (useful for large datasets with lots of redundant measurements). Describes the approximate number of clusters for clustering the data')
parser.add_argument('--bootstrap', type=int, default=0, help='Number of bootstrap iterations to perform')
parser.add_argument('--titrate', type=str, default=None, help='For morph-and-map experiments. Name of chemical titrated. Must also specify kdfile when using this option.')
parser.add_argument('--nomutrepeat', default=False, action='store_true', help='Skip repeating mutants')
parser.add_argument('--clipzeros', default=False, action='store_true', help='Clip data to non-zero regions')
parser.add_argument('--postmodel', default=False, action='store_true', help='Perform SHAPE-directed modeling after analysis using the calculated hidden reactivities for each structures. Useful if the hidden reactivities do not match well with the prior structures')
parser.add_argument('--worker', default=False, action='store_true', help='Worker mode (non-verbose, simple output). Used for MC model selection.')
parser.add_argument('--kdfile', default=None, type=argparse.FileType('r'), help='File with the dissociation constants for each structure, for titrating a chemical specified in the titrate option'.)
parser.add_argument('--splitplots', default=-1, type=int, help='Plot subsets of data and predicted data rather than the whole set')
parser.add_argument('--detailedplots', default=False, action='store_true', help='Plots log-likelihood trace, all predicted data vs real data separately, and comparison plots between initial and final structure weights')
parser.add_argument('--nonormalization', default=False, action='store_true', help='Do not perform box-plot normalization of the data. Useful for MAP-seq datasets')
parser.add_argument('--dpi', type=int, default=200, help='DPI resolution for plots')
parser.add_argument('--scalemethod', type=str, default='linear', help='Scaling method to perform after fits')
parser.add_argument('--priorweights', type=str, default='rnastructure', help='Algorithm to use for starting structure weights. Can be "rnastructure", "viennarna", and "uniform"')
parser.add_argument('--interpreter', type=str, default='python', help='For MC model selection. Python interpreter to use for the worker files')
parser.add_argument('--njobs', type=int, default=None, help='For soft EM analysis. Number of parallel jobs to run the E-step on')
parser.add_argument('--csize', type=int, default=3, help='Number of sequence positions to be allowed for contact sites.')
parser.add_argument('--msnsim', type=int, default=100, help='For MC model selection. Number of Monte Carlo simulations per worker')
parser.add_argument('--msnworkers', type=int, default=10, help='For MC model selection. Number of workers.')
parser.add_argument('--msmaxsamples', type=int, default=inf, help='For MC model selection. Maximum number of structures per sample in MC simulation')
parser.add_argument('--msworkerformat', type=str, default='sh', help='For MC model selection. File format of worker files. Can be "sh" (for simple shell script) and "gridengine" (for use with grid engine')

args = parser.parse_args()

# Global variables
model_selection_names = {'mc':'Monte Carlo (includes MCMC)', 'heuristic': 'Heuristic'}
worker_file_formats = ['gridengine', 'sh']

print 'Parsing RDAT'
rdat = RDATFile()
rdat.load(args.rdatfile)
construct = rdat.constructs.values()[0]

seqpos = construct.seqpos
sorted_seqpos = sorted(seqpos)
offset = construct.offset
sequence = construct.sequence[min(seqpos) - offset - 1:max(seqpos) - offset].upper()
sequence = construct.sequence.upper()


print 'Parsing mutants and data'
data = []

mut_labels = []
mutants = []
mutpos = []
wt_idx = 0
use_struct_clusters = False
concentrations = []
kds = []
for idx, d in enumerate(construct.data):
    if 'warning' in d.annotations:
        continue
    if args.titrate:
        if 'chemical' in d.annotations:
            chemfound = False
            for item in d.annotations['chemical']:
                chem, conc = item.split(':')[:2]
                if chem == args.titrate:
                    print 'Found concentration of chemical of interest (%s) for data at index %s: %s' % (chem, idx+1, conc)
                    concentrations.append(float(conc.replace('uM', '')))
                    chemfound = True
            if not chemfound:
                concentrations.append(0)
    if 'mutation' in d.annotations:
        label = d.annotations['mutation'][0]
    else:
        label = 'WT'
    if label == 'WT':
        if 'sequence' in d.annotations:
            mutant = d.annotations['sequence'][0]
        else:
            mutant = sequence
        wt_idx = idx
        pos = -1
        mutpos.append(pos)
    else:
        pos = int(label[1:len(label)-1]) - 1 - construct.offset
        if 'sequence' in d.annotations:
            mutant = d.annotations['sequence'][0]
        else:
            mutant = sequence[:pos] + label[-1] + sequence[pos+1:]
        mutpos.append(pos)
    if mutant in mutants and args.nomutrepeat:
        continue
    mut_labels.append(label)
    mutants.append(mutant)
    if args.nonormalization:
        nd_tmp = normalize([d.values[seqpos.index(i)] for i in sorted_seqpos])
        nd = [d.values[seqpos.index(i)] if d.values[seqpos.index(i)] >= 0 else 0.001 for i in sorted_seqpos]
        #nd = [nd[i] if nd[i] < 4 else max(nd_tmp) for i in range(len(nd))]
        #nd = [d.values[seqpos.index(i)] for i in sorted_seqpos]
    else:
        nd = normalize([d.values[seqpos.index(i)] for i in sorted_seqpos])
    if args.clipzeros:
        last_seqpos = len(seqpos)
        for i in xrange(len(nd)):
            if nd[i] == 0:
                last_seqpos = max(last_seqpos, i)
                break
    data.append(nd)
if args.clipzeros:
    sorted_seqpos = sorted_seqpos[:last_seqpos]
    data = array(data)[:,:last_seqpos]
else:
    data = array(data)
if args.splitplots < 0:
    args.splitplots = data.shape[0]
if args.cutoff > 0:
    if args.start or args.end:
        print 'Cannot specify sequence range start and end and cutoff at the same time!'
        exit()
    data_cutoff = data[:, args.cutoff:-args.cutoff]
    seqpos_cutoff = sorted_seqpos[args.cutoff:-args.cutoff]
    seqpos_start = args.cutoff
    seqpos_end = data.shape[1] - args.cutoff
    mutpos_cutoff = [pos - args.cutoff for pos in mutpos]
    seqpos_range = (seqpos_start, seqpos_end)
elif args.start != None or args.end != None:
    if not (args.start and args.end):
        print 'Must specify both sequence range start and end!'
        exit()
    seqpos_start = sorted_seqpos.index(args.start)
    seqpos_end = sorted_seqpos.index(args.end) + 1
    seqpos_cutoff = sorted_seqpos[seqpos_start:seqpos_end]
    data_cutoff = data[:, seqpos_start:seqpos_end]
    mutpos_cutoff = [pos - seqpos_start for pos in mutpos]
    seqpos_range = (seqpos_start, seqpos_end)
else:
    data_cutoff = data
    seqpos_cutoff = sorted_seqpos
    seqpos_start = 0
    seqpos_end = len(seqpos_cutoff)
    mutpos_cutoff = mutpos
    seqpos_range = None

if args.structfile != None:
    if args.clusterfile != None:
        print 'Both structure and structure cluster were specified! Need only one of the two to run on single structure or cluster modalities'
        exit()
    else:
        if args.structset != None:
            structures = []
            start_reading_structs = False
            for line in args.structfile.readlines():
                if line[0] == '#':
                    if start_reading_structs:
                        break
                    if line[1:].strip() == args.structset:
                        start_reading_structs = True
                else:
                    if start_reading_structs:
                        structures.append(line.strip().replace('A', '.'))

        else:
            structures = [line.strip().replace('A', '.') for line in args.structfile.readlines() if line[0] != '#']
else:
    if args.clusterfile != None:
        use_struct_clusters = True
        if not args.medoidfile:
            print 'Need a medoid file when using structure clusters!'
            exit()
        else:
            structure_medoids = [line.strip() for line in args.medoidfile.readlines()]
        print 'Reading cluster structure file'
        # Cluster structure files are given as tab-delimited:
        # cluster id, structure (dot-bracket)
        # OR
        # cluster id, structure (dot-bracket), weights for each mutant (in comma separated values)
        structure_clusters = defaultdict(list)
        structures = []
        struct_weights_by_clust = {}
        struct_medoid_indices = {}
        cluster_indices = defaultdict(list)
        struct_idx = 0
        for line in args.clusterfile.readlines():
            fields = line.strip().split('\t')
            structure_clusters[fields[0]].append(fields[1])
            structures.append(fields[1])
            if len(fields) == 3:
                if fields[0] in struct_weights_by_clust:
                    struct_weights_by_clust[fields[0]].append([float(weight) for weight in fields[2].split(',')])
                else:
                    struct_weights_by_clust[fields[0]] = [[float(weight) for weight in fields[2].split(',')]]
                cluster_indices[fields[0]].append(struct_idx)
                struct_idx += 1
        for c, structs in structure_clusters.iteritems():
            for i, s in enumerate(structs):
                if s in structure_medoids:
                    struct_medoid_indices[c] = i
                    break
        for c in struct_weights_by_clust:
            struct_weights_by_clust[c] = array(struct_weights_by_clust[c]).T
    else:
        print 'Getting suboptimal ensemble for ALL mutants'
        #structures, deltas = ss.subopt(sequence[seqpos_start:seqpos_end], nstructs=args.nsubopt, algorithm=args.priorweights)
        structures, deltas = ss.subopt(sequence, nstructs=args.nsubopt, algorithm=args.priorweights)
        structures = list(set([s.dbn for s in structures]))
        for i, m in enumerate(mutants):
            print 'Doing mutant %s: %s' % (i, m)
            #mut_structures, deltas = ss.subopt(m[seqpos_start:seqpos_end], nstructs=args.nsubopt, algorithm=args.priorweights)
            mut_structures, deltas = ss.subopt(m, nstructs=args.nsubopt, algorithm=args.priorweights)
            mut_structures = list(set([s.dbn for s in mut_structures if s.dbn not in structures]))
            if len(mut_structures) > 0:
                print 'Found new structures: %s' % mut_structures
            structures.extend(mut_structures)
        structures = list(set(structures))
print 'Structures to consider are:'
for i, s in enumerate(structures):
    print '%s:%s' % (i, s)

if args.titrate != None:
    if args.kdfile == None:
        print 'Expected a Kd file when analyzing titrations (the titrate parameter was set), but the kdfile option is not set!'
        exit()
    else:
        print 'Reading Kd file'
        kds = zeros([len(concentrations), len(structures)])
        filestructs = args.kdfile.readline().strip().split('\t')
        try:
            order = [structures.index(s) for s in filestructs]
        except ValueError:
            print 'Structures in Kd file do not match structures in structure file!'
            exit()
        line = args.kdfile.readline()
        linenum = 0
        while line:
            fields = [float(val) for val in line.strip().split('\t')]
            for i, idx in enumerate(order):
                kds[linenum,i] = fields[idx]
            line = args.kdfile.readline()
            linenum += 1
if args.modelselect != None:
    #energies = get_free_energy_matrix(structures,[m[seqpos_start:seqpos_end] for m in  mutants], algorithm=args.priorweights)
    energies = get_free_energy_matrix(structures,[m for m in  mutants], algorithm=args.priorweights)
else:
    energies = get_free_energy_matrix(structures, mutants, algorithm=args.priorweights)
fa = mapping_analysis.FAMappingAnalysis(data, structures, mutants, mutpos=mutpos_cutoff, energies=energies, concentrations=concentrations, kds=kds, seqpos_range=seqpos_range, c_size=args.csize)
unpaired_data, paired_data = fa.set_priors_by_rvs(SHAPE_unpaired_sample, SHAPE_paired_sample)
if args.clusterfile:
    fa.set_structure_clusters(structure_clusters, struct_medoid_indices, struct_weights_by_clust=struct_weights_by_clust)
    W_0 = fa.W
else:
    W_0 = calculate_weights(energies)

def prepare_model_select_simulation_structure_file(idx, nsim):
    sf = open('%sstructures%s.txt' % (args.outprefix, idx), 'w')
    for i in xrange(nsim):
        n = randint(2, min(args.msmaxsamples, len(structures)))
        sf.write('#%s\n' % i)
        sf.write('\n'.join(set(sample(structures, n))))
        if i < nsim - 1:
            sf.write('\n')
    sf.close()
    return sf.name

def prepare_worker_file(idx, nsim, simfilename):
    wf = open('%sworker%s.txt' % (args.outprefix, idx), 'w')
    general_options = '%s %sworker_%s --worker --structfile=%s' % (os.path.abspath(args.rdatfile.name), args.outprefix, idx, os.path.abspath(simfilename))
    carry_on_options = ['nsim', 'refineiter', 'structest', 'clusterdatafactor', 
            'bootstrap', 'cutoff', 'start', 'end', 'hardem', 'energydelta', 'titrate',
            'nomutrepeat', 'clipzeros', 'kdfile', 'nonormalization', 'priorweights', 'njobs', 'csize']
    for opt in carry_on_options:
        val = args.__dict__[opt]
        if val != None:
            if type(val) == bool:
                if val:
                    general_options += ' --%s ' % opt
            else:
                general_options += ' --%s=%s ' % (opt, val)

    for i in xrange(nsim):
        wf.write('%s %s %s --structset=%s\n' % (args.interpreter, os.environ['REEFFIT_HOME'] + '/bin/reeffit ' + general_options, i))
    return wf.name

if args.modelselect != None:
    print 'Model selection using %s' % model_selection_names[args.modelselect]
    if args.modelselect == 'mc':
        if args.msworkerformat not in worker_file_formats:
            print 'Unrecognized worker file format %s, aborting!' % args.msworkerformat
        print 'Preparing worker files for MC sampling'
        print 'Number of worker files %s' % args.msnworkers
        print 'Number of samples per file %s' % args.msnsim
        print 'Worker file format is %s' % args.msworkerformat
        for i in xrange(args.msnworkers):
            sfname = prepare_model_select_simulation_structure_file(i, args.msnsim)
            prepare_worker_file(i, args.msnsim, sfname)
        print 'Finished preparing worker files, run them using your scheduler and them use compile_worker_results.py to compile the results'
        exit()
    if args.modelselect in ['heuristic']:
        selected_structures, assignments = fa.model_select(greedy_iter=args.greedyiter, max_iterations=2, prior_swap=not args.nopriorswap, expstruct_estimation=args.structest, G_constraint=args.energydelta, n_jobs=args.njobs, apply_pseudoenergies=not args.nopseudoenergies, algorithm=args.priorweights, hard_em=args.hardem, mode=args.modelselect)
        print 'Getting sequence energies'
        energies = get_free_energy_matrix(structures, mutants)
        print 'Getting cluster energies'
        W, struct_weights_by_clust = calculate_weights(energies, clusters=assignments)
        outclustfile = open(args.outprefix + 'structure_clusters.txt', 'w')
        outmedoidsfile = open(args.outprefix + 'structure_medoids.txt', 'w')
        for m in selected_structures:
            outmedoidsfile.write(structures[m] + '\n')
        for c, indices in assignments.iteritems():
            for i, idx in enumerate(indices):
                energies_str = ','.join([str(w) for w in struct_weights_by_clust[c][:,i]])
                outclustfile.write('%s\t%s\t%s\n' % (c, structures[idx], energies_str))
        outclustfile.close()
        outmedoidsfile.close()
        print 'Plotting PCA structure clusters'
        PCA_structure_plot(structures, assignments, selected_structures)
        savefig(args.outprefix + 'pca_cluster_plot.png', dpi=300)
        print 'Done, check out cluster file %s and medoids file %s' % (outclustfile.name, outmedoidsfile.name)
        exit()
else:
    selected_structures = range(len(structures))

nstructs = len(selected_structures)
nmeas =  data_cutoff.shape[0]
npos = data_cutoff.shape[1]

# Boostrap variables
Wboot = zeros([nmeas, nstructs, args.bootstrap])
Psiboot = zeros([npos, nmeas, nmeas, args.bootstrap])
E_dboot = zeros([nstructs, npos, args.bootstrap])
E_ddTboot = zeros([nstructs, nstructs, npos, args.bootstrap])

def remove_non_cannonical(structure, sequence):
    cannonical_bp = [('G','C'), ('C','G'), ('G','U'), ('U','G'), ('A','U'), ('U','A')]
    bp_dict = ss.SecondaryStructure(dbn=structure).base_pair_dict()
    res_struct = ['.']*len(sequence)
    for n1, n2 in bp_dict.iteritems():
        if (sequence[n1], sequence[n2]) in cannonical_bp:
            if n1 < n2:
                res_struct[n1] = '('
                res_struct[n2] = ')'
            else:
                res_struct[n1] = ')'
                res_struct[n2] = '('
    return ''.join(res_struct)

def make_struct_figs(structures, fprefix):
    for i, s in enumerate(structures):
        VARNA.cmd(mutants[0], remove_non_cannonical(s, mutants[0]), args.outprefix + fprefix + 'structure%s.svg' % i, options={'baseOutline':rgb2hex(STRUCTURE_COLORS[i]), 'fillBases':False, 'resolution':'10.0', 'flat':True, 'offset':offset + seqpos_start, 'bp':'#000000'})

if not args.worker:
    make_struct_figs(structures, '')
for b_iter in xrange(args.bootstrap + 1):

    if b_iter > 0:
        dirname = '%s/boot%s' % (args.outprefix, b_iter)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        prefix = dirname
        bootstrap_range = range(data_cutoff.shape[1])
        I = [choice(bootstrap_range) for i in bootstrap_range]
    else:
        prefix = args.outprefix
        I = arange(data_cutoff.shape[1])
    seqpos_iter = [seqpos_cutoff[i] for i in I]

    lhood_traces, W_fa, W_fa_std, Psi_fa, E_d_fa, E_c_fa, sigma_d_fa, E_ddT_fa, M_fa = fa.analyze(max_iterations=args.refineiter, nsim=args.nsim, G_constraint=args.energydelta, cluster_data_factor=args.clusterdatafactor, use_struct_clusters=use_struct_clusters, seq_indices=I, n_jobs=args.njobs, return_loglikes=True, hard_em=args.hardem)

    loglikes, Psi_reinits = lhood_traces

    if args.clusterfile:
        selected_structures = [cluster_indices[c][struct_medoid_indices[c]] for c in structure_clusters]
    print 'Selected structures were:'
    for i in selected_structures:
        print '%s %s' % (i,fa.structures[i])

    corr_facs, data_pred, sigma_pred = fa.correct_scale(stype=args.scalemethod)
    missed_indices, missed_vals = fa.calculate_missed_predictions(data_pred=data_pred)
    chi_sq, rmsea, aic = fa.calculate_fit_statistics()

    if args.clusterdatafactor and not args.worker:

        CI = fa.data_cluster_medoids.values()

        figure(1)
        clf()
        plot_mutxpos_image(fa.cluster_data_pred[:,I], sequence, seqpos_iter, offset, [mut_labels[k] for k in CI])
        savefig('%s/cluster_data_pred.png' % (prefix), dpi=100)

        figure(1)
        clf()
        plot_mutxpos_image(fa.data_cutoff[CI,I], sequence, seqpos_iter, offset, [mut_labels[k] for k in CI])
        savefig('%s/cluster_real_data.png' % (prefix), dpi=100)

        for cid in set(fa.data_clusters):
            figure(1)
            clf()
            CI = [idx for idx, cidx in enumerate(fa.data_clusters) if cidx == cid]
            plot_mutxpos_image(fa.data_cutoff[CI,I], sequence, seqpos_iter, offset, [mut_labels[k] for k in CI])
            savefig('%s/cluster_%s_real_data.png' % (prefix, cid), dpi=100)

    if b_iter > 0:
        Wboot[:,:,b_iter-1] = W_fa
        E_dboot[:,:,b_iter-1] = E_d_fa
        E_ddTboot[:,:,:,b_iter-1] = E_ddT_fa
        Psiboot[:,:,:,b_iter-1] = Psi_fa

    print 'Saving data'
    if not args.worker:
        pickle.dump(fa.data, open('%sdata.pickle' % prefix, 'w'))
        pickle.dump(W_fa_std, open('%sW_std.pickle' % prefix, 'w'))
        pickle.dump(Psi_fa, open('%sPsi.pickle' % prefix, 'w'))
        pickle.dump(E_c_fa, open('%sE_c.pickle' % prefix, 'w'))
        pickle.dump(E_ddT_fa, open('%sE_ddT.pickle' % prefix, 'w'))
        pickle.dump(M_fa, open('%sM.pickle' % prefix, 'w'))
        pickle.dump(data_pred, open('%sdata_pred.pickle' % prefix, 'w'))
        pickle.dump(sigma_pred, open('%ssigma_pred.pickle' % prefix, 'w'))
        if args.bootstrap_indices > 0:
            pickle.dump(I, open('%sbootstrap_indices.pickle' % prefix, 'w'))
    pickle.dump(W_fa, open('%sW.pickle' % prefix, 'w'))
    pickle.dump(E_d_fa, open('%sE_d.pickle' % prefix, 'w'))

    if args.worker:
        report = open('%s_results.txt' % prefix,'a')
        report.write('%s\t%s\t%s\t%s\t%s\n' % (max(loglikes), chi_sq, rmsea, aic, ','.join(structures)))
        report.close()
        print 'Worker results were appended to %s' % report.name
    else:
        print 'Printing report'
        report = open('%s/report.txt' % prefix,'w')
        report.write('Arguments: %s\n' % args)
        if b_iter > 0:
            report.write('Bootstrap iteration %s. Selected sequence positions were:\n' % b_iter)
            for i in I:
                report.write('%s: %s\n' % (i, i + offset + 1))
        if wt_idx != -1:
            report.write('Structure index\tStructure\tWild type weight\tWeight std\n')
        else:
            report.write('No Wild type found in data set\n')
            report.write('Structure index\tStructure\tSeq 0 weight\tSeq 0 std\n')
        for i, s in enumerate(selected_structures):
            report.write('%s\t%s\t%.3f\t%.3f\n' % (i, fa.structures[s], W_fa[0,i], W_fa_std[0,i]))
        report.write('Likelihood: %s\n' % max(loglikes))
        report.write('chi squared/df: %s\n' % chi_sq)
        report.write('RMSEA: %s\n' % rmsea)
        report.write('AIC: %s\n' % aic)
        report.close()

        r = range(data_cutoff.shape[1])
        for i, s in enumerate(selected_structures):
            f = figure(2)
            f.set_size_inches(15, 5)
            clf()
            title('Structure %s: %s' % (i, s))
            if args.hardem:
                expected_reactivity_plot(E_d_fa[i,:], fa.structures[s], yerr=sigma_d_fa[i,:], seq_indices=I)
            else:
                expected_reactivity_plot(E_d_fa[i,:], fa.structures[s], yerr=sigma_d_fa[i,:]/sqrt(args.nsim), seq_indices=I)
            xticks(r[0:len(r):5], seqpos_iter[0:len(seqpos_iter):5], rotation=90)
            savefig('%s/exp_react_struct_%s.png' % (prefix, i), dpi=args.dpi)

        for i in arange(0, data_cutoff.shape[0], args.splitplots):
            if i == 0 and args.splitplots == data_cutoff.shape[0]:
                isuffix = ''
            else:
                isuffix = '_%s' % (i/args.splitplots)
            figure(1)
            clf()
            plot_mutxpos_image(data_cutoff[i:i+args.splitplots, I], sequence, seqpos_iter, offset, [mut_labels[k] for k in xrange(i, i+args.splitplots)])
            savefig('%s/real_data%s.png' % (prefix, isuffix), dpi=args.dpi)

            figure(1)
            clf()
            plot_mutxpos_image(data_pred[i:i+args.splitplots], sequence, seqpos_iter, offset, [mut_labels[k] for k in xrange(i, i+args.splitplots)], vmax=data_cutoff[i:i+args.splitplots,:].mean())
            savefig('%s/reeffit_data_pred%s.png' % (prefix, isuffix), dpi=args.dpi)

            figure(1)
            clf()
            plot_mutxpos_image(data_pred[i:i+args.splitplots], sequence, seqpos_iter, offset, [mut_labels[k] for k in xrange(i, i+args.splitplots)], 
                    #missed_indices=missed_indices, contact_sites=fa.contact_sites, weights=W_fa[i:i+args.splitplots,:])
                    missed_indices=missed_indices, weights=W_fa[i:i+args.splitplots,:])
            savefig('%s/data_pred_annotated%s.png' % (prefix,isuffix), dpi=args.dpi)

            """
            figure(1)
            clf()
            imshow(C[i:i+args.splitplots,:], cmap=get_cmap('jet'), interpolation='nearest')
            savefig('%s/contacts%s.png' % (prefix,isuffix), dpi=args.dpi)
            """
            for s in xrange(len(structures)):
                figure(1)
                clf()
                imshow(E_c_fa[i:i+args.splitplots,s,:] - fa.calculate_data_pred(no_contacts=True)[0], aspect='auto', interpolation='nearest')
                xticks(range(len(seqpos_iter)), ['%s%s' % (pos, sequence[pos - offset - 1]) for pos in seqpos_iter], fontsize='xx-small', rotation=90)
                ml = [mut_labels[k] for k in xrange(i, i+args.splitplots)]
                yticks(range(len(ml)), ml, fontsize='xx-small')
                savefig('%s/E_c_%s_structure_%s.png' % (prefix,isuffix,s), dpi=args.dpi)


            figure(3)
            clf()
            ax = subplot(111)
            weights_by_mutant_plot(W_fa[i:i+args.splitplots,:], W_fa_std[i:i+args.splitplots,:], [mut_labels[k] for k in xrange(i, i+args.splitplots)])
            savefig('%s/weights_by_mutant%s.png' % (prefix,isuffix), dpi=args.dpi)

            if args.detailedplots:
                # Plot Log-likelihood trace
                figure(3)
                clf()
                plot(loglikes, linewidth=2)
                scatter(Psi_reinits, [loglikes[k] for k in Psi_reinits], label='$Psi$ reinit.')
                ylabel('log-likelihood')
                xlabel('iteration')
                savefig('%s/loglike_trace.png' % (prefix), dpi=args.dpi)
                # Plot weights by mutant by structure, compared to initial weight values
                for j in xrange(nstructs):
                    figure(3)
                    clf()
                    weights_by_mutant_plot(W_fa[i:i+args.splitplots,[j]], W_fa_std[i:i+args.splitplots,[j]], [mut_labels[k] for k in xrange(i, i+args.splitplots)], W_ref=W_0[i:i+args.splitplots,[j]], idx_offset=j)
                    savefig('%s/weights_by_mutant_structure%s_%s.png' % (prefix, isuffix, j), dpi=100)

                if b_iter == 0:
                    # Plot prior histograms
                    x = linspace(0,8,100)
                    figure(4)
                    clf()
                    hist(unpaired_data, 100, alpha=0.6, normed=1)
                    plot(x, fa.unpaired_pdf(x), 'r', linewidth=2)
                    ylim(0, max(fa.unpaired_pdf(x)) + 0.5)
                    savefig('%sprior_unpaired.png' % args.outprefix, dpi=args.dpi)
                    figure(4)
                    clf()
                    hist(paired_data, 100, alpha=0.6, normed=1)
                    plot(x, fa.paired_pdf(x), 'r', linewidth=2)
                    ylim(0, max(fa.paired_pdf(x)) + 0.5)
                    savefig('%sprior_paired.png' % args.outprefix, dpi=args.dpi)
                    # Plot data vs predicted for each measurement
                    r = range(data_cutoff.shape[1])
                    for i in xrange(data_cutoff.shape[0]):
                        f = figure(5)
                        f.set_size_inches(15, 5)
                        clf()
                        plot(r, data_cutoff[i,I], color='r', label='Data', linewidth=2)
                        errorbar(r, data_pred[i,:], yerr=sigma_pred[i,:], color='b', label='Predicted', linewidth=2)
                        title('%s' % mut_labels[i])
                        xticks(r[0:len(r):5], seqpos_iter[0:len(seqpos_iter):5], rotation=90)
                        ylim(0,3)
                        legend()
                        savefig('%s/data_vs_predicted_%s_%s.png' % (prefix, i, mut_labels[i]))

if args.bootstrap > 0:
    print 'Compiling bootstrap results'

    E_dcompile = zeros([nstructs, npos, args.bootstrap])

    Wcompile = zeros([nmeas, nstructs, args.bootstrap])

    for b_iter in xrange(args.bootstrap):
        dirname = '%s/boot%s' % (args.outprefix, b_iter + 1)
        Wcompile[:,:,b_iter] = pickle.load(open('%s/W.pickle' % (dirname)))
        #E_dcompile[:,:,b_iter] = pickle.load(open('%s/E_d.pickle' % (dirname)))


    Wcompile_std = Wcompile.std(axis=2)
    Wcompile_mean = Wcompile.mean(axis=2)
    """
    E_dcompile_std = E_dcompile.std(axis=2)
    E_dcompile_mean = E_dcompile.mean(axis=2)
    
    r = range(data_cutoff.shape[1])
    for i, s in enumerate(selected_structures):
        f = figure(2)
        f.set_size_inches(15, 5)
        clf()
        title('Structure %s: %s' % (i, s))
        expected_reactivity_plot(E_dcompile_mean[i,:], fa.structures[s], yerr=E_dcompile_std[i,:])
        xticks(r[0:len(r):5], seqpos_cutoff[0:len(seqpos_cutoff):5], rotation=90)
        savefig('%s/bootstrap_exp_react_struct_%s.png' % (args.outprefix, i), dpi=args.dpi)

    """
    if not args.worker:
        figure(3)
        clf()
        ax = subplot(111)
        weights_by_mutant_plot(Wcompile_mean, Wcompile_std, mut_labels)
        savefig('%s/bootstrap_weights_by_mutant.png' % (args.outprefix), dpi=args.dpi)

        if args.detailedplots:
            # Plot weights by mutant by structure, compared to initial weight values
            for j in xrange(nstructs):
                figure(3)
                clf()
                weights_by_mutant_plot(Wcompile_mean, Wcompile_std, mut_labels, W_ref=W_0, idx_offset=j)
                savefig('%s/bootstrap_weights_by_mutant_structure_%s.png' % (args.outprefix, j), dpi=100)
        print 'Printing bootstrap report'
        report = open('%s/bootstrap_report.txt' % args.outprefix,'w')
        if wt_idx != -1:
            report.write('Structure index\tStructure\tWild type weight\tWeight std\n')
        else:
            report.write('No Wild type found in data set\n')
            report.write('Structure index\tStructure\tSeq 0 weight\tSeq 0 std\n')
        for i, s in enumerate(selected_structures):
            report.write('%s\t%s\t%.3f\t%.3f\n' % (i, fa.structures[s], Wcompile_mean[0,i], Wcompile_std[0,i]))
print 'Performing post 1D secondary structure modeling'
if args.postmodel:
    if args.worker:
        print 'Post 1D structure modeling has no effect in worker mode!'
    else:
        new_structures = []
        for i in xrange(E_d_fa.shape[0]):
            md = mapping.MappingData(data=E_d_fa[i,:], enforce_positives=True)
            new_structures.append(ss.fold(sequence, mapping_data=md, algorithm=args.priorweights)[0].dbn)
        open('%s/postmodel_structures.txt' % args.outprefix, 'w').write('\n'.join(new_structures))
        make_struct_figs(new_structures, 'postmodel_')
print 'Done!'
print '\a'
