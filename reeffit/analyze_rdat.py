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
import map_analysis_utils
from random import choice, sample
import mapping_analysis
from itertools import chain

parser = argparse.ArgumentParser()
parser.prog = 'reeffit'

# General options
parser.add_argument('rdatfile', type=argparse.FileType('r'), help='The RDAT file that has the multi-dimensional chemical mapping data')
parser.add_argument('outprefix', type=str, help='Prefix (e.g. directory) that all resulting reports, plot files, etc. will have')
parser.add_argument('--addrdatfiles', type=argparse.FileType('r'), nargs='+', default=[], help='Additional rdat files that contain the multi-dimensional chemical mapping data')
parser.add_argument('--priorweights', type=str, default='rnastructure', help='Algorithm to use for starting structure weights. Can be "rnastructure", "viennarna", and "uniform"')
parser.add_argument('--njobs', type=int, default=None, help='For soft EM analysis. Number of parallel jobs to run the E-step on')

# Input structures options
parser.add_argument('--structfile', default=None, type=argparse.FileType('r'), help='Text files with structures to analyze, one per line, in dot-bracket notation')
parser.add_argument('--clusterfile', default=None, type=argparse.FileType('r'), help='File with clusters of structures: one structure per line, with tab-delimited fields. Format is cluster_id, dot-bracket structure, comma-separated energies per cluster' )
parser.add_argument('--medoidfile', default=None, type=argparse.FileType('r'), help='File specifying the medoid structures of each cluster.')
parser.add_argument('--structset', default=None, type=str, help='Subset of structures in the specified structfile to use in the analysis. Each subset is identified by a "header" specified after a hash (#) preceding the set of structures. This option will search all headers for structset and analyze the data with those structures. Used in worker mode.')
parser.add_argument('--structlabelfile', default=None, type=argparse.FileType('r'), help='File specifying labels (names) for the structures.')

# Pre-processing options
parser.add_argument('--cutoff', default=0, type=int, help='Number of data points and nucleotides to cut off from the beginning and end of the data and sequences: these will not be used in the analysis. Useful to avoid saturated "outliers" in the data.')
parser.add_argument('--start', default=None, type=int, help='Seqpos starting position of the data and sequences from where to analyze. Useful to focus the analysis on a particular location. Must specify both start and end options.')
parser.add_argument('--end', default=None, type=int, help='Seqpos ending position of the data and sequences to analyze. Useful to focus the analysis on a particular location. Must specify both start and end options.')
parser.add_argument('--nomutrepeat', default=False, action='store_true', help='Skip repeating mutants')
parser.add_argument('--clipzeros', default=False, action='store_true', help='Clip data to non-zero regions')
parser.add_argument('--csize', type=int, default=3, help='Number of sequence positions to be allowed for contact sites.')
parser.add_argument('--boxnormalize', default=False, action='store_true', help='Perform box-plot normalization of the data. Useful for capillary sequencing datasets')
parser.add_argument('--nworkers', type=int, default=10, help='For cross validation, MC model selection or bootstrapping. Number of workers.')
parser.add_argument('--workerformat', type=str, default='sh', help='File format of worker files. Can be "sh" (for simple shell script) and "gridengine" (for use with grid engine')
parser.add_argument('--ntasks', type=int, default=100, help='Number of tasks (cross-validation parameters to try, bootstrap iterations, or Monte Carlo simulations) per worker')

# Model selection options
parser.add_argument('--modelselect', type=str, default=None, help='Model selection mode. Can be one of "sample" (recommended), "heuristic", "mc" (Monte Carlo, including MCMC).')
parser.add_argument('--nsubopt', default=100, type=int, help='For model selection. Number of maximum suboptimal structures to take from each sequence\'s structural ensemble in the RDAT file')
parser.add_argument('--greedyiter', default=10, type=int, help='For heuristic model selection. Number of greedy iterations in which REEFFIT tries to add more structures to make the model better.')
parser.add_argument('--nopriorswap', action='store_true', help='For heuristic model selection. Do not swap structure medoids in each cluster, take the default medoids found by centrality maximization')
parser.add_argument('--nopseudoenergies', action='store_true', help='For heuristic model selection. Do not use pseudoenergies (i.e. SHAPE-directed modeling) to score the structures. Structures will be scored using regular RNAstructure energies, with no data.')
parser.add_argument('--structest', default='hclust', type=str, help='For heuristic model selection. Method for estimating the number of structures underlying the data, and later used for clustering. Available methods are "hclust" (hierarchical clustering) and "fa" (standard, gaussian factor analysis)')
parser.add_argument('--worker', default=False, action='store_true', help='Worker mode (non-verbose, simple output). Used for MC model selection.')
parser.add_argument('--cvfold', default=0, type=int, help='Number of cross-validations to select the reactivity and weight regularization tuning parameters (lam_reacts and lam_weights)')
parser.add_argument('--crossvalidate', type=str, default=None, help='Prepare cross-validation for a regularization parameter')
parser.add_argument('--maxcvparam', type=int, default=15, help='For cross-validation setup. Maximum value of the cross-validation parameter to try.')
parser.add_argument('--maxlamweights', type=int, default=10, help='For cross-validation setup. Maximum value of lam_weights to try')
parser.add_argument('--msmaxsamples', type=int, default=inf, help='For MC model selection. Maximum number of structures per sample in MC simulation')
parser.add_argument('--interpreter', type=str, default='python', help='For MC model selection. Python interpreter to use for the worker files')
parser.add_argument('--nostructlabels', default=False, action='store_true', help='No labels (names) for the structures in PCA plot.')

# Fitting options
parser.add_argument('--nsim', default=1000, type=int, help='Number of simulations used for each E-step when performing soft EM.')
parser.add_argument('--refineiter', default=10, type=int, help='Maximum number of EM iterations to perform')
parser.add_argument('--hardem', default=False, action='store_true', help='Perform hard EM instead of soft EM, finding a MAP of the hidden reactivities, rather than simulating from the posterior. This makes REEFFIT run considerably faster, but does not yield a rigorous, posterior distribution estimation of the hidden reactivities of each structure.')
parser.add_argument('--energydelta', default=None, type=float, help='Kcal/mol free energy limit that the structure weights are allowed to deviate from the initial weights -- this is a hard limit, as opposed to lamweights option.')
parser.add_argument('--clusterdatafactor', type=int, default=None, help='For clustering the data to reduce dimensionality (useful for large datasets with lots of redundant measurements). Describes the approximate number of clusters for clustering the data')
parser.add_argument('--preparebootstrap', action='store_true', default=False, help='Prepare bootstrap worker files.')
parser.add_argument('--titrate', type=str, default=None, help='For morph-and-map experiments. Name of chemical titrated. Must also specify kdfile when using this option.')
parser.add_argument('--postmodel', default=False, action='store_true', help='Perform SHAPE-directed modeling after each EM iteration using the calculated hidden reactivities for each structures. Useful if the hidden reactivities do not match well with the prior structures')
parser.add_argument('--kdfile', default=None, type=argparse.FileType('r'), help='File with the dissociation constants for each structure, for titrating a chemical specified in the titrate option')
parser.add_argument('--lamreacts', default=0.0, type=float, help='Regularization parameter controlling the similarity between reactivities of similar structures. Higher values will force similar reactivities between structures, depending on their base pair distance.')
parser.add_argument('--lamweights', default=0.0, type=float, help='Regularization parameter controlling how far from the initial weight estimates (e.g. from RNAstructure or ViennaRNA). Higher values will force weights to be closer to initial estimates')
parser.add_argument('--lammut', default=5.0, type=float, help='Regularization parameter controlling similarities of delta delta Gs of weights of wild type and mutants to the ones calculated by the secondary structure algorithm (e.g. RNAstructure or ViennaRNA). Higher values will force delta delta G values between wild type and mutant structures to be similar.')
parser.add_argument('--lamridge', default=0.26, type=float, help='Regularization parameter controlling the signal strength (smoothened sparsity) of the weights. Higher values will penalize large weights.')
parser.add_argument('--lamfile', default=None, type=argparse.FileType('r'), help='File with results of a cross-validation run. Format is lines with cross_validation_error, lam_reacts, lam_weights; tab-delimited. Values with lowest cross_validation_error will be taken for values of lam_reacts and lam_weights [UNTESTED].')
parser.add_argument('--decompose', action='store_true', default=False, help='Decompose structures into a set of overlapping motifs to reduce number of variables to fit.')
parser.add_argument('--scalemethod', type=str, default='linear', help='Scaling method to perform after fits')
parser.add_argument('--seqindices', type=str, default=None, help='Sequence indices used in the fitting. Used mainly for bootstrapping.')
parser.add_argument('--compilebootstrap', action='store_true', default=False, help='Compile bootstrapping results. Will locate all boot* subdirectories in outprefix and extract the weights of each bootstrap iteration and compile a summary.')

# Plotting options
parser.add_argument('--splitplots', default=-1, type=int, help='Plot subsets of data and predicted data rather than the whole set')
parser.add_argument('--detailedplots', default=False, action='store_true', help='Plots log-likelihood trace, all predicted data vs real data separately, and comparison plots between initial and final structure weights')
parser.add_argument('--dpi', type=int, default=200, help='DPI resolution for plots')

args = parser.parse_args()

# Global variables
model_selection_names = {'mc':'Monte Carlo (includes MCMC)', 'heuristic': 'Heuristic', 'cv':'Cross-validation', 'sample':'Suboptimal structure sampling'}
worker_file_formats = ['gridengine', 'sh']
carry_on_options = ['nsim', 'refineiter', 'structest', 'clusterdatafactor',
        'decompose', 'cutoff', 'start', 'end', 'hardem', 'energydelta', 'titrate',
        'nomutrepeat', 'clipzeros', 'kdfile', 'boxnormalize', 'priorweights', 'njobs', 'csize', 'addrdatfiles', 'crossvalidate']
regparams = {'lamreacts':args.lamreacts, 'lamweights':args.lamweights, 'lammut':args.lammut, 'lamridge':args.lamridge}
MAX_STRUCTURES_PLOT = 10
rdatname = args.rdatfile.name[args.rdatfile.name.rfind('/')+1:].split('_')[0]

idxoffset = 0
nwarnings = 0

# Helper functions
def valid(seq):
    return 'G' in seq.upper() or 'C' in seq.upper() or 'A' in seq.upper() or 'U' in seq.upper()

def make_struct_figs(structures, fprefix, indices=None, base_annotations=None, helix_function=lambda x,y:x, helix_fractions=None, annotation_color='#FF0000'):
    options = {'drawBases':False, 'fillBases':False, 'resolution':'10.0', 'flat':True, 'offset':offset + seqpos_start}
    if indices == None:
        indices = range(len(structures))
    for i, s in enumerate(structures):
        print s
        options['bp'] = rgb2hex(STRUCTURE_COLORS[i])
        varna = VARNA(sequences=[mutants[wt_idx]], structures=[ss.SecondaryStructure(dbn=remove_non_cannonical(s, mutants[0]))])
        if base_annotations == None:
            CMD = varna.render(output=args.outprefix + fprefix + 'structure%s.svg' % indices[i], annotation_by_helix=True, helix_function=helix_function, cmd_options=options)
        else:
            varna.annotation_font_size = 13
            varna.annotation_color = annotation_color
            if helix_fractions == None:
                helix_frac_annotations = ''
                base_weight_annotations = varna._get_base_annotation_string([base_annotations[i]], annotation_by_helix=True, helix_function=helix_function)
            else:
                helix_frac_annotations = varna._get_base_annotation_string([helix_fractions[i]], annotation_by_helix=True, helix_function=helix_function, stype='B', helix_side=0)
                #varna.annotation_color = '#0033CC'
                #base_weight_annotations = varna._get_base_annotation_string([base_annotations[i]], annotation_by_helix=True, helix_function=helix_function, stype='B', helix_side=0, base_offset=-2)
            options['annotations'] = helix_frac_annotations.strip('"')
            #options['annotations'] += base_weight_annotations.strip('"')
            CMD = varna.render(output=args.outprefix + fprefix + 'structure%s.svg' % indices[i], annotation_by_helix=True, helix_function=helix_function, cmd_options=options)
        print CMD
        os.system(CMD)

def carry_on_option_string():
    options = ''
    for opt in carry_on_options:
        val = args.__dict__[opt]
        if val != None:
            if type(val) == bool:
                if val:
                    options += ' --%s ' % opt
            elif type(val) == list:
                if len(val) > 0:
                    if opt == 'addrdatfiles':
                        options += ' --%s %s' % (opt, ' '.join([os.path.abspath(x.name) for x in val]))
                    else:
                        options += ' --%s %s' % (opt, ' '.join([str(x) for x in val]))
            else:
                options += ' --%s=%s ' % (opt, val)

    for k, v in regparams.iteritems():
        if k != args.crossvalidate:
            options += ' --%s=%s' % (k, v)

    return options

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

def prepare_mc_worker_file(idx, nsim, simfilename):
    wf = open('%smc_worker%s.sh' % (args.outprefix, idx), 'w')
    general_options = '%s %sworker_%s --worker --structfile=%s' % (os.path.abspath(args.rdatfile.name), args.outprefix, idx, os.path.abspath(simfilename))
    general_options += carry_on_option_string()
    for i in xrange(nsim):
        wf.write('%s %s %s --structset=%s\n' % (args.interpreter, os.environ['REEFFIT_HOME'] + '/reeffit/analyze_rdat.py ', general_options, i))
    return wf.name

def prepare_cv_worker_file(idx, all_parameters):
    wf = open('%s%s_cv_worker%s.sh' % (args.outprefix, args.crossvalidate, idx), 'w')
    general_options = '%s %s --structfile %sall_structures.txt --worker' % (os.path.abspath(args.rdatfile.name), args.outprefix, args.outprefix)
    general_options += carry_on_option_string()
    for i in xrange(len(all_parameters)):
        wf.write('%s %s %s --cvfold %s --%s=%s\n' % (args.interpreter, os.environ['REEFFIT_HOME'] + '/reeffit/analyze_rdat.py ', general_options, args.cvfold, args.crossvalidate, all_parameters[i]))
    return wf.name

def prepare_bootstrap_worker_file(idx, all_indices, idxoffset):
    wf = open('%sbootstrap_worker%s.sh' % (args.outprefix, idx), 'w')
    for i in xrange(len(all_indices)):
        general_options = '%s %s/boot%s/ --structfile %s --worker' % (os.path.abspath(args.rdatfile.name), args.outprefix, i + idxoffset, os.path.abspath(args.structfile.name))
        general_options += carry_on_option_string()
        wf.write('%s %s %s --seqindices="%s"\n' % (args.interpreter, os.environ['REEFFIT_HOME'] + '/reeffit/analyze_rdat.py ', general_options, ','.join([str(x) for x in all_indices[i]])))
    return wf.name

def write_worker_master_script(workerfiles, wtype):
    mwf = open('%smaster_script_%s.sh' % (args.outprefix, wtype), 'w')
    mwf.write('#!/bin/bash\n')
    mwf.write('if [ "$1" = "execute" ]\nthen\n')
    mwf.write('\t' + ' &\n\t'.join(['sh %s' % w for w in workerfiles]) + ' & \n')
    general_options = '%s %s/ --structfile %s' % (os.path.abspath(args.rdatfile.name), args.outprefix, os.path.abspath(args.structfile.name))
    general_options += carry_on_option_string()
    if wtype == 'bootstrap':
        mwf.write('elif [ "$1" = "compile" ]\nthen\n')
        general_options += ' --compilebootstrap '
        mwf.write('\t%s %s %s\n' % (args.interpreter, os.environ['REEFFIT_HOME'] + '/reeffit/analyze_rdat.py ', general_options))
        mwf.write('else\n\techo "Option $1 not recognized, must be either \'execute\' or \'compile\'"\n')
    """
    if wtype == 'cv':
        general_options += ' --lamfile=%scross_validation_results.txt ' % args.outprefix
        mwf.write('\t%s %s %s\n' % (args.interpreter, os.environ['REEFFIT_HOME'] + '/reeffit/analyze_rdat.py ', general_options))
    """
    mwf.write('fi')
    mwf.write('\nwait\n')
    return mwf.name

def print_worker_options():
    if args.workerformat not in worker_file_formats:
        print 'Unrecognized worker file format %s, aborting!' % args.workerformat
        exit()
    print 'Preparing worker files'
    print 'Number of worker files %s' % args.nworkers
    print 'Number of samples per file %s' % args.ntasks
    print 'Worker file format is %s' % args.workerformat



for rdatidx, rdatfile in enumerate([args.rdatfile] + args.addrdatfiles):

    print 'Parsing RDAT %s' % rdatfile.name
    rdat = RDATFile()
    rdat.load(rdatfile)
    construct = rdat.constructs.values()[0]

    seqpos = construct.seqpos
    offset = construct.offset
    sequence = construct.sequence[min(seqpos) - offset - 1:max(seqpos) - offset].upper()
    if rdatidx == 0:
        sequence = construct.sequence.upper()
        sorted_seqpos = sorted(seqpos)
        data = []
        mut_labels = []
        mutants = []
        mutpos = []
        wt_idx = -1
        wt_indices = []
        use_struct_clusters = False
        concentrations = []
        kds = []
        last_seqpos = 0



    print 'Parsing mutants and data'

    for idx, d in enumerate(construct.data):
        if 'warning' in d.annotations:
            nwarnings += 1
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
            label = ';'.join(d.annotations['mutation'])
        else:
            label = 'WT'
        pos = []
        if label == 'WT':
            if 'sequence' in d.annotations:
                mutant = d.annotations['sequence'][0]
                pos = []
                if not valid(sequence):
                    sequence = mutant
                for i, ms in enumerate(mutant):
                    if ms != sequence[i] and len(pos) < 1:
                        pos.append(i)
                        if label == 'WT':
                            label = '%s%s%s' % (sequence[i], i + construct.offset + 1, mutant[i])
            if pos == []:
                mutant = sequence
                wt_indices.append(idx + idxoffset - nwarnings)

        else:
            if 'sequence' in d.annotations:
                mutant = d.annotations['sequence'][0]
                pos = []
                if not valid(sequence):
                    sequence = mutant
                for i, ms in enumerate(mutant):
                    if ms != sequence[i] and len(pos) < 1:
                        pos.append(i)
            else:
                mutant = sequence
                pos = []
                for mutation in d.annotations['mutation']:
                    if mutation != 'WT':
                        pos.append(int(mutation[1:len(mutation)-1]) - 1 - construct.offset)
                        mutant = mutant[:pos[0]] + mutation[-1] + mutant[pos[0]+1:]
        mutpos.append(pos)
        if mutant in mutants and args.nomutrepeat:
            continue
        mut_labels.append(label)
        mutants.append(mutant)
        if not args.boxnormalize:
            nd_tmp = normalize([d.values[seqpos.index(i)] for i in sorted_seqpos])
            nd = [d.values[seqpos.index(i)] if d.values[seqpos.index(i)] >= 0 else 0.001 for i in sorted_seqpos]
            #nd = [nd[i] if nd[i] < 4 else max(nd_tmp) for i in range(len(nd))]
            #nd = [d.values[seqpos.index(i)] for i in sorted_seqpos]
        else:
            nd = normalize([d.values[seqpos.index(i)] for i in sorted_seqpos])
        if args.clipzeros:
            nd_last_sepos = len(seqpos)
            for i in xrange(len(nd)-1, -1, -1):
                if nd[i] != 0:
                    break
                nd_last_seqpos = i
            last_seqpos = max(nd_last_seqpos, last_seqpos)
        data.append(nd)
    idxoffset += len(construct.data)

wt_idx = wt_indices[0]
if args.clipzeros:
    sorted_seqpos = sorted_seqpos[:last_seqpos]
    data = array(data)[:,:last_seqpos]
else:
    data = array(data)

for i in xrange(data.shape[0]):
    for j in xrange(data.shape[1]):
        if data[i,j] == 0:
            data[i,j] = rand()*0.001

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
    mutpos_cutoff = [[p - args.cutoff for p in pos] for pos in mutpos]
    seqpos_range = (seqpos_start, seqpos_end)
elif args.start != None or args.end != None:
    if not (args.start and args.end):
        print 'Must specify both sequence range start and end!'
        exit()
    seqpos_start = sorted_seqpos.index(args.start)
    seqpos_end = sorted_seqpos.index(args.end) + 1
    seqpos_cutoff = sorted_seqpos[seqpos_start:seqpos_end]
    data_cutoff = data[:, seqpos_start:seqpos_end]
    mutpos_cutoff = [[p - seqpos_start for p in pos] for pos in mutpos]
    seqpos_range = (seqpos_start, seqpos_end)
else:
    data_cutoff = data
    seqpos_cutoff = sorted_seqpos
    seqpos_start = 0
    seqpos_end = len(seqpos_cutoff)
    mutpos_cutoff = mutpos
    seqpos_range = None

if args.crossvalidate != None and not args.worker:
    print_worker_options()
    print 'Preparing worker files for cross-validation tuning of parameter %s' % args.crossvalidate
    parameter_grid = []
    tot_num_tries = args.ntasks*args.nworkers
    param_delta = args.maxcvparam/float(tot_num_tries)
    print param_delta
    workerfiles = []
    for i in arange(0, args.maxcvparam, param_delta):
        parameter_grid.append(i)
    for i in xrange(args.nworkers):
        workerfiles.append(prepare_cv_worker_file(i, parameter_grid[i*args.ntasks:i*args.ntasks+args.ntasks]))
    write_worker_master_script(workerfiles, '%s_cv' % args.crossvalidate)
    print 'Finished preparing worker files!'
    exit()

if args.preparebootstrap:
    print_worker_options()
    bootstrap_range = range(data_cutoff.shape[1])
    workerfiles = []
    idxoffset = 0
    for idx in xrange(args.nworkers):
        indices = []
        for i in xrange(args.ntasks):
            I = [choice(bootstrap_range) for j in bootstrap_range]
            indices.append(I)
        workerfiles.append(prepare_bootstrap_worker_file(idx, indices, idxoffset))
        idxoffset += len(indices)
    print 'Finished preparing bootstrap worker files, run them using your scheduler and them use then rerun REEFFIT replacing the preparebootstrap option with the compilebootstrap option!'
    write_worker_master_script(workerfiles, 'bootstrap')
    exit()

if args.lamfile != None:
    print 'Getting best tunning parameters, lam_reacts and lam_weights, from cross-validation file'
    min_cv_error = inf
    for line in args.lamfile.readlines():
        cv_error, lam_reacts, lam_weights = [float(f) for f in line.strip().split('\t')]
        if cv_error < min_cv_error:
            min_cv_error = cv_error
            args.lam_reacts = lam_reacts
            args.lam_weights = lam_weights
    print 'Parameters were lam_reacts=%s, lam_weights=%s' % (args.lam_reacts, args.lam_weights)

if args.structfile != None:
    if args.clusterfile != None:
        print 'Both structure and structure cluster were specified! Need only one of the two to run on single structure or cluster modalities'
        exit()
    else:
        if args.structset != None:
            structures = []
            original_structures = []
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
                        original_structures.append(line.strip().replace('A', '.'))

        else:
            structures = [line.strip().replace('A', '.') for line in args.structfile.readlines() if line[0] != '#']
            original_structures = list(structures)
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
"""
all_struct_file = open('%sall_structures.txt' % args.outprefix, 'w')
all_struct_file.write('\n'.join(structures))
all_struct_file.close()
exit()
"""
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
#if args.modelselect != None:
#    #energies = get_free_energy_matrix(structures,[m[seqpos_start:seqpos_end] for m in  mutants], algorithm=args.priorweights)
#    energies = get_free_energy_matrix(structures,[m for m in  mutants], algorithm=args.priorweights)
if args.modelselect == None or args.modelselect == 'heuristic':
    energies = get_free_energy_matrix(structures, mutants, algorithm=args.priorweights)
if args.medoidfile:
    struct_medoids = [s.strip() for s in args.medoidfile]
    struct_medoid_indices = [-1]*len(struct_medoids)
    for sm in struct_medoids:
        if sm not in structures:
            structures.append(sm)
    for i, sm in enumerate(struct_medoids):
        for j, s in enumerate(structures):
            if s == sm:
                struct_medoid_indices[i] = j

fa = mapping_analysis.FAMappingAnalysis(data, structures, mutants, mutpos=mutpos_cutoff, concentrations=concentrations, kds=kds, seqpos_range=seqpos_range, c_size=args.csize, njobs=args.njobs, lam_reacts=args.lamreacts, lam_weights=args.lamweights, lam_mut=args.lammut, lam_ridge=args.lamridge)
if args.decompose:
    fa.perform_motif_decomposition()
unpaired_data, paired_data = fa.set_priors_by_rvs(SHAPE_unpaired_sample, SHAPE_paired_sample)

if args.modelselect != None:
    print 'Model selection using %s' % model_selection_names[args.modelselect]
    if args.modelselect == 'mc':
        print_worker_options()
        for i in xrange(args.nworkers):
            sfname = prepare_model_select_simulation_structure_file(i, args.ntasks)
            prepare_ms_worker_file(i, args.ntasks, sfname)
        print 'Finished preparing worker files, run them using your scheduler and them use compile_worker_results.py to compile the results'
        write_worker_master_script(workerfiles, 'mc')
        exit()
    if args.modelselect == 'sample':
        allstructfile = open(args.outprefix + 'all_structures.txt', 'w')
        allstructfile.write('\n'.join(structures))
        print 'Done, check out the structures in %s' % (allstructfile.name)
        exit()

    if args.modelselect == 'heuristic':
        fa.energies = energies
        selected_structures, assignments = fa.model_select(greedy_iter=args.greedyiter, max_iterations=2, prior_swap=not args.nopriorswap, expstruct_estimation=args.structest, G_constraint=args.energydelta, apply_pseudoenergies=not args.nopseudoenergies, algorithm=args.priorweights, hard_em=args.hardem, method=args.modelselect, post_model=args.postmodel)
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
        savefig(args.outprefix + 'pca_cluster_plot_ms.png', dpi=300)
        print 'Done, check out cluster file %s and medoids file %s' % (outclustfile.name, outmedoidsfile.name)
        exit()
else:
    selected_structures = range(len(structures))

if args.clusterfile:
    fa.set_structure_clusters(structure_clusters, struct_medoid_indices, struct_weights_by_clust=struct_weights_by_clust)
    W_0 = fa.W
else:
    W_0 = calculate_weights(energies)


fa.energies = energies
nstructs = len(selected_structures)
nmeas =  data_cutoff.shape[0]
npos = data_cutoff.shape[1]

#If we are doing cross validation, then do the cross validation and exit
if args.cvfold > 0:
    print 'Performing %s-fold cross validation for parameters %s' % (args.cvfold, regparams)
    fold_sets = [arange(i, data_cutoff.shape[1], args.cvfold) for i in xrange(args.cvfold)]
    cv_errors = []
    data_comp = data.copy()
    data_comp[data_comp >= data_comp.mean()] = 1
    for fold in xrange(args.cvfold):
        cv_indices = [x for x in chain(*[fold_sets [i] for i in xrange(args.cvfold) if i != fold])]
        pred_indices = fold_sets[fold]

        fa_cv = mapping_analysis.FAMappingAnalysis(data, structures, mutants, mutpos=mutpos_cutoff, energies=energies, concentrations=concentrations, kds=kds, seqpos_range=seqpos_range, c_size=args.csize, lam_reacts=args.lamreacts, lam_weights=args.lamweights, lam_mut=args.lammut, lam_ridge=args.lamridge, njobs=args.njobs)
        if args.decompose:
            fa_cv.perform_motif_decomposition()
        lhood_traces, W_cv, W_fa_std, Psi_fa, E_d_fa, E_c_fa, sigma_d_fa, E_ddT_fa, M_fa, post_structures = fa_cv.analyze(max_iterations=args.refineiter, nsim=args.nsim, G_constraint=args.energydelta, cluster_data_factor=args.clusterdatafactor, use_struct_clusters=use_struct_clusters, seq_indices=cv_indices, return_loglikes=True, hard_em=args.hardem, post_model=args.postmodel)

        fa_cv = mapping_analysis.FAMappingAnalysis(data, structures, mutants, mutpos=mutpos_cutoff, energies=energies, concentrations=concentrations, kds=kds, seqpos_range=seqpos_range, c_size=args.csize, lam_reacts=args.lamreacts, lam_weights=args.lamweights, lam_mut=args.lammut, lam_ridge=args.lamridge, njobs=args.njobs)
        if args.decompose:
            fa_cv.perform_motif_decomposition()
        lhood_traces, W_fa, W_fa_std, Psi_fa, E_d_fa, E_c_fa, sigma_d_fa, E_ddT_fa, M_fa, post_structures = fa_cv.analyze(max_iterations=1, W0=W_cv, nsim=args.nsim, G_constraint=args.energydelta, cluster_data_factor=args.clusterdatafactor, use_struct_clusters=use_struct_clusters, seq_indices=pred_indices, return_loglikes=True, hard_em=args.hardem, post_model=args.postmodel)

        data_pred_cv, sigma_pred_cv = fa_cv.calculate_data_pred()
        data_pred_cv[data_pred_cv >= data_pred_cv.mean()] = 1
        err = asarray(data_comp[:, pred_indices] - asarray(data_pred_cv))**2/asarray(sigma_pred_cv)**2
        """
        print 'Max'
        print err.max()
        print 'Min'
        print err.min()
        print 'Sum'
        print err.sum()
        print 'Mean'
        print err.mean()
        figure(1)
        plot_mutxpos_image(data_pred_cv, sequence, pred_indices, 0, mut_labels)
        figure(2)
        plot_mutxpos_image(data_comp[:, pred_indices], sequence, pred_indices, 0, mut_labels)
        show()
        print 'Corrected error'
        #corr_err = array([((e - err.mean())**4)/(err.std()**2) for e in err]).mean() - 3
        m =  median(err)
        corr_err = err[err >= m].sum()
        print corr_err
        exit()
        """
        cv_errors.append(err.mean())
    report = open('%s%s_cv_results.txt' % (args.outprefix, args.crossvalidate),'a')
    report.write('%s\t%s\n' % (array(cv_errors).mean(), '\t'.join(['%s=%s' % (k, v) for k, v in regparams.iteritems()])))
    exit()


if args.medoidfile != None:
    struct_medoids = [s.strip() for s in args.medoidfile]
    medoid_dict = {}
    assignments = defaultdict(list)
    for i, s in enumerate(structures):
        mindist = inf
        for j in struct_medoid_indices:
            dist = map_analysis_utils._mutinf([s[i] for s in fa.struct_types], [s[j] for s in fa.struct_types])
            if dist < mindist:
                mindist = dist
                clust_idx = j
        assignments[clust_idx].append(i)
    for j in struct_medoid_indices:
        medoid_dict[j] = j

else:
    medoid_dict, assignments = cluster_structures(fa.struct_types, structures=structures)

   
# Set sequence indices, if bootstrapping, use input sequence indices
# otherwise, use all indices
if args.seqindices != None:
    if not os.path.exists(args.outprefix):
        os.mkdir(args.outprefix)
    I = [int(x) for x in args.seqindices.split(',')]
else:
    I = arange(data_cutoff.shape[1])

# prefix, short for args.outprefix for now on
prefix = args.outprefix
seqpos_iter = [seqpos_cutoff[i] for i in I]

# Perform the analysis

lhood_traces, W_fa, W_fa_std, Psi_fa, E_d_fa, E_c_fa, sigma_d_fa, E_ddT_fa, M_fa, post_structures = fa.analyze(max_iterations=args.refineiter, nsim=args.nsim, G_constraint=args.energydelta, cluster_data_factor=args.clusterdatafactor, use_struct_clusters=use_struct_clusters, seq_indices=I, return_loglikes=True, hard_em=args.hardem, post_model=args.postmodel)

# Get the most frequent medoid per structure cluster
if args.medoidfile != None:
    maxmedoids = medoid_dict.keys()
else:
    maxmedoids = [where(W_fa[wt_idx,:] == W_fa[wt_idx,stindices].max())[0][0] for stindices in assignments.values()]

# Plot the medoids using VARNA
if not args.worker:
    if len(structures) > MAX_STRUCTURES_PLOT:
        make_struct_figs([structures[i] for i in maxmedoids], '', indices=maxmedoids)
    else:
        make_struct_figs(structures, '')

# If we used motif decomposition, plot the number of motifs used per sequence position
if args.decompose:
    figure(1)
    clf()
    title('Number of variables fitted per position\n (number of motifs per position)')
    bar(0.5+arange(len(fa.nmotpos)), fa.nmotpos, linewidth=0, color='lightblue')
    xticks(range(len(seqpos_cutoff)), ['%s%s' % (pos, sequence[pos - offset - 1]) for pos in seqpos], fontsize='xx-small', rotation=90)
    savefig('%s/number_of_variables_per_sequence_position.png' % prefix, dpi=args.dpi)



# Change structures to post facto SHAPE-directed modeling structures
# (if no postmodeling option, post_structures are the same as structures)
if args.postmodel:
    structures = post_structures

# Get the likelihood traces and points where the covariance reaches negative values
# and needs to be reinitialized
loglikes, Psi_reinits = lhood_traces

# Get the selected structures on the clusters if we are provided a structure cluster file
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

print 'Saving data'
if args.worker:
    worker_dict = {'W':W_fa, 'E_d':E_d_fa, 'data_pred':data_pred, 'sigma_pred':sigma_pred, 'Psi':Psi_fa, 'data':fa.data}
    if args.structset != None:
        pickle.dump(worker_dict, open('%s_%s_dict.pickle' % (prefix, args.structset), 'w'))
    else:
        pickle.dump(worker_dict, open('%sworker_dict.pickle' % prefix, 'w'))
    if args.seqindices != None:
        pickle.dump(I, open('%sbootstrap_indices.pickle' % prefix, 'w'))
else:
    pickle.dump(fa.data, open('%sdata.pickle' % prefix, 'w'))
    pickle.dump(W_fa_std, open('%sW_std.pickle' % prefix, 'w'))
    pickle.dump(Psi_fa, open('%sPsi.pickle' % prefix, 'w'))
    pickle.dump(E_c_fa, open('%sE_c.pickle' % prefix, 'w'))
    pickle.dump(E_ddT_fa, open('%sE_ddT.pickle' % prefix, 'w'))
    pickle.dump(M_fa, open('%sM.pickle' % prefix, 'w'))
    pickle.dump(data_pred, open('%sdata_pred.pickle' % prefix, 'w'))
    pickle.dump(sigma_pred, open('%ssigma_pred.pickle' % prefix, 'w'))
    pickle.dump(W_fa, open('%sW.pickle' % prefix, 'w'))
    pickle.dump(E_d_fa, open('%sE_d.pickle' % prefix, 'w'))


if args.worker:
    report = open('%s_results.txt' % prefix,'a')
    report.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (max(loglikes), chi_sq, rmsea, aic, ','.join(original_structures), ','.join([str(w) for w in W_fa[wt_idx,:].tolist()])))
    report.close()
    if args.seqindices != None:
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
            plot_mutxpos_image(data_pred[i:i+args.splitplots, I], sequence, seqpos_iter, offset, [mut_labels[k] for k in xrange(i, i+args.splitplots)], vmax=data_cutoff[i:i+args.splitplots,:].mean())
            savefig('%s/reeffit_data_pred%s.png' % (prefix, isuffix), dpi=args.dpi)

            figure(1)
            clf()
            if len(structures) > MAX_STRUCTURES_PLOT:
                weights_by_mutant_plot(W_fa[i:i+args.splitplots,:], W_fa_std[i:i+args.splitplots,:], [mut_labels[k] for k in xrange(i, i+args.splitplots)], assignments=assignments, medoids=maxmedoids)
            else:
                weights_by_mutant_plot(W_fa[i:i+args.splitplots,:], W_fa_std[i:i+args.splitplots,:], [mut_labels[k] for k in xrange(i, i+args.splitplots)])

            savefig('%s/weights_by_mutant%s.png' % (prefix,isuffix), dpi=args.dpi)


    print 'Worker results were appended to %s' % report.name
else:
    print 'Printing report'
    report = open('%s/report.txt' % prefix,'w')
    report.write('Arguments: %s\n' % args)
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
    r = arange(data_cutoff.shape[1])

    # Structure cluster landscape plot for wild type
    if args.nostructlabels:
        struct_labels = None
    else:
        struct_labels = [ '%s_%s' % (rdatname.upper(), m) for m in maxmedoids]
        if args.structlabelfile:
            for line in args.structlabelfile.readlines():
                label, s = line.strip().split('\t')
                struct_labels[[structures[m] for m in maxmedoids].index(s)] = label
    PCA_structure_plot(structures, assignments, maxmedoids, weights=W_fa[wt_idx,:], names=struct_labels)
    savefig('%s/pca_landscape_plot_WT.png' % prefix, dpi=args.dpi)

    for widx in wt_indices:

        figure(1)
        clf()
        bpp_matrix_plot(structures, W_fa[widx,:], ref_weights=W_0[widx,:], weight_err=W_fa_std[wt_idx,:], offset=offset)
        savefig('%s/bppm_plot_WT%s.png' % (prefix, widx), dpi=args.dpi)

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


        figure(1)
        clf()
        unstruct_indices = []
        for selstruct, structidx in enumerate(selected_structures):
            for seqidx, struct in enumerate(structures[structidx]):
                if struct == '.':
                    unstruct_indices.append((selstruct,seqidx))
        plot_mutxpos_image(E_d_fa[:, I], sequence, seqpos_iter, offset, [str(k) for k in xrange(E_d_fa.shape[0])], missed_indices=unstruct_indices, aspect=None)
        ylabel('Structure')
        xlabel('Sequence position')
        savefig('%s/E_d%s.png' % (prefix, isuffix), dpi=args.dpi+100)

        # Plot weights

        overall_wt_fractions_file = open('%s/overall_wt_fractions.txt' % (args.outprefix), 'w')
        figure(3)
        clf()
        ax = subplot(111)
        if len(structures) > MAX_STRUCTURES_PLOT:
            weights_by_mutant_plot(W_fa[i:i+args.splitplots,:], W_fa_std[i:i+args.splitplots,:], [mut_labels[k] for k in xrange(i, i+args.splitplots)], assignments=assignments, medoids=maxmedoids)
            savefig('%s/weights_by_mutant%s.png' % (prefix,isuffix), dpi=args.dpi)
            # Plot weights by mutant by structure, compared to initial weight values
            for j in maxmedoids:
                figure(3)
                clf()
                weights_by_mutant_plot(W_fa[i:i+args.splitplots,:], W_fa_std[i:i+args.splitplots,:], [mut_labels[k] for k in xrange(i, i+args.splitplots)], W_ref=W_0[i:i+args.splitplots,:], idx=j, assignments=assignments, medoids=maxmedoids)
                savefig('%s/weights_by_mutant_structure%s_%s.png' % (prefix, isuffix, j), dpi=100)
                for structs in assignments.values():
                    if j in structs:
                        overall_wt_fractions_file.write('%s\t%s\t%s\n' % (j, W_fa[wt_idx, structs].sum(), sqrt((W_fa_std[wt_idx, structs]**2).sum())))


        else:
            weights_by_mutant_plot(W_fa[i:i+args.splitplots,:], W_fa_std[i:i+args.splitplots,:], [mut_labels[k] for k in xrange(i, i+args.splitplots)])
            savefig('%s/weights_by_mutant%s.png' % (prefix,isuffix), dpi=args.dpi)
            # Plot weights by mutant by structure, compared to initial weight values
            for j in xrange(nstructs):
                figure(3)
                clf()
                weights_by_mutant_plot(W_fa[i:i+args.splitplots,:], W_fa_std[i:i+args.splitplots,:], [mut_labels[k] for k in xrange(i, i+args.splitplots)], W_ref=W_0[i:i+args.splitplots,:], idx=j)
                savefig('%s/weights_by_mutant_structure%s_%s.png' % (prefix, isuffix, j), dpi=100)
                overall_wt_fractions_file.write('%s\t%s\t%s\n' % (j, W_fa[wt_idx, j], W_fa_std[wt_idx, j]))



        # Plot Log-likelihood trace
        figure(3)
        clf()
        plot(loglikes, linewidth=2)
        scatter(Psi_reinits, [loglikes[k] for k in Psi_reinits], label='$Psi$ reinit.')
        ylabel('log-likelihood')
        xlabel('iteration')
        savefig('%s/loglike_trace.png' % (prefix), dpi=args.dpi)

        # Individual plots for expected reactivities for medoid structures
        for s in maxmedoids:
            f = figure(2)
            f.set_size_inches(15, 5)
            clf()
            title('Structure %s: %s' % (s, structures[s]))
            if args.hardem:
                expected_reactivity_plot(E_d_fa[s,:], fa.structures[s], yerr=sigma_d_fa[s,:], seq_indices=I)
            else:
                expected_reactivity_plot(E_d_fa[s,:], fa.structures[s], yerr=sigma_d_fa[i,:]/sqrt(args.nsim), seq_indices=I)
            xticks(r[0:len(r):5] + 1, seqpos_iter[0:len(seqpos_iter):5], rotation=90)
            savefig('%s/exp_react_struct_%s.png' % (prefix, s), dpi=args.dpi)

            # Plot WT predicted vs real
            f = figure(5)
            f.set_size_inches(15, 5)
            clf()
            plot(r, data_cutoff[wt_idx,I], color='r', label='Data', linewidth=2)
            errorbar(r, data_pred[wt_idx,:], yerr=sigma_pred[wt_idx,:], color='b', label='Predicted', linewidth=2)
            title('WT')
            xticks(r[0:len(r):5], seqpos_iter[0:len(seqpos_iter):5], rotation=90)
            ylim(0,4)
            legend()
            savefig('%s/data_vs_predicted_WT.png' % prefix)



        if args.detailedplots:
            # Plot contact maps
            for s in xrange(len(structures)):
                figure(1)
                clf()
                imshow(E_c_fa[i:i+args.splitplots,s,:] - fa.calculate_data_pred(no_contacts=True)[0], aspect='auto', interpolation='nearest')
                xticks(range(len(seqpos_iter)), ['%s%s' % (pos, sequence[pos - offset - 1]) for pos in seqpos_iter], fontsize='xx-small', rotation=90)
                ml = [mut_labels[k] for k in xrange(i, i+args.splitplots)]
                yticks(range(len(ml)), ml, fontsize='xx-small')
                savefig('%s/E_c_%s_structure_%s.png' % (prefix,isuffix,s), dpi=args.dpi)
            # Invidual plots for expected reactivities
            for i, s in enumerate(selected_structures):
                f = figure(2)
                f.set_size_inches(15, 5)
                clf()
                title('Structure %s: %s' % (s, structures[s]))
                if args.hardem:
                    expected_reactivity_plot(E_d_fa[i,:], fa.structures[s], yerr=sigma_d_fa[i,:], seq_indices=I)
                else:
                    expected_reactivity_plot(E_d_fa[i,:], fa.structures[s], yerr=sigma_d_fa[i,:]/sqrt(args.nsim), seq_indices=I)
                xticks(r[0:len(r):5] + 1, seqpos_iter[0:len(seqpos_iter):5], rotation=90)
                savefig('%s/exp_react_struct_%s.png' % (prefix, i), dpi=args.dpi)


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
            # Plot all data vs predicted for each measurement
            r = arange(data_cutoff.shape[1])
            for i in xrange(data_cutoff.shape[0]):
                f = figure(5)
                f.set_size_inches(15, 5)
                clf()
                plot(r, data_cutoff[i,I], color='r', label='Data', linewidth=2)
                errorbar(r, data_pred[i,:], yerr=sigma_pred[i,:], color='b', label='Predicted', linewidth=2)
                title('%s' % mut_labels[i])
                xticks(r[0:len(r):5], seqpos_iter[0:len(seqpos_iter):5], rotation=90)
                ylim(0,4)
                legend()
                savefig('%s/data_vs_predicted_%s_%s.png' % (prefix, i, mut_labels[i]))
if args.compilebootstrap:
    print 'Compiling bootstrap results'

    nboot = 0

    Wcompile = zeros([nmeas, nstructs, 1])

    for fname in os.listdir(args.outprefix):
        dirname = args.outprefix + fname
        if 'boot' in fname and os.path.isdir(dirname):
            nboot += 1
            print 'Doing %s' % fname
            if os.path.exists('%s/worker_dict.pickle' % dirname):
                worker_dict = pickle.load(open('%s/worker_dict.pickle' % dirname))
                Wtmp = zeros([nmeas, nstructs, 1])
                Wtmp[:,:,0] = worker_dict['W']
                Wcompile = append(Wcompile, Wtmp, axis=2)

    # This has to be done due to some weirdness in how CVXOPT converts
    # numpy arrays to its own matrices...voodoo
    Wcompile = array(Wcompile.tolist())

    print 'Number of bootstrapping results found: %s' % nboot

    E_dcompile = zeros([nstructs, npos, nboot])

    Wcompile_std = Wcompile.std(axis=2)
    Wcompile_mean = Wcompile.mean(axis=2)

    pickle.dump(Wcompile, open('%sW_bootstrap.pickle' % args.outprefix, 'w'))


    if len(structures) > MAX_STRUCTURES_PLOT:
        base_annotations = []
        helix_fractions = []
        all_struct_objs  = [ss.SecondaryStructure(dbn=s) for s in structures]
        for m in maxmedoids:
            for structs in assignments.values():
                if m in structs:
                    struct_objs = [ss.SecondaryStructure(dbn=structures[s]) for s in structs]
                    bp_weights = ss.base_pair_fractions_in_structures(ss.SecondaryStructure(dbn=structures[m]), all_struct_objs, factors=Wcompile_mean[wt_idx, :])
                    bp_weights_std = ss.base_pair_fractions_in_structures(ss.SecondaryStructure(dbn=structures[m]), all_struct_objs, factors=Wcompile_std[wt_idx, :])
                    bp_fractions = ss.base_pair_fractions_in_structures(ss.SecondaryStructure(dbn=structures[m]), struct_objs)
                    bp_fractions_str = {}
                    bp_weights_str = {}
                    for bp in bp_fractions:
                        bp_fractions_str[bp] = ('%3.2f' % bp_fractions[bp]).rstrip('0').rstrip('.') + '%'
                    for bp in bp_weights:
                        bp_weights_str[bp] = ('%3.2f' % bp_weights[bp]).rstrip('0').rstrip('.') + '% +/-' + ('%3.2f' % sqrt(bp_weights_std[bp])).rstrip('0').rstrip('.')
                    base_annotations.append(bp_weights_str)
                    helix_fractions.append(bp_weights_str)
        def helix_function(x,y):
            extract_val = lambda v: float(v.split('%')[0])
            if extract_val(x) > extract_val(y):
                return x
            else:
                return y
        make_struct_figs([structures[m] for m in maxmedoids], 'bootstrap_', indices=maxmedoids, base_annotations=base_annotations, helix_fractions=helix_fractions, helix_function=helix_function)
        #make_struct_figs([structures[m] for m in maxmedoids], 'bootstrap_', base_annotations=base_annotations, helix_function=lambda x,y: '%3.2f%%' % max(float(x.strip('%')),float(y.strip('%'))))
        #make_struct_figs([structures[m] for m in maxmedoids], 'cluster_', base_annotations=helix_fractions, helix_function=lambda x,y: '%3.2f%%' % max(float(x.strip('%')),float(y.strip('%'))), annotation_color='#0033CC')


    fa.analyze(max_iterations=1, nsim=args.nsim, G_constraint=args.energydelta, cluster_data_factor=args.clusterdatafactor, use_struct_clusters=use_struct_clusters, hard_em=args.hardem, W0=Wcompile_mean)

    _, data_pred_boot, sigma_pred_boot = fa.correct_scale(stype=args.scalemethod)
    stats_boot = {}
    pickle.dump(data_pred_boot, open('%sdata_pred_bootstrap.pickle' % args.outprefix, 'w'))
    pickle.dump(sigma_pred_boot, open('%ssigma_pred_bootstrap.pickle' % args.outprefix, 'w'))
    stats_boot['chi_sq_df'], stats_boot['rmsea'], stats_boot['aic'] = fa.calculate_fit_statistics(data_pred=data_pred_boot, sigma_pred=sigma_pred_boot)
    pickle.dump(stats_boot, open('%sstatistics_bootstrap.pickle' % args.outprefix, 'w'))


    E_dcompile_std = fa.sigma_d
    E_dcompile_mean = fa.E_d



    bootfactor = sqrt(nboot)
    bootfactor = 1

    # Plot WT predicted vs real
    f = figure(5)
    f.set_size_inches(15, 5)
    clf()
    plot(r, data_cutoff[wt_idx,:], color='r', label='Data', linewidth=2)
    errorbar(r, data_pred_boot[wt_idx,:], yerr=sigma_pred_boot[wt_idx,:], color='b', label='Predicted', linewidth=2)
    title('WT')
    xticks(r[0:len(r):5], seqpos_iter[0:len(seqpos_iter):5], rotation=90)
    ylim(0,4)
    legend()
    savefig('%s/bootstrap_data_vs_predicted_WT.png' % args.outprefix)

    r = arange(data_cutoff.shape[1])
    for s in maxmedoids:
        f = figure(2)
        f.set_size_inches(15, 5)
        clf()
        title('Structure %s: %s' % (s, structures[s]))
        expected_reactivity_plot(E_dcompile_mean[s,:], fa.structures[s], yerr=E_dcompile_std[s,:]/bootfactor)
        xticks(r[0:len(r):5] + 1, seqpos_cutoff[0:len(seqpos_cutoff):5], rotation=90)
        savefig('%s/bootstrap_exp_react_struct_%s.png' % (args.outprefix, s), dpi=args.dpi)

    overall_wt_fractions_file = open('%s/bootstrap_overall_wt_fractions.txt' % (args.outprefix), 'w')

    # Plot weights and data pred
    for i in arange(0, data_cutoff.shape[0], args.splitplots):
        if i == 0 and args.splitplots == data_cutoff.shape[0]:
            isuffix = ''
        else:
            isuffix = '_%s' % (i/args.splitplots)
        
        # Plot bootstrap data pred
        figure(1)
        clf()
        plot_mutxpos_image(data_pred_boot[i:i+args.splitplots], sequence, seqpos_cutoff, offset, [mut_labels[k] for k in xrange(i, i+args.splitplots)], vmax=data_cutoff[i:i+args.splitplots,:].mean())
        savefig('%s/bootstrap_reeffit_data_pred%s.png' % (args.outprefix, isuffix), dpi=args.dpi)

        # Plot bootstrap weights
        figure(3)
        clf()
        ax = subplot(111)
        if len(structures) > MAX_STRUCTURES_PLOT:
            weights_by_mutant_plot(Wcompile_mean[i:i+args.splitplots,:], Wcompile_std[i:i+args.splitplots,:]/bootfactor,[mut_labels[k] for k in xrange(i, i+args.splitplots)], W_samples=Wcompile[i:i+args.splitplots,:,:], assignments=assignments, medoids=maxmedoids)
            savefig('%s/bootstrap_weights_by_mutant%s.png' % (args.outprefix, isuffix), dpi=args.dpi)
            for j in maxmedoids:
                figure(3)
                clf()
                weights_by_mutant_plot(Wcompile_mean[i:i+args.splitplots,:], Wcompile_std[i:i+args.splitplots,:]/bootfactor,[mut_labels[k] for k in xrange(i, i+args.splitplots)], W_samples=Wcompile[i:i+args.splitplots,:,:], W_ref=W_0[i:i+args.splitplots,:], idx=j, assignments=assignments, medoids=maxmedoids)
                savefig('%s/bootstrap_weights_by_mutant_structure_%s%s.png' % (args.outprefix, j, isuffix), dpi=100)
                for structs in assignments.values():
                    if j in structs:
                        overall_wt_fractions_file.write('%s\t%s\t%s\n' % (j, Wcompile_mean[wt_idx, structs].sum(), Wcompile[:,structs,:].sum(axis=1).std(axis=1)[0]))

        else:
            weights_by_mutant_plot(Wcompile_mean[i:i+args.splitplots,:], Wcompile_std[i:i+args.splitplots,:]/bootfactor,[mut_labels[k] for k in xrange(i, i+args.splitplots)])
            savefig('%s/bootstrap_weights_by_mutant%s.png' % (args.outprefix, isuffix), dpi=args.dpi)
            for j in selected_structures:
                figure(3)
                clf()
                weights_by_mutant_plot(Wcompile_mean[i:i+args.splitplots,:], Wcompile_std[i:i+args.splitplots,:]/bootfactor, [mut_labels[k] for k in xrange(i, i+args.splitplots)], W_ref=W_0[i:i+args.splitplots,:], idx=j)
                savefig('%s/bootstrap_weights_by_mutant_structure_%s%s.png' % (args.outprefix, j, isuffix), dpi=100)

    if len(structures) > MAX_STRUCTURES_PLOT:
        for j in maxmedoids:
            for structs in assignments.values():
                if j in structs:
                    overall_wt_fractions_file.write('%s\t%s\t%s\n' % (j, Wcompile_mean[wt_idx, structs].sum(), Wcompile[:,structs,:].sum(axis=1).std(axis=1)[0]))

    else:
        for j in selected_structures:
            overall_wt_fractions_file.write('%s\t%s\t%s\n' % (j, Wcompile_mean[wt_idx, j], Wcompile_std[wt_idx, j]))

    PCA_structure_plot(structures, assignments, maxmedoids, weights=Wcompile_mean[wt_idx,:], names=struct_labels)
    savefig('%s/bootstrap_pca_landscape_plot_WT.png' % prefix, dpi=args.dpi)

    figure(1)
    clf()
    bpp_matrix_plot(structures, Wcompile_mean[wt_idx,:], ref_weights=W_0[wt_idx,:], weight_err=Wcompile_std[wt_idx,:], offset=offset)
    savefig('%s/bootstrap_bppm_plot_WT.png' % prefix, dpi=args.dpi)


    print 'Printing bootstrap report'
    report = open('%s/bootstrap_report.txt' % args.outprefix,'w')
    if wt_idx != -1:
        report.write('Structure index\tStructure\tWild type weight\tWeight std\n')
    else:
        report.write('No Wild type found in data set\n')
        report.write('Structure index\tStructure\tSeq 0 weight\tSeq 0 std\n')
    for i, s in enumerate(selected_structures):
        report.write('%s\t%s\t%.3f\t%.3f\n' % (i, fa.structures[s], Wcompile_mean[0,i], Wcompile_std[0,i]/bootfactor))

for s in structures:
    print s

if not args.worker and args.postmodel:
    if len(structures) > MAX_STRUCTURES_PLOT:
        make_struct_figs([structures[i] for i in maxmedoids], 'postmodel_', indices=maxmedoids)
    else:
        make_struct_figs(structures, 'postmodel_')
    make_struct_figs(structures, 'postmodel_')
print 'Done!'
print '\a'
