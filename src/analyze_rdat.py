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
from random import choice
import mapping_analysis

parser = argparse.ArgumentParser()
parser.add_argument('rdatfile', type=argparse.FileType('r'))
parser.add_argument('outprefix', type=str)
parser.add_argument('--structfile', default=None, type=argparse.FileType('r'))
parser.add_argument('--clusterfile', default=None, type=argparse.FileType('r'))
parser.add_argument('--medoidfile', default=None, type=argparse.FileType('r'))
parser.add_argument('--nsubopt', default=100, type=int)
parser.add_argument('--nsim', default=1000, type=int)
parser.add_argument('--cutoff', default=0, type=int)
parser.add_argument('--start', default=None, type=int)
parser.add_argument('--end', default=None, type=int)
parser.add_argument('--greedyiter', default=10, type=int)
parser.add_argument('--nopriorswap', action='store_false')
parser.add_argument('--nopseudoenergies', action='store_false')
parser.add_argument('--modelselect', default=False, action='store_true')
parser.add_argument('--hardem', default=False, action='store_true')
parser.add_argument('--energydelta', default=None, type=float)
parser.add_argument('--refineiter', default=10, type=int)
parser.add_argument('--structest', default='hclust', type=str)
parser.add_argument('--clusterdatafactor', type=int, default=None)
parser.add_argument('--bootstrap', type=int, default=0)
parser.add_argument('--titrate', type=str, default=None)
parser.add_argument('--nomutrepeat', default=False, action='store_true')
parser.add_argument('--clipzeros', default=False, action='store_true')
parser.add_argument('--postmodel', default=False, action='store_true')
parser.add_argument('--kdfile', default=None, type=argparse.FileType('r'))
parser.add_argument('--splitplots', default=-1, type=int)
parser.add_argument('--detailedplots', default=False, action='store_true')
parser.add_argument('--nonormalization', default=False, action='store_true')
parser.add_argument('--dpi', type=int, default=200)
parser.add_argument('--scalemethod', type=str, default='linear')
parser.add_argument('--priorweights', type=str, default='rnastructure')
parser.add_argument('--njobs', type=int, default=None)
parser.add_argument('--csize', type=int, default=3)

args = parser.parse_args()


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
        structures = [line.strip().replace('A', '.') for line in args.structfile.readlines()]
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
        structures, deltas = ss.subopt(sequence[seqpos_start:seqpos_end], nstructs=args.nsubopt, algorithm=args.priorweights)
        structures = list(set([s.dbn for s in structures]))
        for i, m in enumerate(mutants):
            print 'Doing mutant %s: %s' % (i, m)
            mut_structures, deltas = ss.subopt(m[seqpos_start:seqpos_end], nstructs=args.nsubopt, algorithm=args.priorweights)
            mut_structures = list(set([s.dbn for s in mut_structures if s.dbn not in structures]))
            if len(mut_structures) > 0:
                print 'Found new structures: %s' % mut_structures
            structures.extend(mut_structures)
            os.popen('rm /tmp/tmp*')
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
if args.modelselect:
    energies = get_free_energy_matrix(structures,[m[seqpos_start:seqpos_end] for m in  mutants], algorithm=args.priorweights)
else:
    energies = get_free_energy_matrix(structures, mutants, algorithm=args.priorweights)
fa = mapping_analysis.FAMappingAnalysis(data, structures, mutants, mutpos=mutpos_cutoff, energies=energies, concentrations=concentrations, kds=kds, seqpos_range=seqpos_range, c_size=args.csize)
unpaired_data, paired_data = fa.set_priors_by_rvs(SHAPE_unpaired_sample, SHAPE_paired_sample)
if args.clusterfile:
    fa.set_structure_clusters(structure_clusters, struct_medoid_indices, struct_weights_by_clust=struct_weights_by_clust)
    W_0 = fa.W
else:
    W_0 = calculate_weights(energies)
if args.modelselect:
    print 'Model selection'
    selected_structures, assignments = fa.model_select(greedy_iter=args.greedyiter, max_iterations=2, prior_swap=args.nopriorswap, expstruct_estimation=args.structest, G_constraint=args.energydelta, n_jobs=args.njobs, apply_pseudoenergies=args.nopseudoenergies, algorithm=args.priorweights, hard_em=args.hardem)
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
    chi_sq, rmsea = fa.calculate_fit_statistics()

    if args.clusterdatafactor:

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
    pickle.dump(fa.data, open('%s/data.pickle' % prefix, 'w'))
    pickle.dump(W_fa, open('%s/W.pickle' % prefix, 'w'))
    pickle.dump(W_fa_std, open('%s/W_std.pickle' % prefix, 'w'))
    pickle.dump(Psi_fa, open('%s/Psi.pickle' % prefix, 'w'))
    pickle.dump(E_d_fa, open('%s/E_d.pickle' % prefix, 'w'))
    pickle.dump(E_c_fa, open('%s/E_c.pickle' % prefix, 'w'))
    pickle.dump(E_ddT_fa, open('%s/E_ddT.pickle' % prefix, 'w'))
    pickle.dump(M_fa, open('%s/M.pickle' % prefix, 'w'))
    pickle.dump(data_pred, open('%s/data_pred.pickle' % prefix, 'w'))
    pickle.dump(sigma_pred, open('%s/sigma_pred.pickle' % prefix, 'w'))
    pickle.dump(I, open('%s/bootstrap_indices.pickle' % prefix, 'w'))

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
    new_structures = []
    for i in xrange(E_d_fa.shape[0]):
        md = mapping.MappingData(data=E_d_fa[i,:], enforce_positives=True)
        new_structures.append(ss.fold(sequence, mapping_data=md, algorithm=args.priorweights)[0].dbn)
    open('%s/postmodel_structures.txt' % args.outprefix, 'w').write('\n'.join(new_structures))
    make_struct_figs(new_structures, 'postmodel_')
print 'Done!'
print '\a'
