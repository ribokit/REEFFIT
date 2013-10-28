import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *
import mapping_analysis
from map_analysis_utils import *
from rdatkit.datahandlers import *
from collections import defaultdict
import os
import argparse
from plot_utils import plot_mutxpos_image
import pickle
from shutil import copy2
from rdatkit.view import VARNA

parser = argparse.ArgumentParser()

parser.add_argument('benchmarkfile', type=argparse.FileType('r'))
parser.add_argument('prefix', type=str)
parser.add_argument('--niter', type=int, default=5)
parser.add_argument('--nsim', type=int, default=1000)
parser.add_argument('--nseq', type=int, default=None)
parser.add_argument('--nosimulate', action='store_true', default=False)
parser.add_argument('--nomodelselect', action='store_true', default=False)
parser.add_argument('--noanalyze', action='store_true', default=False)
parser.add_argument('--nosummarize', action='store_true', default=False)
parser.add_argument('--usetruestructures', action='store_true', default=False)
parser.add_argument('--priorweights', type=str, default='rnastructure')

args = parser.parse_args()
if args.usetruestructures:
    structfile = 'structures.txt'
else:
    structfile = 'structure_medoids.txt'

complement = {'A':'U', 'C':'G', 'U':'A', 'G':'C'}

BENCHMARKDIR = 'insilico_benchmark/'
def simulate_dataset(sequence, structures, name, pathname, outname):
    print 'Simulating dataset'
    mutants = [sequence] + [sequence[:i] + complement[sequence[i]] + sequence[i+1:] for i in xrange(len(sequence))]
    struct_types = []
    for i in xrange(len(sequence)):
        struct_types.append(['u' if structures[j][i] == '.' else 'p' for j in xrange(len(structures))])

    mutpos = [-1] + range(len(sequence))
    seqpos = range(len(sequence))
    mut_labels = ['%s%s%s' % (sequence[pos], pos+1, complement[sequence[pos]]) if pos >= 0 else 'WT' for pos in mutpos]
    true_data, perturbed_data, data_noised, true_energies, weights_noised, reacts, obs_noise_sigmas = mock_data(mutants, structures, energy_mu=0, energy_sigma=1, obs_sigma=0.02, mutpos=mutpos, c_size=3, correlate_regions=False, return_steps=True)

    figure(1)
    clf()
    plot_mutxpos_image(true_data, sequence, seqpos, 0, mut_labels)
    savefig('%s/simulated_no_noise.png' % pathname, dpi=300)
    plot_mutxpos_image(perturbed_data, sequence, seqpos, 0, mut_labels)
    savefig('%s/simulated_noised_weights.png' % pathname, dpi=300)
    plot_mutxpos_image(data_noised, sequence, seqpos, 0, mut_labels)
    savefig('%s/simulated_fully_noised.png' % pathname, dpi=300)
    plot_mutxpos_image(data_noised, sequence, seqpos, 0, mut_labels)
    savefig('%s/%s_results/%s.png' % (BENCHMARKDIR, args.prefix, outname), dpi=300)

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

def find_cluster(clusters, structure):
    minscore = inf
    bestclust = None
    for k, stlist in clusters.iteritems():
        for st in stlist:
            score = 0
            for i in xrange(len(st)):
                if st[i] != structure[i]:
                    score += 1
            if score < minscore:
                minscore = score
                bestclust = k
    return bestclust, minscore

count = 1 
statnames = ['Name','Sequence length', 'Number of structures',  'Number of predicted structures', 'Number of correct structures', '$\Chi^2/df$', 'RMSEA', 'Weight error sum', 'Min weight error', 'Max weight error', 'Mean weight error', 'Percentage of weights correctly predicted']
stats = []
ids = []
for line in args.benchmarkfile.readlines():
    if args.nseq and count > args.nseq:
        break
    count += 1
    if line.strip()[0] == '#':
        continue
    else:
        name, sequence, structures = line.strip().split('\t')
        structures = structures.split(',')
        outname = name.lower().replace(' ','_').replace(';','_').replace('/','_')
        pathname = BENCHMARKDIR + outname + '/'  + args.prefix
        ids.append(pathname)
        if not os.path.exists(pathname):
            os.mkdir(pathname)
        if not os.path.exists('%s/%s_results/' % (BENCHMARKDIR, args.prefix)):
            os.mkdir('%s/%s_results/' % (BENCHMARKDIR, args.prefix))

            
        if not args.nosimulate:
            weights_noised, mut_labels, obs_noise_sigmas, reacts = simulate_dataset(sequence, structures, name, pathname, outname)

            pickle.dump(weights_noised, open('%s/weights_noised.pickle' % pathname, 'w'))
            pickle.dump(mut_labels, open('%s/mut_labels.pickle' % pathname, 'w'))
            pickle.dump(obs_noise_sigmas, open('%s/obs_noise_sigmas.pickle' % pathname, 'w'))
            pickle.dump(reacts, open('%s/reacts.pickle' % pathname, 'w'))

        else:
            weights_noised = pickle.load(open('%s/weights_noised.pickle' % pathname))
            mut_labels = pickle.load(open('%s/mut_labels.pickle' % pathname))
            obs_noise_sigmas = pickle.load(open('%s/obs_noise_sigmas.pickle' % pathname))
            reacts = pickle.load(open('%s/reacts.pickle' % pathname))
        
        sfile = open('%s/structures.txt' % (pathname), 'w')
        sfile.write('\n'.join(structures))
        sfile.close()

        true_structures = [l.strip() for l in open('%s/structures.txt' % (pathname)).readlines()]
        for i, s in enumerate(true_structures):
            VARNA.cmd(sequence, s, pathname + '/true_structure%s.svg' % i, options={'fillBases':False, 'resolution':'10.0', 'flat':True, 'bp':'#000000'})
        if not args.nomodelselect:
            CMD = 'time python analyze_rdat.py %s/%s.rdat %s/  --modelselect --njobs 63 --nonormalization --nopseudoenergies --priorweights %s' % (pathname, outname, pathname, args.priorweights)
            print 'Executing command'
            print CMD
            os.system(CMD)

        if not args.noanalyze:
            print 'Executing command:'
            #CMD = 'time python analyze_rdat.py %s/%s.rdat %s/  --nsim 1000 --refineiter %s --structfile %s/structures.txt --njobs 15 --nonormalization --detailedplots --csize 3 --psi "%s"' % (outname, outname, outname, args.niter, outname, ','.join([str(x**2) for x in obs_noise_sigmas]))
            CMD = 'time python analyze_rdat.py %s/%s.rdat %s/  --nsim %s --refineiter %s --structfile %s/%s --njobs 63 --nonormalization --detailedplots --csize 3 --priorweights %s' % (pathname, outname, pathname, args.nsim, args.niter, pathname, structfile, args.priorweights)
            print CMD
            os.system(CMD)
        if not args.nosummarize:
            copy2(pathname + '/simulated_fully_noised.png', '%s/%s_results/%s.png' % (BENCHMARKDIR, args.prefix, outname))
            copy2(pathname + '/reeffit_data_pred.png', '%s/%s_results/%s_reefit_data_pred.png' % (BENCHMARKDIR, args.prefix, outname))
            copy2(pathname + '/loglike_trace.png', '%s/%s_results/%s_loglike_trace.png' % (BENCHMARKDIR, args.prefix, outname))
            copy2(pathname + '/real_data.png', '%s/%s_results/%s.png' % (BENCHMARKDIR, args.prefix, outname))

            select_structs =  [l.strip() for l in open('%s/%s' % (pathname, structfile)).readlines()]
            
            clusters =  defaultdict(list)
            cluster_indices =  defaultdict(list)
            for l in open('%s/structure_clusters.txt' % pathname).readlines():
                k, v = l.strip().split('\t')[:2]
                clusters[k].append(v)

            scores = {}
            assignments = defaultdict(list)
            for i, s in enumerate(structures):
                clust, score = find_cluster(clusters, s)
                scores[s] = score
                assignments[clust].append(i)

            select_scores = {}
            select_assignments = defaultdict(list)
            for i, s in enumerate(select_structs):
                clust, score = find_cluster(clusters, s)
                select_scores[s] = score
                select_assignments[clust].append(i)

            def struct_compare(s1, s2):
                matches = 0.
                for i, s in enumerate(s1):
                    if s1[i] == s2[i]:
                        matches += 1.
                return matches/len(s1) == 1

            nstructcorr = 0
            for i, s in enumerate(select_structs):
                for s2 in structures:
                    if struct_compare(s, s2):
                        nstructcorr += 1

            print 'Number of correct structures: %s out of %s in %s' % (nstructcorr, len(structures), len(select_structs))
            print 'Plotting comparison plots'

            W_fa = pickle.load(open('%s/W.pickle' % pathname))
            W_fa_std = pickle.load(open('%s/W_std.pickle' % pathname))
            Psi_fa = pickle.load(open('%s/Psi.pickle' % pathname))
            E_d_fa = pickle.load(open('%s/E_d.pickle' % pathname))
            E_c_fa = pickle.load(open('%s/E_c.pickle' % pathname))
            data_pred = pickle.load(open('%s/data_pred.pickle' % pathname))
            sigma_pred = pickle.load(open('%s/sigma_pred.pickle' % pathname))

            print 'Parsing RDAT'
            rdat = RDATFile()
            rdat.load(open('%s/%s.rdat' % (pathname, outname)))
            construct = rdat.constructs.values()[0]
            seqpos = construct.seqpos
            sorted_seqpos = sorted(seqpos)

            data = []
            for idx, d in enumerate(construct.data):
                nd_tmp = normalize([d.values[seqpos.index(i)] for i in sorted_seqpos])
                nd = [d.values[seqpos.index(i)] if d.values[seqpos.index(i)] > 0 else min(nd_tmp) for i in sorted_seqpos]
                data.append(nd)
            data = array(data)


            if nstructcorr == len(structures) and len(select_structs) == len(structures):
                W_pred = W_fa
                W_pred_std = W_fa_std
                W_obs = weights_noised
                exact_match = True
            else:
                W_pred = []
                W_pred_std = []
                W_obs = []
                if len(assignments) == 1:
                    pdb.set_trace()

                for c in assignments:
                    obs = zeros([W_fa.shape[0]])
                    pred = zeros([W_fa.shape[0]])
                    pred_std = zeros([W_fa.shape[0]])
                    for i in assignments[c]:
                        obs += weights_noised[:,i]
                    W_obs.append(obs.tolist())
                    for i in select_assignments[c]:
                        pred += W_fa[:,i]
                        #pred_std = array([max(W_fa_std[j,i], pred_std[j]) for j in xrange(W_fa_std.shape[0])])
                    pred_std = array([W_fa_std[j,:].sum() for j in xrange(W_fa_std.shape[0])])
                    W_pred.append(pred.tolist())
                    W_pred_std.append(pred_std.tolist())

                W_pred = array(W_pred).T
                W_pred_std = array(W_pred_std).T
                W_obs = array(W_obs).T

                exact_match = False


            print pathname
            figure(1)
            clf()
            r = arange(len(obs_noise_sigmas))
            est_sigmas = [sqrt(Psi_fa[i,0,0]) for i in r]
            scatter(obs_noise_sigmas, est_sigmas, color='b')
            xlabel('True noise variance')
            ylabel('Estimated noise variance')
            maxval = max(max(est_sigmas), max(obs_noise_sigmas))
            xlim(0, maxval)
            ylim(0, maxval)
            savefig('%s/Psi_comparison.png' % pathname, dpi=300)
            savefig('%s/%s_results/%s_Psi_comparison.png' % (BENCHMARKDIR, args.prefix, outname), dpi=300)

            if exact_match:
                for i, s in enumerate(structures):
                    binarized_react = [1 if x == '.' else 0 for x in s]
                    f = figure(2)
                    f.set_size_inches(15, 5)
                    clf()
                    title('Structure %s: %s' % (i, s))
                    plot(1+arange(len(s)), binarized_react, linewidth=2, c='r')
                    plot(1+arange(len(s)), E_d_fa[i,:], linewidth=2, color='k')
                    plot(1+arange(len(s)), reacts[i,:], '--', linewidth=2, color='k')
                    xlim(1,len(s))
                    #ylim(0,2)
                    savefig('%s/%s_results/%s_react_%s_comparison.png' % (BENCHMARKDIR, args.prefix, outname, i), dpi=300)
                    savefig('%s/exp_react_struct_%s_comparison.png' % (pathname, i), dpi=300)


            figure(3)
            clf()
            cm = get_cmap('Paired')
            if exact_match:
                nstructs = W_pred.shape[1]
            else:
                nstructs = len(assignments)
            for i in xrange(W_pred.shape[1]):
                errorbar(range(W_pred.shape[0]), W_pred[:,i], yerr=W_pred_std[:,i], linewidth=2, label='Pred. structure %s ' % i, color=cm(50*i))
                plot(range(W_obs.shape[0]), W_obs[:,i], '--', linewidth=4,  label='Sim. structure %s ' % i, color=cm(50*i))

            ylim(0,1)
            xticks(range(len(mut_labels)), mut_labels, fontsize='x-small', rotation=90)
            xlim(-1, len(mut_labels))
            #legend()
            savefig('%s/%s_results/%s_weights_comparison.png' % (BENCHMARKDIR, args.prefix, outname), dpi=300)
            savefig('%s/weights_by_mutant_comparison.png' % pathname, dpi=300)

            print 'Saving stats'
            weight_errors = (W_pred - W_obs)**2
            mutants_missed = ones([W_fa.shape[0]])
            weights_missed = zeros(W_fa.shape)
            for j in xrange(W_pred.shape[0]):
                for s in xrange(W_pred.shape[1]):
                    if W_obs[j,s] <= W_pred[j,s] + W_pred_std[j,s] and W_obs[j,s] >= W_pred[j,s] - W_pred_std[j,s]:
                        mutants_missed[j] = 0
                    else:
                        weights_missed[j,s] = 1
            # The following needs to be corrected
            chi_sq = ((data - data_pred)**2/(sigma_pred)**2).sum()
            df = data.size - W_pred.size - E_d_fa.size - E_c_fa[logical_not(isnan(E_c_fa))].size - data.shape[0] - 1
            rmsea = sqrt(max((chi_sq/df - 1)/(data.shape[1] - 1), 0.0))
            stats.append([name, len(sequence), len(structures), len(select_structs), nstructcorr, chi_sq/df, rmsea, weight_errors.sum(), weight_errors.min(), weight_errors.max(), weight_errors.mean(), 100 - (weights_missed.mean())*100])
if not args.nosummarize:
    print 'Doing benchmark summary'
    outfile = open('%s/%s_results/summary.txt' % (BENCHMARKDIR, args.prefix), 'w')
    outfile.write('%s\n' % '\t'.join(statnames))
    for s in stats:
        outfile.write('%s\n' % '\t'.join([str(x) for x in s]))
else:
    print 'Finished simulating'

print 'Writing log file'
logfile = open('%s/%s_results/benchmark.log' % (BENCHMARKDIR, args.prefix), 'w')
logfile.write(str(args) + '\n')
logfile.write('\n'.join(ids))

print 'Done!'
