#This file is part of the REEFFIT package.
#    Copyright (C) 2013 Pablo Cordero <tsuname@stanford.edu>
import argparse
import matplotlib
matplotlib.use('Agg')
import os
import pickle
from matplotlib.pylab import *
from rdatkit.datahandlers import RDATFile
import rdatkit.secondary_structure as ss
from plot_utils import *
from map_analysis_utils import *
import scipy.cluster.hierarchy
import map_analysis_utils
import mapping_analysis

parser = argparse.ArgumentParser()
parser.prog = 'reeffit-inspect'

# General options
parser.add_argument('rdatfile', type=argparse.FileType('r'), help='The RDAT file that has the multi-dimensional chemical mapping data')
parser.add_argument('resultdir', type=str, help='Prefix (e.g. directory) that all resulting reports, plot files, etc. are.')
parser.add_argument('structfile', type=argparse.FileType('r'), help='Files with all secondary structures, one per line, in dot-bracket notation')
parser.add_argument('outdir', type=str, help='Output directory')
parser.add_argument('--addrdatfiles', type=argparse.FileType('r'), nargs='+', default=[], help='Additional rdat files that contain the multi-dimensional chemical mapping data')
parser.add_argument('--splitplots', default=-1, type=int, help='Plot subsets of data and predicted data rather than the whole set')
parser.add_argument('--dpi', type=int, default=200, help='DPI resolution for plots')
parser.add_argument('--bootstrap', action='store_true', default=False , help='Get bootstrap results')
parser.add_argument('--topstructs', type=int, default=-1)
parser.add_argument('--topclusterstructs', type=int, default=-1)
parser.add_argument('--topclusterreport', type=int, default=-1)
parser.add_argument('--structfig', type=int, default=-1)
parser.add_argument('--structsearch', type=str, default='')
parser.add_argument('--medoidfractions', action='store_true', default=False)
parser.add_argument('--bpfraction', type=str, default='')
parser.add_argument('--mutant', type=str, default='WT')
parser.add_argument('--seqrange', type=str, default='')
parser.add_argument('--weightplots', action='store_true', default=False)
parser.add_argument('--bppmplots', action='store_true', default=False)
parser.add_argument('--dendrogram', action='store_true', default=False)
parser.add_argument('--medoidfigs', action='store_true', default=False)
parser.add_argument('--reports', action='store_true', default=False)
parser.add_argument('--reclusterbp', action='store_true', default=False, help='Recluster by base pair distance')

args = parser.parse_args()


class Results(object):
    pass

class RDATdata(object):
    pass

def save_results(results):
    for name, obj in results.__dict__.iteritems():
        fname= '%s/%s.pickle' % (args.resultdir, name.replace('_fa',''))
        if not os.path.exists(fname):
            pickle.dump(obj, open(fname, 'w'))
        else:
            print 'File %s exists, skipping save' % fname


def read_results(rdat_data):
    print 'Loading results from previous run'
    results = Results()

    results.structures = [line.strip() for line in args.structfile.readlines() if line[0] != '#']
    results.struct_types = get_struct_types(results.structures)
    if args.bootstrap:
        print 'Bootstrap option set: Getting bootstrapped results'
        W_boot = pickle.load(open('%s/W_bootstrap.pickle' % args.resultdir))
        results.W_fa = W_boot.mean(axis=2)
        results.W_fa_std = W_boot.std(axis=2)
    else:
        results.W_fa = pickle.load(open('%s/W.pickle' % args.resultdir))
        results.W_fa_std = pickle.load(open('%s/W_std.pickle' % args.resultdir))
    results.data_pred = pickle.load(open('%s/data_pred.pickle' % args.resultdir))
    try:
        results.linkage = pickle.load(open('%s/linkage.pickle' % args.resultdir))
        results.assignments = pickle.load(open('%s/assignments.pickle' % args.resultdir))
        results.medoid_dict = pickle.load(open('%s/medoid_dict.pickle' % args.resultdir))
    except:
        print 'No clustering objects saved, skipping loading them...should probably use reclusterbp option'
    try:
        results.W_ref= pickle.load(open('%s/W_ref.pickle' % args.resultdir))
    except:
        print 'No reference weight file, computing RNAstructure weights as reference'
        results.ref_energies = get_free_energy_matrix(results.structures, rdat_data.mutants)
        results.W_ref = calculate_weights(results.ref_energies)
    """
    results.Psi_fa = pickle.load(open('%s/Psi.pickle' % args.resultdir))
    results.E_d_fa = pickle.load(open('%s/E_d.pickle' % args.resultdir))
    results.E_c_fa = pickle.load(open('%s/E_c.pickle' % args.resultdir))
    results.E_ddT_fa = pickle.load(open('%s/E_ddT.pickle' % args.resultdir))
    results.M_fa = pickle.load(open('%s/M.pickle' % args.resultdir))
    results.sigma_pred = pickle.load(open('%s/sigma_pred.pickle' % args.resultdir))
    """
    return results


def parse_rdat_metadata():
    rdatfiles = [args.rdatfile]
    if args.addrdatfiles is not None:
        rdatfiles += args.addrdatfiles
    rdat_data = RDATdata()
    idxoffset = 0
    for rdat_idx, rdatfile in enumerate(rdatfiles):
        print 'Parsing RDAT metadata %s' % rdatfile
        rdat = RDATFile()
        rdat.load(rdatfile)
        construct = rdat.constructs.values()[0]

        seqpos = construct.seqpos
        sequence = construct.sequence.upper()
        if rdat_idx  == 0:
            rdat_data.offset = construct.offset
            rdat_data.sequence = sequence
            rdat_data.sorted_seqpos = sorted(seqpos)
            rdat_data.mut_labels = []
            rdat_data.mutants = []
            rdat_data.mutpos = []
            rdat_data.wt_indices = []

        nwarnings = 0

        for idx, d in enumerate(construct.data):
            if 'warning' in d.annotations:
                nwarnings += 1
                continue
            if 'mutation' in d.annotations:
                label = ';'.join(d.annotations['mutation'])
            else:
                label = 'WT'
            pos = []
            if label == 'WT':
                if 'sequence' in d.annotations:
                    mutant = d.annotations['sequence'][0]
                    pos = []
                    for i, ms in enumerate(mutant):
                        if ms != sequence[i] and len(pos) < 1:
                            pos.append(i)
                            if label == 'WT':
                                label = '%s%s%s' % (sequence[i], i + construct.offset + 1, mutant[i])
                if pos == []:
                    mutant = sequence
                    rdat_data.wt_indices.append(idx + idxoffset - nwarnings)

            else:
                if 'sequence' in d.annotations:
                    mutant = d.annotations['sequence'][0]
                    pos = []
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
            rdat_data.mutpos.append(pos)
            rdat_data.mut_labels.append(label)
            rdat_data.mutants.append(mutant)
            idxoffset += len(construct.data)

    return rdat_data

def dendrogram(results, rdat_data):
    print 'Doing dedrogram'
    scipy.cluster.hierarchy.dendrogram(results.linkage)
    savefig('%s/structures_dendrogram.png' % (args.outdir), dpi=args.dpi)


def weight_plots(results, rdat_data):
    print 'Doing weight plots'
    for i in arange(0, results.data_pred.shape[0], args.splitplots):
        if i == 0 and args.splitplots == results.data_pred.shape[0]:
            isuffix = ''
        else:
            isuffix = '_%s' % (i/args.splitplots)
        figure(1)
        clf()
        weights_by_mutant_plot(results.W_fa[i:i+args.splitplots,:], results.W_fa_std[i:i+args.splitplots,:]*40, [rdat_data.mut_labels[k] for k in xrange(i, i+args.splitplots)], assignments=results.assignments, medoids=results.maxmedoids)
        savefig('%s/weights_by_mutant%s.png' % (args.outdir,isuffix), dpi=args.dpi)

def structure_reports(results, rdat_data):
    print 'Structure weight report by mutant'
    for j in xrange(results.W_fa.shape[0]):
        print 'Report for %s' % rdat_data.mut_labels[j]
        struct_report = open('%s/structure_weights%s_%s.txt' % (args.outdir, j, rdat_data.mut_labels[j]),'w')
        struct_report.write('Structure index\tStructure\tCluster medoid\tWeight\tWeight error (stdev)\n')
        for i in xrange(results.W_fa.shape[1]):
            if results.W_fa[j,i] >= 0.05:
                for structs in results.assignments.values():
                    if i in structs:
                        for m in results.maxmedoids:
                            if m in structs:
                                break
                        break
                struct_report.write('%s\t%s\t%s\t%s\t%s\n' % (i, results.structures[i], results.structures[m], results.W_fa[j,i], results.W_fa_std[j,i]))
        struct_report.close()

def strip_bp(struct, start, end):
    bp_dict = ss.SecondaryStructure(dbn=struct).base_pair_dict()
    struct_list = list(struct)
    for i in range(0, start) + range(end, len(struct)):
        if i in bp_dict:
            j = bp_dict[i]
            struct_list[i] = '.'
            struct_list[j] = '.'
    return ''.join(struct_list)[start:end]


def struct_figs(results, rdat_data, structures, fprefix, indices=None, base_annotations=None, helix_function=lambda x,y:x, helix_fractions=None, annotation_color='#FF0000', no_colors=False, titles=None):
    print 'Doing structure figures'
    if args.seqrange:
        seqrange = (args.seqrange[0] - rdat_data.offset - 1, args.seqrange[1] - rdat_data.offset - 1)
        offset = args.seqrange[0] - 1
    else:
        seqrange = (0, len(rdat_data.sequence))
        offset = rdat_data.offset
    cstructures = [strip_bp(remove_non_cannonical(s, rdat_data.sequence), seqrange[0], seqrange[1]) for s in structures]
    if titles is None:
        titles = ['']*len(cstructures)
    make_struct_figs(cstructures, rdat_data.sequence[seqrange[0]:seqrange[1]], offset, args.outdir + '/' + fprefix, indices=indices, base_annotations=base_annotations, helix_function=helix_function, helix_fractions=helix_fractions, annotation_color=annotation_color, no_colors=no_colors, titles=titles)


def bppm_plots(results, rdat_data):
    for j in xrange(results.W_fa.shape[0]):
        if rdat_data.mut_labels[j] == args.mutant:
            print 'BPPM plot for %s' % args.mutant
            figure(1)
            clf()
            bpp_matrix_plot(results.structures, results.W_fa[j,:], ref_weights=results.W_ref[j,:], weight_err=results.W_fa_std[j,:], offset=rdat_data.offset, hard_thresh=0.0)
            xlim(args.seqrange[0] - rdat_data.offset - 1, args.seqrange[1] - rdat_data.offset - 1)
            ylim(args.seqrange[1] - rdat_data.offset - 1, args.seqrange[0] - rdat_data.offset - 1)
            savefig('%s/bppm_plot_%s_%s.png' % (args.outdir, j, rdat_data.mut_labels[j]), dpi=args.dpi)

if __name__ == '__main__':
    rdat_data = parse_rdat_metadata()
    results = read_results(rdat_data)

    if args.seqrange:
        args.seqrange = [int(s) for s in args.seqrange.split(':')]


    plots = set()
    reports = set()
    if args.splitplots < 0:
        args.splitplots = results.data_pred.shape[0]
    if args.reclusterbp:
        results.medoid_dict, results.assignments, results.linkage = cluster_structures(results.struct_types, structures=results.structures, distance='basepair')

    results.maxmedoids = [where(results.W_fa[rdat_data.wt_indices[0],:] == results.W_fa[rdat_data.wt_indices[0], stindices].max())[0][0] for stindices in results.assignments.values()]
    save_results(results)

    if args.weightplots:    
        plots.add(weight_plots)
    if args.bppmplots:
        plots.add(bppm_plots)
    if args.dendrogram:
        plots.add(dendrogram)
    if args.reports:
        reports.add(structure_reports)
    if args.topclusterreport >= 0:
        for cluster, indices in results.assignments.iteritems():
            print 'For cluster %s' % cluster
            sorted_indices = sorted(indices, key=lambda x:results.W_fa[rdat_data.wt_indices[0], x], reverse=True)[:args.topclusterreport]
            for i in sorted_indices:
                print '%s %s' % (results.W_fa[rdat_data.wt_indices[0], i], results.structures[i])
    if args.topclusterstructs >= 0:
        for cluster, indices in results.assignments.iteritems():
            sorted_indices = sorted(indices, key=lambda x:results.W_fa[rdat_data.wt_indices[0], x], reverse=True)[:args.topclusterstructs]
            weight_strs = ['%3.5f' % results.W_fa[rdat_data.wt_indices[0], x] for x in sorted_indices]
            titles = ['Cluster %s, WT weight %s' % (cluster, w) for w in weight_strs]
            struct_figs(results, rdat_data, [results.structures[i] for i in sorted_indices], 'cluster_%s_' % cluster, indices=weight_strs, titles=titles, no_colors=True)
    if args.medoidfigs:
        struct_figs(results, rdat_data, [results.structures[i] for i in results.maxmedoids], 'medoid_', indices=['_cluster_%s_idx_%s' % (results.assignments.keys()[i], results.maxmedoids[i]) for i in range(len(results.maxmedoids))])
    if args.topstructs >= 0:
        sorted_struct_indices, sorted_weights = zip(*[(idx, weight) for idx, weight in sorted(enumerate(results.W_fa[args.topstructs, :]), key=lambda x: x[1], reverse=True) if weight >= 0.05])
        sorted_weights = ['%3.2f' % w for w in sorted_weights]
        struct_figs(results, rdat_data, [results.structures[i] for i in sorted_struct_indices], '%s_topstruct_' % (rdat_data.mut_labels[args.topstructs]), indices=sorted_weights)
    if args.structfig >= 0:
        print 'Plotting structure %s with WT weight %s' % (args.structfig, results.W_fa[rdat_data.wt_indices[0], args.structfig])
        struct_figs(results, rdat_data, [results.structures[args.structfig]], '', indices=[args.structfig])

    if args.structsearch:
        for idx, structure in enumerate(results.structures):
            if args.structsearch.strip('"') in structure:
                print 'Plotting structure %s with WT weight %s' % (idx, results.W_fa[rdat_data.wt_indices[0], idx])
                struct_figs(results, rdat_data, [results.structures[idx]], 'search_', indices=[idx], no_colors=True)


    if args.medoidfractions:
        for i, label in enumerate(rdat_data.mut_labels):
            if label == args.mutant:
                for values, cluster in results.assignments.iteritems():
                    w = results.W_fa[i, values].sum()
                    w = sqrt((results.W_fa_std[i, values]**2).sum())
                    print 'For cluster %s, medoid fraction of %s is %s +/- %s' % (cluster, label, w, w_std)
    
    if args.bpfraction:
        for i, label in enumerate(rdat_data.mut_labels):
            if label == args.mutant:
                bppm, bppm_std = bpp_matrix_from_structures(results.structures, results.W_fa[i,:], weight_err=results.W_fa[i,:])
                bp = [int(b) - rdat_data.offset - 1 for b in args.bpfraction.split(',')]
                print 'Base-pair fraction of %s is %s +/- %s' % (label, bppm[bp[0], bp[1]], bppm_std[bp[0], bp[1]])


    for doplot in plots:
        figure(1)
        clf()
        doplot(results, rdat_data)

    for doreport in reports:
        doreport(results, rdat_data)


