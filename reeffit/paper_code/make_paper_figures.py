import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *
import mapping_analysis
from map_analysis_utils import *
from plot_utils import *
from rdatkit.datahandlers import *
from collections import defaultdict
import os
import argparse
from plot_utils import plot_mutxpos_image
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('benchmarkfile', type=argparse.FileType('r'))
parser.add_argument('prefix', type=str)

args = parser.parse_args()

complement = {'A':'U', 'C':'G', 'U':'A', 'G':'C'}

BENCHMARKDIR = 'insilico_benchmark/'

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
    return bestclust
templatefname = 'svg_templates/insilico_comparison.svg'
currdir = os.getcwd()
for line in args.benchmarkfile.readlines():
    if line.strip()[0] == '#':
        continue
    else:
        name, sequence, structures = line.strip().split('\t')
        structures = structures.split(',')
        outname = name.lower().replace(' ','_').replace(';','_').replace('/','_')
        pathname = BENCHMARKDIR + outname
        result_dir = '%s/%s_results/' % (BENCHMARKDIR, args.prefix)
        result_prefix = '%s/%s/%s_' % (BENCHMARKDIR, outname, args.prefix)
        select_structs =  [l.strip() for l in open('%sstructure_medoids.txt' % result_prefix).readlines()]
        print 'Doing %s' % outname

        select_struct_vecs = []
        for s in select_structs:
            select_struct_vecs.append([1 if x == '.' else 0 for x in s])
        select_struct_vecs = array(select_struct_vecs)

        
        clusters =  defaultdict(list)
        cluster_indices =  defaultdict(list)
        all_structs = []
        all_struct_vecs = []
        cluster_colors = {}
        cluster_ordering = []
        for l in open('%s/%s/structure_clusters.txt' % (BENCHMARKDIR, outname)).readlines():
            k, v = l.strip().split('\t')[:2]
            cluster_colors[k] = (205./255, 201./255, 201./255, 1.0)
            cluster_ordering.append(k)
            clusters[k].append(v)
            all_structs.append(v)
            all_struct_vecs.append([1 if s == '.' else 0 for s in v])
        all_struct_vecs = array(all_struct_vecs)

        energies = get_free_energy_matrix(all_structs, [sequence])
        struct_weights = calculate_weights(energies)[0]

        assignments = {}
        true_struct_vecs = []
        for i, s in enumerate(structures):
            clust = find_cluster(clusters, s)
            assignments[s] = clust
            true_struct_vecs.append([1 if x == '.' else 0 for x in s])
        true_struct_vecs = array(true_struct_vecs)


        select_assignments = {}
        for i, s in enumerate(select_structs):
            select_assignments[s] = clust

        if len(structures) > len(select_structs):
            for i, s in enumerate(select_structs):
                clust = find_cluster(clusters, s)
                cluster_colors[clust] = STRUCTURE_COLORS[i]
        else:
            for i, s in enumerate(structures):
                clust = find_cluster(clusters, s)
                cluster_colors[clust] = STRUCTURE_COLORS[i]

        print 'Making PCA plot for %s' % outname 
        U, s, Vh = svd(all_struct_vecs.T)
        basis = U[:,:2]
        all_struct_coordinates = dot(all_struct_vecs, basis)
        true_struct_coordinates = dot(true_struct_vecs, basis)
        select_struct_coordinates = dot(select_struct_vecs, basis)

        figure(1)
        clf()

        all_sizes = [max(50, struct_weights[i]*50000) for i in range(len(struct_weights))]
        scatter(all_struct_coordinates[:,0], all_struct_coordinates[:,1], c=[cluster_colors[c] for c in cluster_ordering], alpha=0.6, linewidth=1, s=all_sizes)
        scatter(true_struct_coordinates[:,0], true_struct_coordinates[:,1], c=[cluster_colors[assignments[s]] for s in structures], linewidth=1.5, marker='*', s=500)
        scatter(select_struct_coordinates[:,0], select_struct_coordinates[:,1], c=[cluster_colors[select_assignments[s]] for s in select_structs], linewidth=1.5, marker='D', s=100)
        savefig('%s/%s_pca_cluster_plot.png' % (result_dir, outname), dpi=300)

        print 'Making SVG file for %s' % outname
        templatefile = open(templatefname)
        svgfile = open('%s/%s_paper_figure.svg' % (result_dir, outname), 'w')
        for line in templatefile.readlines():
            newline = line.replace('{exportdir}', result_dir).replace('{name}', outname).replace('{figdir}', '').replace('{id}', outname)
            svgfile.write(newline)
        svgfile.close()
        templatefile.close()

        print 'Exporting to PDF'
        os.chdir(result_dir)
        os.system('inkscape -z -D --file=%s_paper_figure.svg --export-pdf=%s_paper_figure.pdf --export-area-drawing  --export-text-to-path' % (outname, outname))
        os.chdir(currdir)

print 'Done'


        
