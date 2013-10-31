import pickle
import networkx as nx
import rdatkit.secondary_structure as ss
import pdb
from matplotlib.pylab import *
import os
import sys
import argparse
from rdatkit.datahandlers import RDATFile
from rdatkit.settings import VARNA
import mdm_analysis
from random import shuffle
from scipy.stats import stats
from math import exp

def normalize(bonuses):
    l = len(bonuses)
    wtdata = array(bonuses)
    if wtdata.min() < 0:
	wtdata -= wtdata.min()
    interquart = stats.scoreatpercentile(wtdata, 75) - stats.scoreatpercentile(wtdata, 25)
    tenperc = stats.scoreatpercentile(wtdata, 90)
    maxcount = 0
    maxav = 0.
    for i in range(l):
	if wtdata[i] >= tenperc:
	    maxav += wtdata[i]
	    maxcount += 1
    maxav /= maxcount
    wtdata = wtdata/maxav
    return wtdata


parser = argparse.ArgumentParser()
parser.add_argument('--nsample', type=int, default=-1, help="Number of structures to sample")
parser.add_argument('--nsubopt', type=int, default=-1, help="Number of suboptimal structures to consider")
parser.add_argument('--structfile', type=argparse.FileType('r'), default=None, help="Specify structures to consider in structure file")
parser.add_argument('--nstructs', type=int, default=-1, help="Number of structures considered (taking the first nstructs from the structfile specified'")
parser.add_argument('--cutoff', type=int, default=0, help="Number of nucleotides of data to strip from the ends of the signals (which are almost always noisy)")
parser.add_argument('--outprefix', type=str, default='.', help="Prefix for output files")
parser.add_argument('--kdfile', type=argparse.FileType('r'), default=None, help="Specify the Kd's of each sequence for mutate-morph-and-map experiments")
parser.add_argument('--chemical', type=str, default=None, help="Specify the chemical for mutate-morph-and-map experiments")
parser.add_argument('--objtype', type=str, default='l2', help="Type of objective norm to use: l1 or l2 (default is l2)")
parser.add_argument('rdatfile', type=argparse.FileType('r'), help="RDAT file with the chemical mapping data")

args = parser.parse_args()

print 'Parsing RDAT'
rdat = RDATFile()
#rdat.load(open('../rdats/TEBOWNED_3D.rdat'))
rdat.load(args.rdatfile)
construct = rdat.constructs.values()[0]

seqpos = construct.seqpos
sorted_seqpos = sorted(seqpos)
sequence = construct.sequence[min(seqpos) - construct.offset - 1:max(seqpos) - construct.offset].upper()

if args.nsample > 0:
    nstructs = int(args.nsample)
    structures, energies = ss.sample(sequence, nstructs=nstructs)
if args.nsubopt > 0:
    nstructs = int(args.nsubopt)
    structures, energies = ss.subopt(sequence, fraction=0.1, nstructs=nstructs)
if args.structfile:
    structures = [ss.SecondaryStructure(dbn=x.strip()) for x in args.structfile.readlines()]
    if args.nstructs > 0:
	nstructs = args.nstructs
	structures = structures[:args.nstructs]
    else:
	nstructs = len(structures)

print 'Structures considered:'
for s in structures:
    print s.dbn

print 'Parsing mutants and data'
sequences = []
concentrations = []
kds = []
data = []
mutlabels = []
for d in construct.data:
    mutseq = list(sequence)
    for anno in d.annotations['mutation']:
        if 'WT' not in anno.upper():
	    frombase, pos, tobase = anno[0], int(anno[1:-1]), anno[1]
	    mutseq[seqpos.index(pos)] = tobase
	mutlabels.append(anno)
    sequences.append(''.join(mutseq))
    nd = abs(normalize([d.values[seqpos.index(i)] for i in sorted_seqpos]))
    data.append(nd)
    if args.chemical:
	found = False
	# Concentrations must be ALL reported in the same units (uM or mM)
	for anno in d.annotations['chemical']:
	    if args.chemical in anno: 
		found = True
		concentrations.append(float(anno.split(':')[-1].replace('uM', '').replace('mM', '')))
	if not found:
	    concentrations.append(0)

data = array(data)

if args.chemical:
    print 'Concentrations of %s are %s' (args.chemical, concentrations)

imshow(data, cmap=get_cmap('Greys'), vmin=0, vmax=data.mean(), aspect='auto', interpolation='nearest')
savefig('%s/real_data.png' % args.outprefix, dpi=300)

print 'Doing SVD to get a lower bound on the number of states'
U, s, V = linalg.svd(data)
S = diag(s)
for i in xrange(nstructs,nstructs+1):
    print '==================================================='
    print 'FOR %s STRUCTURES' % i
    print '==================================================='
    for j, s in enumerate([0, 0.5,1,5,10,20,50,100,200,500,1000,2000]):
	#svd_pred = dot(U[:,:i], dot(S[:i,:i], V[:i,:]))
	#data_pred_lin, d_pred_lin, alpha_pred_lin = mdm_analysis.linear_reactivity_model(sequences, structures[:i], data.T, objtype=args.objtype, concentrations=concentrations, kds=kds, cutoff=args.cutoff)
	data_pre = data[(48*j):(48+48*j),:]
	data_pred_fact, d_pred_fact, alpha_pred_fact = mdm_analysis.factor_analysis(data_pre.T, i, cutoff=args.cutoff)
	if j == 0:
	    full_alpha = alpha_pred_fact
	else:
	    tmp = []
	    for p in xrange(48):
		g = nx.Graph()
		g.add_nodes_from(range(6))
		for ii in xrange(3):
		    g.add_weighted_edges_from([(ii, x, 1./abs(full_alpha[ii,p] - alpha_pred_fact[x-3,p])) for x in xrange(3, 6)])
		mdict = nx.algorithms.max_weight_matching(g)
		order = [mdict[x] - 3 for x in xrange(3)]
		tmp.append(alpha_pred_fact[order,p].tolist())
	    full_alpha = append(full_alpha, array(tmp).T, axis=1)
	#imshow(data_pred_lin.T, cmap=get_cmap('Greys'), vmin=0, vmax=data_pred_lin.mean(), aspect='auto', interpolation='nearest')
	#savefig('%s/predicted_data_linear_mod_%s_structures.png' % (args.outprefix, j), dpi=300)
	imshow(data_pred_fact.T, cmap=get_cmap('Greys'), vmax=data_pred_fact.mean(), aspect='auto', interpolation='nearest')
	xticks(range(len(sequence)), ['%s%s' % (i, x) for i, x in enumerate(sequence)], fontsize='x-small', rotation=90)
	yticks(range(len(mutlabels[(j*48):(48 + j*48)])), mutlabels[(j*48):(48 + j*48)], fontsize='x-small')
	savefig('%s/predicted_data_factor_analysis_%s_structures.png' % (args.outprefix, j), dpi=300)
	#imshow(svd_pred, cmap=get_cmap('Greys'), vmin=0, vmax=svd_pred.mean(), aspect='auto', interpolation='nearest')
	#savefig('%s/predicted_data_svd_%s_structures.png' % (args.outprefix, j), dpi=300)
	imshow(data_pre, cmap=get_cmap('Greys'), vmin=0, vmax=data.mean(), aspect='auto', interpolation='nearest')
	xticks(range(len(sequence)), ['%s%s' % (i, x) for i, x in enumerate(sequence)], fontsize='x-small', rotation=90)
	yticks(range(len(mutlabels[(j*48):(48 + j*48)])), mutlabels[(j*48):(48 + j*48)], fontsize='x-small')
	savefig('%s/real_data_%s.png' % (args.outprefix, j), dpi=300)
"""
print 'Graphing structures'
for i, s in enumerate(structs):
    cmd = 'java -cp %s fr.orsay.lri.varna.applications.VARNAcmd -sequenceDBN %s -structureDBN "%s" -o addtest/%s.png' % (VARNA, sequence, s.dbn, i)
    print cmd
    os.popen(cmd)
"""
pickle.dump(full_alpha, open('weights.pickle','w'))
for i in xrange(2):
    clf()
    plot(full_alpha.T[range(i,576,48),:])
    savefig('%s/struct_weight_plot__%s.png' % (args.outprefix, i), dpi=300)
for i in range(12):
    clf()
    plot(full_alpha.T[(i*48):(48 + i*48),:])
    savefig('weights_%s.png' % i, dpi=120)
