# Cluster by mutant profile similarity
import matplotlib
matplotlib.use('Agg')
import sys
import scipy.cluster.hierarchy as sch
from rdatkit.datahandlers import RDATFile
from map_analysis_utils import *
from plot_utils import plot_mutxpos_image

fname = sys.argv[1]
outname = sys.argv[2]

print 'Parsing RDAT'
rdat = RDATFile()
rdat.load(open(fname))
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
for idx, d in enumerate(construct.data):
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
    mut_labels.append(label)
    mutants.append(mutant)
    #if args.nonormalization:
        #nd_tmp = normalize([d.values[seqpos.index(i)] for i in sorted_seqpos])
        #nd = [d.values[seqpos.index(i)] if d.values[seqpos.index(i)] >= 0 else 0.001 for i in sorted_seqpos]
        #nd = [nd[i] if nd[i] < 4 else max(nd_tmp) for i in range(len(nd))]
        #nd = [d.values[seqpos.index(i)] for i in sorted_seqpos]
    nd = normalize([d.values[seqpos.index(i)] for i in sorted_seqpos])
    data.append(nd)
data = array(data)

print 'Calculating distance matrix'
D = zeros([data.shape[0], data.shape[0]])

for i in xrange(D.shape[0]):
    for j in xrange(D.shape[1]):
        D[i,j] = abs(data[i,:] - data[j,:]).sum()

Y = sch.linkage(D, method='centroid')
Z = sch.dendrogram(Y, orientation='right')
indices = Z['leaves']

print 'Plotting'
figure()
plot_mutxpos_image(data[indices,:], sequence, seqpos, offset, [mut_labels[i] for i in indices], cmap=get_cmap('Greys'))
savefig(outname, dpi=300)
