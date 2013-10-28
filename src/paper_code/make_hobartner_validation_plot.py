import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *
from rdatkit.datahandlers import RDATFile
import plot_utils


print 'Parsing RDAT'
rdat = RDATFile()
rdat.load(open('../rdat/HOBIST_SHP_0003.rdat'))
construct = rdat.constructs.values()[0]
seqpos = construct.seqpos
sorted_seqpos = sorted(seqpos)
start = sorted_seqpos.index(1)
end = sorted_seqpos.index(25)+1

data = []
for idx, d in enumerate(construct.data):
    nd_tmp = normalize([d.values[seqpos.index(i)] for i in sorted_seqpos])
    nd = array([d.values[seqpos.index(i)] if d.values[seqpos.index(i)] > 0 else 0.001 for i in sorted_seqpos])
    if 'mutation' in d.annotations:
        if 'WT' in d.annotations['mutation']:
            wt = nd[start:end]
        if 'C20G' in d.annotations['mutation']:
            c20g = nd[start:end]
        if 'C5G' in d.annotations['mutation']:
            c5g = nd[start:end]




f = figure(1)
f.set_size_inches(15, 5)
clf()
r = arange(1,26)
plot(r, c20g, color=plot_utils.STRUCTURE_COLORS[1], linewidth=2)
savefig('hobartner_analysis/C20G.png', dpi=200)

f = figure(2)
f.set_size_inches(15, 5)
clf()
r = arange(1,26)
plot(r, c5g, color=plot_utils.STRUCTURE_COLORS[0], linewidth=2)
savefig('hobartner_analysis/C5G.png', dpi=200)

f = figure(2)
f.set_size_inches(15, 5)
clf()
r = arange(1,26)
plot(r, wt, 'r', linewidth=2)
plot(r, 0.7*c20g + 0.3*c5g, 'b', linewidth=2)
savefig('hobartner_analysis/WT_C20_C5G.png', dpi=200)
