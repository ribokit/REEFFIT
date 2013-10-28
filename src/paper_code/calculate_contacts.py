import argparse
import pdb
import matplotlib
import os
matplotlib.use('Agg')
from matplotlib.pylab import *
from rdatkit.datahandlers import RDATFile
import mapping_analysis
from map_analysis_utils import *
import rdatkit.secondary_structure as ss
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('rdatfile', type=argparse.FileType('r'))
parser.add_argument('structfile', type=argparse.FileType('r'))
parser.add_argument('--cutoff', default=0, type=int)
parser.add_argument('outprefix', type=str)

args = parser.parse_args()

def plot_mutxpos_image(d, cm='Greys'):
        imshow(d, cmap=get_cmap(cm), vmin=0, vmax=d.mean(), aspect='auto', interpolation='nearest')
        #imshow(d, cmap=get_cmap('Greys'), vmin=0, vmax=1.0, aspect='auto', interpolation='nearest')
        #imshow(d, cmap=cmap, aspect='auto', interpolation='nearest')
        xticks(range(len(sequence)), ['%s%s' % (pos, sequence[pos - offset - 1]) for pos in seqpos], fontsize='x-small', rotation=90)
        yticks(range(len(mut_labels)), mut_labels, fontsize='x-small')


print 'Parsing RDAT'
rdat = RDATFile()
rdat.load(args.rdatfile)
construct = rdat.constructs.values()[0]

seqpos = construct.seqpos
sorted_seqpos = sorted(seqpos)
offset = construct.offset
sequence = construct.sequence[min(seqpos) - offset - 1:max(seqpos) - offset].upper()


print 'Parsing mutants and data'
data = []

mut_labels = []
mutants = []
mutpos = []
wt_idx = 0
for idx, d in enumerate(construct.data):
    label = d.annotations['mutation'][0]
    if label == 'WT':
        mutant = sequence
        wt_idx = idx
        pos = -1
    else:
        pos = int(label[1:len(label)-1]) - 1 - construct.offset
        mutant = sequence[:pos] + label[-1] + sequence[pos+1:]
    if mutant in mutants:
        continue
    mutpos.append(pos)
    mut_labels.append(label)
    mutants.append(mutant)
    nd = normalize([d.values[seqpos.index(i)] for i in sorted_seqpos])
    data.append(nd)

data = array(data)
if args.cutoff > 0:
    data_cutoff = data[:, args.cutoff:-args.cutoff]
    seqpos_cutoff = sorted_seqpos[args.cutoff:-args.cutoff]
else:
    data_cutoff = data
    seqpos_cutoff = sorted_seqpos

selected_structures = [line.strip() for line in args.structfile.readlines()]

energies = get_free_energy_matrix(selected_structures, mutants)
W_0 = calculate_weights(energies)


print 'Reading pickled data'
W = pickle.load(open('%s/W.pickle' % args.outprefix))
W_std = pickle.load(open('%s/W_std.pickle' % args.outprefix))
Psi = pickle.load(open('%s/Psi.pickle' % args.outprefix))
D = pickle.load(open('%s/D.pickle' % args.outprefix))
E_ddT = pickle.load(open('%s/E_ddT.pickle' % args.outprefix))
M = pickle.load(open('%s/M.pickle' % args.outprefix))

fa = mapping_analysis.FAMappingAnalysis(data, selected_structures, cutoff=args.cutoff, mutpos=mutpos)
fa.W = W
fa.E_d = D
fa.Psi = Psi
fa.E_ddT = E_ddT
fa.M = M

corr_facs, data_pred = fa.correct_scale()
C = fa.calculate_contacts()
data_pred_contacts = data_pred + C

figure(1)
clf()
plot_mutxpos_image(data_cutoff)
savefig('%s/real_data.png' % (args.outprefix), dpi=300)

figure(1)
clf()
plot_mutxpos_image(data_pred)
savefig('%s/fa_data_pred.png' % (args.outprefix), dpi=300)

figure(1)
clf()
plot_mutxpos_image(data_pred_contacts)
savefig('%s/data_pred_contacts.png' % (args.outprefix), dpi=300)

figure(1)
clf()
imshow(C, cmap=get_cmap('jet'), interpolation='nearest')
savefig('%s/contacts.png' % (args.outprefix), dpi=300)


figure(3)
clf()
for i in xrange(len(selected_structures)):
    errorbar(range(W.shape[0]), W[:,i], yerr=W_std[:,i], linewidth=2, label='structure %s ' % i)
ylim(0,1)
legend()
savefig('%s/weights_by_mutant.png' % (args.outprefix), dpi=300)

for i in xrange(len(selected_structures)):
    figure(3)
    clf()
    errorbar(range(W.shape[0]), W[:,i], yerr=W_std[:,i], linewidth=2, label='structure %s ' % i)
    plot(range(W_0.shape[0]), W_0[:,i], linewidth=2, label='starting weight structure %s ' % i)
    legend()
    ylim(0,1)
    savefig('%s/weights_by_mutant_structure%s.png' % (args.outprefix, i), dpi=300)

"""
r = range(data_cutoff.shape[1])
for i in xrange(data_cutoff.shape[0]):
    figure(5)
    clf()
    plot(r, data_cutoff[i,:], color='r', label='Data', linewidth=2)
    plot(r, data_pred[i,:], color='b', label='Predicted', linewidth=2)
    title('%s' % mut_labels[i])
    xticks(r[0:len(r):5], seqpos_cutoff[0:len(seqpos_cutoff):5], rotation=90)
    legend()
    savefig('%s/data_vs_predicted_%s_%s.png' % (args.outprefix, i, mut_labels[i]))
"""
