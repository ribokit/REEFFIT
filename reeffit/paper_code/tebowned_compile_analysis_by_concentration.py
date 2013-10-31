import mcmc_factor_analysis as mcmcfa
import scipy.optimize
from scipy.stats import zscore
import pickle
import os
import pdb
from rdatkit.datahandlers import RDATFile
from rdatkit import secondary_structure as ss
from matplotlib.pylab import *
from numpy import linalg
from scipy import stats

def get_base_pair_indices(struct, mutpos):
    bp_indices = []
    stack = []
    for i, s in enumerate(struct):
	if s == '(':
	    stack.append(i)
	if s == ')':
	    j = stack.pop()
	    if i in mutpos:
		idx = mutpos.index(i)
		bp_indices.append([mutpos.index(i),j])
		bp_indices.append([mutpos.index(j),i])
    return bp_indices

def plot_mutxpos_image(d, cmap=get_cmap('Greys')):
    imshow(d, cmap=cmap, vmin=0, vmax=d.mean(), aspect='auto', interpolation='nearest')
    xticks(range(len(sequence[cutoff:-cutoff])), ['%s%s' % (i+1, x) for i, x in enumerate(sequence[cutoff:-cutoff])], fontsize='x-small', rotation=90)
    yticks(range(48), mut_labels[:48], fontsize='x-small')

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


print 'Parsing RDAT'
rdat = RDATFile()
rdat.load(open('../rdat/TEBOWNED3D.rdat'))
construct = rdat.constructs.values()[0]

seqpos = construct.seqpos
sorted_seqpos = sorted(seqpos)
sequence = construct.sequence[min(seqpos) - construct.offset - 1:max(seqpos) - construct.offset].upper()


print 'Parsing mutants and data'
data = []

mut_labels = []
cutoff = 1 
for d in construct.data:
    mut_labels.append(d.annotations['mutation'][0])
    nd = normalize([d.values[seqpos.index(i)] for i in sorted_seqpos])[cutoff:-cutoff]
    data.append(nd)

data = mat(array(data))
npos = data.shape[1]
structures = ['..........................(((((((............)))))))....................',\
'.....((((aaaaaa((((....))))aaaaa))))....................................',\
'.....((((......((((....)))).....))))....................................',\
'.....((((......((((....)))).....)))).(((((....))))).....................']
structures = [s[cutoff:-cutoff-20] for s in structures]

sets, mut_labels, nmeas  = pickle.load(open('factor_analysis_variables.pickle'))
mut_labels = mut_labels*len(sets)
nstructs = len(structures)

E_d_all = pickle.load(open('E_d_all.pickle'))
E_ddT_all = pickle.load(open('E_ddT_all.pickle'))
W_all = pickle.load(open('W_all.pickle'))

# Set a 'mask' to the indices in the data thare are perturbed given the mutations and structures
mutpos = [int(label[1:-1])-1 if label != 'WT' else 'WT' for label in mut_labels[:nmeas]]
struct_mut_mask = [[x,x] for x in mutpos if x != 'WT']
for i, s in enumerate(structures):
    struct_mut_mask += get_base_pair_indices(s, mutpos)
struct_mut_mask_rows, struct_mut_mask_cols = zip(*struct_mut_mask)

for i, s in enumerate(sets):
    indices = arange(nmeas*i, nmeas*(i+1))
    W = W_all[indices, :]
    E_d = E_d_all[:,:,i]
    data_pred = array(dot(W, E_d))
    contact_map = zeros(data_pred.shape)
    diff = abs(data[indices,:] - data_pred)[struct_mut_mask_rows, struct_mut_mask_cols]
    contact_map[struct_mut_mask_rows, struct_mut_mask_cols] = array(diff)[0]
    figure(2)
    clf()
    matshow(contact_map)
    xticks(range(len(sequence[cutoff:-cutoff])), ['%s%s' % (i+1, x) for i, x in enumerate(sequence[cutoff:-cutoff])], fontsize='x-small', rotation=90)
    yticks(range(48), mut_labels[:48], fontsize='x-small')
    savefig('compiled/%suM_contact_map.png' % s, dpi=300)
    figure(2)
    clf()
    matshow(zscore(data[indices,:] - data_pred, axis=0))
    xticks(range(len(sequence[cutoff:-cutoff])), ['%s%s' % (i+1, x) for i, x in enumerate(sequence[cutoff:-cutoff])], fontsize='x-small', rotation=90)
    yticks(range(48), mut_labels[:48], fontsize='x-small')
    savefig('compiled/%suM_diff.png' % s, dpi=300)


E_d_mean =  mat(E_d_all.mean(axis=2))

E_ddT_mean = E_ddT_all.mean(axis=2)

data_E_d = mat(zeros([data.shape[0], nstructs]))

for i in xrange(npos):
    data_E_d += dot(data[:,i], E_d_mean[:,i].T)

W_mean = dot(data_E_d, linalg.inv(E_ddT_mean))

print 'Plotting expected reactivities'
E_d_std = E_d_all.std(axis=2)

colors = ('blue','green','red','teal')
for i, s in enumerate(structures):
    binarized_react = [1 if x == '.' else 0 for x in s]
    f = figure(10)
    f.set_size_inches(15, 5)
    clf()
    title('Structure %s: %s' % (i, s))
    exp_react = (E_d_mean[i,:]).tolist()[0]
    plot(1+arange(len(s)), binarized_react, linewidth=2, c='r')
    bar(0.5+arange(len(s)), exp_react, linewidth=0, width=1, color=colors[i])
    errorbar(1+arange(len(s)), exp_react, yerr=E_d_std[i,:], fmt='black')
    xticks(1 + arange(len(s)), ['%s%s' % (j+1, x) for j, x in enumerate(sequence[cutoff:-cutoff])], fontsize='x-small', rotation=90)
    xlim(1,len(s))
    ylim(0,2)
    savefig('compiled/exp_react_struct_%s.png' %  i, dpi=300)


f = figure(1)
f.set_size_inches(45, 5)
clf()
for i in xrange(len(structures)):
    plot(W_mean[:,i], label='structure %s' % i, linewidth=2)
#legend()
xticks(xrange(len(mut_labels)), mut_labels, fontsize='xx-small', rotation=90)
savefig('compiled/mean_weights_by_mutant.png', dpi=300)
f = figure(1)
f.set_size_inches(45, 5)
clf()
for i in xrange(len(structures)):
    plot(W_all[:,i], label='structure %s' % i, linewidth=2)
#legend()
xticks(xrange(len(mut_labels)), mut_labels, fontsize='xx-small', rotation=90)
savefig('compiled/weights_by_mutant.png', dpi=300)

print 'Plotting mean WT weights'
figure(5)
clf()
semilogx(sets, W_mean[arange(0, len(sets)*nmeas, nmeas)])
savefig('compiled/mean_WT_weights.png', dpi=300)
print 'Plotting WT weights'
figure(5)
clf()
kd = 8.888777e+00
weights = W_all[arange(0, len(sets)*nmeas, nmeas)]
"""
for i,c in enumerate(sets):
    if weights[i,1] != 0:
	print weights[i,:]
	scale = (c/(c + kd))/weights[i,1]
	weights[i,:] *= scale
	totleft = 1 - weights[i,1]
	scaleleft = totleft/weights[i,[0,2,3]].sum()
	weights[i,[0,2,3]] *= scaleleft
	print weights[i,:]
        """
semilogx(sets, weights)
savefig('compiled/WT_weights.png', dpi=300)
exit()
print 'Plotting mean predicted data'
for k, r in enumerate(sets):
    indices = arange(48*k, 48*k + 48)
    data_r = array(data[indices,:])
    data_pred = array(dot(W_mean[indices,:], E_d_mean))
    figure(2)
    clf()
    plot_mutxpos_image(data_r)
    savefig('compiled/%suM_realdata.png' % r, dpi=300)
    clf()
    plot_mutxpos_image(data_pred)
    savefig('compiled/%suM_predicteddata.png' % r, dpi=300)
    f = figure(3)
    f.set_size_inches(15, 5)
    clf()
    for i in xrange(len(structures)):
	plot(range(48), array(W_mean[indices,i].T)[0], linewidth=2, label='structure %s' % i)
    #legend()
    xticks(xrange(len(mut_labels[:48])), mut_labels[:48], fontsize='x-small', rotation=90)
    savefig('compiled/%uM_weights_by_mutant.png' % r, dpi=300)
    figure(4)
    zdiff = abs(zscore(data_r, axis=1) - zscore(data_pred, axis=1))
    plot_mutxpos_image(zdiff, cmap=get_cmap('jet'))
    savefig('compiled/%suM_zscore_diff.png' % r, dpi=300)
