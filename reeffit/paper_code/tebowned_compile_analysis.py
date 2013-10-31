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
'.....((((......((((....)))).....))))....................................',\
'.....((((......((((....)))).....)))).(((((....))))).....................']
structures = [s[cutoff:-cutoff-20] for s in structures]

sets, mut_labels, nmeas  = pickle.load(open('factor_analysis_variables.pickle'))
mut_labels = mut_labels*len(sets)
nstructs = len(structures)

E_d_all = pickle.load(open('E_dboot.pickle'))
E_ddT_all = pickle.load(open('E_ddTboot.pickle'))


E_d_mean =  mat(E_d_all.mean(axis=2))

E_ddT_mean = E_ddT_all.mean(axis=2)

data_E_d = mat(zeros([data.shape[0], nstructs]))

for i in xrange(npos):
    data_E_d += dot(data[:,i], E_d_mean[:,i].T)

W_mean = dot(data_E_d, linalg.inv(E_ddT_mean))

# Proper scaling

scale = 2.5

W_mean *= 1/scale
E_d_mean *= scale
W_all = zeros([nmeas*len(sets), nstructs, E_d_all.shape[2]])

for i in xrange(E_d_all.shape[2]):
    data_E_d = mat(zeros([48*len(sets), nstructs]))
    E_d_b = mat(E_d_all[:,:,i])
    for j in xrange(npos):
	data_E_d += dot(data[:,j], E_d_b[:,j].T)
    W_all[:,:,i] = dot(data_E_d, linalg.inv(E_ddT_all[:,:,i]))
    W_all[:,:,i] *= 1/scale
 
Wstd = W_all.std(axis=2)


print 'Plotting expected reactivities'
E_d_std = (E_d_all*scale).std(axis=2)

colors = ('blue','green','red')
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
savefig('compiled/weights_by_mutant.png', dpi=300)
exit()
print 'Plotting WT weights'
figure(5)
semilogx(sets, W_mean[arange(0, len(sets)*nmeas, nmeas)], 'o')
savefig('compiled/WT_weights.png', dpi=300)

print 'Plotting predicted data'
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
	#errorbar(range(48), array(W_mean[indices,i].T)[0], yerr=Wstd[indices,i], linewidth=2, label='structure %s' % i)
	plot(range(48), array(W_mean[indices,i].T)[0], linewidth=2, label='structure %s' % i)
    #legend()
    xticks(xrange(len(mut_labels[:48])), mut_labels[:48], fontsize='x-small', rotation=90)
    savefig('compiled/%uM_weights_by_mutant.png' % r, dpi=300)
    figure(4)
    zdiff = abs(zscore(data_r, axis=1) - zscore(data_pred, axis=1))
    plot_mutxpos_image(zdiff, cmap=get_cmap('jet'))
    savefig('compiled/%suM_zscore_diff.png' % r, dpi=300)

print 'Plotting WT weights boxplot'
for s in xrange(len(structures)): 
    figure(6)
    clf()
    Wforbox = [ W_all[i,s,:] for i in arange(0, nmeas*len(sets), nmeas)]
    boxplot(Wforbox)
    title('WT weights for structure %s' % s)
    savefig('compiled/structure_%s_WT_weights.png' % s, dpi=300)



