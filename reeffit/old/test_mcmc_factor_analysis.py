import mcmc_factor_analysis as mcmcfa
import pickle
from rdatkit.datahandlers import RDATFile
from rdatkit import secondary_structure as ss
from matplotlib.pylab import *
from numpy import linalg
from scipy import stats

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
data = array(data)
#data = array(data[576-48:576])
selected = range(48) + range(48*4, 48*4+48)+ range(48*7, 48*7+48)+ range(48*11, 48*11+48)

data = data[selected,:]
mut_labels = mut_labels[:48]


structures = ['..........................(((((((............)))))))....................',\
'.....((((......((((....)))).....))))....................................',\
'.....((((......((((....)))).....)))).(((((....))))).....................']

structures = [s[cutoff:-cutoff-20] for s in structures]
print 'Considering the following structures:'
for s in structures:
    print s

nstructs = len(structures)
npos = data.shape[1]
nmeas = data.shape[0]
nboot = 1

Wboot = zeros([nmeas, nstructs, nboot])
Psiboot = zeros([nmeas, nmeas, nboot])
E_dboot = zeros([nstructs, npos, nboot])
print 'Starting bootstrap...'
for b in xrange(nboot):
    print 'Calling factor analysis module'
    W, Psi, E_d = mcmcfa.analyze(data, structures, max_iterations=2)

    scale_factors = linalg.lstsq(W[:,:nstructs], ones(nmeas))[0]
    scale_factors = [norm(E_d[s,:],ord=2) for s in xrange(nstructs)]

    for s in xrange(nstructs):
	W[:,s] *= scale_factors[s]
	E_d[s,:] /= scale_factors[s]

    data_pred = array(dot(W, E_d))
    Wboot[:,:,b] = W
    E_dboot[:,:,b] = E_d
    Psiboot[:,:,b] = Psi

    if b == 0:
	print 'Plotting data'
	imshow(data, cmap=get_cmap('Greys'), vmin=0, vmax=data.mean(), aspect='auto', interpolation='nearest')
	savefig('realdata.png')
	imshow(data_pred, cmap=get_cmap('Greys'), vmin=0, vmax=data_pred.mean(), aspect='auto', interpolation='nearest')
	savefig('predicteddata.png')

E_d = E_dboot.mean(axis=2)
W = Wboot.mean(axis=2)
Psi = Psiboot.mean(axis=2)

E_dstd = E_dboot.std(axis=2)
Wstd = Wboot.std(axis=2)
Psistd = Psiboot.std(axis=2)

for i, s in enumerate(structures):
    binarized_react = [1 if x == '.' else 0 for x in s]
    figure(1)
    clf()
    title('Structure %s: %s' % (i, s))
    exp_react = array(E_d[i,:])
    plot(1+arange(len(s)), binarized_react, linewidth=2, c='r')
    bar(0.5+arange(len(s)), exp_react, linewidth=0, width=1, color='gray')
    errorbar(1+arange(len(s)), exp_react, yerr=E_dstd[i,:], fmt='black')
    xticks(1 + arange(len(s)), ['%s%s' % (j+1, x) for j, x in enumerate(sequence[cutoff:-cutoff])], fontsize='x-small', rotation=90)
    xlim(1,len(s))
    ylim(0,1.5)
    savefig('exp_react_struct_%s.png' % i)

clf()
for i in xrange(len(structures)):
    #plot(W[:,i], label='structure %s' % i, linewidth=2)
    errorbar(range(nmeas), W[:,i], yerr=Wstd[:,i], linewidth=2, label='structure %s' % i)
legend()
xticks(xrange(len(mut_labels)), mut_labels, fontsize='x-small', rotation=90)
savefig('weights_by_mutant.png')
pickle.dump(W, open('W.pickle', 'w'))
