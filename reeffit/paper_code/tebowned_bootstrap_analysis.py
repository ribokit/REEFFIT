import mcmc_factor_analysis as mcmcfa
import scipy.optimize
import pickle
import os
import pdb
from rdatkit.datahandlers import RDATFile
from rdatkit import secondary_structure as ss
from matplotlib.pylab import *
from numpy import linalg
from scipy import stats
import random

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

structures = ['..........................(((((((............)))))))....................',\
'.....((((......((((....)))).....))))....................................',\
'.....((((......((((....)))).....)))).(((((....))))).....................']
nstructs = len(structures)
npos = data.shape[1]
nmuts = 48 #we've got 48 mutants
nmeas = 48*3 #number of measurements for bootstrap


print 'Considering the following structures:'
structures = [s[cutoff:-cutoff-20] for s in structures]

for s in structures:
    print s


# We have sets = [0,0.5,1,5,10,20,50,100,200,500,1000,2000] for uM concentrations of FMN
sets = [0,0.5,1,5,10,20,50,100,200,500,1000,2000]
totnmeas = len(sets)*nmuts
nboot = 5
Wboot = zeros([nmeas, nstructs, nboot])
Psiboot = zeros([nmeas, nmeas, nboot])
E_dboot = zeros([nstructs, npos, nboot])
E_ddTboot = zeros([nstructs, nstructs, nboot])

all_indices = range(totnmeas)
print 'Starting bootstrap...'
for b in xrange(nboot):
    print 'Creating directory'
    if not os.path.exists('bootstrap/boot_%s' % b):
	os.mkdir('bootstrap/boot_%s' %b)
    indices = random.sample(all_indices, nmeas)
    data_r = array(data[indices,:])

    print 'Calling factor analysis module'
    W, Psi, E_d, E_ddT = mcmcfa.analyze(data_r, structures, max_iterations=5)
    
    # We can either find the scale factors that force
    # the weights to sum up to one...
    #scale_factors = linalg.lstsq(W[:,:nstructs], ones(nmeas))[0]
    # Or normalize the expected reaclitivities
    """
    scale_factors = [norm(E_d[s,:],ord=2)/7. for s in xrange(nstructs)]

    for s in xrange(nstructs):
	W[:,s] *= scale_factors[s]
	E_d[s,:] /= scale_factors[s]
	E_ddT[s,:] /= scale_factors[s]
    """
    data_pred = array(dot(W, E_d))
    Wboot[:,:,b] = W
    E_dboot[:,:,b] = E_d
    E_ddTboot[:,:,b] = E_ddT
    Psiboot[:,:,b] = Psi
    mut_labelsboot = [mut_labels[i] for i in indices]
    print 'Plotting data'
    figure(1)
    clf()
    imshow(data_r, cmap=get_cmap('Greys'), vmin=0, vmax=data_r.mean(), aspect='auto', interpolation='nearest')
    xticks(range(len(sequence)), ['%s%s' % (i, x) for i, x in enumerate(sequence)], fontsize='x-small', rotation=90)
    yticks(range(len(mut_labelsboot)), mut_labelsboot, fontsize='x-small')
    savefig('bootstrap/boot_%s/realdata.png' % b, dpi=300)
    clf()
    imshow(data_pred, cmap=get_cmap('Greys'), vmin=0, vmax=data_pred.mean(), aspect='auto', interpolation='nearest')
    xticks(range(len(sequence)), ['%s%s' % (i, x) for i, x in enumerate(sequence)], fontsize='x-small', rotation=90)
    yticks(range(len(mut_labelsboot)), mut_labelsboot, fontsize='x-small')
    savefig('bootstrap/boot_%s/predicteddata.png' % b, dpi=300)

print 'Analysis finished, plotting and saving final results'
pickle.dump(Wboot, open('Wboot.pickle', 'w'))
pickle.dump(E_dboot, open('E_dboot.pickle', 'w'))
pickle.dump(E_ddTboot, open('E_ddTboot.pickle', 'w'))


