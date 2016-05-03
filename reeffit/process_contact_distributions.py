from matplotlib.pylab import *
import os
import scipy.stats as stats

from rdatkit import RDATFile, SecondaryStructure

import map_analysis_utils as utils

rdatdir = '../rdat/mutate_and_map_for_training/'
diagdata, offdiagdata, alldata, contactdeltadata = [], [], [], []

for fname in os.listdir(rdatdir):
    if fname == '.svn':
        continue
    print 'Doing %s' % fname
    rdat = RDATFile()
    rdat.load(open(rdatdir + fname))
    construct = rdat.constructs.values()[0]
    struct = SecondaryStructure(dbn=construct.structure)
    bp_dict = struct.base_pair_dict()
    sorted_seqpos = sorted(construct.seqpos)
    wtread = False
    for d in construct.data:
        label = d.annotations['mutation'][0].replace('Lib1-', '').replace('Lib2-', '').replace('Lib3-', '')

        if label == 'WT':
            wt_nd = utils.normalize([d.values[construct.seqpos.index(i)] for i in sorted_seqpos])
            wtread = True
        else:
            pos = int(label[1:len(label)-1]) - 1 - construct.offset
            nd = utils.normalize([d.values[construct.seqpos.index(i)] for i in sorted_seqpos])
            alldata += nd.tolist()
            diagdata.append(nd[pos])
            if wtread:
                #contactdeltadata.append((nd - wt_nd)[pos])
                contactdeltadata += (nd - wt_nd).tolist()
            if pos in bp_dict:
                if wtread:
                    pass
                    #contactdeltadata.append((nd - wt_nd)[bp_dict[pos]])
                offdiagdata.append(nd[bp_dict[pos]])

print 'Fitted gammas (shape, loc, scale):'
print 'All data'
allparams = stats.expon.fit(alldata)
print allparams
print 'Diagonal'
diagparams = stats.gamma.fit(diagdata)
print diagparams
print 'Off-diagonal'
offdiagparams = stats.gamma.fit(offdiagdata)
print offdiagparams
print 'Contact delta'
contactdeltaparams = stats.cauchy.fit(contactdeltadata)
#contactdeltaparams = [0.036036085880561453, 3.0564874002215925]
print contactdeltaparams

x = linspace(0, 5, 100)
x2 = linspace(-1, 1, 1000)
diagpdf = stats.gamma.pdf(x, diagparams[0], loc=diagparams[1], scale=diagparams[2])
offdiagpdf = stats.gamma.pdf(x, offdiagparams[0], loc=offdiagparams[1], scale=offdiagparams[2])
contactdeltapdf = stats.cauchy.pdf(x2, loc=contactdeltaparams[0], scale=contactdeltaparams[1])
allpdf = stats.expon.pdf(x, loc=allparams[0], scale=allparams[1])

figure(1)
plot(x, diagpdf, 'r')
hist(diagdata, 100, normed=True, alpha=0.3)
savefig('diagonal_reactivities.png')
clf()
plot(x, offdiagpdf, 'r')
hist(offdiagdata, 100, normed=True, alpha=0.3)
savefig('offdiagonal_reactivities.png')
clf()
plot(x, allpdf, 'r')
hist(alldata, 100, normed=True, alpha=0.3, range=(0, 3))
xlim(0, 3)
savefig('all_reactivities.png')
clf()
plot(x2, contactdeltapdf, 'r')
hist(contactdeltadata, 200, normed=True, alpha=0.3, range=(-1, 1))
xlim(-1, 1)
savefig('contactdelta_reactivities.png', dpi=300)

