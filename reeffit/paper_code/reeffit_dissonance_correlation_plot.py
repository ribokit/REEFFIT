from matplotlib.pylab import *
from rdatkit.datahandlers import RDATFile
from non_parametric_tests_of_stability import *
from map_analysis_utils import normalize
import pickle


dirs = ['16s_4way_junction_analysis', 'HHRz_G12A_analysis', 'hobartner_analysis', 'medloop_eterna_analysis', 'medloop_delta_analysis', 'tcf21_analysis', 'tristable_analysis']
rdats =['r16S_20130531_2D_1M7_new.rdat', 'ModifiedFirstSeqToWTG12A.rdat', 'HOBIST_SHP_0003.rdat', 'MDLOOP_ETERNA_0001.rdat', 'Medloop_EteRNA.rdat', 'TCF21_ETERNA_0001.rdat', 'TRIFOR_1M7_0000.rdat']
do_normalize = [True, False, False, False, False, False, False]
def get_rdat_data(fname, norm):
    print 'Parsing %s' % fname
    rdat = RDATFile()
    rdat.load(open(fname))
    construct = rdat.constructs.values()[0]
    seqpos = construct.seqpos
    sorted_seqpos = sorted(seqpos)
    offset = construct.offset
    sequence = construct.sequence[min(seqpos) - offset - 1:max(seqpos) - offset].upper()
    for idx, d in enumerate(construct.data):
        if 'warning' not in d.annotations:
            tmp = normalize([d.values[seqpos.index(i)] for i in sorted_seqpos])
            if norm:
                nd = tmp.tolist()
            else:
                nd = [d.values[seqpos.index(i)] if d.values[seqpos.index(i)] > 0 else tmp.mean() for i in sorted_seqpos]
            if 'mutation' not in d.annotations or 'WT' in d.annotations['mutation']:
                return nd
        else:
            print 'Skipping %s data, marked as badQuality' % idx
    return None

reeffit_ents = []
diss_ents = []
diss_means = []
labels = []
data = []
pdata = []
edata = []
alldata = []
for i, dname in enumerate(dirs):
    d = get_rdat_data('../rdat/' + rdats[i], do_normalize[i])
    alldata += d
    d = array(d)
    data.append(d)
    pdata += d[d <= d.mean()].tolist()
    edata += d[d > d.mean()].tolist()
    W = pickle.load(open(dname + '/W.pickle'))
    ent = sum([-W[0,j]*log(W[0,j]) if W[0,j] > 0 else 0 for j in xrange(W.shape[1])])
    reeffit_ents.append(ent)
    labels.append(dname[:dname.find('_analysis')])

pdata = array(pdata)
edata = array(edata)
alldata = array(alldata)
for d in data:
    diss_ents.append(dissonance_entropy(array(d), edata.mean(),  pdata.mean()))
    diss_means.append(dissonance_mean(d, refmean=alldata.mean()))

figure(1)
clf()
scatter(diss_ents, reeffit_ents, color='blue', alpha=0.6)
ylabel('REEFFIT entropy')
xlabel('Dissonance entropy')
for i, name in enumerate(labels):
    t = text(diss_ents[i], reeffit_ents[i], name)
    t.set_color('b')


savefig('reeffit_vs_dissonance_entropies.png', dpi=200)
figure(1)
clf()
scatter(diss_means, reeffit_ents, color='blue', alpha=0.6)
ylabel('REEFFIT entropy')
xlabel('Dissonance mean')
for i, name in enumerate(labels):
    t = text(diss_means[i], reeffit_ents[i], name)
    t.set_color('b')


savefig('reeffit_vs_dissonance_mean.png', dpi=200)
