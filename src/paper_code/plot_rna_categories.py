from matplotlib.pylab import *
import pickle

outdir = 'insilico_simulations'
figdir = 'paper_figures'
energies = pickle.load(open(outdir + '/all_energies_dict.pickle'))
weights = pickle.load(open(outdir + '/all_weights_dict.pickle'))

def diff(v):
    res = []
    if v.shape[1] == 1:
        res = [1]
    for i in xrange(v.shape[1]):
        for j in xrange(v.shape[1]):
            if i != j:
                res.append(v[0][i] - v[0][j])
    return array(res)

def size(v):
    if sum(v >= 0.9999):
        return 1
    else:
        return v.size

numstates = [size(v) for v in weights.values()]
max_weight_diff = [abs(diff(v)).max() for v in weights.values()]
min_weight_diff = [abs(diff(v)).min() for v in weights.values()]
mean_weight_diff = [abs(diff(v)).mean() for v in weights.values()]

figure(1)
clf()
hist(numstates, 10, color='blue', alpha=0.6,  linewidth=0)
savefig(figdir + '/numstates_hist.png', dpi=100)

figure(1)
clf()
hist(max_weight_diff, 20, color='blue', alpha=0.6,  linewidth=0)
savefig(figdir + '/maxweightdiff_hist.png', dpi=100)

figure(1)
clf()
hist(min_weight_diff, 20, color='blue', alpha=0.6,  linewidth=0)
savefig(figdir + '/minweightdiff_hist.png', dpi=100)


figure(1)
clf()
hist(mean_weight_diff, 20, color='blue', alpha=0.6,  linewidth=0)
savefig(figdir + '/meanweightdiff_hist.png', dpi=100)
