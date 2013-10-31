from matplotlib.pylab import *
from numpy import linalg
import scipy.optimize
import pickle

def hill(conc, Kd):
    return conc/(Kd + conc)
structures = ['..........................(((((((............)))))))....................',\
'.....((((......((((....)))).....))))....................................',\
'.....((((......((((....)))).....)))).(((((....))))).....................']

sets, mut_labels, nmeas  = pickle.load(open('factor_analysis_variables.pickle'))
W_all = pickle.load(open('W_all.pickle'))
E_d_all = pickle.load(open('E_d_all.pickle'))

Kds = [0]*nmeas
conc_range = arange(0, max(sets) + 1000, 0.2)
colors = ('b','g','r')
for i, m in enumerate(mut_labels):
    print 'Fitting Hill function to mutant %s' % m
    Wm = W_all[arange(i, nmeas*len(sets), nmeas),:]
    clf()
    for j, s in enumerate(structures):
	Wm[:,j] /= Wm[:,j].max()
	popt = scipy.optimize.curve_fit(hill, sets, Wm[:,j])
	if j == 2:
	    Kds[i] = popt[0]
	this_hill = lambda conc: hill(conc, Kds[i])
	semilogx(conc_range, [this_hill(c) for c in conc_range], color=colors[j], linewidth=2)
	semilogx(sets, Wm[:,j], 'o', color=colors[j], label='structure %s' % j)
    print 'Kd for %s = %s' % (mut_labels[i], Kds[i])
    legend()
    xlabel('Concentration')
    ylabel('Fraction')
    title('Mutant %s, Kd=%s' % (m, Kds[i]))
    savefig('hill_fits/hill_fit_%s.png' % m)

