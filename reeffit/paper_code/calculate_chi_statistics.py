import pickle
import pdb
from matplotlib.pylab import *
test_cases = [ ('M-stable', 'tristable_analysis/'), ('MedLoop', 'medloop_eterna_analysis/'), ('MedLoop$\Delta$', 'medloop_delta_analysis/'), ('Bistable', 'hobartner_analysis/')]
outfile = open('experimental_chi_statistics_table.txt', 'w')
outfile.write('Name\t$\chi^2/df$\tRMSEA\n')
for name, pathname in test_cases:
    data = asarray(pickle.load(open('%s/data.pickle' % pathname)))
    W_fa = pickle.load(open('%s/W.pickle' % pathname))
    W_fa_std = pickle.load(open('%s/W_std.pickle' % pathname))
    Psi_fa = pickle.load(open('%s/Psi.pickle' % pathname))
    E_d_fa = pickle.load(open('%s/E_d.pickle' % pathname))
    E_c_fa = pickle.load(open('%s/E_c.pickle' % pathname))
    data_pred = pickle.load(open('%s/data_pred.pickle' % pathname))
    sigma_pred = pickle.load(open('%s/sigma_pred.pickle' % pathname))
    chi_sq = ((data - data_pred)**2/(sigma_pred)**2).sum()
    df = data.size - W_fa.size - E_d_fa.size - E_c_fa[logical_not(isnan(E_c_fa))].size - data.shape[0] - 1
    rmsea = sqrt(max((chi_sq/df - 1)/(data.shape[1] - 1), 0.0))
    outfile.write('%s\t%s\t%s\n' % (name, chi_sq/df, rmsea))

outfile.close()



