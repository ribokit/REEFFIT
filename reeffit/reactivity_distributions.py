from numpy.random import rand
import os
from scipy.stats import gamma, laplace, expon


distfile = open(os.environ['REEFFIT_HOME'] + '/reeffit/reactivity_distributions/SHAPEdist.txt')
distfile.readline()
#Read parameters for paired distribution
pparams = [float(x) for x in distfile.readline().strip().split()]
#Read parameters for unpaired distribution
uparams = [float(x) for x in distfile.readline().strip().split()]

distfile = open(os.environ['REEFFIT_HOME'] + '/reeffit/reactivity_distributions/SHAPEdist_all.txt')
#Read parameters for all reactivities distribution
distfile.readline()
allparams = [float(x) for x in distfile.readline().strip().split()]

distfile = open(os.environ['REEFFIT_HOME'] + '/reeffit/reactivity_distributions/SHAPEdist_diagonal.txt')
distfile.readline()
#Read parameters for the diagonal and contact distribution
dparams = [float(x) for x in distfile.readline().strip().split()]

contact_diff_params = [0.036036085880561453, 3.0564874002215925]

ugamma1 = gamma(uparams[0], loc=uparams[1], scale=uparams[2])
ugamma2 = gamma(uparams[3], loc=uparams[4], scale=uparams[5])

pgamma1 = gamma(pparams[0], loc=pparams[1], scale=pparams[2])
pgamma2 = gamma(pparams[3], loc=pparams[4], scale=pparams[5])

SHAPE_contacts_pdf = gamma(dparams[0], loc=dparams[1], scale=dparams[2]).pdf
SHAPE_contacts_sample = gamma(dparams[0], loc=dparams[1], scale=dparams[2]).rvs

SHAPE_contacts_diff_pdf = laplace(loc=contact_diff_params[0], scale=contact_diff_params[1]).pdf
SHAPE_contacts_diff_sample = laplace(loc=contact_diff_params[0], scale=contact_diff_params[1]).rvs

SHAPE_all_pdf = expon(loc=allparams[0], scale=allparams[1]).pdf
SHAPE_all_sample = expon(loc=allparams[0], scale=allparams[1]).rvs


def _sample_from_mixture(p1, p2, w1, w2):
    if rand() > w1:
        return p1.rvs()
    else:
        return p2.rvs()


def SHAPE_unpaired_sample():
    return _sample_from_mixture(ugamma1, ugamma2, uparams[5], uparams[6])


def SHAPE_paired_sample():
    return _sample_from_mixture(pgamma1, pgamma2, pparams[5], pparams[6])


def SHAPE_unpaired_pdf(x):
    return uparams[5] * ugamma1.pdf(x) + uparams[6] * ugamma2.pdf(x)


def SHAPE_paired_pdf(x):
    return pparams[5] * pgamma1.pdf(x) + pparams[6] * pgamma2.pdf(x)


