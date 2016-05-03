import argparse
from itertools import chain
import os
# import pdb

import matplotlib
from matplotlib.colors import rgb2hex
from matplotlib.pylab import *
from scipy.stats.mstats import mquantiles

from event_utils import AnnoteFinder

parser = argparse.ArgumentParser()
parser.add_argument('inprefix', type=str)
parser.add_argument('outprefix', type=str)
parser.add_argument('--trueensemble', type=argparse.FileType('r'))
parser.add_argument('--method', type=str, default='mc')
parser.add_argument('--quantile', type=float, default=0.5)

args = parser.parse_args()


opt_aic = inf
opt_structs, chi_sq_traces, aic_traces, struct_traces, weight_traces, nrejected = [], [], [], [], [], []
i = 0
for fname in os.listdir(args.inprefix):
    if 'results.txt' not in fname:
        continue
    print 'Doing %s' % fname
    wfile = open(args.inprefix + '/' + fname)
    if args.method == 'mc':
        struct_trace, chi_sq_trace, aic_trace, weight_trace = [], [], [], []
        for line in wfile.readlines():
            lhood, chi_sq, rmsea, aic, structstr, weights = line.strip().split('\t')
            structs = structstr.split(',')
            weights = array([float(x) for x in weights.split(',')])
            chi_sq = float(chi_sq)
            aic = float(aic.strip('[]'))
            chi_sq_trace.append(chi_sq)
            aic_trace.append(aic)
            weight_trace.append(weights)
            struct_trace.append(structs)
            if aic < opt_aic:
                opt_structs = structs
                opt_aic = aic

    elif args.method == 'mcmc':
        nrejected.append(0)
        struct_trace, chi_sq_trace, aic_trace, weight_trace = [], [], [], []
        first = True
        for line in wfile.readlines():
            lhood, chi_sq, rmsea, aic, structstr = line.strip().split('\t')
            chi_sq = float(chi_sq)
            weights = array([float(x) for x in weights.split(',')])
            aic = float(aic.strip('[]'))
            structs = structstr.split(',')
            if first:
                chi_sq_trace.append(chi_sq)
                aic_trace.append(aic)
                struct_trace.append(structs)
                weight_trace.append(weights)
                if aic < opt_aic:
                    opt_structs = structs
                    opt_aic = aic
                first = False
            else:
                alpha = aic_trace[-1]/aic
                if alpha >= rand():
                    chi_sq_trace.append(chi_sq)
                    aic_trace.append(aic)
                    struct_trace.append(structs)
                    weight_trace.append(weights)
                    if aic < aic_sq:
                        opt_structs = structs
                        aic_sq = aic
                else:
                    nrejected[i] += 1
    chi_sq_traces.append(chi_sq_trace)
    aic_traces.append(aic_trace)
    struct_traces.append(struct_trace)
    i += 1


if args.method == 'mcmc':
    for i, aic_trace in enumerate(aic_traces):
        struct_trace = struct_traces[i]
        chainfile = open('%s_chain%s.txt' % (args.outprefix, i), 'w')
        for j, aic in enumerate(aic_trace):
            structs = struct_trace[j]
            chainfile.write('%s\t%s\n' % (aic, ','.join(structs)))
        chainfile.close()


print 'Finished compiling traces'
if args.method == 'mcmc':
    print 'Number of rejected samples per trace: %s' % nrejected
print 'Optimal AIC was: %s' % opt_aic
print 'Optimal structures were:'
for s in opt_structs:
    print s
print
opt_structfile = open('%sstructures.txt' % args.outprefix, 'w')
print 'Saving optimal structures to file %s' % opt_structfile.name
opt_structfile.write('#Optimal AIC: %s\n' % opt_aic)
for s in opt_structs:
    opt_structfile.write(s + '\n')
opt_structfile.close()
print 'Plotting traces'
figure(1)
clf()
xmin = inf
xmax = -inf
all_aic_traces = []
for aic_trace in aic_traces:
    all_aic_traces += aic_trace
    aic_trace = array(aic_trace)
    xmin = min(xmin, aic_trace.min())
    xmax = max(xmax, aic_trace.min() + aic_trace.std())
    plot(aic_trace, alpha=0.6, linewidth=3)
xlabel('Sample')
ylabel('$AIC$')
ylim([xmin, xmax])
cutoff = mquantiles(all_aic_traces, prob=[args.quantile])
ylim([xmin - 100, cutoff])
savefig('%s%s_aic_traces.png' % (args.outprefix, args.method), dpi=200)
close()


figure(1)
clf()
for chi_sq_trace in chi_sq_traces:
    plot(array(chi_sq_trace), alpha=0.6, linewidth=3)
xlabel('Sample')
ylabel('$\chi^2/df$')
savefig('%s%s_chi_sq_traces.png' % (args.outprefix, args.method), dpi=200)
close()



if args.trueensemble is not None:
    def struct_distance(s1, s2):
        res = 0.
        for i in xrange(len(s1)):
            if s1[i] != s2[i]:
                res += 1
        return res / len(s1)

    true_structs = [l.strip() for l in args.trueensemble.readlines() if l[0] != '#']
    sdistance_traces = []
    for struct_trace in struct_traces:
        sdistance_trace = []
        matches = [False] * len(true_structs)
        for structs in struct_trace:
            currdist = 0.
            for s in structs:
                mindist = inf
                minidx = 0
                for i, ts in enumerate(true_structs):
                    dist = struct_distance(s, ts)
                    if dist < mindist:
                        mindist = dist
                        minidx = i
                matches[minidx] = True
                currdist += mindist
            for i, m in enumerate(matches):
                if not m:
                    maxdist = -inf
                    for s in structs:
                        dist = struct_distance(s, true_structs[i])
                        if dist > maxdist:
                            maxdist = dist
                    currdist += maxdist

            sdistance_trace.append(currdist)
        sdistance_traces.append(sdistance_trace)

    figure(2)
    clf()
    for i, aic_trace in enumerate(aic_traces):
        scatter(sdistance_traces[i], aic_trace, alpha=0.6, linewidth=0, color='r')
    ylabel('$AIC$')
    xlabel('Ensemble distance')
    ylim([xmin - 100, cutoff])
    savefig('%s%s_aic_vs_ensemble_dist.png' % (args.outprefix, args.method), dpi=200)
    close()

    figure(3)
    clf()
    for i in xrange(len(aic_traces)):
        indices = [idx for idx, aic in enumerate(aic_traces[i]) if aic <= cutoff]
        sdistance_trace = array(sdistance_traces[i])
        aic_trace = array(aic_traces[i])
        scatter(sdistance_trace[indices], aic_trace[indices], alpha=0.6, linewidth=0, color='r')
        annotes = ['\n' + '\n'.join(structs) + '\n$\chi^2/df$: %s' % chi_sq_traces[i][j] for j, structs in enumerate(struct_traces[i]) if j in indices]
        af = AnnoteFinder(sdistance_trace[indices], aic_trace[indices], annotes)
        connect('button_press_event', af)
    ylabel('$AIC$')
    xlabel('Ensemble distance')
    ylim([xmin - 100, cutoff])
    show()


