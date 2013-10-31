import argparse
import pdb
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import rgb2hex
import os
from matplotlib.pylab import *

parser = argparse.ArgumentParser()
parser.add_argument('inprefix', type=str)
parser.add_argument('outprefix', type=str)
parser.add_argument('--method', type=str, default='mc')

args = parser.parse_args()

opt_chi_sq = inf
opt_structs = []
chi_sq_traces = []
struct_traces = []
nrejected = []
i = 0
for fname in os.listdir(args.inprefix):
    if 'results.txt' not in fname:
        continue
    print 'Doing %s' % fname
    wfile = open(args.inprefix + '/' + fname)
    if args.method == 'mc':
        struct_trace = []
        chi_sq_trace = []
        for line in wfile.readlines():
            lhood, chi_sq, rmsea, aic, structstr = line.strip().split('\t')
            structs = structstr.split(',')
            chi_sq = float(chi_sq)
            chi_sq_trace.append(chi_sq)
            struct_trace.append(structs)
            if chi_sq < opt_chi_sq:
                opt_structs = structs
                opt_chi_sq = chi_sq
    elif args. method == 'mcmc':
        nrejected.append(0)
        struct_trace = []
        chi_sq_trace = []
        first = True
        for line in wfile.readlines():
            lhood, chi_sq, rmsea, aic, structstr = line.strip().split('\t')
            chi_sq = float(chi_sq)
            structs = structstr.split(',')
            if first:
                chi_sq_trace.append(chi_sq)
                struct_trace.append(structs)
                if chi_sq < opt_chi_sq:
                    opt_structs = structs
                    opt_chi_sq = chi_sq
                first = False
            else:
                alpha = chi_sq_trace[-1]/chi_sq
                if alpha >= rand():
                    chi_sq_trace.append(chi_sq)
                    struct_trace.append(structs)
                    if chi_sq < opt_chi_sq:
                        opt_structs = structs
                        opt_chi_sq = chi_sq
                else:
                    nrejected[i] += 1
    chi_sq_traces.append(chi_sq_trace)
    struct_traces.append(struct_trace)
    i += 1



if args.method == 'mcmc':
    for i, chi_sq_trace in enumerate(chi_sq_traces):
        struct_trace = struct_traces[i]
        chainfile = open('%s_chain%s.txt' % (args.outprefix, i), 'w')
        for j, chi_sq in enumerate(chi_sq_trace):
            structs = struct_trace[j]
            chainfile.write('%s\t%s\n' % (chi_sq, ','.join(structs)))
        chainfile.close()

            
print 'Finished compiling traces'
if args.method == 'mcmc':
    print 'Number of rejected samples per trace: %s' % nrejected
print 'Optimal chi-squared was: %s' % opt_chi_sq
print 'Optimal structures were:'
for s in opt_structs:
    print s
print
opt_structfile = open('%sstructures.txt' % args.outprefix, 'w')
print 'Saving optimal structures to file %s' % opt_structfile.name
opt_structfile.write('#Optimal chi-squared: %s\n' % opt_chi_sq)
for s in opt_structs:
    opt_structfile.write(s + '\n')
opt_structfile.close()
print 'Plotting traces'
figure(1)
clf()
for chi_sq_trace in chi_sq_traces:
    plot(1/array(chi_sq_trace), alpha=0.6)
xlabel('Sample')
ylabel('$1/(\chi^2/df)$')
savefig('%schi_sq_traces.png' % args.outprefix, dpi=200)
