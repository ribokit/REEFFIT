import map_analysis_utils as utils
import rdatkit.secondary_structure as ss
from Bio import SeqIO
import argparse
from random import choice, random
import pdb
import os

parser = argparse.ArgumentParser()

parser.add_argument('fastafile', type=str)
parser.add_argument('outfile', type=argparse.FileType('w'))
parser.add_argument('--nsubopt', type=int, default=200)
parser.add_argument('--nseqs', type=int, default=None)

args = parser.parse_args()
h = SeqIO.parse(args.fastafile, format='fasta')

probseqfile = open(args.fastafile + '.problematic', 'w')

count = 0
rc = {'A':'U','U':'A','G':'C','C':'G'}
def get_structures(sequence):
    print 'Obtaining suboptimal structures'
    mutants = [sequence[:i-1] + rc[sequence[i]] + sequence[i:] for i in range(len(sequence))[1:]]
    structures = []
    for m in [sequence] + mutants:
        mutstructs, deltas = ss.subopt(m, nstructs=args.nsubopt)
        #mutstructs, deltas = ss.sample(m, nstructs=args.nsubopt)
        mutstructs = [s.dbn for s in mutstructs]
        for ms in mutstructs:
            if ms not in structures:
                structures.append(ms)
    if len(structures) <= 5:
        return structures
    struct_types = utils.get_struct_types(structures)
    print 'Getting clusters'
    medoids, assignments = utils.cluster_structures(struct_types, structures=structures)
    resstructs = [structures[i] for i in medoids.values()]
    l = range(len(structures))
    """
    if random() > 1:
        resstructs.append(structures[choice(l)])
        if random() > 0.5:
            resstructs.append(structures[choice(l)])
    """
    return resstructs

for r in h:
    if args.nseqs and count > args.nseqs:
        break
    sequence = str(r.seq)
    if len(sequence) > 300:
        continue
    print 'Doing %s' % r.id
    print 'Length: %s'%  len(sequence)
    try:
        structures = get_structures(sequence)
    except Exception:
        probseqfile.write('>%s\n%s' % (r.id, sequence))
    if len(structures) <= 0:
        print 'Found only one structure, skipping...'
        probseqfile.write('>%s\n%s' % (r.id, sequence))
        continue
    args.outfile.write('%s\t%s\t%s\n' % (r.id, sequence, ','.join(structures)))
    count += 1
    os.system('rm /tmp/tmp*')
args.outfile.close()

