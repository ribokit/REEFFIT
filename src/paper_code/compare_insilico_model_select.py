import sys


logfile = open(sys.argv[1])
for path in logfile.readlines()[1:]:
    try:
        path = path.strip()
        print '======================================================'
        print path
        print '"Real" structures used to simulate:'
        realstructs = []
        reeffitstructs = []
        for l in open(path + '/structures.txt').readlines():
            l = l.strip()
            realstructs.append(l)
            print l
        print 'Predicted structures:'
        for l in open(path + '/' + sys.argv[2] + 'structure_medoids.txt').readlines():
            l = l.strip()
            reeffitstructs.append(l)
            print l
        ncorr = len([s for s in reeffitstructs if s in realstructs])
        print 'Number of real structures: %s' % len(realstructs)
        print 'Number of predicted structures: %s' % len(reeffitstructs)
        print 'Correctly predicted %s out of %s' % (ncorr, len(realstructs))
    except IOError:
        pass


print '======================================================'
