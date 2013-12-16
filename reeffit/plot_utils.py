from matplotlib.pylab import *
import matplotlib.pylab as pl
from matplotlib.patches import Rectangle
import pdb

STRUCTURE_COLORS = [get_cmap('Paired')(i*50) for i in xrange(100)]
def plot_mutxpos_image(d, sequence, seqpos, offset, mut_labels, cmap=get_cmap('Greys'), vmin=0, vmax=None, missed_indices=None, contact_sites=None, structure_colors=STRUCTURE_COLORS, weights=None, aspect='auto'):
    ax = subplot(111)
    if vmax == None:
        vmax = d.mean()
    ax.imshow(d, cmap=get_cmap('Greys'), vmin=0, vmax=vmax, aspect=aspect, interpolation='nearest')
    if missed_indices != None:
        for x, y in missed_indices:
            ax.add_artist(Rectangle(xy=(y-0.5, x-0.5), facecolor='none', edgecolor='r', linewidth=0.25, width=1, height=1))
    if contact_sites != None:
        if weights == None:
            weights = ones(d.shape[0], len(contact_sites))
        for k, v in contact_sites.iteritems():
            for x, y in zip(*where( v != 0)):
                ax.add_artist(Rectangle(xy=(y-0.5, x-0.5), facecolor='none', edgecolor=structure_colors[k], linewidth=1, width=1, height=1, alpha=weights[x,k]))
    xticks(range(len(seqpos)), ['%s%s' % (pos, sequence[pos - offset - 1]) for pos in seqpos], fontsize='xx-small', rotation=90)
    yticks(range(len(mut_labels)), mut_labels, fontsize='xx-small')
    return ax

def expected_reactivity_plot(react, struct, yerr=None, ymin=0, ymax=5, seq_indices=None):
    if seq_indices == None:
        seq_indices = [i for i in xrange(len(react))]
    binarized_react = [1 if x == '.' else 0 for i, x in enumerate(struct[:len(react)]) if i in seq_indices]
    plot(1+arange(len(binarized_react)), binarized_react, linewidth=2, c='r')
    bar(0.5+arange(len(react)), react, linewidth=0, width=1, color='gray')
    if yerr != None:
        errorbar(1+arange(len(react)), react, yerr=yerr, linewidth=2, c='k')
    ylim(0,5)
    xlim(1,len(react))

def weights_by_mutant_plot(W, W_err, mut_labels, structure_colors=STRUCTURE_COLORS, W_ref=None, idx=-1, assignments=None, medoids=None):
    ax = subplot(111)
    if assignments == None:
        _W = W
        _W_err = W_err
        _W_ref = W_ref
        nstructs = W.shape[1]
        struct_indices = range(nstructs)
    else:
        _W = zeros([W.shape[0], len(medoids)])
        _W_err = zeros([W.shape[0], len(medoids)])
        if W_ref != None:
            _W_ref = zeros([W.shape[0], len(medoids)])
        else:
            _W_ref = None
        nstructs = len(medoids)
        struct_indices = medoids
        i = 0
        setidx = False
        for c, si in assignments.iteritems():
            if idx in si:
                if not setidx:
                    idx = i
                    setidx = True
                
            for s in si:
                _W[:,i] += W[:,s]
                if _W_ref != None:
                    _W_ref[:,i] += W_ref[:,s]
                _W_err[:,i] += W_err[:,s]**2
            _W_err[:,i] = sqrt(_W_err[:,i])
            i += 1
    if idx >= 0:
        weight_range = [idx]
    else:
        weight_range = xrange(nstructs)

    for j in weight_range:
        ax.errorbar(arange(W.shape[0])+1, _W[:,j], yerr=_W_err[:,j], linewidth=3, label='structure %s ' % (struct_indices[j]), color=structure_colors[j])
        if _W_ref != None:
            ax.errorbar(arange(_W_ref.shape[0])+1, _W_ref[:,j], linestyle='--', linewidth=3, label='reference %s ' % (struct_indices[j]), color=structure_colors[j])
    ylim(0,1)
    xlim(0, _W.shape[0]+1)
    xticks(arange(_W.shape[0])+1, mut_labels, fontsize='xx-small', rotation=90)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def PCA_structure_plot(structures, assignments, medoids, colorbyweight=False, weights=None, names=None):
    all_struct_vecs = []
    all_struct_indices = []
    select_struct_vecs = []
    cluster_colors = {}
    struct_to_clust = {}
    for i, c in enumerate(assignments):
        cluster_colors[c] = STRUCTURE_COLORS[i]
    for c, indices in assignments.iteritems():
        for i in indices:
            struct_to_clust[i] = c
            all_struct_indices.append(i) 
            vec = [1 if s == '.' else 0 for s in structures[i]]
            all_struct_vecs.append(vec)
            if i in medoids:
                select_struct_vecs.append(vec)
    all_struct_vecs = array(all_struct_vecs)
    select_struct_vecs = array(select_struct_vecs)

    U, s, Vh = svd(all_struct_vecs.T)
    basis = U[:,:2]
    all_struct_coordinates = dot(all_struct_vecs, basis)
    select_struct_coordinates = dot(select_struct_vecs, basis)
    
    all_sizes = 50
    medoid_sizes = 100
    if weights == None:
        all_structure_colors = [cluster_colors[struct_to_clust[i]] for i in all_struct_indices]
        medoid_colors = [cluster_colors[struct_to_clust[i]] for i in medoids]
        linewidth = 1
    else:
        cmap = get_cmap('jet')
        normf = Normalize(vmin=0, vmax=max(weights), clip=True)
        if colorbyweight:
            all_structure_colors = [cmap(normf(weights[i])) for i in all_struct_indices]
            medoid_colors = [cmap(normf(weights[i])) for i in medoids]
            linewidth = 0
        else:
            all_structure_colors = [cluster_colors[struct_to_clust[i]] for i in all_struct_indices]
            medoid_colors = [cluster_colors[struct_to_clust[i]] for i in medoids]
            all_sizes = array([max(50, log(1 + weights[i])*5e4) for i in all_struct_indices])
            medoid_sizes = array([max(50, log(1 + weights[i])*5e4) for i in medoids])
            linewidth = 1

    figure(1)
    clf()

    scatter(all_struct_coordinates[:,0], all_struct_coordinates[:,1], c=all_structure_colors, alpha=0.6, linewidth=linewidth, s=all_sizes)
    scatter(select_struct_coordinates[:,0], select_struct_coordinates[:,1], c=medoid_colors, linewidth=2, s=medoid_sizes)
    if names != None:
        for i, m in enumerate(medoids):
            text(select_struct_coordinates[i,0], select_struct_coordinates[i,1], names[i], style='italic')
