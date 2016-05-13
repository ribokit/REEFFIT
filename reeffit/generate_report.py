import argparse
import os
import pickle
# import pdb

import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *

from rdatkit.handler import *

import mapping_analysis
from map_analysis_utils import *
from plot_utils import *
from collections import defaultdict
from plot_utils import plot_mutxpos_image

parser = argparse.ArgumentParser()

parser.add_argument('resultdir', type=str)
parser.add_argument('name', type=str)
parser.add_argument('prefix', type=str)

args = parser.parse_args()

result_dir = args.resultdir + '/'
template_dir = os.environ['REEFFIT_HOME'] + '/reeffit/svg_templates/'


def fill_in_template(data_dict, templatefile, svgfile):
    for line in templatefile.readlines():
        newline = line
        for tag, value in data_dict.iteritems():
            newline = newline.replace('{%s}' % tag, str(value))
        svgfile.write(newline)
    return svgfile


def linesplit(line):
    res = []
    inatt = False
    currf = ''
    for i, s in enumerate(line):
        if s == ' ':
            if not inatt:
                res.append(currf)
                currf = ''
            else:
                currf += s
        else:
            currf += s
        if s == '"':
            inatt = not inatt
    return res


def get_attributes(line):
    att_dict = {}
    k = ''
    readval = False
    for x in linesplit(line):
        if '=' in x:
            k, v = x.strip(' ">\n').split('=')
            att_dict[k] = v.strip('"')
    return att_dict


def generate_structure_grid(structure_files, titles, fractions, fractions_std):
    svgstrings = ['']*len(structure_files)
    bounding_boxes = []
    colors = ['rgb(0,0,0)'] * len(structure_files)
    for i, sfile in enumerate(structure_files):
        read = False
        bounding_box = [0, 0]
        found_color = False
        for line in sfile.readlines():
            if 'line' in line and not read:
                read = True
            if read:
                attributes = get_attributes(line)
                if 'circle' in line and not found_color:
                    if attributes['stroke'] not in ['rgb(0%, 0%, 0%)', 'rgb(100%, 100%, 100%)', 'none']:
                        colors[i] = attributes['stroke']
                        found_color = True
                kx, ky = '', ''
                if 'x2' in attributes:
                    kx = 'x2'
                    ky = 'y2'
                elif 'x' in attributes:
                    kx = 'x'
                    ky = 'y'
                else:
                    pass
                if kx:
                    if float(attributes[kx]) > bounding_box[0]:
                        bounding_box[0] = float(attributes[kx])
                    if float(attributes[ky]) > bounding_box[1]:
                        bounding_box[1] = float(attributes[ky])
                if '</svg>' in line:
                    break
                svgstrings[i] += line
        bounding_boxes.append(bounding_box)
    text_padding = 50
    padding = 100
    xoffset = 0
    yoffset = 0
    j = 0
    gridstr = ''
    max_bounding_box = [0, 0]
    for i, svgs in enumerate(svgstrings):
        if j >= 3:
            j = 0
            xoffset += max_bounding_box[0] + padding
            yoffset = text_padding
            max_bounding_box[0] = 0
        yoffset += padding

        name, subidx = titles[i].split('_')
        title = '<text x="%s" y="%s" text-anchor="start" font-family="Garamond"' % (xoffset, yoffset)
        title += ' font-style="italic" font-size="35" fill="%s">%s<tspan baseline-shift ="sub">%s</tspan></text>\n' % (colors[i], name, subidx)
        yoffset += text_padding
        title += '<text x="%s" y="%s" text-anchor="start" font-family="Garamond"' % (xoffset, yoffset)
        title += ' font-style="italic" font-size="35" fill="%s">%3.2f%% +/- %3.2f</text>\n' % (colors[i], fractions[i], fractions_std[i])

        yoffset += text_padding
        group_head = '<g\n   transform="translate(%s, %s)">\n' % (xoffset, yoffset)
        gridstr += title + group_head + svgs + '</g>\n'
        yoffset += bounding_boxes[i][1]
        max_bounding_box[0] = max(max_bounding_box[0], bounding_boxes[i][0])
        max_bounding_box[1] = max(yoffset + padding, max_bounding_box[1])
        j += 1
    max_bounding_box[0] += xoffset
    return gridstr, max_bounding_box[0], max_bounding_box[1]

all_svg_files = []
print 'Generating structures page'
struct_indices, struct_fractions, struct_fractions_std = [], {}, {}

for line in open('%s%s_overall_wt_fractions.txt' % (result_dir, args.prefix)).readlines():
    idx, fraction, fraction_std = [float(x) for x in line.strip().split('\t')]
    idx = str(int(idx))
    struct_indices.append(idx)
    struct_fractions[idx] = fraction * 100
    struct_fractions_std[idx] = fraction_std * 100

structure_files, titles, fractions, fractions_std = [], [], [], []
for fname in os.listdir(result_dir):
    if '%s_structure' % args.prefix in fname and '.svg' in fname and '_page' not in fname:
        idx = fname.replace('%s_structure' % args.prefix, '').replace('.svg', '')
        if idx in struct_indices:
            structure_files.append(open(result_dir + fname))
            titles.append(args.name + '_' + idx)
            fractions.append(struct_fractions[idx])
            fractions_std.append(struct_fractions_std[idx])

data_dict = {}
data_dict['structure_grid'], grid_xoffset, grid_yoffset = generate_structure_grid(structure_files, titles, fractions, fractions_std)
data_dict['pca_img'] = os.path.abspath(result_dir + 'pca_landscape_plot_WT.png')
data_dict['pca_x'] = grid_xoffset + 200
data_dict['pca_y'] = 25
data_dict['pca_height'] = 1200
data_dict['pca_width'] = 1600
data_dict['pc1_labelx'] = data_dict['pca_width'] * 0.5
data_dict['pc1_labely'] = data_dict['pca_height'] - 20
data_dict['pc2_labelx'] = 100
data_dict['pc2_labely'] = data_dict['pca_height'] * 0.5
data_dict['size_x'] = grid_xoffset + data_dict['pca_width']
data_dict['size_y'] = max(data_dict['pca_height'], grid_yoffset)
svgfile = fill_in_template(data_dict, open(template_dir + 'structures.svg'), open('%s%s_structure_page.svg' % (result_dir, args.prefix), 'w'))
svgfile.close()
all_svg_files.append(svgfile.name)

print 'Generating weights by mutant and predicted data page'
data_dict = {}
for img in ['weights_by_mutant', 'reeffit_data_pred']:
    data_dict['%s_img' % img] = os.path.abspath('%s%s_%s.png' % (result_dir, args.prefix, img))

data_dict['real_data_img'] = os.path.abspath('%sreal_data.png' % result_dir)
data_dict['data_vs_predicted_img'] = os.path.abspath('%s%s_data_vs_predicted_WT.png' % (result_dir, args.prefix))
svgfile = fill_in_template(data_dict, open(template_dir + 'weights_and_data.svg'), open('%s%s_weights_and_data_page.svg' % (result_dir, args.prefix), 'w'))
svgfile.close()
all_svg_files.append(svgfile.name)

print 'Generating all latent reactivities page'
data_dict = {}
data_dict['all_reactivities_img'] = os.path.abspath('%sE_d.png' % result_dir)
svgfile = fill_in_template(data_dict, open(template_dir + 'all_latent_reactivities.svg'), open('%s%s_all_latent_reactivities_page.svg' % (result_dir, args.prefix), 'w'))
svgfile.close()
all_svg_files.append(svgfile.name)

print 'Generating reactivities and weights pages'

data_dict = {}
page_idx = 1
i = 0
MAX_STRUCTS = 5
for fname in os.listdir(result_dir):
    if '%s_exp_react_struct_' % args.prefix in fname and '.png':
        struct_idx = fname.replace('%s_exp_react_struct_' % args.prefix, '').replace('.png', '')
        if struct_idx in struct_indices:
            if i > MAX_STRUCTS - 1:
                svgfile = fill_in_template(data_dict, open(template_dir + 'reactivities_and_weights_%s.svg' % (i)), open('%s%s_reactivities_and_weights_page%s.svg' % (result_dir, args.prefix, page_idx), 'w'))
                svgfile.close()
                all_svg_files.append(svgfile.name)
                i = 0
                page_idx += 1
                data_dict = {}
            data_dict['exp_react_%s_img' % i] = os.path.abspath(result_dir + fname)
            data_dict['weights_%s_img' % i] = os.path.abspath(result_dir + '%s_weights_by_mutant_structure_%s.png' % (args.prefix, struct_idx))
            i += 1
if i <= MAX_STRUCTS:
    svgfile = fill_in_template(data_dict, open(template_dir + 'reactivities_and_weights_%s.svg' % (i)), open('%s%s_reactivities_and_weights_page%s.svg' % (result_dir, args.prefix, page_idx), 'w'))
    svgfile.close()
    all_svg_files.append(svgfile.name)

print 'Making PDFs from SVGs'
for svgfname in all_svg_files:
    os.system('inkscape -z -D --file=%s --export-pdf=%s --export-area-drawing  --export-text-to-path' % (svgfname, svgfname.replace('.svg', '.pdf')))

print 'Compiling full report'
os.system('pdfunite %s %s%s_report.pdf' % (' '.join([x.replace('.svg', '.pdf') for x in all_svg_files]), result_dir, args.prefix))

print 'Done'


