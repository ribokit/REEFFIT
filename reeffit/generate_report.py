import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *
import mapping_analysis
from map_analysis_utils import *
from plot_utils import *
from rdatkit.datahandlers import *
from collections import defaultdict
import os
import argparse
from plot_utils import plot_mutxpos_image
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('resultdir', type=argparse.FileType('r'))
parser.add_argument('name', type=str)
parser.add_argument('prefix', type=str)

args = parser.parse_args()

result_dir = args.result_dir + '/'
template_dir = os.environ['REEFFIT_HOME'] + '/svg_templates/'

def fill_in_template(data_dict, templatefile, svgfile):
    for line in templatefile.readlines():
        newline = line
        for tag, value in data_dict.iteritems():
            newline = newline.replace('{%s}' % tag, value)
        svgfile.write(newline)
    return svgfile

def get_attributes(line):
    att = ''
    attributes = []
    for s in line:
        if read:
            att += s
        if s == '"' and not read:
            read = True
        if s == '"' and read:
            read = False
            attributes.append(att)
            att = ''
    return attributes

def generate_structure_grid(structure_files, titles):
    svgstrings = ['']*len(structure_files)
    bounding_boxes = []
    for i, sfile in enumerate(structure_files):
        read = False
        bounding_box = [0,0]
        for line in sfile.readlines():
            if 'line' in line and not read:
                read = True
            if read:
                attributes =  get_attributes(line)
                for att in attributes:
                    try:
                        val = float(att)
                        if val < bounding_box[0]:
                            bounding_box[0] = val
                        if val > bounding_box[1]:
                            bounding_box[1] = val
                    except ValueError:
                        pass
                if '</svg>' in line:
                    break
                svgstrings[i] += line
        bounding_boxes.append(bounding_box)
    xoffset = 0
    yoffset = 0
    max_bounding_x = 0
    j = 0
    gridstr = ''
    max_bounding_x = 0
    for i, svgs in enumerate(svgstrings):
        if j > 3:
            j = 0
            xoffset += max_bounding_x
            yoffset = 0
            max_bounding_x = 0
        title = '<text x="%s" y="%s" text-anchor="middle" font-family="Verdana"' % (xoffset, yoffset)
        title += 'font-size="12" fill="rgb(25%%, 25%%, 25%%)">%s</text>\n' % titles[i]
        yoffset += 15
        group_head = '<g\n   transform="translate(%s, %s)">\n' % (xoffset, yoffset)
        gridstr += title + group_head + svgs + '</g>\n'
        yoffset += bounding_boxes[i][1]
        max_bounding_x = max(max_bounding_x, bounding_boxes[i][0])
        j += 1
    return gridstr
    

# Structure plot
structure_files = []
titles = []
for fname in os.listdir(result_dir):
    if '%s_structure' % args.prefix in fname and '.svg' in fname:
        structure_files.append(open(result_dir + fname))
        idx = fname.replace('%s_structure', '').replace('.svg','')
        titles.append(args.name + idx)
data_dict['structure_grid'] = generate_structure_grid(structure_files, titles)
svgfile = fill_in_template(data_dict, template_dir + 'structures.svg', open('%s%s_structure_page.svg' % (result_dir, args.prefix)))
svgfile.close()


"""
currdir = os.getcwd()

        newline = line.replace('{exportdir}', result_dir).replace('{name}', outname).replace('{figdir}', '').replace('{id}', outname)
print 'Making SVG file for %s' % outname
templatefile = open(templatefname)
svgfile = open('%s/%s_paper_figure.svg' % (result_dir, outname), 'w')
svgfile.close()
templatefile.close()

print 'Exporting to PDF'
os.chdir(result_dir)
os.system('inkscape -z -D --file=%s_paper_figure.svg --export-pdf=%s_paper_figure.pdf --export-area-drawing  --export-text-to-path' % (outname, outname))
os.chdir(currdir)
"""
print 'Done'


        
