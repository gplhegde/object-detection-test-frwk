"""This tool generates fixed point object detection model from floating point model.
The model is in the form of xml file. The input model must follow OpenCV cascade detector model format.

Usage : python generate_fix_point_model.py <float_point_model_file.xml>
"""

import xml.dom.minidom as minidom
from xml.etree.ElementTree import ElementTree, dump
import sys

model_file = sys.argv[1]
fix_point_model_file = 'lbp_fp_face_model.xml'

# no of fraction bits to represent the floating point number
frac_bits = 8
# no of integer bits 
int_bits = 1
# most of the models contain signed numbers. hence we need 1 bit for sign
sign_bit = 1

total_bits = frac_bits + int_bits + sign_bit

scale_factor = 2**frac_bits
print ('Scaling the floating point values by {:d}'.format(scale_factor))

# read and parse the xml model file
doc = minidom.parse(model_file)
root = doc.getElementsByTagName('opencv_storage')[0]

# root node for stages.
stages = root.getElementsByTagName('stages')[0]
all_stages = stages.getElementsByTagName('_')

# list of stage threshold nodes. Stage thresholds are in floating point. hence we need to convert them
stage_thr = root.getElementsByTagName('stageThreshold')

# left and right weights of classifier stump
leaf_values = root.getElementsByTagName('leafValues')

# fixed point conversion of stage thresholds
for thr in stage_thr:
    th = float(thr.childNodes[0].data)
    th = int(scale_factor*th)
    thr.childNodes[0].data = th

# fixed point conversion of stump weights
for lv in leaf_values:
    vstr = lv.childNodes[0].data
    vstr = vstr.strip().split(' ')
    vstr = str(int(float(vstr[0])*scale_factor)) + ' ' + str(int(float(vstr[1])*scale_factor))
    lv.childNodes[0].data = vstr

# store the new fixed point model file
new_model = open(fix_point_model_file, 'w')
new_model.write(doc.toxml())
new_model.close()

print('Wrote fixed point model to {:s}'.format(fix_point_model_file))
