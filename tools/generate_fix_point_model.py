import xml.dom.minidom as minidom
from xml.etree.ElementTree import ElementTree, dump
import sys

model_file = sys.argv[1]
fix_point_model_file = 'lbp_fp_face_model.xml'

frac_bits = 8
int_bits = 1
sign_bit = 1
total_bits = frac_bits + int_bits + sign_bit
scale_factor = 2**frac_bits
print ('Scaling the floating point values by {:d}'.format(scale_factor))

doc = minidom.parse(model_file)
root = doc.getElementsByTagName('opencv_storage')[0]

stages = root.getElementsByTagName('stages')[0]
all_stages = stages.getElementsByTagName('_')


stage_thr = root.getElementsByTagName('stageThreshold')
leaf_values = root.getElementsByTagName('leafValues')

for thr in stage_thr:
    th = float(thr.childNodes[0].data)
    th = int(scale_factor*th)
    thr.childNodes[0].data = th

for lv in leaf_values:
    vstr = lv.childNodes[0].data
    vstr = vstr.strip().split(' ')
    vstr = str(int(float(vstr[0])*scale_factor)) + ' ' + str(int(float(vstr[1])*scale_factor))
    lv.childNodes[0].data = vstr

new_model = open(fix_point_model_file, 'w')
new_model.write(doc.toxml())
new_model.close()

print('Wrote fixed point model to {:s}'.format(fix_point_model_file))
