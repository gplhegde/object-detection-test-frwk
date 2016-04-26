import sys
sys.path.insert(0, './lib')
import dataset
from dataset import CaltechFace
from obj_evaluator import ObjectEvaluator

dset = CaltechFace('caltech-face-1999')
print ('No of images in this dataset = {:d}'.format(dset.no_images))

#a = dset.annotations
#dset.visualize_annoatations()

paths = dset.image_paths

# LBP cascade face detector
det = ObjectEvaluator('lbp_face', 'cascade')

# iterate thru all images in dataset and detect the faces.
det.visualize_detections(paths)
