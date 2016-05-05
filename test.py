import sys
sys.path.insert(0, './lib')
from dataset import CaltechFace
from obj_evaluator import ObjectEvaluator

dset = CaltechFace('caltech-face-1999')
print ('No of images in this dataset = {:d}'.format(dset.no_images))

a = dset.annotations
#dset.visualize_annoatations()
print a[0]
print a[1]

paths = dset.image_paths
print paths[0]
print paths[1]
# LBP cascade face detector
det = ObjectEvaluator('custom_lbp_face', 'cascade')

# iterate thru all images in dataset and detect the faces.
#det.visualize_detections(paths)

det.evaluate_model(paths, a)
