import sys
sys.path.insert(0, './lib')
import dataset
from dataset import  AFWFace
from obj_evaluator import ObjectEvaluator

dset =  AFWFace('afw')
print ('No of images in this dataset = {:d}'.format(dset.no_images))

a = dset.annotations
#dset.visualize_annoatations()
print a[0]
print a[1]

paths = dset.image_paths
print paths[0]
print paths[1]
# LBP cascade face detector
det = ObjectEvaluator('lbp_face', 'cascade')

# iterate thru all images in dataset and detect the faces.
#det.visualize_detections(paths)

det.evaluate_model(paths, a)
