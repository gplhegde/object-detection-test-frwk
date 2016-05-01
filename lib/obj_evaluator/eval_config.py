#-------------------------------------------------------------------
#  Author: Gopalakrishna Hegde, NTU Singapore  
#  Date:  30 April 2016
#
#  Licensed under The MIT License [see LICENSE for details]
#
#
#-------------------------------------------------------------------
from easydict import EasyDict as edict
import sys, os

this_dir = os.path.dirname(__file__)

__EVAL_CONFIG = edict()

eval_cfg = __EVAL_CONFIG

__MODELS = {}

__MODELS['lbp_face'] = '/opt/opencv/data/lbpcascades/lbpcascade_frontalface.xml'

__MODELS['custom_lbp_face'] = os.path.join(this_dir, '../../models/lbp_fp_face_model.xml')

__MODELS['haar_face'] = '/opt/opencv/data/haarcascades/haarcascade_frontalface_default.xml'


__EVAL_METRICS = {}

# intersection over union threshold. Basically % of verlap with the ground truth annotations.
__EVAL_METRICS['iou_thr'] = 0.3

# evaluate false +ve rate
__EVAL_METRICS['fpr'] = True

# classifier models
__EVAL_CONFIG.MODELS = __MODELS

# object detection evaluation metrics
__EVAL_CONFIG.METRICS = __EVAL_METRICS

# image scaling factor for image pyramid
__EVAL_CONFIG.SCALE_FACTOR = 1.1

# min neighbors for a detection to be considered as true detection
__EVAL_CONFIG.MIN_NEIGHBORS = 5

# min size of(w, h) the detected object. object smaller than  this are neglected.
__EVAL_CONFIG.MIN_OBJ_SIZE = (30, 30)

# max size of detected object.
__EVAL_CONFIG.MAX_OBJ_SIZE = ()
