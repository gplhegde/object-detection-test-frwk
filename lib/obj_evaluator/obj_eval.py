#-------------------------------------------------------------------
#  Author: Gopalakrishna Hegde, NTU Singapore  
#  Date:  30 April 2016
#
#  Licensed under The MIT License [see LICENSE for details]
#
#
#-------------------------------------------------------------------
import os, sys
import cv2
from eval_config import eval_cfg
import utils
import numpy as np
import cPickle

class ObjectEvaluator(object):
    """Object evaluator classi. This has methods to evaluate images in a dataset and given 
    dataset images and ground thruth annotations.
    This produces accuracy figures using specified evaluation criteria.
    """
    def __init__(self, obj_type, det_type):
        # type of the object under consideration. Example: face, ball.
        self.obj_type = obj_type

        # detector type. ex: cascade detector
        self.det_type = det_type

        # retrive the trained model path
        self.model_path = eval_cfg.MODELS[self.obj_type]
        assert(os.path.exists(self.model_path)), 'Model definition file not found'
        print('Using {:s} as model for detection'.format(self.model_path))

        # create a detector
        if self.det_type == 'cascade':
            self.detector = cv2.CascadeClassifier(self.model_path)
        else:
            raise ValueError('Unsupported detector type')

        self.cache_dir = './cache'
        self.detections_file = 'caltech_face_detections.pkl'
        if (not os.path.isdir(self.cache_dir)):
            os.mkdir(self.cache_dir)

    def __detect_objects(self, image):  
        """Given a BGR/GRAY image, this method returns detected boxes
        Output:
        A list of rectangles in the following format
        objs = [[x1, y1, w1, h1], [x2, y2, w2, h2], ....[]]
        """
        # convert to gray scale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # set the max object size to image size if not specified in config.
        if (eval_cfg.MAX_OBJ_SIZE):
            max_size = tuple(eval_cfg.MAX_OBJ_SIZE)
        else:
            max_size = (gray_img.shape[1], gray_img.shape[0])
        # detect objects using initialized model
        objs = self.detector.detectMultiScale(gray_img,
            scaleFactor=eval_cfg.SCALE_FACTOR,
            minNeighbors=eval_cfg.MIN_NEIGHBORS,
            minSize=tuple(eval_cfg.MIN_OBJ_SIZE),
            maxSize=max_size
        )
        return objs

    def visualize_detections(self, img_paths):
        """Iterates thru all images in the given image path list and detects objects/faces
        using initialized detector and draws boxes around detected objects.
        """
        for im in img_paths:
            img = cv2.imread(im)
            objs = self.__detect_objects(img)
            for (x, y, w, h) in objs:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

            cv2.imshow('detections', img)
            print('Press any key to move to next image...')
            cv2.waitKey()

    def __evaluate_iou_accuracy(self, detections, annotations):
        """Given the detections and corresponding annotations for a dataset, this method
        does intersectio over union evaluations to determine the detection accuracy using
        configured IoU threshold
        """
        def get_iou_list(gt_box, det_boxes):
            iou_list = [utils.overlap_area(gt_box, np.array(d, dtype=np.float64)) for d in det_boxes]
            return iou_list

        total_gt_boxes = 0
        total_false_pos = 0
        total_detections = 0
        match_list = []
        false_det_list = []
        for i, ann in enumerate(annotations):
             print('Evaluating detections of image {:d}'.format(i))
             total_gt_boxes += len(ann['rects'])
             ann_matches = []
             det_boxes = detections[i]

             det_idx = range(len(detections[i]))

             for gt_box in ann['rects']:
                 # get list of IoU for the current ground truth box wrt all detections
                 iou = get_iou_list(gt_box, det_boxes)
                 matched_det_idx = []
                 # all detections with IoU > threshold are detection matches for the current gt box
                 match = []
                 for e in det_idx:
                     if iou[e] >= eval_cfg.METRICS['iou_thr']:
                         match.append(det_boxes[e])
                         if e not in matched_det_idx:
                             matched_det_idx.append(e)

                 ann_matches.append(match)
                 # object is detected if atleast one matching box is found
                 if (len(match) != 0):
                     total_detections += 1
                         

             match_list.append(ann_matches)
             # find false +ves. Those detections whose IoU < threshold for all gt boxes
             fp_list = []
             for e in det_idx:
                 if e not in matched_det_idx:
                     total_false_pos += 1
                     fp_list.append(det_boxes[e])
             false_det_list.append(fp_list)

        accuracy = float(total_detections) / total_gt_boxes
        print('---------IoU Test Summary---------')
        print('No of images tested = {:d}'.format(len(annotations)))
        print('No of ground truth objects in the entire test set = {:d}'.format(total_gt_boxes))
        print('No of correctly detected boxes = {:d}'.format(total_detections))
        print('No of false +ves = {:d}'.format(total_false_pos))
        print('Detection accuracy = {:f}'.format(accuracy))
        return (accuracy, match_list, false_det_list)

    def __get_all_detections(self, img_paths):
        """Given set of image paths, iterates thru all images to find the objects in them.
        Returns list of objects detected as co-ordinates of rectangles.
        Input:
        img_paths : list of absolute image paths
        Output:
        detections: [[(xmin, ymin, xmax, ymax), ()..()], [(), (), ...()], .... [(), ...]]
        """
        det_file = os.path.join(self.cache_dir, self.detections_file)
        if (os.path.exists(det_file)):
            with open(det_file, 'r') as f:
                detections = cPickle.load(f)['det']
                return detections

        detections = []
        for im in img_paths:
            print('Processing {:s} for objects'.format(im.split('/')[-1]))
            img = cv2.imread(im)
            # find objects in the image
            objs = self.__detect_objects(img)
            # convert (xmin, ymin, w, h) format to (xmin, ymin, xmax, ymax) format
            objs = [(obj[0], obj[1], obj[0]+obj[2], obj[1]+obj[3]) for obj in objs]
            
            # append the detections for the current image
            detections.append(objs)

        # save detections
        with open(det_file, 'w') as f:
            cPickle.dump({'det': detections}, f)
            print('Saved all detections into {:s}'.format(det_file))

        return detections

    def evaluate_model(self, img_paths, annotations, show_img=False):
        """Object detector model evaluator. Tests the model on a set of images with ground thruth annotations
        and computes detection accuracy, false +ve rate and other metrics
        """
        detections = self.__get_all_detections(img_paths)
        accuracy, match_list, fp_list = self.__evaluate_iou_accuracy(detections, annotations)
        if (show_img == True):
            for i, im in enumerate(img_paths):
                img = cv2.imread(im)
                ann = annotations[i]
                fp = fp_list[i]
                this_img_match = match_list[i]
                # draw false +ves if any
                for f in fp:
                    cv2.rectangle(img, (f[0], f[1]), (f[2], f[3]), (0, 0, 255), 2)
                # draw annotations and detections
                for i, a in enumerate(ann['rects']):
                    this_gt_match = this_img_match[i]
                    for m in this_gt_match:
                        cv2.rectangle(img, (m[0], m[1]), (m[2], m[3]), (255, 0, 0), 2)
                    
                    cv2.rectangle(img, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (0, 255, 0), 2)

                cv2.imshow('detections', img)
                cv2.waitKey()

        print('--------------Performance Statistics------------')
        print('No of images tested = {:d}'.format(len(img_paths)))
        print('No of ground truth objects  =')
        print('No of false +ves = ')
        print('No of detections(true +ve) = ')
        print('Detection accuracy = {:.3f}'.format(accuracy))

if __name__=='__main__':
    #img_file = sys.argv[1]
    det = ObjectEvaluator('lbp_face', 'cascade')
    #objs = det.__detect_objects(cv2.imread(img_file))
    #print objs
    paths = ['/opt/obj-detection/face-datasets/caltech-face-1999/images/image_0001.jpg',
             '/opt/obj-detection/face-datasets/caltech-face-1999/images/image_0002.jpg']

    ann = [{'rects': np.array([[ 434.0688738 ,   40.30860215,  783.27815507,  532.09445388]])},
           {'rects': np.array([[ 190.24227504,   60.97187323,  533.25257499,  523.82914544]])}]

    det.evaluate_model(paths, ann, True)
