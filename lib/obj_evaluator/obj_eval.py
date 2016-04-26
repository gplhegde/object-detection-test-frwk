import os, sys
import cv2
from eval_config import eval_cfg

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






if __name__=='__main__':
    img_file = sys.argv[1]
    det = ObjectEvaluator('lbp_face', 'cascade')
    objs = det.__detect_objects(cv2.imread(img_file))
    print objs
