"""Cascade object detector using LBP cascade algorithm. Even though OpenCV has object detector,
this module will implement a custom controlled detector so that many parameters can be changed.
This implementation is not performance optimal in any sense
"""
import os
import xml.dom.minidom as minidom
import cv2
import numpy as np

class ObjectDetector(object):
    def __init__(self, model_file, model_type='float_pt'):
        # only xml files are accepted by this detector
        assert(model_file.split('.')[-1] == 'xml'), 'Model file mush be in the xml format.'
        self.model_file = model_file
        # parse the model file
        self.model_root = minidom.parse(self.model_file).getElementsByTagName('opencv_storage')[0]
        # stages root node
        self.stages = self.model_root.getElementsByTagName('stages')[0]
        # features root node
        self.features = self.model_root.getElementsByTagName('features')[0]
        # check if the model file is in the correct OpenCV format
        assert(self.model_root != None), 'Invalid model file'
        # model type. either float or fixed
        self.model_type = model_type

    @property
    def no_stages(self):
        return int(self.model_root.getElementsByTagName('stageNum')[0].childNodes[0].data)

    @property
    def win_height(self):
        return int(self.model_root.getElementsByTagName('height')[0].childNodes[0].data)
        
    @property
    def win_width(self):
        return int(self.model_root.getElementsByTagName('width')[0].childNodes[0].data)

    def _get_stage_thr(self, stage_no):
        """Returns stage threshold of the given cascade stage.
        """
        stage_thr_nodes = self.model_root.getElementsByTagName('stageThreshold')
        if(self.model_type == 'fixed_pt'):
            return int(stage_thr_nodes[stage_no].childNodes[0].data)
        else:
            return float(stage_thr_nodes[stage_no].childNodes[0].data)

    def _no_stage_stumps(self, stage_no):
        """Given the stage number, returns the number of weak classifiers/stumps present in the stage
        """
        stage_stump_nodes = self.stages.getElementsByTagName('maxWeakCount')
        return int(stage_stump_nodes[stage_no].childNodes[0].data)

    def _feat_no(self, stage_no, stump_no):
        """Given the stage number and the stump number, it returns the LBP feature number for that stump
        """
        stump_node = self.stages.getElementsByTagName('weakClassifiers')[stage_no]
        int_node_list = stump_node.getElementsByTagName('internalNodes')
        feat_no = int_node_list[stump_no].childNodes[0].data
        
        feat_no = int(feat_no.strip().split(' ')[2])
        return feat_no

    def _stump_weights(self, stage_no, stump_no):
        """Given the stage no and stump no, returns the left and right node weights for the decision stump
        """
        stump_node = self.stages.getElementsByTagName('weakClassifiers')[stage_no]
        leaf_node_list = stump_node.getElementsByTagName('leafValues')
        leaf_data = leaf_node_list[stump_no].childNodes[0].data
        leaf_vals = leaf_data.strip().split(' ')
        if(self.model_type == 'fixed_pt'):
            return (int(leaf_vals[0]), int(leaf_vals[1]))
        else:
            return (float(leaf_vals[0]), float(leaf_vals[1]))

    def _stump_luts(self, stage_no, stump_no):
        """Given the stage no and stump no, returns the lookup table(8 32 bit integers) for the decision stump
        """
        stump_node = self.stages.getElementsByTagName('weakClassifiers')[stage_no]
        int_node_list = stump_node.getElementsByTagName('internalNodes')
        int_data = int_node_list[stump_no].childNodes[0].data.strip().replace('\n', '')

        # first 3 numbers represent left and right index, feat no
        lut_str = int_data.split()[3:]
        luts = [int(e) for e in lut_str]
        return tuple(luts)

    def _feat_params(self, feat_no):
        """Given the feature number, returns the rectangular feature information.
        Input: feat_no: LBP feature number returned by self._feat_no()
        Output: (x, y, w, h) of the feature.
        """
        rect_nodes = self.features.getElementsByTagName('rect')
        rect_data = rect_nodes[feat_no].childNodes[0].data.strip()
        params = [int(e) for e in rect_data.split()]
        return tuple(params)

    def _compute_lbp_feature(self, ii_img, feat_param, win_pos_x, win_pos_y):
        """Returns lbp code given the feature parameters (x, y, w, h) in the image and 
        integral image.
        Output: lab code (np.uint8)
        """
        def _get_cell_sum(ii_img, x, y, w, h)
            left_pt = x-1
            top_pt = y-1
            # a --> top left corner of ii image 
            if (left_pt < 0 or top_pt < 0):
                a = 0
            else:
                a = ii_img[top_pt, left_pt]
            # top right corner
            if(top_pt < 0):
                b = 0
            else:
                b = ii_img[top_pt, left_pt + w]
            # bottom left corner
            if(left_pt < 0):
                d = 0
            else:
                d = ii_img[top_pt+h, left_pt]
            # bottom right point
            c = ii_img[top_pt+h, left_pt+w]
            # cell sum
            return (a+c) - (b+d)

        # move the co-ordinated from window's top left corner to feature block's top left corner
        feat_x = win_pos_x + feat_param[0]
        feat_y = win_pos_y + feat_param[1]
        cell_width = feat_param[2]
        cell_height = feat_param[3]
        # cell positions in terms of multiple of cell height and width in the LBP order. (x, y) format
        cell_pos = ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1))

        # sum of center cell
        cent_x = feat_x + cell_width
        cent_y = feat_y + cell_height
        cent_sum = _get_cell_sum(ii_img, cent_x, cent_y, cell_width, cell_height)
        # compute sum of all 8 cells and generate LBP code
        lbp_code = 0
        for n in range(8):
            pos = cell_pos[n]
            cell_x = feat_x + pos[0] * cell_width
            cell_y = feat_y + pos[1] * cell_height
            cell_sum = _get_cell_sum(ii_img, cell_x, cell_y, cell_width, cell_height)
            if(cell_sum > cent_sum):
                lbp_code += (2**(7-n))

        return np.uint8(lbp_code)

    def detect_objects(self, in_img, scale_factor=1.1, min_neighbors=3, min_size=(30,30), max_size=())
        """Detect objects using the LBP cascade classifier present in the given grayscale image.
        This has similar functionality as that of cv2.detectMultiScale() method
        """
        objs = []
        # convert to gray scale if the image is color 
        if(len(gray_img.shape) == 3):
            gray_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = in_img

        
if __name__=='__main__':
    this_dir = os.path.dirname(__file__)
    model_file = os.path.join(this_dir, '../../models/lbp_fp_face_model.xml')

    det = ObjectDetector(model_file, 'fixed_pt')

    print('Total no of stages = {:d}'.format(det.no_stages))
    print('Detector height = {:d}'.format(det.win_height))
    print('Detector width = {:d}'.format(det.win_width))
    for s in range(det.no_stages):
        print ('stage {:d} : thr = {:f}'.format(s, det._get_stage_thr(s)))
        print ('stage {:d} : no of stumps = {:d}'.format(s, det._no_stage_stumps(s)))

    print (det._feat_no(1, 1))

    print(det._stump_weights(0,0))
    print(det._stump_luts(6,2))

    print(det._feat_params(3))
