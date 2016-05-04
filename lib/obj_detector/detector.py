#-------------------------------------------------------------------
#  Author: Gopalakrishna Hegde, NTU Singapore  
#  Date:  30 April 2016
#
#  Licensed under The MIT License [see LICENSE for details]
#
#
#-------------------------------------------------------------------
"""Cascade object detector using LBP cascade algorithm. Even though OpenCV has object detector,
this module will implement a custom controlled detector so that many parameters can be changed.
This implementation is not performance optimal in any sense
"""
import os,sys
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

        # decode the model file and create data structures
        self.no_stages = int(self.model_root.getElementsByTagName('stageNum')[0].childNodes[0].data)
        self.win_height = int(self.model_root.getElementsByTagName('height')[0].childNodes[0].data)
        self.win_width = int(self.model_root.getElementsByTagName('width')[0].childNodes[0].data)

        self.stage_thr = tuple([self._get_stage_thr(s) for s in range(self.no_stages)])
        self.stage_stumps = tuple([self._no_stage_stumps(s) for s in range(self.no_stages)])
        # get all feature index for all stages and form a list of tuple.
        # each tuple contains feature indices for one stage
        self.feat_idx = []
        for stg in range(self.no_stages):
            stage_feat_idx = [self._feat_no(stg, s) for s in range(self.stage_stumps[stg])]
            self.feat_idx.append(tuple(stage_feat_idx))
        # get all rectangle features and form  a list of rects (tuple)
        self.feat_params = []
        rect_nodes = self.features.getElementsByTagName('rect')
        for rect in rect_nodes:
            rect_data = rect.childNodes[0].data.strip()
            params = [int(e) for e in rect_data.split()]
            self.feat_params.append(tuple(params))
        # collect LUTs and weights of all stumps and store them in list of list of tuples
        # each tuple is for one stump and a list over all stumps in that stage
        self.luts = []
        self.weights = []
        for stg in range(self.no_stages):
            stage_luts = [self._stump_luts(stg, s)  for s in range(self.stage_stumps[stg])]
            stage_wts = [self._stump_weights(stg, s)  for s in range(self.stage_stumps[stg])]
            self.luts.append(stage_luts)
            self.weights.append(stage_wts)
                 
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
        def _get_cell_sum(ii_img, x, y, w, h):
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

        # move the co-ordinate from window's top left corner to feature block's top left corner
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

    def _evaluate_window(self, win_x, win_y, ii_img):
        """Given a window top left corner(x, y), evaluates it for presense of any objects.
        Input:
        win_x : column no of top left corner
        win_y : row no of top left corner
        ii_img: integral image of the image to which the window belongs
        Output: True --> contains object, False -> no objects found
        """
        no_stages = self.no_stages
        for stage in range(no_stages):
            no_stumps = self.stage_stumps[stage]
            stage_thr =  self.stage_thr[stage]
            weight = 0
            for s in range(no_stumps):
                # get parameters of current stump
                feat_no = self.feat_idx[stage][s]
                feat_params = self.feat_params[feat_no]
                stump_luts = self.luts[stage][s]
                stump_wts = self.weights[stage][s]
                # compute LBP feature 
                lbp_code = self._compute_lbp_feature(ii_img, feat_params, win_x, win_y)
                # decide whether to add left or right leaf value
                flag = stump_luts[lbp_code >> 5] & (1 << (lbp_code & 31))
                if ( flag > 0):
                    weight += stump_wts[0]
                else:
                    weight += stump_wts[1]
            # if the weight is less than the stage threshold, stage is failed
            if (weight < stage_thr):
                return False

        # window is passed if it passes all stages
        return True

    def block_integral_object_detector(self, in_img, scale_factor=1.1, blk_height=60, blk_width=80, min_neighbors=3, min_size=(30,30)):
        """This uses block integral image instead of full integral image. 
        """
        v_stride = 1
        h_stride = 1
        objs = []
        # convert to gray scale if the image is color 
        if(len(in_img.shape) == 3):
            gray_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = in_img

        org_height = gray_img.shape[0]
        org_width = gray_img.shape[1]
        cur_width = org_width
        cur_height = org_height
        win_width = self.win_width
        win_height = self.win_height
        blk_horz_stride = blk_width - win_width
        blk_vert_stride = blk_height - win_height

        # initial scale 1 as we process  original image
        scale = 1.0
        # downscale image and detect objects until one of the image dimension
        # becomes less  than the window size
        while(cur_width > (win_width+1) and cur_height > (win_height+1)):
            # max possible window top left corner positions.
            x_max = cur_width - win_width + 1
            y_max = cur_height - win_height + 1
            # extract a sliding image block and compute integral image on that.
            # detect the objects in the current block
            print('Current scale = {:f}'.format(scale))
            blk_y = 0
            while blk_y < y_max:
                blk_x = 0
                while blk_x < x_max:
                    print ('Block position (y,x) = ({:d},{:d})'.format(blk_y, blk_x))
                    # we cannot have full block in the edge of the image
                    max_blk_width = min(blk_width, cur_width - blk_x)
                    max_blk_height = min(blk_height, cur_height - blk_y)
                    # extract a block and
                    img_blk = gray_img[blk_y:blk_y+max_blk_height, blk_x:blk_x+max_blk_width]
                    ii_img = cv2.integral(img_blk)
                    # now use sliding window detector to find objects in the current block
                    for row in range(0, max_blk_height-win_height+1, v_stride):
                        for col in range(0, max_blk_width-win_width+1, h_stride):
                            # detect if the current window contains any objects
                            win_pass = self._evaluate_window(col, row, ii_img)
                            # record the window if it passes
                            if(win_pass):
                                objs.append(tuple([int((col+blk_x)*scale),
                                     int((row+blk_y)*scale),
                                     int(scale*win_width),
                                     int(scale*win_height)]))
                    # slide the block horizontally
                    blk_x += blk_horz_stride
                # slide the block vertically        
                blk_y += blk_vert_stride
            # down scale the image
            cur_width = int(cur_width/scale_factor)
            cur_height = int(cur_height/scale_factor)
            scale *= scale_factor
            gray_img = cv2.resize(gray_img, dsize=(cur_width, cur_height), interpolation=cv2.INTER_LINEAR)
            # perform new detections on the rescaled image.

        # perform NMS 
        print len(objs)
        return objs

    def detect_objects(self, in_img, scale_factor=1.1, min_neighbors=3, min_size=(30,30), max_size=()):
        """Detect objects using the LBP cascade classifier present in the given grayscale image.
        This has similar functionality as that of cv2.detectMultiScale() method
        """
        v_stride = 1
        h_stride = 1
        objs = []
        # convert to gray scale if the image is color 
        if(len(in_img.shape) == 3):
            gray_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = in_img

        org_height = gray_img.shape[0]
        org_width = gray_img.shape[1]
        cur_width = org_width
        cur_height = org_height
        win_width = self.win_width
        win_height = self.win_height

        # initial scale 1 as we process  original image
        scale = 1.0
        # downscale image and detect objects until one of the image dimension
        # becomes less  than the window size
        while(cur_width > (win_width+1) and cur_height > (win_height+1)):
            # max possible window top left corner positions.
            x_max = cur_width - win_width + 1
            y_max = cur_height - win_height + 1
            # compute integral image
            ii_img = cv2.integral(gray_img)
            print ('current scale = {:f}'.format(scale))
            for row in range(0, y_max, v_stride):
                for col in range(0, x_max, h_stride):
                    # detect if the current window contains any objects
                    win_pass = self._evaluate_window(col, row, ii_img)
                    # record the window if it passes
                    if(win_pass):
                        objs.append(tuple([int(col*scale),
                                     int(row*scale),
                                     int(scale*win_width),
                                     int(scale*win_height)]))
 
            # down scale the image
            cur_width = int(cur_width/scale_factor)
            cur_height = int(cur_height/scale_factor)
            scale *= scale_factor
            gray_img = cv2.resize(gray_img, dsize=(cur_width, cur_height), interpolation=cv2.INTER_LINEAR)
            # perform new detections on the rescaled image.

        # perform NMS 
        print len(objs)
        return objs

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
    print('Looking for objects....')
    clr_img = cv2.imread(sys.argv[1])
    #objs = det.detect_objects(in_img=clr_img,
    #    scale_factor=2.0)

    objs = det.block_integral_object_detector(in_img=clr_img,
        scale_factor=2.0,
        blk_height=120,
        blk_width=160)

    for (x, y, w, h) in objs:
        cv2.rectangle(clr_img, (x, y), (x+w, y+h), (255, 0, 0), 3)

    cv2.imshow('detections', clr_img)
    print('Press any key to move to next image...')
    cv2.waitKey()
