"""Cascade object detector using LBP cascade algorithm. Even though OpenCV has object detector,
this module will implement a custom controlled detector so that many parameters can be changed.
This implementation is not performance optimal in any sense
"""
import os
import xml.dom.minidom as minidom
import cv2

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

if __name__=='__main__':
    this_dir = os.path.dirname(__file__)
    model_file = os.path.join(this_dir, '../../models/lbp_fp_face_model.xml')

    det = ObjectDetector(model_file, 'fixed_pt')

    print ('Total no of stages = {:d}'.format(det.no_stages))
    for s in range(det.no_stages):
        print ('stage {:d} : thr = {:f}'.format(s, det._get_stage_thr(s)))
        print ('stage {:d} : no of stumps = {:d}'.format(s, det._no_stage_stumps(s)))

    print (det._feat_no(1, 1))

    print(det._stump_weights(0,0))
    print(det._stump_luts(6,2))

    print(det._feat_params(3))
