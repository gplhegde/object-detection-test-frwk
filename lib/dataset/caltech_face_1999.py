#-------------------------------------------------------------------
#  Author: Gopalakrishna Hegde, NTU Singapore  
#  Date:  30 April 2016
#
#  Licensed under The MIT License [see LICENSE for details]
#
#
#-------------------------------------------------------------------
import os
import obj_dataset
from datasets import get_dataset
import scipy.io as scio
import cv2
import numpy as np

class CaltechFace(obj_dataset.ObjectDataset):
    def __init__(self, dataset_name):
        
        obj_dataset.ObjectDataset.__init__(self, dataset_name)
        # get the path to the current dataset. The dataset directory must contain following directories
        # <dataset_path>
        #     - annotations    : contains annotation file/files. The format and file type is left to the dataset.
        #                        This is taken care by populate_annotations method.
        #     - images         : flat directory containing all images listed in image_list.txt
        #     - image_list.txt : list of image names without file extension
        self.dataset_path = self.get_dataset_path(self._dset_name)
        # all image formats present in the images directory.
        self._img_format = ('jpg', 'jpeg', 'png')
        # names of all images is are loaded into a list from image_list.txt
        self._img_names = self.load_image_names()
        # annotations for all objects in all images are loaded into list of dictionaries.
        self._populate_anotations = self.load_annotations
        # method to form and return absolute path of all images in the dataset
        self._get_image_paths = self.get_image_paths

    def get_dataset_path(self, dset_name):
        return get_dataset(dset_name)


    def load_image_names(self):
        """Reads image list file and creates a list of image names.
        """
        img_list_file = os.path.join(self.dataset_path, 'image_list.txt')
        assert(os.path.exists(img_list_file)), 'Image list file not found'

        with open(img_list_file) as f:
            img_names = [line.strip() for line in f.readlines()]

        return img_names

    def load_annotations(self):
        """ Load annotations from .mat file and form a list dictionaries.
        [x_bot_left y_bot_left x_top_left y_top_left ... 
            x_top_right y_top_right x_bot_right y_bot_right]
        """
        # load annotations mat file
        annotation_file = os.path.join(self.dataset_path, 'annotations', 'annotations.mat')
        # the mat file contains a variable named SubDir_Data which holds the co-ordinates of the box
        # 1 column is for 1 image. The column number and image name suffix are in sync. 
        raw_ann = scio.loadmat(annotation_file)['SubDir_Data']

        # we need only top left and bottom right corner co-ordinates
        valid_rows = (2, 3, 6, 7)
        annotations = []
        for im in self.img_names:
            img_no = int(im.split('_')[-1])
            # image number starts from 1. Hence subtract 1 to match with row 0
            # in matlab, array co-ordinates start from 1 hence subtract 1
            box = raw_ann[valid_rows, img_no-1] - 1
            # co-ordinates are in a column vector. reshape them into a row
            box = box.reshape(1, -1)
            # add the box to the annotation list as a new dictionary
            annotations.append({'rects': box})

        return annotations

    def get_image_paths(self):
        """makes list of absolute image paths of all images in the dataset
        """
        img_paths = []
        for im in self.img_names:
            img_paths.append(self.get_image_path(im))

        return img_paths

    def get_image_path(self, img_name):
        name = ''
        for ext in self._img_format:
            # see if the image exists in any of the image format listed in _img_format
            name =  os.path.join(self.dataset_path, 'images', img_name+'.'+ext)
            if(os.path.exists(name)):
                break
        assert(name != ''), 'Cannot locate the image path'
        return name
        
    def visualize_annoatations(self):
        """Draw rectange as per the annotations and show the image. Just for cross checking
        """
        ann = self.annotations
        for i, im in enumerate(self.img_names):
            img_path = self.get_image_path(im)
            img = cv2.imread(img_path)
            box = ann[i]['rects'][0].astype(np.int32)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.imshow('with_box', img)
            print('Press any key to move to next image...')
            cv2.waitKey()

if __name__=='__main__':
    data = CaltechFace('caltech-face-1999')
    print ('No of images in this dataset = {:d}'.format(data.no_images))

    a = data.annotations
    #data.visualize_annoatations()
    paths = data.image_paths
    print (paths)

