import os
import obj_dataset
from datasets import get_dataset
import cv2

class AFWFace(obj_dataset.ObjectDataset):
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
        """ Load annotations from .txt file and form a list dictionaries.
        each line of annotation file is as follows
        img_name, xmin, ymin, xmax, ymax, xmin, ymin,...... '\n'
        """
        # load annotations mat file
        annotation_file = os.path.join(self.dataset_path, 'annotations', 'annotations.txt')
        # we need only top left and bottom right corner co-ordinates
        annotations = []
        line_cnt = 0
        with open(annotation_file, 'r') as af:
            for line in af:
                img_box_list = []
                ann = line.split(',')
                # make sure that the formed image list and annotations are in order by 
                # comparing image name from the annotation file as well
                assert(ann[0].split('.')[0] == self.img_names[line_cnt]), 'Image name and annotation mismatch'
                # last element is '\n'. hance remove it. first element is image name
                ann_box = ann[1:-1]
                # number of boxes is lenght of annotations / 4
                for n in range(0, len(ann_box), 4):
                    img_box_list.append([int(ann_box[n]), int(ann_box[n+1]), int(ann_box[n+2]), int(ann_box[n+3])])

                line_cnt += 1
                annotations.append({'rects': img_box_list})

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
            boxes = ann[i]['rects']
            for box in boxes:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.imshow('with_box', img)
            print('Press any key to move to next image...')
            cv2.waitKey()

if __name__=='__main__':
    data = AFWFace('afw')
    print ('No of images in this dataset = {:d}'.format(data.no_images))

    paths = data.image_paths
    #print (paths)
    a = data.annotations
    data.visualize_annoatations()

