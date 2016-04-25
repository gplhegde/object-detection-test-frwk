import os
from datasets import get_dataset

class CaltechFace(ObjectDataset)
    def __init__(self, dataset_name):
        
        ObjectDataset.__init__(dataset_name)
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

        
