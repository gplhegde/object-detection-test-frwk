
class ObjectDataset(object):
    """ Object detection(mainly face detection) image dataset class. Includes methods and properties general to all datasets.
    The dataset specific class mush implement methods to populate annotations
    """
    def __init__(self, dset_name):
        self._dset_name = dset_name
        self._img_names = []
        self._annotations = None
        self._populate_anotations = self.not_implemeted_handler

    @property
    def dset_name(self):
        return self._dset_name

    @property
    def no_images(self):
        return len(self._img_names)

    @property
    def img_names(self):
        return self._img_names

    @property
    def annotations(self):
        """Object annotations in each image of the dataset.
        The annotations are a list of dictionaries of the following form.
        {'rects' : [[x, y, w, h],... [....]]}
        """
        if self._annotations == None:
            self._annotations = self._populate_anotations()
        return self._annotations

    def not_implemented_handler(self):
        NotImplementedError
