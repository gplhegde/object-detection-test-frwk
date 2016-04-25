
__DATASETS = {}


__DATASETS['caltech-face-1999'] = {'path': '/opt/obj-detection/face-datasets/caltech-face-1999'}


def get_dataset(name):
    for key in __DATASETS.keys():
        if name == key:
            return __DATASETS[key]['path']

    raise KeyError('Dataset not found')
