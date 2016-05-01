#-------------------------------------------------------------------
#  Author: Gopalakrishna Hegde, NTU Singapore  
#  Date:  30 April 2016
#
#  Licensed under The MIT License [see LICENSE for details]
#
#
#-------------------------------------------------------------------
__DATASETS = {}


__DATASETS['caltech-face-1999'] = {'path': '/opt/obj-detection/face-datasets/caltech-face-1999'}
__DATASETS['afw'] = {'path': '/opt/obj-detection/face-datasets/AFW'}


def get_dataset(name):
    for key in __DATASETS.keys():
        if name == key:
            return __DATASETS[key]['path']

    raise KeyError('Dataset not found')
