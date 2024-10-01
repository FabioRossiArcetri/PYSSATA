
from pyssata.data_objects.base_data_obj import BaseDataObj


class BaseList(list, BaseDataObj):
    '''Base data objects for lists'''
    def __init__(self,  device_idx=None):
        BaseDataObj.__init__(self,  device_idx=device_idx)
