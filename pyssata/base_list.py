
from pyssata.base_data_obj import BaseDataObj


class BaseList(list, BaseDataObj):
    '''Base data objects for lists'''
    def __init__(self,  target_device_idx=None):
        BaseDataObj.__init__(self,  target_device_idx=target_device_idx)
