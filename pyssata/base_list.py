
from pyssata.base_time_obj import BaseTimeObj

class BaseList(list, BaseTimeObj):
    '''Base data objects for lists'''
    def __init__(self):
        BaseTimeObj.__init__(self)
