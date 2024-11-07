from astropy.io import fits

import os
import numpy as np

from specula import xp
from collections import OrderedDict
import pickle
import yaml
import time

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.ef import ElectricField
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes

obj_type = {'atmo_phase':ElectricField,
            'res_ef':ElectricField,
            'ccd_pixels': Pixels,
            'sr': BaseValue}

class DataSource(BaseProcessingObj):
    '''Data source object'''

    def __init__(self,
                outputs,
                store_dir: str,
                data_format: str = 'fits'):
        super().__init__()
        self.items = {}
        self.storage = {}
        self.decimation_t = 0
        self.data_filename = ''
        self.tn_dir = store_dir
        self.data_format = data_format
        self.headers = {}

        for aout in outputs:            
            self.loadFromFile(aout)
        for k in self.storage.keys():
            if not obj_type[k] is BaseValue:
                self.outputs[k] = obj_type[k].from_header(self.headers[k])
            else:
                self.outputs[k] = BaseValue()

    def loadFromFile(self, name):
        if name in self.items:
            raise ValueError(f'Storing already has an object with name {name}')
        if self.data_format=='fits':
            self.load_fits(name)
        elif self.data_format=='pickle':
            self.load_pickle(name)

    def load_pickle(self, name):
        filename = os.path.join(self.tn_dir,name + '.pickle')
        with open( filename, 'rb') as handle:
            unserialized_data = pickle.load(handle)
        times = unserialized_data['times']
        data = unserialized_data['times']
        self.storage[name] = { t:data.data[i] for i, t in enumerate(times.data.tolist())}

    def load_fits(self, name):
        filename = os.path.join(self.tn_dir, name+'.fits')
        self.headers[name] = fits.getheader(filename)
        hdul = fits.open(filename)        
        times = hdul[1]
        data = hdul[0]
        self.storage[name] = { t:data.data[i] for i, t in enumerate(times.data.tolist())}

    def get(self, name):
        return self.storage[name]

    def keys(self):
        return list(self.storage.keys())

    def has_key(self, name):
        return name in self.storage

    def times(self, name, as_list=False):
        if not self.has_key(name):
            print(f'The key: {name} is not stored in the object!')
            return -1
        times = self.storage.get(f'{name}_times')
        if times is None:
            h = self.storage[name]
            times = list(h.keys())
        return times if as_list else self.t_to_seconds(np.array(times, dtype=self.dtype))

    def values(self, name, as_list=False, init=0):
        init = int(init)
        if not self.has_key(name):
            print(f'The key: {name} is not stored in the object!')
            return -1
        h = self.storage[name]
        if isinstance(h, OrderedDict):
            values = list(h.values())[init:]
        else:
            values = h[init:]
        return values if as_list else np.array(values, dtype=self.dtype)

    def size(self, name, dimensions=False):
        if not self.has_key(name):
            print(f'The key: {name} is not stored in the object!')
            return -1
        h = self.storage[name]
        return h.shape if not dimensions else h.shape[dimensions]

    def mean(self, name, init=0, dim=None):
        values = self.values(name, init=init)
        return np.mean(values, axis=dim)

    def stddev(self, name, init=0, dim=None):
        values = self.values(name, init=init)
        return np.std(values, axis=dim)

    def rms(self, name, init=0, dim=None):
        values = self.values(name, init=init)
        return np.sqrt(np.mean(np.square(values), axis=dim))

    def variance(self, name, init=0, dim=None):
        values = self.values(name, init=init)
        return np.var(values, axis=dim)

    def trigger_code(self):
        for k in self.storage.keys():            
            self.outputs[k].set_value(self.outputs[k].xp.array(self.storage[k][self.current_time]))
            self.outputs[k].generation_time = self.current_time
        
    def run_check(self, time_step, errmsg=''):
        return True

    def finalize(self):
        pass
        