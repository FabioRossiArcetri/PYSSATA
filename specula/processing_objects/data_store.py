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


class DataStore(BaseProcessingObj):
    '''Data storage object'''

    def __init__(self,
                store_dir: str,
                data_format: str = 'fits'):
        super().__init__()
        self.items = {}
        self.storage = {}
        self.decimation_t = 0
        self.data_filename = ''
        self.tn_dir = store_dir
        self.data_format = data_format
        self.inputs['atmo_phase'] = InputValue(type=ElectricField)
        self.inputs['res_ef'] = InputValue(type=ElectricField)                
        self.inputs['ccd_pixels'] = InputValue(type=Pixels)
        self.inputs['sr'] = InputValue(type=BaseValue)

    def setParams(self, params):
        self.params = params

    def setReplayParams(self, replay_params):
        self.replay_params = replay_params

    def add(self, data_obj, name=None):
        if name is None:
            name = data_obj.__class__.__name__
        if name in self.items:
            raise ValueError(f'Storing already has an object with name {name}')
        self.items[name] = data_obj
        self.storage[name] = OrderedDict()

    def add_array(self, data, name):
        if name in self.items:
            raise ValueError(f'Storing already has an object with name {name}')
        self.storage[name] = data

    def save_pickle(self, compress=False):
        times = {k: np.array(list(v.keys()), dtype=self.dtype) for k, v in self.storage.items() if isinstance(v, OrderedDict)}
        data = {k: np.array(list(v.values()), dtype=self.dtype) for k, v in self.storage.items() if isinstance(v, OrderedDict)}        
        for k,v in times.items():            
            filename = os.path.join(self.tn_dir,k+'.pickle')
            hdr = self.local_inputs[k].get_fits_header()
            with open(filename, 'wb') as handle:
                data_to_save = {'data': data[k], 'times': times[k], 'hdr':hdr}
                pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    def save_params(self):
        filename = os.path.join(self.tn_dir, 'params.yml')
        with open(filename, 'w') as outfile:
            yaml.dump(self.params, outfile,  default_flow_style=False, sort_keys=False)

        self.replay_params['data_source']['store_dir'] = self.tn_dir

        filename = os.path.join(self.tn_dir, 'replay_params.yml')
        with open(filename, 'w') as outfile:
            yaml.dump(self.replay_params, outfile,  default_flow_style=False, sort_keys=False)

    def save_fits(self, compress=False):
        times = {k: np.array(list(v.keys()), dtype=self.dtype) for k, v in self.storage.items() if isinstance(v, OrderedDict)}
        data = {k: np.array(list(v.values()), dtype=self.dtype) for k, v in self.storage.items() if isinstance(v, OrderedDict)}        
        for k,v in times.items():            
            filename = os.path.join(self.tn_dir,k+'.fits')            
            hdr = self.local_inputs[k].get_fits_header()
            hdu_time = fits.ImageHDU(times[k], header=hdr)
            hdu_data = fits.PrimaryHDU(data[k], header=hdr)
            hdul = fits.HDUList([hdu_data, hdu_time])
            hdul.writeto(filename, overwrite=True)


    def create_TN_folder(self):
        today = time.strftime("%Y%m%d_%H%M%S")
        while True:
            tn = f'{today}'
            prefix = os.path.join(self.tn_dir, tn)
            if not os.path.exists(prefix):
                os.makedirs(prefix)
                break            
        self.tn_dir = prefix        


    def restore(self, filename, params=None):
        data = loadmat(filename)
        self.storage = data.get('data', {})
        if 'params' in data:
            if params is not None:
                params.update(data['params'])
            else:
                self.storage['params'] = data['params']

    def restore_tracknum(self, tn, dir='.', params=None, no_phi=False):
        tndir = os.path.join(dir, tn)
        if os.path.isdir(tndir):
            self.restore(os.path.join(tndir, 'old_format.mat'))
        else:
            for filename in os.listdir(tndir):
                if filename.endswith('.fits') and not no_phi and 'Phi' in filename:
                    continue
                name = filename.split('.')[0]
                data = loadmat(os.path.join(tndir, filename))
                self.storage[name] = data

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
        for k, item in self.items.items():
            if item is not None and item.generation_time == self.current_time:
                if isinstance(item, BaseValue):
                    v = cpuArray(item.value)
                elif isinstance(item, Slopes):
                    v = cpuArray(item.slopes)
                elif isinstance(item, Pixels):
                    v = cpuArray(item.pixels)
                elif isinstance(item, ElectricField):
                    v = np.stack( (cpuArray(item.A), cpuArray(item.phaseInNm)) )
                else:
                    raise TypeError(f"Error: don't know how to save an object of type {type(item)}")
                self.storage[k][self.current_time] = v

    def run_check(self, time_step, errmsg=''):
        return True

    def finalize(self):        
        self.create_TN_folder()
        self.save_params()
        if self.data_format=='pickle':
            self.save_pickle()
        elif self.data_format=='fits':
            self.save_fits()
        else:
            raise TypeError(f"Error: unsupported file format {self.data_format}")