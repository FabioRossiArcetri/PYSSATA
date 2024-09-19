import os
import numpy as np
from pyssata import gpuEnabled
from pyssata import xp
from collections import OrderedDict
import pickle
import time

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.base_value import BaseValue
from pyssata.data_objects.ef import ElectricField
from pyssata.data_objects.pixels import Pixels
from pyssata.data_objects.slopes import Slopes


class Datastore(BaseProcessingObj):
    '''Data storage object'''

    def __init__(self, store_dir: str):
        super().__init__()
        self._items = {}
        self._storage = {}
        self._decimation_t = 0
        self._data_filename = ''
        self._tn_dir = store_dir

    def add(self, data_obj, name=None):
        if name is None:
            name = data_obj.__class__.__name__
        if name in self._items:
            raise ValueError(f'Storing already has an object with name {name}')
        self._items[name] = data_obj
        self._storage[name] = OrderedDict()

    def add_array(self, data, name):
        if name in self._items:
            raise ValueError(f'Storing already has an object with name {name}')
        self._storage[name] = data

    def save(self, filename, compress=False):
        times = {k: xp.array(list(v.keys())) for k, v in self._storage.items() if isinstance(v, OrderedDict)}
        data = self._storage
        with open(filename, 'wb') as handle:
            data_to_save = {'data': data, 'times': times}
            pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_tracknum(self, dir='.', params=None, nofits=False, nosav=False, nodlm=False, nooldformat=False, compress=False, saveFloat=False):
        today = time.strftime("%Y%m%d")
        num = 0
        while True:
            tn = f'{today}.{num}'
            prefix = os.path.join(dir, tn)
            if not os.path.exists(prefix):
                os.makedirs(prefix)
                break
            num += 1
        self._tn_dir = prefix

        if params is not None:
            with open(os.path.join(prefix, 'params.txt'), 'w') as f:
                for k, v in params.items():
                    f.write(f'{k}: {v}\n')
            savemat(os.path.join(prefix, 'params.mat'), {'params': params}, do_compression=compress)

        savemat(os.path.join(prefix, 'routines.mat'), {'routines': self._routines}, do_compression=compress)

        if not nofits:
            for k, v in self._storage.items():
                if isinstance(v, OrderedDict) and len(v) > 0:
                    filename = os.path.join(prefix, f'{k}.fits')
                    times = xp.array(list(v.keys())) / float(self._time_resolution)
                    data = xp.array(list(v.values()))
                    if saveFloat and data.dtype == xp.float64:
                        data = data.astype(xp.float32)
                    savemat(filename, {'data': data, 'times': times}, do_compression=compress)
                else:
                    self._save_to_sav(prefix, k, v, compress)

        if not nooldformat:
            filename = os.path.join(prefix, 'old_format.mat')
            self.save(filename, compress=compress)
        return tn

    def _save_to_sav(self, prefix, name, data, compress):
        filename = os.path.join(prefix, 'data.mat')
        savemat(filename, {name: data}, do_compression=compress)

    def restore(self, filename, params=None):
        data = loadmat(filename)
        self._storage = data.get('data', {})
        if 'params' in data:
            if params is not None:
                params.update(data['params'])
            else:
                self._storage['params'] = data['params']

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
                self._storage[name] = data

    @property
    def decimation_t(self):
        return self._decimation_t

    @decimation_t.setter
    def decimation_t(self, value):
        self._decimation_t = self.seconds_to_t(value)

    @property
    def tn_dir(self):
        return self._tn_dir

    @property
    def data_filename(self):
        return self._data_filename

    def get(self, name):
        return self._storage[name]

    def keys(self):
        return list(self._storage.keys())

    def has_key(self, name):
        return name in self._storage

    def times(self, name, as_list=False):
        if not self.has_key(name):
            print(f'The key: {name} is not stored in the object!')
            return -1
        times = self._storage.get(f'{name}_times')
        if times is None:
            h = self._storage[name]
            times = list(h.keys())
        return times if as_list else self.t_to_seconds(xp.array(times))

    def values(self, name, as_list=False, init=0):
        init = int(init)
        if not self.has_key(name):
            print(f'The key: {name} is not stored in the object!')
            return -1
        h = self._storage[name]
        if isinstance(h, OrderedDict):
            values = list(h.values())[init:]
        else:
            values = h[init:]
        return values if as_list else xp.array(values)

    def size(self, name, dimensions=False):
        if not self.has_key(name):
            print(f'The key: {name} is not stored in the object!')
            return -1
        h = self._storage[name]
        return h.shape if not dimensions else h.shape[dimensions]

    def mean(self, name, init=0, dim=None):
        values = self.values(name, init=init)
        return xp.mean(values, axis=dim)

    def stddev(self, name, init=0, dim=None):
        values = self.values(name, init=init)
        return xp.std(values, axis=dim)

    def rms(self, name, init=0, dim=None):
        values = self.values(name, init=init)
        return xp.sqrt(xp.mean(xp.square(values), axis=dim))

    def variance(self, name, init=0, dim=None):
        values = self.values(name, init=init)
        return xp.var(values, axis=dim)

    def plot(self, name, init=0, map=False, over=False, resolution=1e9, **kwargs):
        import matplotlib.pyplot as plt
        values = self.values(name, init=init)
        times = self.times(name)
        if resolution != 1e9:
            if (times[1] - times[0]) * resolution <= 1:
                times *= resolution
            elif (times[1] - times[0]) / resolution > 1e-4:
                times /= resolution
        if map:
            plt.imshow(values, **kwargs)
        else:
            if len(values.shape) > 1:
                plt.plot(times[init:], values, **kwargs)
            elif over:
                plt.plot(times[init:], values, **kwargs)
            else:
                plt.plot(times[init:], values, **kwargs)
        plt.show()

    def trigger(self, t):
        do_store_values = self._decimation_t == 0 or t % self._decimation_t == 0
        if do_store_values:
            for k, item in self._items.items():
                if item is not None and item.generation_time == t:
                    if isinstance(item, BaseValue):
                        v = item.value
                    elif isinstance(item, Slopes):
                        v = item.slopes
                    elif isinstance(item, Pixels):
                        v = item.pixels
                    elif isinstance(item, ElectricField):
                        v = item.phaseInNm
                    else:
                        raise TypeError(f"Error: don't know how to save an object of type {type(item)}")
                    self._storage[k][t] = v

    def run_check(self, time_step, errmsg=''):
        return True

    def cleanup(self, destroy_items=False):
        if destroy_items:
            for item in self._items.values():
                if item:
                    item.cleanup()
        self._storage.clear()
        self._items.clear()
        super().cleanup()
        if self._verbose:
            print('Datastore has been cleaned up.')
