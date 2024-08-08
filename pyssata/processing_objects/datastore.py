import numpy as np
import pickle
from collections import OrderedDict
from pathlib import Path

class DataStore(BaseProcessingObj):
    def __init__(self):
        super().__init__('store', 'Storing object')
        self._items = {}
        self._storage = {}
        self._decimation_t = 0
        self._data_filename = ''
        self._tn_dir = ''

    def add(self, data_obj, name=None):
        if name is None:
            name = type(data_obj).__name__
        if name in self._items:
            raise ValueError(f'Storing already has an object with name {name}')
        self._items[name] = data_obj
        self._storage[name] = OrderedDict()

    def add_array(self, data, name):
        if name in self._items:
            raise ValueError(f'Storing already has an object with name {name}')
        self._storage[name] = data

    def save(self, filename, compress=False):
        times = {k: np.array(list(v.keys())) for k, v in self._storage.items() if isinstance(v, OrderedDict)}
        data = self._storage
        with open(filename, 'wb') as f:
            pickle.dump({'data': data, 'times': times}, f, protocol=pickle.HIGHEST_PROTOCOL if compress else pickle.DEFAULT_PROTOCOL)

    def save_tracknum(self, dir='.', params=None, nofits=False, nosav=False, nooldformat=False, compress=False, save_float=False):
        Path(dir).mkdir(parents=True, exist_ok=True)
        today = tracknum()
        num = 0
        while True:
            tn = f"{today}.{num}"
            prefix = Path(dir) / tn
            if not prefix.exists():
                break
            num += 1
        prefix.mkdir(parents=True, exist_ok=True)
        self._tn_dir = str(prefix)

        if params is not None:
            with open(prefix / 'params.txt', 'w') as f:
                f.write(str(params))
            with open(prefix / 'params.pkl', 'wb') as f:
                pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL if compress else pickle.DEFAULT_PROTOCOL)

        # Save routines (not applicable in Python)
        # Save main (not applicable in Python)

        sav = []
        for k, item in self._storage.items():
            if k in self._items:
                if len(item) > 0:
                    if not nofits:
                        filename = prefix / f"{k}.fits"
                        times = np.array(list(item.keys())) / self._time_resolution
                        data = np.array(list(item.values()))
                        if save_float and data.dtype == np.float64:
                            data = data.astype(np.float32)
                        writefits(filename, data)
                        writefits(filename, times, append=True)
                    else:
                        sav.append(k)
                        temp = np.array(list(item.keys()))
                        if len(temp) > 1:
                            sav.append(f"{k}_times")

        if sav and not nosav:
            filename = prefix / 'data.pkl'
            with open(filename, 'wb') as f:
                pickle.dump({k: self._storage[k] for k in sav}, f, protocol=pickle.HIGHEST_PROTOCOL if compress else pickle.DEFAULT_PROTOCOL)
            self._data_filename = str(filename)

        if not nooldformat:
            filename = prefix / 'old_format.pkl'
            self.save(filename, compress=compress)

        return tn

    def restore(self, filename, params=None):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        if 'params' in data:
            if params is not None:
                params = data.pop('params')
            self._storage['params'] = params
        self._storage.update(data)

    def restore_tracknum(self, tn, dir='.', params=None, no_phi=False):
        tndir = Path(dir) / tn
        self.restore(tndir / 'old_format.pkl', params=params)
        if (tndir / 'params.pkl').exists():
            with open(tndir / 'params.pkl', 'rb') as f:
                params = pickle.load(f)
        for fits_file in tndir.glob('*.fits'):
            name = fits_file.stem
            if no_phi and 'Phi' in name:
                continue
            if name in self._storage:
                raise ValueError(f'Storing already has an object with name {name}')
            data = readfits(fits_file)
            times = readfits(fits_file, ext=1)
            self._storage[name] = OrderedDict(zip(times, data))

    @staticmethod
    def restore(filename, dir='.', params=None, no_phi=False):
        store = DataStore()
        if Path(dir).is_dir():
            store.restore_tracknum(filename, dir=dir, params=params, no_phi=no_phi)
        else:
            store.restore(Path(dir) / filename, params=params)
        return store

    def set_property(self, decimation_t=None, **kwargs):
        if decimation_t is not None:
            self._decimation_t = self.seconds_to_t(decimation_t)
        super().set_property(**kwargs)

    def get_property(self, tn_dir=None, data_filename=None, **kwargs):
        super().get_property(**kwargs)
        if tn_dir is not None:
            tn_dir = self._tn_dir
        if data_filename is not None:
            data_filename = self._data_filename

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
        times = self._storage.get(f'{name}_TIMES')
        if times is None:
            times = np.array(list(self._storage[name].keys()))
        return times.tolist() if as_list else self.t_to_seconds(times)

    def values(self, name, as_list=False, init=0):
        if not self.has_key(name):
            print(f'The key: {name} is not stored in the object!')
            return -1
        values = self._storage[name]
        if isinstance(values, OrderedDict):
            values = np.array(list(values.values()))
        if init > 0:
            values = values[init:]
        return values.tolist() if as_list else values

    def size(self, name, dimensions=False):
        if not self.has_key(name):
            print(f'The key: {name} is not stored in the object!')
            return -1
        values = self._storage[name]
        if isinstance(values, OrderedDict):
            values = np.array(list(values.values()))
        return values.shape if dimensions else values.size

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

    def plot(self, name, init=0, map=False, over=False, resolution=1e9, **kwargs):
        values = self.values(name, init=init)
        if map:
            plt.imshow(values, **kwargs)
        else:
            times = self.times(name)
            if resolution != 1e9:
                if (times[1] - times[0]) * resolution <= 1:
                    times *= resolution
                elif (times[1] - times[0]) / resolution > 1e-4:
                    times /= resolution
            if over:
                plt.plot(times[init:], values, **kwargs)
            else:
                plt.plot(times[init:], values, **kwargs)
        plt.show()

    def trigger(self, t):
        do_store_values = self._decimation_t == 0 or t % self._decimation_t == 0
        if do_store_values:
            for k, item in self._items.items():
                if item is not None and item.generation_time == t:
                    v = self._get_value_from_item(item)
                    self._storage[k][t] = v

    def _get_value_from_item(self, item):
        class_name = type(item).__name__
        if class_name in ['BaseValue', 'BaseGPUValue', 'Cheat', 'Slopes', 'Pixels', 'Ef', 'I', 'Layer']:
            return getattr(item, 'value', None) or getattr(item, 'read', None)() or getattr(item, 'slopes', None) or getattr(item, 'pixels', None) or getattr(item, 'phase_in_nm', None) or getattr(item, 'i', None)
        else:
            raise ValueError(f"Error: don't know how to save an object of type {class_name}")

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
            print('datastore has been cleaned up.')

# Helper functions
def tracknum():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d%H%M%S")

def writefits(filename, data, append=False):
    from astropy.io import fits
    hdu = fits.PrimaryHDU(data)
    if append and Path(filename).exists():
        hdu_list = fits.open(filename, mode='append')
        hdu_list.append(hdu)
        hdu_list.writeto(filename, overwrite=True)
    else:
        hdu.writeto(filename, overwrite=True)

def readfits(filename, ext=0):
    from astropy.io import fits
    with fits.open(filename) as hdul:
        return hdul[ext].data
