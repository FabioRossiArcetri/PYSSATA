import numpy as np
from astropy.io import fits

from pyssata.data_objects.base_data_obj import BaseDataObj

class Intensity(BaseDataObj):
    '''Intensity field object'''
    def __init__(self, dimx, dimy, precision=0, type_str=None):
        super().__init__()
        
        if type_str is None:
            self.type = np.float32 if precision == 0 else np.float64
        else:
            self.type = np.dtype(type_str)
        
        self._i = np.zeros((dimx, dimy), dtype=self.type)

    @property
    def i(self):
        return self._i

    @i.setter
    def i(self, value):
        self._i = value

    @property
    def ptr_i(self):
        return self._i

    @property
    def size(self):
        return self._i.shape

    @property
    def value(self):
        return self._i

    @property
    def ptr_value(self):
        return self._i

    def sum(self, i2, factor=1.0):
        self._i += i2.i * factor

    def revision_track(self):
        return '$Rev$'

    def cleanup(self):
        del self._i
        super().cleanup()

    def save(self, filename, hdr):
        hdr = fits.Header()
        hdr.append(('VERSION', 1))
        super().save(filename, hdr)
        fits.writeto(filename, self._i, hdr, overwrite=True)

    def read(self, filename):
        hdr = fits.getheader(filename)
        super().read(filename, hdr)
        self._i = fits.getdata(filename)
