import numpy as np

from pyssata import xp

from astropy.io import fits

from pyssata.data_objects.base_data_obj import BaseDataObj

class Intensity(BaseDataObj):
    '''Intensity field object'''
    def __init__(self, dimx, dimy, precision=None):
        super().__init__(precision=precision)
                
        self._i = xp.zeros((dimx, dimy), dtype=self.dtype)

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

    def save(self, filename, hdr):
        hdr = fits.Header()
        hdr.append(('VERSION', 1))
        super().save(filename, hdr)
        fits.writeto(filename, self._i, hdr, overwrite=True)

    def read(self, filename):
        hdr = fits.getheader(filename)
        super().read(filename, hdr)
        self._i = fits.getdata(filename)
