import numpy as np
from astropy.io import fits

from pyssata.base_time_obj import BaseTimeObj

class BaseDataObj(BaseTimeObj):
    def __init__(self, objname, objdescr, precision=0):
        """
        Initialize the base data object.

        Parameters:
        objname (str): object name
        objdescr (str): object description
        precision (int, optional): double 1 or single 0, defaults to single precision
        """
        super().__init__(objname, objdescr, precision)
        self._tag = ''

    @property
    def tag(self):
        return self._tag

    @tag.setter
    def tag(self, value):
        self._tag = value

    def save(self, filename):
        hdr = fits.Header()
        hdr['TAG'] = self._tag
        super().save(filename)
        with fits.open(filename, mode='update') as hdul:
            hdr = hdul[0].header
            hdr['TAG'] = self._tag
            hdul.flush()

    def read(self, filename):
        super().read(filename)
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            self._tag = hdr.get('TAG', '').strip()

    def cleanup(self):
        pass
