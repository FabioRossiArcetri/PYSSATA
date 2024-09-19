from astropy.io import fits

from pyssata.base_time_obj import BaseTimeObj
from pyssata import float_dtype

class BaseDataObj(BaseTimeObj):
    def __init__(self, precision=None):
        """
        Initialize the base data object.

        Parameters:
        precision (int, optional):if None will use the global_precision, otherwise pass 0 for double, 1 for single
        """
        super().__init__(precision)
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
        exten = 1
        return hdr, exten

    def cleanup(self):
        pass
