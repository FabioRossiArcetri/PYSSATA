import numpy as np
from astropy.io import fits

class BaseTimeObj:
    def __init__(self, objname, objdescr, precision=0):
        """
        Creates a new base_time object.

        Parameters:
        objname (str): object name
        objdescr (str): object description
        precision (int, optional): double 1 or single 0, defaults to single precision
        """
        self._objname = objname
        self._objdescr = objdescr
        self._time_resolution = int(1e9)
        self._generation_time = -1
        self._precision = precision

    def __repr__(self):
        return f"{self._objdescr} ({self._objname})"

    @property
    def generation_time(self):
        return self._generation_time

    @generation_time.setter
    def generation_time(self, value):
        self._generation_time = value

    @property
    def time_resolution(self):
        return self._time_resolution

    @time_resolution.setter
    def time_resolution(self, value):
        self._time_resolution = value

    @property
    def objdescr(self):
        return self._objdescr

    @objdescr.setter
    def objdescr(self, value):
        self._objdescr = value

    @property
    def objname(self):
        return self._objname

    @objname.setter
    def objname(self, value):
        self._objname = value

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, value):
        self._precision = value

    def t_to_seconds(self, t):
        return float(t) / float(self._time_resolution)

    def seconds_to_t(self, seconds):
        if self._time_resolution == 0:
            return 0

        ss = f"{float(seconds):.9f}".rstrip('0').rstrip('.')
        if '.' not in ss:
            ss += '.'

        dotpos = ss.find('.')
        intpart = ss[:dotpos]
        fracpart = ss[dotpos + 1:]

        return (int(intpart) * self._time_resolution +
                int(fracpart) * (self._time_resolution // (10 ** len(fracpart))))

    def save(self, filename):
        hdr = fits.Header()
        hdr['GEN_TIME'] = self._generation_time
        hdr['TIME_RES'] = self._time_resolution
        hdr['OBJNAME'] = self._objname
        hdr['OBJDESCR'] = self._objdescr
        hdr['PRECISION'] = self._precision

        primary_hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(filename, overwrite=True)

    def read(self, filename):
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            self._generation_time = int(hdr.get('GEN_TIME', 0))
            self._time_resolution = int(hdr.get('TIME_RES', 0))
            self._objname = hdr.get('OBJNAME', '')
            self._objdescr = hdr.get('OBJDESCR', '')
            self._precision = int(hdr.get('PRECISION', 0))

    def get_properties_list(self):
        return vars(self)

    def cleanup(self):
        pass
