import numpy as np
from pyssata import gpuEnabled
from pyssata import xp

from astropy.io import fits

from pyssata.data_objects.base_data_obj import BaseDataObj

class PupData(BaseDataObj):
    def __init__(self):
        super().__init__()
        self._radius = xp.zeros(4, dtype=float)
        self._cx = xp.zeros(4, dtype=float)
        self._cy = xp.zeros(4, dtype=float)
        self._ind_pup = xp.empty((4, 0), dtype=int)
        self._framesize = xp.zeros(2, dtype=int)
        
    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value

    @property
    def cx(self):
        return self._cx

    @cx.setter
    def cx(self, value):
        self._cx = value

    @property
    def cy(self):
        return self._cy

    @cy.setter
    def cy(self, value):
        self._cy = value

    @property
    def framesize(self):
        return self._framesize

    @framesize.setter
    def framesize(self, value):
        self._framesize = value

    @property
    def ind_pup(self):
        return self._ind_pup

    @ind_pup.setter
    def ind_pup(self, value):
        self._ind_pup = self.zcorrection(value)

    @property
    def n_subap(self):
        return self._ind_pup.shape[1] // 4

    def zcorrection(self, indpup):
        tmp = indpup.copy()
        tmp[2, :], tmp[3, :] = indpup[3, :], indpup[2, :]
        return tmp

    def single_mask(self):
        f = xp.zeros(self._framesize, dtype=float)
        f[self._ind_pup[0, :]] = 1
        return f

    def complete_mask(self):
        f = xp.zeros(self._framesize, dtype=float)
        for i in range(4):
            f[self._ind_pup[i, :]] = 1
        return f

    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = fits.Header()
        hdr['VERSION'] = 2
        hdr['FSIZEX'] = self._framesize[0]
        hdr['FSIZEY'] = self._framesize[1]

        super().save(filename, hdr)

        fits.append(filename, self._ind_pup)
        fits.append(filename, self._radius)
        fits.append(filename, self._cx)
        fits.append(filename, self._cy)

    def read(self, filename, hdr=None, exten=0):
        hdr, exten = super().read(filename)

        self._ind_pup = fits.getdata(filename, ext=exten)
        self._radius = fits.getdata(filename, ext=exten + 1)
        self._cx = fits.getdata(filename, ext=exten + 2)
        self._cy = fits.getdata(filename, ext=exten + 3)
        exten += 4

    @staticmethod
    def restore(filename):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])

        if version > 2:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        p = PupData()
        if version >= 2:
            p.framesize = [int(hdr['FSIZEX']), int(hdr['FSIZEY'])]

        p.read(filename, hdr)
        return p
