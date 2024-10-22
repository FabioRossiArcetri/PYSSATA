import math

import numpy as np
from astropy.io import fits

from pyssata import cpuArray
from pyssata.base_data_obj import BaseDataObj


class SubapData(BaseDataObj):
    def __init__(self,
                 idxs,
                 map,
                 nx: int,
                 ny: int,
                 energy_th: float = 0,
                 target_device_idx=None,
                 precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self._idxs = idxs.astype(int)
        self._map = map.astype(int)
        self._nx = int(nx)
        self._ny = int(ny)
        self._energy_th = float(energy_th)

    @property
    def idxs(self):
        return self._idxs

    @idxs.setter
    def idxs(self, value):
        self._idxs = value

    @property
    def n_subaps(self):
        return self._idxs.shape[0]

    @property
    def np_sub(self):
        return int(math.sqrt(self._idxs.shape[1]))

    @property
    def map(self):
        return self._map

    @property
    def energy_th(self):
        return self._energy_th

    @property
    def nx(self):
        return self._nx

    @property
    def ny(self):
        return self._ny

    def subap_idx(self, n):
        """Returns the indices of subaperture `n`."""
        return self._idxs[n, :]

    def map_idx(self, n):
        """Returns the position of subaperture `n`."""
        return self._map[n]

    def save(self, filename):
        """Saves the subaperture data to a file."""
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['ENRGYTH'] = self.energy_th
        hdr['NP_SUB'] = self.np_sub
        hdr['NX'] = self.nx
        hdr['NY'] = self.ny
        fits.writeto(filename, np.zeros(2), hdr)
        fits.append(filename, cpuArray(self._idxs))
        fits.append(filename, cpuArray(self._map))

    @classmethod
    def restore(cls, filename, target_device_idx=None):
        """Restores the subaperture data from a file."""
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            version = hdr.get('VERSION')
            if version != 1:
                raise ValueError(f"Unknown version {version} in file {filename}")
            energy_th = hdr.get('ENRGYTH')
            nx = hdr.get('NX')
            ny = hdr.get('NY')
            idxs = hdul[1].data
            map = hdul[2].data
        return SubapData(idxs=idxs, map=map, nx=nx, ny=ny, energy_th=energy_th,
                         target_device_idx=target_device_idx)
