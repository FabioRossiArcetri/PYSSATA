
import math
from pyssata.data_objects.base_data_obj import BaseDataObj


class SubapData(BaseDataObj):
    def __init__(self, np_sub=None, n_subaps=None, target_device_idx=None, precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        if np_sub is not None and n_subaps is not None:
            self._idxs = self.xp.zeros((n_subaps, np_sub ** 2), dtype=int)
            self._map = self.xp.zeros(n_subaps, dtype=int)
        else:
            self._idxs = None
            self._map = None

        self._energy_th = 0.0
        self._nx = 0
        self._ny = 0

    @property
    def idxs(self):
        return self._idxs

    @idxs.setter
    def idxs(self, value):
        self._idxs = value

    @property
    def n_subaps(self):
        return self._idxs.shape[0] if self._idxs is not None else 0

    @property
    def np_sub(self):
        return int(math.sqrt(self._idxs.shape[1])) if self._idxs is not None else 0

    @property
    def map(self):
        return self._map

    @map.setter
    def map(self, value):
        self._map = value

    @property
    def energy_th(self):
        return self._energy_th

    @energy_th.setter
    def energy_th(self, value):
        self._energy_th = value

    @property
    def nx(self):
        return self._nx

    @nx.setter
    def nx(self, value):
        self._nx = value

    @property
    def ny(self):
        return self._ny

    @ny.setter
    def ny(self, value):
        self._ny = value

    def subap_idx(self, n):
        """Returns the indices of subaperture `n`."""
        return self._idxs[n, :]

    def map_idx(self, n):
        """Returns the position of subaperture `n`."""
        return self._map[n]

    def set_subap_idx(self, n, idx):
        """Sets the indices of subaperture `n`."""
        self._idxs[n, :] = idx

    def set_subap_map(self, n, pos):
        """Sets the mapping position of subaperture `n`."""
        self._map[n] = pos

    def save(self, filename, hdr):
        """Saves the subaperture data to a file."""
        hdr['VERSION'] = 1
        hdr['ENRGYTH'] = self._energy_th
        hdr['NX'] = self._nx
        hdr['NY'] = self._ny
        super().save(filename, hdr)
        self.xp.savez_compressed(filename, idxs=self._idxs, map=self._map)

    def read(self, filename, hdr, exten=0):
        """Reads subaperture data from a file."""
        super().read(filename, hdr, exten)
        data = self.xp.load(filename + ".npz")
        self._idxs = data['idxs']
        self._map = data['map']

    @classmethod
    def restore(cls, filename):
        """Restores the subaperture data from a file."""
        p = cls()
        hdr = {}
        data = self.xp.load(filename + ".npz")
        version = int(hdr.get('VERSION', 1))
        if version != 1:
            raise ValueError(f"Unknown version {version} in file {filename}")
        p._nx = int(hdr.get('NX', 0))
        p._ny = int(hdr.get('NY', 0))
        p.read(filename, hdr)
        return p
