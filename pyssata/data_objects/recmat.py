
import numpy as np
from astropy.io import fits

from pyssata import cpuArray
from pyssata.base_data_obj import BaseDataObj


class Recmat(BaseDataObj):
    def __init__(self,
                 recmat,
                 modes2recLayer=None,
                 norm_factor: float = 0,
                 target_device_idx=None,
                 precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self._recmat = self.xp.array(recmat)
        self._modes2recLayer = modes2recLayer
        self._im_tag = ''
        self._proj_list = []
        self._norm_factor = norm_factor

    @property
    def recmat(self):
        return self._recmat

    @property
    def modes2recLayer(self):
        return self._modes2recLayer

    @modes2recLayer.setter
    def modes2recLayer(self, modes2recLayer):
        self._modes2recLayer = modes2recLayer
        self._proj_list = []
        n = modes2recLayer.shape
        for i in range(n[0]):
            idx = self.xp.where(modes2recLayer[i, :] > 0)[0]
            proj = self.xp.zeros((n[1], len(idx)), dtype=self.dtype)
            proj[idx, :] = self.xp.identity(len(idx))
            self._proj_list.append(proj)
            
    @property
    def proj_list(self):
        return self._proj_list

    @proj_list.setter
    def proj_list(self, value):
        self._proj_list = value

    @property
    def im_tag(self):
        return self._im_tag

    @im_tag.setter
    def im_tag(self, value):
        self._im_tag = value

    @property
    def norm_factor(self):
        return self._norm_factor

    @norm_factor.setter
    def norm_factor(self, value):
        self._norm_factor = value

    def reduce_size(self, nModesToBeDiscarded):
        recmat = self._recmat
        nmodes = recmat.shape[1]
        if nModesToBeDiscarded >= nmodes:
            raise ValueError(f"nModesToBeDiscarded should be less than nmodes (<{nmodes})")
        self._recmat = recmat[:, :nmodes - nModesToBeDiscarded]

    def save(self, filename, hdr=None):
        
        if not filename.endswith('.fits'):
            filename += '.fits'

        if hdr is None:
            hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['IM_TAG'] = self._im_tag
        hdr['NORMFACT'] = self._norm_factor

        fits.writeto(filename, np.zeros(2), hdr)
        fits.append(filename, cpuArray(self._recmat.T))
        if self._modes2recLayer is not None:
            fits.append(filename, cpuArray(self._modes2recLayer))

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        norm_factor = float(hdr['NORMFACT'])
        recmat = fits.getdata(filename, ext=1)
        if len(fits.open(filename)) >= 3:
            mode2reLayer = fits.getdata(filename, ext=2)
        else:
            mode2reLayer = None
        return Recmat(recmat, mode2reLayer, norm_factor, target_device_idx=target_device_idx)


