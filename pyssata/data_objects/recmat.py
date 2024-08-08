import numpy as np
from astropy.io import fits

from pyssata.data_objects.base_data_obj import BaseDataObj


class Recmat(BaseDataObj):
    def __init__(self):
        self._recmat = None
        self._modes2recLayer = None
        self._im_tag = ''
        self._gpu_recmat = None
        self._proj_list = []
        self._norm_factor = 0.0
        self._doNotPutOnGpu = False

        if not super().__init__():
            return

    @property
    def recmat(self):
        return self._recmat

    @recmat.setter
    def recmat(self, value):
        self.set_recmat(value)

    @property
    def modes2recLayer(self):
        return self._modes2recLayer

    @modes2recLayer.setter
    def modes2recLayer(self, value):
        self.set_modes2recLayer(value)

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

    def set_recmat(self, recmat, doNotPutOnGpu=False):
        self.free()
        self._doNotPutOnGpu = doNotPutOnGpu
        if recmat is not None:
            self._recmat = recmat
            if self.has_gpu():
                self._gpu_recmat = self.create_gpu_matrix(recmat, doNotPutOnGpu)

    def set_modes2recLayer(self, modes2recLayer):
        self._modes2recLayer = modes2recLayer
        self._proj_list = []
        n = modes2recLayer.shape
        for i in range(n[0]):
            idx = np.where(modes2recLayer[i, :] > 0)[0]
            proj = np.zeros((n[1], len(idx)), dtype=float)
            proj[idx, :] = np.identity(len(idx))
            self._proj_list.append(proj)

    def reduce_size(self, nModesToBeDiscarded):
        recmat = self._recmat
        nmodes = recmat.shape[1]
        if nModesToBeDiscarded >= nmodes:
            raise ValueError(f"nModesToBeDiscarded should be less than nmodes (<{nmodes})")
        self._recmat = recmat[:, :nmodes - nModesToBeDiscarded]
        if self.has_gpu():
            self._gpu_recmat = self.create_gpu_matrix(self._recmat, self._doNotPutOnGpu)

    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['IM_TAG'] = self._im_tag
        hdr['TAG'] = self._tag
        hdr['NORMFACT'] = self._norm_factor

        super().save(filename, hdr)

        fits.append(filename, self._recmat)
        if self._modes2recLayer is not None:
            fits.append(filename, self._modes2recLayer)

    def read(self, filename, hdr=None, exten=0, doNotPutOnGpu=False):
        super().read(filename, hdr, exten)

        self._recmat = fits.getdata(filename, ext=exten)
        self.set_recmat(self._recmat, doNotPutOnGpu)

        mode2reLayer = fits.getdata(filename, ext=exten + 1)
        if mode2reLayer.size > 1:
            self.set_modes2recLayer(mode2reLayer)

        self._norm_factor = float(hdr['NORMFACT'])
        exten += 1

    @staticmethod
    def restore(filename, doNotPutOnGpu=False):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])

        if version != 1:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        rec = Recmat()
        rec.im_tag = str(hdr['IM_TAG']).strip()
        rec.read(filename, hdr, doNotPutOnGpu=doNotPutOnGpu)

        return rec

    def revision_track(self):
        return '$Rev$'

    def free(self):
        self._recmat = None
        if self._gpu_recmat is not None:
            self._gpu_recmat.cleanup()
            self._gpu_recmat = None

    def cleanup(self):
        self.free()
        super().cleanup()

    def has_gpu(self):
        # Placeholder for actual GPU check
        return False

    def create_gpu_matrix(self, array, doNotPutOnGpu):
        # Placeholder for actual GPU matrix creation
        return None
