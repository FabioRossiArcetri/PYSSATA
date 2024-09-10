import numpy as np
from astropy.io import fits

from pyssata.data_objects.recmat import Recmat

class Intmat:
    def __init__(self, intmat):
        self._intmat = intmat
        self._slope_mm = None
        self._slope_rms = None
        self._pupdata_tag = ''
        self._norm_factor = 0.0

    def set_intmat(self, intmat):
        self._intmat = intmat

    def row(self, row):
        r = np.reshape(self._intmat[row, :], (-1,))
        return {'slopes': r, 'pupdata_tag': self._pupdata_tag}

    def reduce_size(self, n_modes_to_be_discarded):
        nmodes = self._intmat.shape[0]
        if n_modes_to_be_discarded >= nmodes:
            raise ValueError(f'nModesToBeDiscarded should be less than nmodes (<{nmodes})')
        self._intmat = self._intmat[:nmodes - n_modes_to_be_discarded, :]

    def reduce_slopes(self, n_slopes_to_be_discarded):
        nslopes = self._intmat.shape[1]
        if n_slopes_to_be_discarded >= nslopes:
            raise ValueError(f'nSlopesToBeDiscarded should be less than nslopes (<{nslopes})')
        self._intmat = self._intmat[:, :nslopes - n_slopes_to_be_discarded]

    def set_start_mode(self, start_mode):
        nmodes = self._intmat.shape[0]
        if start_mode >= nmodes:
            raise ValueError(f'start_mode should be less than nmodes (<{nmodes})')
        self._intmat = self._intmat[start_mode:, :]

    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = {}
        hdr['VERSION'] = 1
        hdr['PUP_TAG'] = self._pupdata_tag
        hdr['TAG'] = self._norm_factor
        # Save fits file
        fits.writeto(filename, self._intmat, hdr, overwrite=True)
        if self._slope_mm is not None:
            fits.append(filename, self._slope_mm)
        if self._slope_rms is not None:
            fits.append(filename, self._slope_rms)

    def read(self, filename, hdr=None, exten=0):
        self.set_intmat(fits.getdata(filename, ext=exten))
        hdr = fits.getheader(filename, ext=exten)
        self._norm_factor = float(hdr.get('NORMFACT', 0.0))
        # Reading additional fits extensions
        info = fits.info(filename)
        if len(info) > 1:
            self._slope_mm = fits.getdata(filename, ext=exten + 1)
            self._slope_rms = fits.getdata(filename, ext=exten + 2)

    def generate_rec(self, nmodes=None, cut_modes=0, w_vec=None, interactive=False):
        if nmodes is not None:
            intmat = self._intmat[:nmodes, :]
        else:
            intmat = self._intmat
        recmat = self.pseudo_invert(intmat, n_modes_to_drop=cut_modes, w_vec=w_vec, interactive=interactive)
        rec = Recmat()
        rec.set_recmat(recmat)
        rec.im_tag = self._norm_factor
        return rec

    def pseudo_invert(self, matrix, n_modes_to_drop=0, w_vec=None, interactive=False):
        # TODO handle n_modes_to_drop, and w_vec
        return np.linalg.pinv(matrix)

    def build_from_slopes(self, slopes, disturbance):
        times = list(slopes.keys())
        nslopes = len(slopes[times[0]])
        nmodes = len(disturbance[times[0]])
        intmat = np.zeros((nmodes, nslopes))
        iter_per_mode = np.zeros(nmodes)
        slope_mm = np.zeros((nmodes, 2))
        slope_rms = np.zeros(nmodes)

        for t in times:
            amp = disturbance[t]
            mode = np.where(amp)[0][0]
            intmat[mode, :] += slopes[t] / amp[mode]
            iter_per_mode[mode] += 1

        for m in range(nmodes):
            if iter_per_mode[m] > 0:
                intmat[m, :] /= iter_per_mode[m]

        im = Intmat(intmat)
        im._slope_mm = slope_mm
        im._slope_rms = slope_rms
        return im
