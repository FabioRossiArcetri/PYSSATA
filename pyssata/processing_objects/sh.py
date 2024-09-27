import numpy as np
from scipy.ndimage import convolve, shift
from scipy.fftpack import fft2, ifft2
import math

from pyssata.base_processing_obj import BaseProcessingObj

class SH(BaseProcessingObj):
    def __init__(self, wavelength_in_nm, lenslet, subap_wanted_fov, sensor_pxscale, subap_npx, FoVres30mas=None, gkern=None, 
                 target_device_idx: int = None, 
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)        

        rad2arcsec = 180 / self.xp.pi * 3600
        self._wavelength_in_nm = wavelength_in_nm
        self._lenslet = lenslet
        self._subap_wanted_fov = subap_wanted_fov / rad2arcsec
        self._sensor_pxscale = sensor_pxscale / rad2arcsec
        self._subap_npx = subap_npx
        self._gkern = False
        self._gkern_size = 0
        if gkern is not None:
            self._gkern = True
            self._gkern_size = gkern / rad2arcsec

        self._ccd_side = self._subap_npx * self._lenslet.n_lenses
        self._out_i = None  # Image output placeholder, to be filled later
        self._kernel_fov_scale = 1.0
        self._squaremask = True
        self._fov_resolution_arcsec = 0.03 if FoVres30mas is not None else 0
        self._kernel_application = ''
        self._kernel_precalc_fft = False
        self._debugOutput = False
        self._noprints = False
        self._subap_idx = None
        self._idx_valid = False
        self._scale_ovs = 1.0
        self._floatShifts = False
        self._rotAnglePhInDeg = 0
        self._xShiftPhInPixel = 0
        self._yShiftPhInPixel = 0
        self._fov_ovs = 1
        self._set_fov_res_to_turbpxsc = False
        self._do_not_double_fov_ovs = False
        self._np_sub = 0
        self._fft_size = 0

    @property
    def wavelength_in_nm(self):
        return self._wavelength_in_nm

    @wavelength_in_nm.setter
    def wavelength_in_nm(self, value):
        self._wavelength_in_nm = value

    @property
    def sensor_pxscale(self):
        return self._sensor_pxscale

    @sensor_pxscale.setter
    def sensor_pxscale(self, value):
        self._sensor_pxscale = value

    @property
    def fov_resolution_arcsec(self):
        return self._fov_resolution_arcsec

    @fov_resolution_arcsec.setter
    def fov_resolution_arcsec(self, value):
        self._fov_resolution_arcsec = value

    @property
    def kernel_application(self):
        return self._kernel_application

    @kernel_application.setter
    def kernel_application(self, str_val):
        if str_val not in ['FFT', 'FOV', 'SUBAP']:
            raise ValueError("Kernel application string must be one of 'FFT', 'FOV', or 'SUBAP'")
        self._kernel_application = str_val

    def set_in_ef(self, in_ef):
        rad2arcsec = 180 / self.xp.pi * 3600

        self._in_ef = in_ef
        lens = self._lenslet.get(0, 0)
        n_lenses = self._lenslet.n_lenses
        ef_size = in_ef.shape[0]
        self._np_sub = max([1, round((ef_size * lens[2]) / 2.0)])
        if self._np_sub * n_lenses > ef_size:
            self._np_sub -= 1
        np_sub = (ef_size * lens[2]) / 2.0

        sensor_pxscale_arcsec = self._sensor_pxscale * rad2arcsec
        dSubApInM = np_sub * in_ef.pixel_pitch
        turbulence_pxscale = self._wavelength_in_nm * 1e-9 / dSubApInM * rad2arcsec
        subap_wanted_fov_arcsec = self._subap_wanted_fov * rad2arcsec
        subap_real_fov_arcsec = self._sensor_pxscale * self._subap_npx * rad2arcsec

        if self._fov_resolution_arcsec == 0:
            if not self._noprints:
                print('FoV internal resolution parameter not set.')
            if self._set_fov_res_to_turbpxsc:
                if turbulence_pxscale >= sensor_pxscale_arcsec:
                    raise ValueError('set_fov_res_to_turbpxsc property should be set to one only if turb. pix. sc. is < sensor pix. sc.')
                self._fov_resolution_arcsec = turbulence_pxscale
            else:
                if turbulence_pxscale < sensor_pxscale_arcsec and sensor_pxscale_arcsec / 2 > 0.5:
                    self._fov_resolution_arcsec = turbulence_pxscale * 0.5
                else:
                    i = 0
                    res_try = turbulence_pxscale / (i + 2)
                    while res_try >= sensor_pxscale_arcsec:
                        res_try = turbulence_pxscale / (i + 2)
                        i += 1
                    self._fov_resolution_arcsec = res_try

    def detect_subaps(self, image, lenslet, energy_th):
        np = image.shape[0]
        mask_subap = self.xp.zeros_like(image)

        idxs = {}
        map = {}
        count = 0
        spot_intensity = self.xp.zeros((self._lenslet.dimx, self._lenslet.dimy))
        x = self.xp.zeros((self._lenslet.dimx, self._lenslet.dimy))
        y = self.xp.zeros((self._lenslet.dimx, self._lenslet.dimy))

        for i in range(self._lenslet.dimx):
            for j in range(self._lenslet.dimy):
                lens = self._lenslet.get(i, j)
                x[i, j] = np / 2.0 * (1 + lens[0])
                y[i, j] = np / 2.0 * (1 + lens[1])
                np_sub = round(np / 2.0 * lens[2])

                mask_subap *= 0
                mask_subap[round(x[i, j] - np_sub / 2):round(x[i, j] + np_sub / 2) - 1,
                           round(y[i, j] - np_sub / 2):round(y[i, j] + np_sub / 2) - 1] = 1

                spot_intensity[i, j] = self.xp.sum(image * mask_subap)

        for i in range(self._lenslet.dimx):
            for j in range(self._lenslet.dimy):
                if spot_intensity[i, j] > energy_th * self.xp.max(spot_intensity):
                    mask_subap *= 0
                    mask_subap[round(x[i, j] - np_sub / 2):round(x[i, j] + np_sub / 2) - 1,
                               round(y[i, j] - np_sub / 2):round(y[i, j] + np_sub / 2) - 1] = 1
                    idxs[count] = self.xp.where(mask_subap == 1)
                    map[count] = j * self._lenslet.dimx + i
                    count += 1

        if count == 0:
            raise ValueError("Error: no subapertures selected")
        print(f'Selected {count} subapertures')

        subaps = SubapData(np_sub=np_sub, n_subaps=len(idxs))
        for k, idx in idxs.items():
            subaps.set_subap_idx(k, idx)
        for k, pos in map.items():
            subaps.set_subap_map(k, pos)

        subaps.energy_th = energy_th
        subaps.nx = self._lenslet.dimx
        subaps.ny = self._lenslet.dimy
        return subaps

    def trigger(self, t):
        if self._in_ef.generation_time != t:
            return

        s = self._in_ef.shape

        fov_oversample = self._fov_ovs
        scale_oversample = self._scale_ovs

        subap_wanted_fov = self._subap_wanted_fov
        sensor_pxscale = self._sensor_pxscale
        subap_npx = self._subap_npx

        rot_angle = self._rotAnglePhInDeg
        xy_shift = [self._xShiftPhInPixel, self._yShiftPhInPixel] * fov_oversample
        if not self._floatShifts:
            xy_shift = self.xp.round(xy_shift)

        M = s[0] * fov_oversample

        if fov_oversample != 1 or self.xp.sum(self.xp.abs([rot_angle, xy_shift])) != 0:
            wf1 = self.xp.zeros((M, M), dtype=complex)
            phase_in_nm_new = self.xp.zeros_like(self._in_ef)  # Placeholder for edge extrapolation
            # Perform congrid and interpolation logic here
        else:
            wf1 = self._in_ef

        wf1_np = wf1.shape[0]
        f = self.xp.zeros((wf1_np, wf1_np))

        np_sub = self._congrid_np_sub
        fft_size = self._fft_size
        nsubap_diam = self._lenslet.dimx

        wf2 = self.xp.zeros((np_sub, np_sub), dtype=complex)
        wf3 = self.xp.zeros((fft_size, fft_size), dtype=complex)

        # Mask to generate the wanted FoV
        fov_complete = fft_size * self._in_ef.pixel_pitch
        fp_mask = self.xp.zeros((fft_size, fft_size))  # Placeholder for mask logic

        tltf = self.get_tlt_f(np_sub, fft_size - np_sub)

        psf_total_at_fft = 0

        for i in range(self._lenslet.dimx):
            for j in range(self._lenslet.dimy):
                lens = self._lenslet.get(i, j)
                x = wf1_np / 2.0 * (1 + lens[0])
                y = wf1_np / 2.0 * (1 + lens[1])

                if not self._idx_valid:
                    f *= 0
                    f[round(x - np_sub / 2):round(x + np_sub / 2) - 1,
                      round(y - np_sub / 2):round(y + np_sub / 2) - 1] = 1
                    idx = self.xp.where(f == 1)
                    self._subap_idx[i, j, :] = idx
                    if i == self._lenslet.dimx - 1 and j == self._lenslet.dimy - 1:
                        self._idx_valid = True
                else:
                    idx = self._subap_idx[i, j, :]

                wf2 = wf1[idx]
                wf3[0, 0] = wf2 * tltf

                tmp_fp4 = fft2(wf3)
                psf_total_at_fft += self.xp.sum(self.xp.abs(tmp_fp4) ** 2)

                if self.xp.any(self.xp.isnan(tmp_fp4)):
                    raise ValueError("FFT result contains NaN values")

        if psf_total_at_fft > 0:
            # Normalize PSF
            pass

    def cleanup(self):
        # Clean up references and memory
        self._in_ef = None
        self._out_i = None
        self._lenslet = None
        self._kernelobj = None

    def get_tlt_f(self, p, c):
        iu = complex(0, 1)
        xx, yy = self.xp.meshgrid(self.xp.arange(-p // 2, p // 2), self.xp.arange(-p // 2, p // 2))
        tlt_g = xx + yy
        tlt_f = self.xp.exp(-2 * self.xp.pi * iu * tlt_g / (2 * (p + c)))
        return tlt_f
