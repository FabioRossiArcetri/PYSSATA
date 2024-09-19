import numpy as np

from pyssata import xp
from scipy.stats import poisson, gamma, norm
from numpy.random import default_rng
from scipy.ndimage import convolve

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.connections import InputValue, OutputValue
from pyssata.data_objects.pixels import Pixels
from pyssata.data_objects.intensity import Intensity

class CCD(BaseProcessingObj):
    '''Simple CCD from intensity field'''
    def __init__(self, dim2d, binning=1, photon_noise=False, readout_noise=False, excess_noise=False,
                 darkcurrent_noise=False, background_noise=False, cic_noise=False, cte_noise=False,
                 readout_level=0, darkcurrent_level=0, background_level=0, cic_level=0, cte_mat=None,
                 quantum_eff=1.0):
        super().__init__()

        self._photon_noise = photon_noise
        self._readout_noise = readout_noise
        self._excess_noise = excess_noise
        self._darkcurrent_noise = darkcurrent_noise
        self._background_noise = background_noise
        self._cic_noise = cic_noise
        self._cte_noise = cte_noise

        self._binning = binning
        self._readout_level = readout_level
        self._darkcurrent_level = darkcurrent_level
        self._background_level = background_level
        self._cic_level = cic_level
        self._cte_mat = cte_mat if cte_mat is not None else xp.zeros((dim2d[0], dim2d[1], 2), dtype=self.dtype)
        self._qe = quantum_eff

        self._pixels = Pixels(dim2d[0] // binning, dim2d[1] // binning)
        self._photon_seed = 1
        self._readout_seed = 2
        self._excess_seed = 3
        self._cic_seed = 4

        self._excess_delta = 1.0

        self._in_i = None
        self._dt = 0
        self._start_time = 0
        self._charge_diffusion = False
        self._charge_diffusion_fwhm = 0
        self._keep_ADU_bias = False
        self._doNotChangeI = False
        self._bg_remove_average = False
        self._do_not_remove_dark = False
        self._ADU_gain = 0
        self._ADU_bias = 0
        self._emccd_gain = 0
        self._bandw = 0
        self._pixelGains = None
        self._notUniformQeMatrix = None
        self._one_over_notUniformQeMatrix = None
        self._notUniformQe = False
        self._normNotUniformQe = False
        self._poidev = None
        self._gaussian_noise = None
        self._useOaalibNoiseSource = False
        self._photon_rng = default_rng(self._photon_seed)

        self.inputs['in_i'] = InputValue(object=self.in_i, type=Intensity)
        self.outputs['out_pixels'] = OutputValue(object=self.out_pixels, type=Pixels)

    @property
    def in_i(self):
        return self._in_i

    @in_i.setter
    def in_i(self, value):
        self._in_i = value
        s = self._pixels.size * self._binning
        self._integrated_i = Intensity(s[0], s[1])

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = self.seconds_to_t(value)

    @property
    def size(self):
        return self._pixels.size

    @property
    def out_pixels(self):
        return self._pixels

    @property
    def bandw(self):
        return self._bandw
    
    @bandw.setter
    def bandw(self, bandw):
        self._bandw = bandw

    @property
    def binning(self):
        return self._binning

    @binning.setter
    def binning(self, value):
        s = self._pixels.size * self._binning
        if (s[0] % self._binning) != 0:
            raise ValueError('Warning: binning requested not applied')
        self._pixels.size = (s[0] // value, s[1] // value)
        self._binning = value

    def trigger(self, t):
        if self._start_time <= 0 or t >= self._start_time:
            if self._in_i.generation_time == t:
                if self._loop_dt == 0:
                    raise ValueError('ccd object loop_dt property must be set.')
                if self._doNotChangeI:
                    self._integrated_i.sum(self._in_i, factor=self._loop_dt / self._dt)
                else:
                    self._integrated_i.sum(self._in_i, factor=self.t_to_seconds(self._loop_dt) * self._bandw)

            if (t + self._loop_dt - self._dt - self._start_time) % self._dt == 0:
                if self._doNotChangeI:
                    self._pixels.pixels = self._integrated_i.i.copy()
                else:
                    if self._emccd_gain == 0:
                        self._emccd_gain = 400 if self._excess_noise else 1

                    if self._ADU_gain == 0:
                        self._ADU_gain = 1 / 20 if self._excess_noise else 8

                    if self._ADU_gain <= 1 and (not self._excess_noise or self._emccd_gain <= 1):
                        print('ATTENTION: ADU gain is less than 1 and there is no electronic multiplication.')

                    if self._ADU_bias == 0:
                        self._ADU_bias = 400

                    self.apply_binning()
                    self.apply_qe()
                    self.apply_noise()

                self._pixels.generation_time = t
                self._integrated_i.i *= 0.0

    def apply_noise(self):
        ccd_frame = self._pixels.pixels
        if self._background_noise or self._darkcurrent_noise:
            ccd_frame += (self._background_level + self._darkcurrent_level)

        if self._cte_noise:
            ccd_frame = xp.dot(xp.dot(self._cte_mat[:, :, 0], ccd_frame), self._cte_mat[:, :, 1])

        if self._cic_noise:
            ccd_frame += xp.random.binomial(1, self._cic_level, ccd_frame.shape)

        if self._charge_diffusion:
            ccd_frame = convolve(ccd_frame, self._chDiffKernel, mode='constant', cval=0.0)

        if self._photon_noise:
            ccd_frame = poisson.rvs(ccd_frame, random_state=self._rng)

        if self._excess_noise:
            ccd_frame = 1.0 / self._excess_delta * gamma.rvs(self._excess_delta * ccd_frame, self._emccd_gain, random_state=self._excess_seed)

        if self._readout_noise:
            ron_vector = norm.rvs(size=ccd_frame.size, random_state=self._readout_seed)
            ccd_frame += ron_vector.reshape(ccd_frame.shape) * self._readout_level

        if self._pixelGains is not None:
            ccd_frame *= self._pixelGains

        if self._notUniformQe and self._normNotUniformQe:
            if self._one_over_notUniformQeMatrix is None:
                self._one_over_notUniformQeMatrix = 1 / self._notUniformQeMatrix
            ccd_frame *= self._one_over_notUniformQeMatrix

        if self._photon_noise:
            ccd_frame = xp.round(ccd_frame * self._ADU_gain) + self._ADU_bias
            ccd_frame[ccd_frame < 0] = 0

            if not self._keep_ADU_bias:
                ccd_frame -= self._ADU_bias

            ccd_frame /= self._ADU_gain
            if self._excess_noise:
                ccd_frame /= self._emccd_gain
            if self._darkcurrent_noise and not self._do_not_remove_dark:
                ccd_frame -= self._darkcurrent_level
            if self._bg_remove_average and not self._do_not_remove_dark:
                ccd_frame -= self._background_level

        self._pixels.pixels = ccd_frame

    def apply_binning(self):
        in_dim = self._integrated_i.ptr_i.shape
        out_dim = self._pixels.size

        if in_dim[0] != out_dim[0] * self._binning:
            ccd_frame = xp.zeros(out_dim * self._binning, dtype=self.dtype)
            ccd_frame[:in_dim[0], :in_dim[1]] = self._integrated_i.i
        else:
            ccd_frame = self._integrated_i.i.copy()

        if self._binning > 1:
            tot_ccd_frame = xp.sum(ccd_frame)
            ccd_frame = ccd_frame.reshape(out_dim[0], self._binning, out_dim[1], self._binning).sum(axis=(1, 3))
            ccd_frame = ccd_frame * self._binning ** 2 * (tot_ccd_frame / xp.sum(ccd_frame))
            self._pixels.pixels = ccd_frame
        else:
            self._pixels.pixels = self._integrated_i.i.copy()

    def apply_qe(self):
        if self._qe != 1:
            self._pixels.multiply(self._qe)
        if self._notUniformQe:
            ccd_frame = self._pixels.pixels * self._notUniformQeMatrix
            self._pixels.pixels = ccd_frame

    def setQuadrantGains(self, quadrantsGains):
        dim2d = self._pixels.pixels.shape
        pixelGains = xp.zeros(dim2d, dtype=self.dtype)
        for i in range(2):
            for j in range(2):
                pixelGains[(dim2d[0] // self._binning // 2) * i:(dim2d[0] // self._binning // 2) * (i + 1),
                           (dim2d[1] // self._binning // 2) * j:(dim2d[1] // self._binning // 2) * (j + 1)] = quadrantsGains[j * 2 + i]
        self._pixelGains = pixelGains

    def run_check(self, time_step, errmsg=''):
        if self._loop_dt == 0:
            self._loop_dt = time_step
        if self._in_i is None:
            errmsg = 'Input intensity object has not been set'
        if self._pixels is None:
            errmsg = 'Pixel object has not been set'
        if self._dt % time_step != 0:
            errmsg = f'integration time dt={self._dt} must be a multiple of the basic simulation time_step={time_step}'
        if self._dt <= 0:
            errmsg = f'dt (integration time) is {self._dt} and must be greater than zero'
        if self._cte_noise and self._cte_mat is None:
            errmsg = 'CTE matrix must be set!'

        is_check_ok = (self._in_i is not None and self._pixels is not None and
                       (self._dt > 0) and (self._dt % time_step == 0) and
                       (not self._cte_noise or self._cte_mat is not None))
        print(errmsg)
        return is_check_ok

    def cleanup(self):
        if ptr_valid(self._cte_mat):
            self._cte_mat = None
        if ptr_valid(self._photon_seed):
            self._photon_seed = None
        if ptr_valid(self._readout_seed):
            self._readout_seed = None
        if ptr_valid(self._excess_seed):
            self._excess_seed = None
        if ptr_valid(self._cic_seed):
            self._cic_seed = None
        if ptr_valid(self._chDiffKernel):
            self._chDiffKernel = None
        if ptr_valid(self._pixelGains):
            self._pixelGains = None
        if ptr_valid(self._notUniformQeMatrix):
            self._notUniformQeMatrix = None
        if ptr_valid(self._one_over_notUniformQeMatrix):
            self._one_over_notUniformQeMatrix = None
        if obj_valid(self._poidev):
            del self._poidev
        if obj_valid(self._gaussian_noise):
            del self._gaussian_noise
        self._integrated_i.cleanup()
        self._pixels.cleanup()

    def revision_track(self):
        return '$Rev$'
