
from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.base_value import BaseValue
from pyssata.data_objects.ef import ElectricField
from pyssata.data_objects.intensity import Intensity
from pyssata.connections import InputValue

import numpy as np

class PSF(BaseProcessingObj):
    def __init__(self,
                 wavelengthInNm: float,
                 nd: int=1,
                 start_time: float=0.0,
                 target_device_idx: int = None, 
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)        
        self._nd = nd
        self._start_time = start_time
        self._wavelengthInNm = wavelengthInNm

        self._sr = BaseValue()
        self._int_sr = BaseValue()
        self._psf = BaseValue()
        self._int_psf = BaseValue()
        self._in_ef = None
        self._ref = None
        self._intsr = 0.0
        self._count = 0

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_sr'] = self._sr
        self.outputs['out_psf'] = self._psf
        self.reset_integration()

    def calc_psf(self, phase, amp, imwidth=None, normalize=False, nocenter=False):
        """
        Calculates a PSF from an electrical field phase and amplitude.

        Parameters:
        phase : ndarray
            2D phase array.
        amp : ndarray
            2D amplitude array (same dimensions as phase).
        imwidth : int, optional
            Width of the output image. If provided, the output will be of shape (imwidth, imwidth).
        normalize : bool, optional
            If set, the PSF is normalized to total(psf).
        nocenter : bool, optional
            If set, avoids centering the PSF and leaves the maximum pixel at [0,0].

        Returns:
        psf : ndarray
            2D PSF (same dimensions as phase).
        """

        # Set up the complex array based on input dimensions and data type
        if imwidth is not None:
            u_ef = self.xp.zeros((imwidth, imwidth), dtype=self.complex_dtype)
            result = amp * self.xp.exp(1j * phase)
            s = result.shape
            u_ef[:s[0], :s[1]] = result
        else:
            u_ef = amp * self.xp.exp(1j * phase)
        # Compute FFT (forward)
        u_fp = self.xp.fft.fft2(u_ef)
        # Center the PSF if required
        if not nocenter:
            u_fp = self.xp.fft.fftshift(u_fp)
        # Compute the PSF as the square modulus of the Fourier transform
        psf = self.xp.abs(u_fp)**2
        # Normalize if required
        if normalize:
            psf /= self.xp.sum(psf)

        return psf


    @property
    def wavelengthInNm(self):
        return self._wavelengthInNm

    @wavelengthInNm.setter
    def wavelengthInNm(self, wavelengthInNm):
        self._wavelengthInNm = wavelengthInNm

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, start_time):
        self._start_time = start_time

    @property
    def out_psf(self):
        return self._psf

    @property
    def out_int_psf(self):
        return self._int_psf

    @property
    def out_sr(self):
        return self._sr

    @property
    def out_int_sr(self):
        return self._int_sr

    @property
    def nd(self):
        return self._nd

    @nd.setter
    def nd(self, nd):
        self._nd = nd

    @property
    def size(self):
        in_ef = self.inputs['in_ef'].get(self._target_device_idx)
        return in_ef.size if in_ef else None

    @property
    def out_ref(self):
        return self._ref

    @property
    def out_count(self):
        return self._count

    def run_check(self, time_step, errmsg=''):
        in_ef = self.inputs['in_ef'].get(self._target_device_idx)
        if not in_ef:
            errmsg += ' Input intensity object has not been set'
        if self._wavelengthInNm == 0:
            errmsg += ' PSF wavelength is zero'
        return bool(in_ef) and (self._wavelengthInNm > 0)

    def reset_integration(self):
        self._count = 0
        in_ef = self.inputs['in_ef'].get(self._target_device_idx)
        if in_ef:
            self._int_psf.value *= 0
        self._intsr = 0

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.in_ef = self.local_inputs['in_ef']
        if self._psf.value is None:
            s = [dim * self._nd for dim in self.in_ef.size]
            self._psf.value = self.xp.zeros(s, dtype=self.dtype)
            self._int_psf.value = self.xp.zeros(s, dtype=self.dtype)
            self._ref = None
        if self.current_time_seconds >= self._start_time:
            self._count += 1

    def trigger_code(self):
        self.out_size = [np.around(dim * self._nd) for dim in self.in_ef.size]
        if not self._ref:
            self._ref = Intensity(self.out_size[0], self.out_size[1])
            self._ref.i = self.calc_psf(self.in_ef.A * 0.0, self.in_ef.A, imwidth=self.out_size[0], normalize=True)
        self._psf.value = self.calc_psf(self.in_ef.phi_at_lambda(self._wavelengthInNm), self.in_ef.A, imwidth=self.out_size[0], normalize=True)
        self._sr.value = self._psf.value[self.out_size[0] // 2, self.out_size[1] // 2] / self._ref.i[self.out_size[0] // 2, self.out_size[1] // 2]

    def post_trigger(self):
        super().post_trigger()
        if self.current_time_seconds >= self._start_time:
            self._intsr += self._sr.value
            self._int_sr.value = self._intsr / self._count
            self._int_psf.generation_time = self.current_time
            self._int_sr.generation_time = self.current_time
            self._int_psf.value += self._psf.value
        self._psf.generation_time = self.current_time
        self._sr.generation_time = self.current_time
