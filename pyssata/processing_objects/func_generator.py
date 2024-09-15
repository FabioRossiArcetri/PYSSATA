import numpy as np

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.base_value import BaseValue


class FuncGenerator(BaseProcessingObj):
    def __init__(self, func_type='SIN', nmodes=None, time_hist=None, psd=None, fr_psd=None, continuous_psd=None, 
                 constant=None, amp=None, freq=None, offset=None, vect_amplitude=None, 
                 seed=None, ncycles=None):
        super().__init__()

        self._type = func_type.upper()

        if self._type == 'PUSHPULLREPEAT':
            self._repeat_ncycles = True
            self._type = 'PUSHPULL'
        else:
            self._repeat_ncycles = False

        if seed is not None:
            if str(seed).strip() == 'auto':
                seed = round(np.random.random() * 1e4)
            self._seed = seed

        # Initialize attributes based on the type
        if self._type == 'SIN':
            self._constant = constant or 0.0
            self._amp = amp or 0.0
            self._freq = freq or 0.0
            self._offset = offset or 0.0

        elif self._type == 'LINEAR':
            self._constant = constant or 0.0
            self._slope = 0.0

        elif self._type == 'RANDOM':
            self._amp = amp or 0.0
            self._constant = constant or 0.0
            self._seed = seed or 0

        elif self._type == 'VIB_HIST':
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type VIB_HIST')
            if time_hist is None:
                raise ValueError('TIME_HIST keyword is mandatory for type VIB_HIST')
            self._vib = Vibrations(nmodes, time_hist=time_hist)

        elif self._type == 'VIB_PSD':
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type VIB_PSD')
            if psd is None and continuous_psd is None:
                raise ValueError('PSD or CONTINUOUS_PSD keyword is mandatory for type VIB_PSD')
            if fr_psd is None:
                raise ValueError('FR_PSD keyword is mandatory for type VIB_PSD')
            self._vib = Vibrations(nmodes, psd=psd, freq=fr_psd, continuous_psd=continuous_psd, seed=seed)

        elif self._type == 'PUSH':
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type PUSH')
            if amp is None and vect_amplitude is None:
                raise ValueError('AMP or VECT_AMPLITUDE keyword is mandatory for type PUSH')
            self._time_hist = modal_pushpull_signal(nmodes, amplitude=amp, vect_amplitude=vect_amplitude, only_push=True, ncycles=ncycles)
            self._amp = amp
            self._vect_amplitude = vect_amplitude

        elif self._type == 'PUSHPULL':
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type PUSHPULL')
            if amp is None and vect_amplitude is None:
                raise ValueError('AMP or VECT_AMPLITUDE keyword is mandatory for type PUSHPULL')
            self._time_hist = modal_pushpull_signal(nmodes, amplitude=amp, vect_amplitude=vect_amplitude, ncycles=ncycles, repeat_ncycles=self._repeat_ncycles)
            self._amp = amp
            self._vect_amplitude = vect_amplitude

        elif self._type == 'TIME_HIST':
            if time_hist is None:
                raise ValueError('TIME_HIST keyword is mandatory for type TIME_HIST')
            self._time_hist = time_hist

        else:
            raise ValueError(f'Unknown function type: {self._type}')

        self._nmodes = nmodes
        self._output = BaseValue()
        self._active = True

    def trigger(self, t):
        seconds = self.t_to_seconds(t)

        if self._type == 'SIN':
            if not self._active:
                s = 0.0
            else:
                s = self._constant + self._amp * np.sin(self._freq * 2 * np.pi * seconds + self._offset)

        elif self._type == 'LINEAR':
            if not self._active:
                s = 0.0
            else:
                s = self._constant + self._slope * seconds

        elif self._type == 'RANDOM':
            if not self._active:
                s = 0.0
            else:
                s = self._constant + np.random.normal(size=len(self._amp)) * self._amp

        elif self._type in ['VIB_HIST', 'VIB_PSD', 'PUSH', 'PUSHPULL', 'TIME_HIST']:
            s = self.get_time_hist_at_time(t)

        else:
            raise ValueError(f'Unknown function generator type: {self._type}')

        self._output.value = s
        self._output.generation_time = t

    def get_time_hist_at_time(self, t):
        if not self._active:
            return np.zeros_like(self._time_hist[0])
        i = round(t / self._loop_dt)
        return self._time_hist[i]

    def cleanup(self):
        # Cleaning up resources
        del self._seed
        del self._constant
        del self._amp
        del self._freq
        del self._offset
        del self._vect_amplitude
        del self._time_hist
        del self._output

    # Getters and Setters for the attributes
    @property
    def type(self):
        return self._type

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        self._active = value

    @property
    def nmodes(self):
        return self._nmodes

    @property
    def constant(self):
        return self._constant

    @constant.setter
    def constant(self, value):
        self._constant = value

    @property
    def amp(self):
        return self._amp

    @amp.setter
    def amp(self, value):
        self._amp = value

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, value):
        self._freq = value

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value

    @property
    def vect_amplitude(self):
        return self._vect_amplitude

    @vect_amplitude.setter
    def vect_amplitude(self, value):
        self._vect_amplitude = value

    @property
    def slope(self):
        return self._slope

    @slope.setter
    def slope(self, value):
        self._slope = value

    @property
    def time_hist(self):
        return self._time_hist

    @property
    def output(self):
        return self._output

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

    def run_check(self, time_step, errmsg=""):
        if hasattr(self, '_vib') and self._vib:
            self._vib.set_niters(self._loop_niters + 1)
            self._vib.set_samp_freq(1.0 / self.t_to_seconds(self._loop_dt))
            self._vib.compute()
            self._time_hist = self._vib.get_time_hist()

        return True

