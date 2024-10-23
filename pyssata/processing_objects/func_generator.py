
import numpy as np

from pyssata.base_value import BaseValue
from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.lib.modal_pushpull_signal import modal_pushpull_signal

class FuncGenerator(BaseProcessingObj):
    def __init__(self, func_type='SIN', nmodes=None, time_hist=None, psd=None, fr_psd=None, continuous_psd=None, 
                constant=None, amp=None, freq=None, offset=None, vect_amplitude=None, 
                seed=None, ncycles=1,
                target_device_idx=None, 
                precision=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.type = func_type.upper()
        if self.type == 'PUSHPULLREPEAT':
            self.repeat_ncycles = True
            self.type = 'PUSHPULL'
        else:
            self.repeat_ncycles = False
        self.active = True

        self.amp = self.xp.array(0)
        self.freq = self.xp.array(0)
        self.offset = self.xp.array(0)
        self.constant = self.xp.array(0)
        self.output = BaseValue(target_device_idx=target_device_idx, value=self.xp.array(0))

        if seed is not None:
            if str(seed).strip() == 'auto':
                seed = self.xp.around(self.xp.random.random() * 1e4)
            self.seed = seed

        # Initialize attributes based on the type
        if self.type == 'SIN':
            self.constant = self.xp.array(constant) if constant is not None else 0.0
            self.amp = self.xp.array(amp) if amp is not None else 0.0
            self.freq = self.xp.array(freq) if freq is not None else 0.0
            self.offset = self.xp.array(offset) if offset is not None else 0.0

        elif self.type == 'LINEAR':
            self.constant = self.xp.array(constant) if constant is not None else 0.0
            self.slope = 0.0

        elif self.type == 'RANDOM':
            self.amp = self.xp.array(amp) if amp is not None else 0.0
            self.constant = self.xp.array(constant) if constant is not None else 0.0
            self.seed = self.xp.array(seed) if seed is not None else 0.0

        elif self.type == 'VIB_HIST':
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type VIB_HIST')
            if time_hist is None:
                raise ValueError('TIME_HIST keyword is mandatory for type VIB_HIST')
            self.vib = Vibrations(nmodes, time_hist=time_hist)

        elif self.type == 'VIB_PSD':
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type VIB_PSD')
            if psd is None and continuous_psd is None:
                raise ValueError('PSD or CONTINUOUS_PSD keyword is mandatory for type VIB_PSD')
            if fr_psd is None:
                raise ValueError('FR_PSD keyword is mandatory for type VIB_PSD')
            self.vib = Vibrations(nmodes, psd=psd, freq=fr_psd, continuous_psd=continuous_psd, seed=seed)

        elif self.type == 'PUSH':
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type PUSH')
            if amp is None and vect_amplitude is None:
                raise ValueError('AMP or VECT_AMPLITUDE keyword is mandatory for type PUSH')
            self.time_hist = modal_pushpull_signal(nmodes, amplitude=amp, vect_amplitude=vect_amplitude, only_push=True, ncycles=ncycles)
            self.amp = self.xp.array(amp)
            self.vect_amplitude = self.xp.array(ect_amplitude)

        elif self.type == 'PUSHPULL':
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type PUSHPULL')
            if amp is None and vect_amplitude is None:
                raise ValueError('AMP or VECT_AMPLITUDE keyword is mandatory for type PUSHPULL')
            self.time_hist = modal_pushpull_signal(nmodes, amplitude=amp, vect_amplitude=vect_amplitude, ncycles=ncycles, repeat_ncycles=self.repeat_ncycles)
            self.amp = self.xp.array(amp) if amp is not None else 0.0
            self.vect_amplitude = self.xp.array(vect_amplitude) if vect_amplitude is not None else 0.0

        elif self.type == 'TIME_HIST':
            if time_hist is None:
                raise ValueError('TIME_HIST keyword is mandatory for type TIME_HIST')
            self.time_hist = self.xp.array(time_hist)

        else:
            raise ValueError(f'Unknown function type: {self.type}')

        self.nmodes = nmodes
        self.outputs['output'] = self.output

#       uncomment when the code is a stream
#        super().build_stream()


    def trigger_code(self):
        if self.type == 'SIN':
            if not self.active:
                s = 0.0
            else:
                s = self.xp.array(self.amp, dtype=self.dtype)*self.xp.sin(self.freq*2*self.xp.pi*self.current_time_seconds + self.offset)+self.xp.array(self.constant, dtype=self.dtype)

        elif self.type == 'LINEAR':
            if not self.active:
                s = 0.0
            else:
                s = self.slope * self.current_time_seconds + self.constant

        elif self.type == 'RANDOM':
            if not self.active:
                s = 0.0
            else:
                s = self.xp.random.normal(size=len(self.amp)) * self.amp + self.constant

        elif self.type in ['VIB_HIST', 'VIB_PSD', 'PUSH', 'PUSHPULL', 'TIME_HIST']:
            s = self.get_time_hist_at_current_time()

        else:
            raise ValueError(f'Unknown function generator type: {self.type}')

        self.output.value = s
        self.output.generation_time = self.current_time

    def get_time_hist_at_current_time(self):
        t = self.current_time
        if not self.active:
            return self.xp.zeros_like(self.time_hist[0])
        i = int(np.round(t / self.loop_dt))
        return self.xp.array(self.time_hist[i])

    def run_check(self, time_step, errmsg=""):
        if hasattr(self, '_vib') and self.vib:
            self.vib.set_niters(self.loop_niters + 1)
            self.vib.set_samp_freq(1.0 / self.t_to_self.current_time_seconds(self.loop_dt))
            self.vib.compute()
            self.time_hist = self.vib.get_time_hist()

        return True

