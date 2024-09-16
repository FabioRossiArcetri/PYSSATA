import numpy as np
from pyssata import gpuEnabled
from pyssata import xp

from pyssata.data_objects.ifunc import IFunc
from pyssata.data_objects.layer import Layer
from pyssata.data_objects.pupilstop import Pupilstop
from pyssata.base_processing_obj import BaseProcessingObj


class DM(BaseProcessingObj):
    def __init__(self,
                 pixel_pitch: float,
                 height: float,
                 ifunc: IFunc=None,
                 type: str=None,
                 nmodes: int=None,
                 nzern: int=None,
                 start_mode: int=None,
                 idx_modes: xp.ndarray=None,
                 npixels: int=None,
                 obsratio: float=None,
                 diaratio: float=None,
                 pupilstop: Pupilstop=None,
                 ):
        super().__init__()
        
        mask = None
        if pupilstop:
            mask = pupilstop.A
            if npixels is None:
                npixels = mask.shape[0]

        if not ifunc:
            ifunc = IFunc(type=type, mask=mask, npixels=npixels,
                           obsratio=obsratio, diaratio=diaratio, nzern=nzern,
                           nmodes=nmodes, start_mode=start_mode, idx_modes=idx_modes)
        self._ifunc = ifunc
        
        s = self._ifunc.mask_inf_func.shape
        nmodes_if = self._ifunc.size[0]
        
        self._if_commands = xp.zeros(nmodes_if, dtype=self._ifunc.type)
        self._layer = Layer(s[0], s[1], pixel_pitch, height)
        self._layer.A = self._ifunc.mask_inf_func
        
        # sign is -1 to take into account the reflection in the propagation
        self._sign = -1

        # Integrator control workaround
        self._history = None
        self._delay = 2
        self._gain = 0.5
    
    def compute_shape(self):
        commands = self._command.value

        if self._history is None:
            self._history = [commands*0] * (self._delay+1)
            self._integrated_commands = commands * 0

        self._history.append(commands)
        self._integrated_commands += self._history[-(self._delay+1)]

        commands = self._integrated_commands * self._gain


        temp_matrix = xp.zeros(self._layer.size, dtype=xp.float64 if self._precision else xp.float32)
        
        # Compute phase only if commands vector is not zero
        if xp.sum(xp.abs(commands)) != 0:
            if len(commands) > len(self._if_commands):
                raise ValueError(f"Error: command vector length ({len(commands)}) is greater than the Influence function size ({len(self._if_commands)})")
            
            self._if_commands[:len(commands)] = self._sign * commands
            
            temp_matrix[self._ifunc.idx_inf_func] = xp.dot(self._if_commands, self._ifunc.ptr_ifunc)

        self._layer.phaseInNm = temp_matrix

    def trigger(self, t):
        if self._verbose:
            print(f"time: {self.t_to_seconds(t)}")
            print(f"command generation time: {self.t_to_seconds(self._command.generation_time)}")
            commands = self._command.value
            
            if commands.size > 0:
                print(f"first {min(6, commands.size)} command values: {commands[:min(5, commands.size)]}")
        
        if self._command.generation_time == t:
            if self._verbose:
                print("---> command applied to DM")
            self.compute_shape()
            self._layer.generation_time = t
        elif self._verbose:
            print(f"command not applied to DM, command generation time: {self._command.generation_time} is not equal to {t}")
    
    # Getters and Setters for the attributes
    @property
    def sign(self):
        return self._sign

    @sign.setter
    def sign(self, value):
        self._sign = value

    @property
    def ifunc(self):
        return self._ifunc.influence_function

    @ifunc.setter
    def ifunc(self, value):
        self._ifunc.influence_function = value

    @property
    def out_layer(self):
        return self._layer

    @property
    def in_command(self):
        return self._command

    @in_command.setter
    def in_command(self, value):
        self._command = value

    @property
    def size(self):
        return self._layer.size

    @property
    def shift_xy_in_pixel(self):
        return self._layer.shift_xy_in_pixel

    @shift_xy_in_pixel.setter
    def shift_xy_in_pixel(self, value):
        self._layer.shift_xy_in_pixel = value

    @property
    def rot_in_deg(self):
        return self._layer.rot_in_deg

    @rot_in_deg.setter
    def rot_in_deg(self, value):
        self._layer.rot_in_deg = value

    @property
    def magnification(self):
        return self._layer.magnification

    @magnification.setter
    def magnification(self, value):
        self._layer.magnification = value

    def cleanup(self):
        if self._if_commands is not None:
            del self._if_commands
        self._ifunc.cleanup()
        self._layer.cleanup()
        self._command.cleanup()
        super().cleanup()
        if self._verbose:
            print("DM has been cleaned up.")

    def run_check(self, time_step, errmsg=""):
        if self._command is None:
            errmsg += f"{self.repr()} No input command defined"
        
        return self._command is not None and self._layer is not None and self._ifunc is not None
    
    @staticmethod
    def revision_track():
        return "$Rev$"
