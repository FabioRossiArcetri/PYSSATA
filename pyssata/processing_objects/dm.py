import numpy as np
from pyssata.base_value import BaseValue
from pyssata.connections import InputValue

from pyssata.data_objects.ifunc import IFunc
from pyssata.data_objects.layer import Layer
from pyssata.data_objects.pupilstop import Pupilstop
from pyssata.base_processing_obj import BaseProcessingObj

class DM(BaseProcessingObj):
    def __init__(self,
                 pixel_pitch: float,
                 height: float,
                 ifunc: IFunc=None,
                 type_str: str=None,
                 nmodes: int=None,
                 nzern: int=None,
                 start_mode: int=None,
                 idx_modes = None,
                 npixels: int=None,
                 obsratio: float=None,
                 diaratio: float=None,
                 pupilstop: Pupilstop=None,
                 target_device_idx=None, 
                 precision=None
                 ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        mask = None
        if pupilstop:
            mask = pupilstop.A
            if npixels is None:
                npixels = mask.shape[0]

        if not ifunc:
            ifunc = IFunc(type_str=type_str, mask=mask, npixels=npixels,
                           obsratio=obsratio, diaratio=diaratio, nzern=nzern,
                           nmodes=nmodes, start_mode=start_mode, idx_modes=idx_modes,
                           target_device_idx=target_device_idx, precision=precision)
        self._ifunc = ifunc
        
        s = self._ifunc.mask_inf_func.shape
        nmodes_if = self._ifunc.size[0]
        
        self._if_commands = self.xp.zeros(nmodes_if, dtype=self._ifunc.dtype)
        self._layer = Layer(s[0], s[1], pixel_pitch, height, target_device_idx=target_device_idx, precision=precision)
        self._layer.A = self._ifunc.mask_inf_func
        
        # sign is -1 to take into account the reflection in the propagation
        self._sign = -1
        self.inputs['in_command'] = InputValue(type=BaseValue)
        self.outputs['out_layer'] = self._layer
#       uncomment when the code is a stream
#        super().build_stream()
    
    def compute_shape(self):
        commands_input = self.inputs['in_command'].get(self._target_device_idx)
        commands = commands_input.value

        temp_matrix = self.xp.zeros(self._layer.size, dtype=self.dtype)
        
        # Compute phase only if commands vector is not zero
        #if self.xp.sum(self.xp.abs(commands)) != 0:
        #    if len(commands) > len(self._if_commands):
        #        raise ValueError(f"Error: command vector length ({len(commands)}) is greater than the Influence function size ({len(self._if_commands)})")
        
        self._if_commands[:len(commands)] = self._sign * commands
        
        temp_matrix[self._ifunc.idx_inf_func] = self.xp.dot(self._if_commands, self._ifunc.ptr_ifunc)

        self._layer.phaseInNm = temp_matrix

    def trigger_code(self):
        command = self.inputs['in_command'].get(self._target_device_idx)
        if self._verbose:
            print(f"time: {self.current_time_seconds}")
            print(f"command generation time: {self.t_to_seconds(command.generation_time)}")
            commands = command.value
            
            if commands.size > 0:
                print(f"first {min(6, commands.size)} command values: {commands[:min(5, commands.size)]}")
        
        if command.generation_time == self.current_time:
            if self._verbose:
                print("---> command applied to DM")
            self.compute_shape()
            self._layer.generation_time = self.current_time
        elif self._verbose:
            print(f"command not applied to DM, command generation time: {command.generation_time} is not equal to {t}")
    
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

    def run_check(self, time_step, errmsg=""):
        commands_input = self.inputs['in_command'].get(self._target_device_idx)
        if commands_input is None:
            errmsg += f"{self.repr()} No input command defined"
        
        return commands_input is not None and self._layer is not None and self._ifunc is not None
