import numpy as np

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.data_objects.layer import Layer

class DM(BaseProcessingObj):
    def __init__(self, pixel_pitch, height, influence_function, GPU=False, objname="dm", objdescr="Deformable Mirror object", PRECISION=0, TYPE=None):
        super().__init__(objname, objdescr)
        
        self._ifunc = influence_function
        self._ifunc.set_precision(self._precision)
        
        s = self._ifunc.mask_inf_func.shape
        nmodes_if = self._ifunc.size[0]
        
        self._if_commands = np.zeros(nmodes_if, dtype=self._ifunc.type)
        self._gpu = GPU
        self._layer = Layer(s[0], s[1], pixel_pitch, height, GPU=GPU, PRECISION=self._precision, TYPE=TYPE)
        self._layer.A = float(self._ifunc.mask_inf_func)
        
        # sign is -1 to take into account the reflection in the propagation
        self._sign = -1
    
    def compute_shape(self):
        if obj_isa(self._command, 'BASE_GPU_VALUE'):
            commands = self._command.read()
        else:
            commands = self._command.value
        
        temp_matrix = np.zeros(self._layer.size, dtype=np.float64 if self._precision else np.float32)
        
        # Compute phase only if commands vector is not zero
        if np.sum(np.abs(commands)) != 0:
            if len(commands) > len(self._if_commands):
                raise ValueError(f"Error: command vector length ({len(commands)}) is greater than the Influence function size ({len(self._if_commands)})")
            
            self._if_commands[:len(commands)] = self._sign * commands
            
            if self.has_gpu():
                temp_matrix.flat[self._ifunc.idx_inf_func] = vecmat_multiply(vector=self._if_commands, matrix=self._ifunc.gpu_ifunc)
            else:
                temp_matrix.flat[self._ifunc.idx_inf_func] = vecmat_multiply(vector=self._if_commands, matrix=self._ifunc.ptr_ifunc)
        
        self._layer.phase_in_nm = temp_matrix
    
    def trigger(self, t):
        if self._verbose:
            print(f"time: {self.t_to_seconds(t)}")
            print(f"command generation time: {self.t_to_seconds(self._command.generation_time)}")
            if obj_isa(self._command, 'BASE_GPU_VALUE'):
                commands = self._command.read()
            else:
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
        if not obj_valid(self._command):
            errmsg += f"{self.repr()} No input command defined"
        
        if self._gpu and not obj_isa(self._command, 'BASE_GPU_VALUE'):
            errmsg += f"WARNING: {self.repr()} Input command is not a BASE_GPU_VALUE object"
        
        return obj_valid(self._command) and obj_valid(self._layer) and obj_valid(self._ifunc)
    
    @staticmethod
    def revision_track():
        return "$Rev$"
