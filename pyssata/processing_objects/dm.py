import numpy as np

class DM(BaseProcessingObj):
    def __init__(self, pixel_pitch, height, influence_function, GPU=False, objname="dm", objdescr="Deformable Mirror object", precision=0, dtype=None):
        super().__init__(objname, objdescr)

        self._ifunc = influence_function
        self._ifunc.set_precision(precision)

        s = self._ifunc.mask_inf_func.shape
        nmodes_if = self._ifunc.size[0]

        self._if_commands = np.zeros(nmodes_if, dtype=self._ifunc.dtype)
        self._gpu = GPU
        self._layer = Layer(s[0], s[1], pixel_pitch, height, GPU=GPU, precision=precision, dtype=dtype)
        self._layer.A = self._ifunc.mask_inf_func.astype(float)

        self._sign = -1

    def compute_shape(self):
        if self._gpu and isinstance(self._command, BaseGpuValue):
            pass  # GPU computation code not implemented
        else:
            if isinstance(self._command, BaseGpuValue):
                commands = self._command.read()
            else:
                commands = self._command.value

            temp_matrix = np.zeros(self._layer.size, dtype=float if self._precision == 0 else np.float64)

            if np.sum(np.abs(commands)) != 0:
                if len(commands) > len(self._if_commands):
                    raise ValueError(f"Command vector length ({len(commands)}) is greater than the Influence function size ({len(self._if_commands)})")

                self._if_commands[:len(commands)] = self._sign * commands

                if has_gpu():
                    temp_matrix[self._ifunc.idx_inf_func] = vecmat_multiply(vector=self._if_commands, matrix=self._ifunc.gpu_ifunc)
                else:
                    temp_matrix[self._ifunc.idx_inf_func] = vecmat_multiply(vector=self._if_commands, matrix=self._ifunc.ptr_ifunc)

            self._layer.phaseInNm = temp_matrix  # Will automatically load on GPU if the output layer is a LAYER_GPU

    def trigger(self, t):
        if self._verbose:
            print('time:', self.t_to_seconds(t))
            print('command generation time:', self.t_to_seconds(self._command.generation_time))
            commands = self._command.read() if isinstance(self._command, BaseGpuValue) else self._command.value
            if len(commands) > 0:
                print('first', min(6, len(commands)), 'command values:', commands[:min(5, len(commands))])

        if self._command.generation_time == t:
            if self._verbose:
                print('---> command applied to DM')
            self.compute_shape()
            self._layer.generation_time = t
        elif self._verbose:
            print('command not applied to DM, command generation time:', self._command.generation_time, 'is not equal to', t)

    def set_property(self, in_command=None, sign=None, shiftXYinPixel=None, rotInDeg=None, magnification=None, aShiftXYinPixel=None, aRotInDeg=None, aMagnification=None, ifunc=None, **kwargs):
        if kwargs:
            super().set_property(**kwargs)

        if in_command is not None:
            self._command = in_command
        if sign is not None:
            self._sign = sign
        if shiftXYinPixel is not None:
            self._layer.shiftXYinPixel = shiftXYinPixel
        if rotInDeg is not None:
            self._layer.rotInDeg = rotInDeg
        if magnification is not None:
            self._layer.magnification = magnification
        if aShiftXYinPixel is not None:
            self._layer.shiftXYinPixel += aShiftXYinPixel
            print('self._layer.shiftXYinPixel', self._layer.shiftXYinPixel)
        if aRotInDeg is not None:
            self._layer.rotInDeg += aRotInDeg
            print('self._layer.rotInDeg', self._layer.rotInDeg)
        if aMagnification is not None:
            self._layer.magnification += aMagnification
            print('self._layer.magnification', self._layer.magnification)
        if ifunc is not None:
            self._ifunc.influence_function = ifunc

    def get_property(self, out_layer=None, ifunc=None, idx_inf_func=None, mask_inf_func=None, in_command=None, size=None, sign=None, shiftXYinPixel=None, rotInDeg=None, magnification=None, obj_ifunc=None, **kwargs):
        if kwargs:
            super().get_property(**kwargs)

        if out_layer is not None:
            out_layer = self._layer
        if ifunc is not None:
            ifunc = self._ifunc.influence_function
        if idx_inf_func is not None:
            idx_inf_func = self._ifunc.idx_inf_func
        if mask_inf_func is not None:
            mask_inf_func = self._ifunc.mask_inf_func
        if in_command is not None:
            in_command = self._command
        if size is not None:
            size = self._layer.size
        if sign is not None:
            sign = self._sign
        if shiftXYinPixel is not None:
            shiftXYinPixel = self._layer.shiftXYinPixel
        if rotInDeg is not None:
            rotInDeg = self._layer.rotInDeg
        if magnification is not None:
            magnification = self._layer.magnification
        if obj_ifunc is not None:
            obj_ifunc = self._ifunc

    def revision_track(self):
        return '$Rev$'

    def run_check(self, time_step, errmsg=''):
        if not obj_valid(self._command):
            errmsg += self.repr() + ' No input command defined'

        if self._gpu:
            if not isinstance(self._command, BaseGpuValue):
                errmsg += 'WARNING: ' + self.repr() + ' Input command is not a BASE_GPU_VALUE object'

        return obj_valid(self._command) and obj_valid(self._layer) and obj_valid(self._ifunc)

    def cleanup(self):
        if ptr_valid(self._if_commands):
            self._if_commands = None
        self._ifunc.cleanup()
        self._layer.cleanup()
        self._command.cleanup()
        super().cleanup()
        if self._verbose:
            print('DM has been cleaned up.')
