import numpy as np

class ModalAnalysis:
    def __init__(self, phase2modes):
        self._phase2modes = phase2modes
        self._modes = BaseValue('modes', 'output modes from modal reconstructor')
        self._rms = BaseValue('modes', 'output RMS of modes from modal reconstructor')
        self._ef = None
        self._dorms = False
        self._wavelengthInNm = 0.0
        self._verbose = False  # Verbose flag for debugging output

    def set_property(self, phase2modes=None, in_ef=None, dorms=None, wavelengthInNm=None, **kwargs):
        if phase2modes is not None:
            self._phase2modes = phase2modes
        if in_ef is not None:
            self._ef = in_ef
        if dorms is not None:
            self._dorms = dorms
        if wavelengthInNm is not None:
            self._wavelengthInNm = wavelengthInNm

    def get_property(self, phase2modes=None, in_ef=None, out_modes=None, rms=None, **kwargs):
        properties = {
            'in_ef': self._ef,
            'phase2modes': self._phase2modes,
            'out_modes': self._modes,
            'rms': self._rms,
        }
        return {k: properties[k] for k in kwargs.keys() if k in properties}

    def unwrap_phase_2d(self, p):
        unwrapped_p = np.copy(p)
        for r in range(p.shape[1]):
            row = unwrapped_p[:, r]
            unwrapped_p[:, r] = self.unwrap_phase(row)
        
        for c in range(p.shape[0]):
            col = unwrapped_p[c, :]
            unwrapped_p[c, :] = self.unwrap_phase(col)
        
        return unwrapped_p

    def unwrap_phase(self, phase):
        # Placeholder function: implement phase unwrapping as needed
        return np.unwrap(phase)

    def trigger(self, t):
        if self._ef.generation_time == t:
            if self._phase2modes.zero_pad:
                m = self.vector_matrix_multiply(
                    self._ef.gpu_phaseInNm if HAS_GPU() else self._ef.phaseInNm,
                    self._phase2modes.gpu_ifunc if HAS_GPU() else self._phase2modes.ptr_ifunc
                )
            else:
                if self._wavelengthInNm > 0:
                    phase_in_rad = self._ef.phaseInNm * (2 * np.pi / self._wavelengthInNm)
                    phase_in_rad *= self._phase2modes.mask_inf_func.astype(float)
                    phase_in_rad = self.unwrap_phase_2d(phase_in_rad)
                    phase_in_nm = phase_in_rad * (self._wavelengthInNm / (2 * np.pi))
                    ph = phase_in_nm[self._phase2modes.idx_inf_func]
                else:
                    ph = self._ef.phaseInNm[self._phase2modes.idx_inf_func]

                m = self.vector_matrix_multiply(
                    ph, 
                    self._phase2modes.gpu_ifunc if HAS_GPU() else self._phase2modes.ptr_ifunc
                )

            self._modes.value = m
            self._modes.generation_time = t

            if self._dorms:
                self._rms.value = np.std(ph)
                self._rms.generation_time = t

            if self._verbose:
                print(f"First residual values: {m[:min(6, len(m))]}")
                if self._dorms:
                    print(f"Phase RMS: {self._rms.value}")

    def revision_track(self):
        return "$Rev$"

    def run_check(self, time_step):
        errmsg = ""
        if not obj_valid(self._ef):
            errmsg += "EF is not valid. "
        if not obj_valid(self._phase2modes):
            errmsg += "phase2modes is not valid. "
        return obj_valid(self._ef) and obj_valid(self._phase2modes), errmsg

    def cleanup(self):
        self._ef.cleanup()
        self._modes.cleanup()
        self._rms.cleanup()
        self._phase2modes.cleanup()

    def __repr__(self):
        return f"{self._objdescr} ({self._objname})"

    def vector_matrix_multiply(self, vector, matrix):
        # Placeholder function: implement GPU or CPU matrix-vector multiplication as needed
        return np.dot(matrix, vector)
