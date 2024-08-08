import numpy as np
from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.base_value import BaseValue
from pyssata.data_objects.recmat import Recmat
from pyssata.data_objects.slopes import Slopes
from pyssata.processing_objects.cheat import Cheat


class ModalRec(BaseProcessingObj):
    '''Modal reconstructor'''

    def __init__(self, recmat=None, projmat=None, intmat=None, polc=None):
        self._recmat = recmat if recmat is not None else Recmat()
        self._projmat = projmat
        self._intmat = intmat
        self._polc = polc

        self.set_layer_modes_list()
        self._control_list = []
        self._past_step_list = []

        self._modes = BaseValue('output modes from modal reconstructor')
        self._pseudo_ol_modes = BaseValue('output POL modes from modal reconstructor')
        self._modes_first_step = BaseValue('output (no projection) modes from modal reconstructor')

        super().__init__()

    def set_layer_modes_list(self):
        if self._recmat.modes2rec_layer is not None:
            self._layer_modes_list = []
            self._layer_idx_list = []
            n = self._recmat.modes2rec_layer.shape[0]
            for i in range(n):
                self._layer_modes_list.append(BaseValue(f'output modes for layer no {i + 1}'))
                self._layer_idx_list.append(np.where(self._recmat.modes2rec_layer[i, :] > 0)[0])

    @property
    def recmat(self):
        return self._recmat

    @recmat.setter
    def recmat(self, value):
        self._recmat = value

    @property
    def in_slopes(self):
        return self._slopes

    @in_slopes.setter
    def in_slopes(self, value):
        self._slopes = value
        if isinstance(value, Slopes):
            size = value.size
        elif isinstance(value, (BaseValue, Cheat)):
            size = len(value.value)
        if self._polc:
            self._pseudo_ol_slopes = Slopes(size)

    @property
    def out_modes(self):
        return self._modes

    @property
    def projmat(self):
        return self._projmat

    @projmat.setter
    def projmat(self, value):
        self._projmat = value

    @property
    def modes2rec_layer(self):
        return self._recmat.modes2rec_layer

    @property
    def out_layer_modes_list(self):
        return self._layer_modes_list

    @property
    def modes_first_step(self):
        return self._modes_first_step

    @modes_first_step.setter
    def modes_first_step(self, value):
        self._modes_first_step = value

    @property
    def pseudo_ol_slopes(self):
        return self._pseudo_ol_slopes

    @property
    def pseudo_ol_modes(self):
        return self._pseudo_ol_modes

    @property
    def intmat(self):
        return self._intmat

    @intmat.setter
    def intmat(self, value):
        self._intmat = value

    @property
    def dm_idx(self):
        return self._dm_idx

    @dm_idx.setter
    def dm_idx(self, value):
        self._dm_idx = value

    def add_control(self, control):
        if not self._polc:
            raise Exception("Control can be added only if POLC is set to 1.")
        self._control_list.append(control)

    def trigger(self, t, slope_ptr=None):
        if not self._recmat.ptr_recmat:
            print("WARNING: modalrec skipping reconstruction because recmat is NULL")
            return

        if self._slopes.generation_time == t:
            comm_new = []
            if len(self._control_list) > 0:
                for control in self._control_list:
                    comm_new.append(control.get_past_state(0))

            if self._modes_first_step.generation_time != t:
                if self._polc:
                    self.compute_pseudo_ol_slopes(t)
                    m = self.compute_modes(self._recmat, self._pseudo_ol_slopes.ptr_slopes)
                    self._pseudo_ol_modes.value = m
                    self._pseudo_ol_modes.generation_time = t

                    if len(m) == len(comm_new):
                        self._modes_first_step.value = m - comm_new
                    else:
                        self._modes_first_step.value = m
                else:
                    m = self.compute_modes(self._recmat, slope_ptr)
                    self._modes_first_step.value = m

                self._modes_first_step.generation_time = t

                if self._layer_modes_list is not None:
                    for i, idx_list in enumerate(self._layer_idx_list):
                        self._layer_modes_list[i].value = self._modes_first_step.value[idx_list]
                        self._layer_modes_list[i].generation_time = t

            if self._projmat is None:
                if self._verbose:
                    n = len(self._modes_first_step.value)
                    print(f"(no projmat) first {min(6, n)} residual values: {self._modes_first_step.value[:min(5, n)]}")
                self._modes.value = self._modes_first_step.value
                self._modes.generation_time = self._modes_first_step.generation_time
            else:
                mp = self.compute_modes(self._projmat, self._modes_first_step.ptr_value)
                if self._verbose:
                    print(f"first {min(6, len(mp))} residual values after projection: {mp[:min(5, len(mp))]}")
                self._modes.value = mp
                self._modes.generation_time = t

            if self._polc and len(self._modes_first_step.value) != len(comm_new):
                self._modes.value -= self._control_list[self._dm_idx].get_past_state(0)
        else:
            if self._verbose:
                print(f"slope generation time: {self._slopes.generation_time} is not equal to {t}")

    def compute_pseudo_ol_slopes(self, t, slopes=None):
        if slopes is None:
            slopes = self._slopes

        if isinstance(slopes, Slopes):
            self._pseudo_ol_slopes.slopes = slopes.slopes
        elif isinstance(slopes, (BaseValue, Cheat)):
            self._pseudo_ol_slopes.slopes = slopes.value

        comm = []
        for control in self._control_list:
            comm.append(control.get_past_state(0))

        if not self._intmat:
            raise Exception("POLC requires intmat, but it is not set")

        comm_slopes = self.compute_modes(self._intmat, np.array(comm), intmat=True)
        self._pseudo_ol_slopes.slopes += comm_slopes
        self._pseudo_ol_slopes.generation_time = t

    def compute_modes(self, matrix, slope_ptr, intmat=False):
        if slope_ptr is None:
            if isinstance(self._slopes, Slopes):
                slope_ptr = self._slopes.ptr_slopes
                if self._verbose:
                    print(f"modalrec.compute_modes slope RMS: {np.sqrt(np.mean(slope_ptr**2))}")
            elif isinstance(self._slopes, BaseValue):
                slope_ptr = self._slopes.ptr_value
                if self._verbose:
                    print(f"modalrec.compute_modes base_value RMS: {np.sqrt(np.mean(slope_ptr**2))}")
            elif isinstance(self._slopes, Cheat):
                slopes = self._slopes.value
                slope_ptr = slopes
                if self._verbose:
                    print(f"modalrec.compute_modes value from cheat RMS: {np.sqrt(np.mean(slope_ptr**2))}")

        if intmat:
            m = VECMAT_MULTIPLY(VECTOR=slope_ptr, MATRIX=matrix.ptr_intmat)
        else:
            m = VECMAT_MULTIPLY(VECTOR=slope_ptr, MATRIX=matrix.ptr_recmat)

        return m

    def run_check(self, time_step):
        errmsg = []
        if not self._slopes:
            errmsg.append("Slopes object not valid")
        if not self._recmat:
            errmsg.append("Recmat object not valid")
        out = bool(self._slopes) and bool(self._recmat)
        if self._polc:
            if not self._intmat:
                errmsg.append("Intmat object not valid")
            if not self._control_list:
                errmsg.append("ControlList object not valid")
            out &= bool(self._intmat) and bool(self._control_list)
        if errmsg:
            print(", ".join(errmsg))
        return out

    def cleanup(self):
        self._modes.cleanup()
        self._slopes.cleanup()
        self._recmat.cleanup()
        self._projmat.cleanup()
        self._layer_modes_list.cleanup()
        self._layer_idx_list.cleanup()
        BaseProcessingObj.cleanup(self)


