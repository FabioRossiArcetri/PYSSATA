import numpy as np
from numba import jit

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.connections import InputValue
from pyssata.base_value import BaseValue
from pyssata.lib.calc_loop_delay import calc_loop_delay

gxp = None
gdtype = None

def compute_comm(input, _outFinite, _ist, _ost, _iirfilter_num, _iirfilter_den):
    global gxp, gdtype
#        nfilter = _iirfilter.num.shape[0]
#        ninput = input.size
    output = input*0
#        if nfilter < ninput:
#            raise ValueError(f"Error: IIR filter needs no more than {nfilter} coefficients ({ninput} given)")

#        if ninput == 1:
#            if xp.isfinite(input):
#                temp_input = input
#                temp_ist = _ist
#                temp_ost = _ost
#                temp_num = _iirfilter.num
#                temp_den = _iirfilter.den
#            else:
#                return output
#        else:
    gxp.isfinite(input, _outFinite)

    _idx_finite = gxp.where(_outFinite)[0]
    temp_input = input[_idx_finite]
    temp_ist = _ist[_idx_finite]
    temp_ost = _ost[_idx_finite]
    temp_num = _iirfilter_num[_idx_finite, :]
    temp_den = _iirfilter_den[_idx_finite, :]

    # online_filter_nojit
    sden = temp_den.shape
    snum = temp_num.shape
    n_input = temp_input.size
    no = sden[1]
    ni = snum[1]
    # Delay the vectors
    temp_ost = gxp.concatenate((temp_ost[:, 1:], gxp.zeros((sden[0], 1), dtype=gdtype)), axis=1)
    temp_ist = gxp.concatenate((temp_ist[:, 1:], gxp.zeros((sden[0], 1), dtype=gdtype)), axis=1)
    # New input
    temp_ist[:n_input, ni - 1] = temp_input
    # New output
    factor = 1/temp_den[:, no - 1]
    temp_ost[:, no - 1] = factor * gxp.sum(temp_num * temp_ist, axis=1)
    temp_ost[:, no - 1] -= factor * gxp.sum(temp_den[:, :no - 1] * temp_ost[:, :no - 1], axis=1)
    temp_output = temp_ost[:n_input, no - 1]
    # return output, ost, ist
    # online_filter_nojit

#        if ninput == 1:
#            output = temp_output
#            _ost = temp_ost
#            _ist = temp_ist
#        else:
    output[_idx_finite] = temp_output
    _ost[_idx_finite] = temp_ost
    _ist[_idx_finite] = temp_ist    
    return output


class IIRControl(BaseProcessingObj):
    '''Infinite Impulse Response filter based Time Control'''
    def __init__(self, iirfilter, delay=0,
                target_device_idx=None,
                precision=None
                ):
        global gxp, gdtype    
        self._verbose = True
        self._iirfilter = iirfilter
        
        super().__init__(target_device_idx=target_device_idx, precision=precision)        

        gxp = self.xp
        gdtype = self.dtype
        if self.xp==np:
            self.compute_comm = jit(compute_comm, nopython=True, cache=True)
        else:
            self.compute_comm = compute_comm

        self._delay = delay if delay is not None else 0
        self._n = iirfilter.nfilter
        self._type = iirfilter.num.dtype
        self.set_state_buffer_length(int(np.ceil(self._delay)) + 1)
        
        # Initialize state vectors
        self._ist = self.xp.zeros_like(iirfilter.num)
        self._ost = self.xp.zeros_like(iirfilter.den)

        self._out_comm = BaseValue(target_device_idx=target_device_idx)
        self.inputs['delta_comm'] = InputValue(type=BaseValue)
        self.outputs['out_comm'] = self._out_comm

        self._opticalgain = None
        self._og_shaper = None
        self._offset = None
        self._bootstrap_ptr = None
        self._modal_start_time = None
        self._time_gmt_imm = None
        self._gain_gmt_imm = None
        self._do_gmt_init_mod_manager = False
        self._skipOneStep = False
        self._StepIsNotGood = False
        self._start_time = 0

        self._outFinite = self.xp.zeros(self._iirfilter.nfilter, dtype=self.dtype)
        self._idx_finite = self.xp.zeros(self._iirfilter.nfilter, dtype=self.dtype)


    def set_state_buffer_length(self, total_length):
        self._total_length = total_length
        if self._n is not None and self._type is not None:
            self._state = self.xp.zeros((self._n, self._total_length), dtype=self.dtype)
            self._comm = self.xp.zeros((self._n, 1), dtype=self.dtype)

    def auto_params_management(self, main_params, control_params, detector_params, dm_params, slopec_params):
        result = control_params.copy()

        if str(result['delay']) == 'auto':
            binning = detector_params.get('binning', 1)
            computation_time = slopec_params.get('computation_time', 0) if slopec_params else 0
            delay = calc_loop_delay(1.0 / detector_params['dt'], dm_set=dm_params['settling_time'],
                                    type=detector_params['name'], bin=binning, comp_tim=computation_time)
            if delay == float('inf'):
                raise ValueError("Delay calculation resulted in infinity")
            result['delay'] = delay * (1.0 / main_params['time_step']) - 1

        return result

    def state_update(self, comm):
        finite_mask = self.xp.isfinite(comm)
        if self.xp.any(finite_mask):
            if self._delay > 0:
                self._state[:, 1:self._total_length] = self._state[:, 0:self._total_length-1]

            self._state[finite_mask, 0] = comm[finite_mask]

        self._comm = self.get_past_state(self._delay)

    @property
    def delay(self):
        return self._delay

    @property
    def last_state(self):
        return self._state[:, 0]

    @property
    def state(self):
        return self._state

    def get_past_state(self, past_step):
        remainder_delay = past_step % 1
        if remainder_delay == 0:
            return self._state[:, int(past_step)]
        else:
            return (remainder_delay * self._state[:, int(np.ceil(past_step))] +
                    (1 - remainder_delay) * self._state[:, int(np.ceil(past_step))-1])

    @property
    def comm(self):
        return self._comm

    @state.setter
    def state(self, state):
        self._state = state

    @property
    def iirfilter(self):
        return self._iirfilter

    @iirfilter.setter
    def iirfilter(self, value):
        self._iirfilter = value

    @property
    def out_comm(self):
        return self._out_comm

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, value):
        self._delay = value

    def set_modal_start_time(self, modal_start_time):
        modal_start_time_ = self.xp.array(modal_start_time, dtype=self.dtype)
        for i in range(len(modal_start_time)):
            modal_start_time_[i] = self.seconds_to_t(modal_start_time[i])
        self._modal_start_time = modal_start_time_

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

    def trigger_code(self):
        if self._opticalgain is not None:
            if self._opticalgain.value > 0:
                delta_comm = self.local_inputs['delta_comm'].value * 1.0 / self._opticalgain.value
                if self._og_shaper is not None:
                    delta_comm *= self._og_shaper
                self.local_inputs['delta_comm'].value = delta_comm
                print(f"WARNING: optical gain compensation has been applied (g_opt = {self._opticalgain.value:.5f}).")
        if self._start_time > 0 and self._start_time > t:
            # newc = self.xp.zeros_like(in_delta_comm.value)
            print(f"delta comm generation time: {self.local_inputs['delta_comm'].generation_time} is not greater than {self._start_time}")
        else:
            # should this be a copy? works like this too
            delta_comm = self.local_inputs['delta_comm'].value

        if self._modal_start_time is not None:
            for i in range(len(self._modal_start_time)):
                if self._modal_start_time[i] > t:
                    delta_comm[i] = 0
                    print(f"delta comm generation time: {self.in_delta_comm.generation_time} is not greater than {self._modal_start_time[i]}")
                    print(f" -> value of mode no. {i} is set to 0.")

        if self._skipOneStep:
            if self._StepIsNotGood:
                delta_comm *= 0
                self._StepIsNotGood = False
                print("WARNING: the delta commands of this step is set to 0 because skipOneStep key is active.")
            else:
                self._StepIsNotGood = True

        if self._bootstrap_ptr is not None:
            bootstrap_array = self._bootstrap_ptr
            bootstrap_time = bootstrap_array[:, 0]
            bootstrap_scale = bootstrap_array[:, 1]
            idx = self.xp.where(bootstrap_time <= self.t_to_seconds(t))[0]
            if len(idx) > 0:
                idx = idx[-1]
                if bootstrap_scale[idx] != 1:
                    print(f"ATTENTION: a scale factor of {bootstrap_scale[idx]} is applied to delta commands for bootstrap purpose.")
                    delta_comm *= bootstrap_scale[idx]
                else:
                    print("no scale factor applied")

        if self._do_gmt_init_mod_manager:
            time_idx = self._time_gmt_imm if self._time_gmt_imm is not None else self.xp.zeros(0, dtype=self.dtype)
            gain_idx = self._gain_gmt_imm if self._gain_gmt_imm is not None else self.xp.zeros(0, dtype=self.dtype)
            delta_comm *= gmt_init_mod_manager(self.t_to_seconds(t), len(delta_comm), time_idx=time_idx, gain_idx=gain_idx)

        n_delta_comm = delta_comm.size
        if n_delta_comm < self._iirfilter.nfilter:
            delta_comm = self.xp.zeros(self._iirfilter.nfilter, dtype=self.dtype)
            delta_comm[:n_delta_comm] = self.local_inputs['delta_comm'].value

        if self._offset is not None:
            n_offset = self._offset.shape[0]
            delta_comm[:n_offset] += self._offset
        
        newc = self.compute_comm(delta_comm, self._outFinite, self._ist, self._ost, self._iirfilter.num, self._iirfilter.den)

            #if self.xp.all(newc == 0) and self._offset is not None:
            #    newc[:n_offset] += self._offset
            #    print("WARNING (IIRCONTROL): newc is a null vector, applying offset.")

#            if self._verbose:
#                n_newc = newc.size
#                print(f"first {min(6, n_delta_comm)} delta_comm values: {delta_comm[:min(6, n_delta_comm)]}")
#                print(f"first {min(6, n_newc)} comm values: {newc[:min(6, n_newc)]}")
#        else:
#            if self._verbose:
#                print(f"delta comm generation time: {self.local_inputs['delta_comm'].generation_time} is not equal to {t}")
#            newc = self.last_state

        self.state_update(newc)

        self._out_comm.value = self.comm
        self._out_comm.generation_time = self.current_time

#        if self._verbose:
#            print(f"first {min(6, len(self._out_comm.value))} output comm values: {self._out_comm.value[:min(6, len(self._out_comm.value))]}")
    
    def run_check(self, time_step, errmsg=""):
        self.prepare_trigger(0)
#        super().build_stream()
        return True
