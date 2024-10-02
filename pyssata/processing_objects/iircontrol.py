import numpy as np

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.connections import InputValue
from pyssata.base_value import BaseValue
from pyssata.lib.calc_loop_delay import calc_loop_delay

class IIRControl(BaseProcessingObj):
    '''Infinite Impulse Response filter based Time Control'''
    def __init__(self, iirfilter, delay=0,
                target_device_idx=None, 
                precision=None
                ):

        self._verbose = True
        self._iirfilter = iirfilter
        
        super().__init__(target_device_idx=target_device_idx, precision=precision)        

        self._delay = delay if delay is not None else 0
        self._n = iirfilter.nfilter
        self._type = iirfilter.num.dtype
        self.set_state_buffer_length(int(self.xp.ceil(self._delay)) + 1)
        
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
            return (remainder_delay * self._state[:, int(self.xp.ceil(past_step))] +
                    (1 - remainder_delay) * self._state[:, int(self.xp.ceil(past_step))-1])

    @property
    def comm(self):
        return self._comm

    @state.setter
    def state(self, state):
        self._state = state

    @property
    def in_delta_comm(self):
        return self._delta_comm

    @in_delta_comm.setter
    def in_delta_comm(self, value):
        self._delta_comm = value

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

    def trigger(self, t):
        ist = self._ist
        ost = self._ost
        in_delta_comm = self.inputs['delta_comm'].get(self._target_device_idx)

        print('in_delta_comm',in_delta_comm)
        print('in_delta_comm.value',in_delta_comm.value)
        print('in_delta_comm.generation_time',in_delta_comm.generation_time)

        if in_delta_comm.generation_time == t:
            if self._opticalgain is not None:
                if self._opticalgain.value > 0:
                    delta_comm = in_delta_comm.value * 1.0 / self._opticalgain.value
                    if self._og_shaper is not None:
                        delta_comm *= self._og_shaper
                    in_delta_comm.value = delta_comm
                    print(f"WARNING: optical gain compensation has been applied (g_opt = {self._opticalgain.value:.5f}).")

            if self._start_time > 0 and self._start_time > t:
                newc = self.xp.zeros_like(in_delta_comm.value)
                print(f"delta comm generation time: {in_delta_comm.generation_time} is not greater than {self._start_time}")
            else:
                delta_comm = in_delta_comm.value

                if self._modal_start_time is not None:
                    for i in range(len(self._modal_start_time)):
                        if self._modal_start_time[i] > t:
                            delta_comm[i] = 0
                            print(f"delta comm generation time: {in_delta_comm.generation_time} is not greater than {self._modal_start_time[i]}")
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

                n_delta_comm = self.xp.size(delta_comm)
                if n_delta_comm < self._iirfilter.nfilter:
                    delta_comm = self.xp.zeros(self._iirfilter.nfilter, dtype=self.dtype)
                    delta_comm[:n_delta_comm] = self._delta_comm.value

                if self._offset is not None:
                    n_offset = self.xp.size(self._offset)
                    delta_comm[:n_offset] += self._offset

                newc = self.compute_comm(delta_comm)                

                if self.xp.all(newc == 0) and self._offset is not None:
                    newc[:n_offset] += self._offset
                    print("WARNING (IIRCONTROL): newc is a null vector, applying offset.")

                if self._verbose:
                    n_newc = self.xp.size(newc)
                    print(f"first {min(6, n_delta_comm)} delta_comm values: {delta_comm[:min(6, n_delta_comm)]}")
                    print(f"first {min(6, n_newc)} comm values: {newc[:min(6, n_newc)]}")
        else:
            if self._verbose:
                print(f"delta comm generation time: {self._delta_comm.generation_time} is not equal to {t}")
            newc = self.last_state

        self.state_update(newc)

        self._out_comm.value = self.comm
        self._out_comm.generation_time = t

        if self._verbose:
            print(f"first {min(6, len(self._out_comm.value))} output comm values: {self._out_comm.value[:min(6, len(self._out_comm.value))]}")

    def compute_comm(self, input):
        nfilter = self._iirfilter.num.shape[0]
        ninput = self.xp.size(input)
        output = input*0

        if nfilter < ninput:
            raise ValueError(f"Error: IIR filter needs no more than {nfilter} coefficients ({ninput} given)")

        if ninput == 1:
            if self.xp.isfinite(input):
                temp_input = input
                temp_ist = self._ist
                temp_ost = self._ost
                temp_num = self._iirfilter.num
                temp_den = self._iirfilter.den
            else:
                return output
        else:
            idx_finite = self.xp.where(self.xp.isfinite(input))[0]
            temp_input = input[idx_finite]
            temp_ist = self._ist[idx_finite]
            temp_ost = self._ost[idx_finite]
            temp_num = self._iirfilter.num[idx_finite, :]
            temp_den = self._iirfilter.den[idx_finite, :]
   
        temp_output, temp_ost, temp_ist = self.online_filter(
            temp_num,
            temp_den,
            temp_input,
            temp_ost,
            temp_ist
        )
        if ninput == 1:
            output = temp_output
            self._ost = temp_ost
            self._ist = temp_ist
        else:
            output[idx_finite] = temp_output
            self._ost[idx_finite] = temp_ost
            self._ist[idx_finite] = temp_ist

        return output
    
    def online_filter(self, num, den, input, ost, ist):
        sden = self.xp.shape(den)
        snum = self.xp.shape(num)
        n_input = self.xp.size(input)
        
        no = sden[1]
        ni = snum[1]

        # Delay the vectors
        ost = self.xp.concatenate((ost[:, 1:], self.xp.zeros((sden[0], 1), dtype=self.dtype)), axis=1)
        ist = self.xp.concatenate((ist[:, 1:], self.xp.zeros((sden[0], 1), dtype=self.dtype)), axis=1)

        # New input
        ist[:n_input, ni - 1] = input

        # New output
        factor = 1/den[:, no - 1]
        ost[:, no - 1] = factor * self.xp.sum(num * ist, axis=1)
        ost[:, no - 1] -= factor * self.xp.sum(den[:, :no - 1] * ost[:, :no - 1], axis=1)

        output = ost[:n_input, no - 1]

        return output, ost, ist

    def run_check(self, time_step, errmsg=""):
        return True
