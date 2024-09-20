import numpy as np

from pyssata import xp

from pyssata.lib.calc_loop_delay import calc_loop_delay


def compute_comm(filter_obj, input, ist=None, ost=None):
    nfilter = filter_obj.num.shape[1]
    ninput = len(input)

    if nfilter < ninput:
        raise ValueError(f"Error: IIR filter needs no more than {nfilter} coefficients ({ninput} given)")

    ordnum = xp.sort(filter_obj.ordnum)
    ordden = xp.sort(filter_obj.ordden)
    idx_onu = xp.unique(ordnum, return_index=True)[1]
    idx_odu = xp.unique(ordden, return_index=True)[1]
    onu = ordden[idx_onu]
    odu = ordden[idx_odu]
    output = xp.zeros_like(input)

    if len(onu) == 1 and len(odu) == 1:
        idx_finite = xp.isfinite(input)
        temp_ist = ist[idx_finite, :]
        temp_ost = ost[idx_finite, :]
        output[idx_finite] = online_filter(filter_obj.num[idx_finite, :onu[0]],
                                           filter_obj.den[idx_finite, :odu[0]], 
                                           input[idx_finite], ost=temp_ost, ist=temp_ist)
        ost[idx_finite, :] = temp_ost
        ist[idx_finite, :] = temp_ist
    else:
        for j in idx_onu:
            for k in idx_odu:
                idx = xp.where((filter_obj.ordnum == onu[j]) & (filter_obj.ordden == odu[k]))[0]
                if len(idx) > 0:
                    ord_num = onu[j]
                    ord_den = odu[k]
                    idx_finite = xp.isfinite(input[idx])
                    if xp.any(idx_finite):
                        idx = idx[idx_finite]
                        temp_ist = ist[idx, :ord_num]
                        temp_ost = ost[idx, :ord_den]
                        output[idx] = online_filter(filter_obj.num[idx, :ord_num], 
                                                    filter_obj.den[idx, :ord_den], 
                                                    input[idx], ost=temp_ost, ist=temp_ist)
                        ost[idx, :ord_den] = temp_ost
                        ist[idx, :ord_num] = temp_ist

    return output

def online_filter(num, den, input, ost=None, ist=None):
    # This function should implement the actual online filtering.
    # The implementation will depend on how the filtering needs to be done.
    # This is a placeholder implementation.
    if ost is None:
        ost = xp.zeros_like(num)
    if ist is None:
        ist = xp.zeros_like(den)
    
    # Example implementation of online filtering (not actual)
    # Compute the new output
    output = xp.dot(num, input) - xp.dot(den[1:], ost)
    
    # Update the states
    ost = xp.roll(ost, -1)
    ost[-1] = output
    ist = xp.roll(ist, -1)
    ist[-1] = input
    
    return output


class TimeControl:
    def __init__(self, delay=0, n=None, type=None, total_length=None):
        self._delay = delay if delay is not None else 0
        self._n = n
        self._type = type

        if total_length is not None:
            self.set_state_buffer_length(total_length)
        else:
            self.set_state_buffer_length(int(xp.ceil(self._delay)) + 1)

    def set_state_buffer_length(self, total_length):
        self._total_length = total_length
        if self._n is not None and self._type is not None:
            self._state = xp.zeros((self._n, self._total_length), dtype=self.dtype)
            self._comm = xp.zeros((self._n, 1), dtype=self.dtype)

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
        finite_mask = xp.isfinite(comm)
        if xp.any(finite_mask):
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
            return (remainder_delay * self._state[:, int(xp.ceil(past_step))] +
                    (1 - remainder_delay) * self._state[:, int(xp.ceil(past_step))-1])

    @property
    def comm(self):
        return self._comm

    @state.setter
    def state(self, state):
        self._state = state

    @staticmethod
    def revision_track():
        return "$Rev$"

    def cleanup(self):
        del self._comm
        del self._state
