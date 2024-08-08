import numpy as np
from pyssata.base_processing_obj import BaseProcessingObj


class IIRControl(TimeControl, BaseProcessingObj):
    def __init__(self, iirfilter, delay=0):
        super().__init__()

        self._iirfilter = iirfilter
        typeIIR = iirfilter.num.dtype
        nIIR = iirfilter.nfilter

        if not super(TimeControl, self).Init(delay=delay, n=nIIR, type=typeIIR):
            raise ValueError("Initialization of TimeControl failed")

        self._ist = np.zeros_like(iirfilter.num)
        self._ost = np.zeros_like(iirfilter.den)

        self._out_comm = BaseValue()

        if not super(BaseProcessingObj, self).Init('iircontrol', 'Infinite Impulse Response filter based Time Control'):
            raise ValueError("Initialization of BaseProcessingObj failed")

        self._delta_comm = None
        self._opticalgain = None
        self._og_shaper = None
        self._offset = None
        self._bootstrap_ptr = None
        self._modal_start_time = None
        self._time_gmt_imm = None
        self._gain_gmt_imm = None
        self._deltaCommHistEx = None
        self._commHistEx = None
        self._deltaCommFutureHistEx = None
        self._ostMatEx = None
        self._istMatEx = None
        self._doExtraPol = False
        self._do_gmt_init_mod_manager = False
        self._nPastStepsEx = 0
        self._gainEx = 0.0
        self._extraPolMinMax = [0, 0]
        self._skipOneStep = False
        self._StepIsNotGood = False

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
        modal_start_time_ = np.array(modal_start_time)
        for i in range(len(modal_start_time)):
            modal_start_time_[i] = self.seconds_to_t(modal_start_time[i])
        self._modal_start_time = modal_start_time_

    def trigger(self, t):
        ist = self._ist
        ost = self._ost

        if self._delta_comm.generation_time == t:
            if self._opticalgain is not None:
                if self._opticalgain.value > 0:
                    delta_comm = self._delta_comm.value * 1.0 / self._opticalgain.value
                    if self._og_shaper is not None:
                        delta_comm *= self._og_shaper
                    self._delta_comm.value = delta_comm
                    print(f"WARNING: optical gain compensation has been applied (g_opt = {self._opticalgain.value:.5f}).")

            if self._start_time > 0 and self._start_time > t:
                newc = np.zeros_like(self._delta_comm.value)
                print(f"delta comm generation time: {self._delta_comm.generation_time} is not greater than {self._start_time}")
            else:
                delta_comm = self._delta_comm.value

                if self._modal_start_time is not None:
                    for i in range(len(self._modal_start_time)):
                        if self._modal_start_time[i] > t:
                            delta_comm[i] = 0
                            print(f"delta comm generation time: {self._delta_comm.generation_time} is not greater than {self._modal_start_time[i]}")
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
                    idx = np.where(bootstrap_time <= self.t_to_seconds(t))[0]
                    if len(idx) > 0:
                        idx = idx[-1]
                        if bootstrap_scale[idx] != 1:
                            print(f"ATTENTION: a scale factor of {bootstrap_scale[idx]} is applied to delta commands for bootstrap purpose.")
                            delta_comm *= bootstrap_scale[idx]
                        else:
                            print("no scale factor applied")

                if self._do_gmt_init_mod_manager:
                    time_idx = self._time_gmt_imm if self._time_gmt_imm is not None else np.zeros(0)
                    gain_idx = self._gain_gmt_imm if self._gain_gmt_imm is not None else np.zeros(0)
                    delta_comm *= gmt_init_mod_manager(self.t_to_seconds(t), len(delta_comm), time_idx=time_idx, gain_idx=gain_idx)

                if len(delta_comm) < self._iirfilter.nfilter:
                    n_delta_comm = len(delta_comm)
                    delta_comm = np.zeros(self._iirfilter.nfilter)
                    delta_comm[:n_delta_comm] = self._delta_comm.value

                if self._offset is not None:
                    delta_comm[:len(self._offset)] += self._offset

                if self._doExtraPol:
                    if self._nPastStepsEx == 0:
                        self._nPastStepsEx = 8

                if self._deltaCommFutureHistEx is not None:
                    if abs(round(self._delay) - self._delay) <= 1e-3:
                        delta_temp = self._deltaCommFutureHistEx[:, self._nPastStepsEx - round(self._delay)]
                    else:
                        delta_temp = (self._delay - np.floor(self._delay)) * self._deltaCommFutureHistEx[:, self._nPastStepsEx - np.ceil(self._delay)] + \
                                     (np.ceil(self._delay) - self._delay) * self._deltaCommFutureHistEx[:, self._nPastStepsEx - np.floor(self._delay)]
                    delta_comm += delta_temp

                newc = compute_comm(self._iirfilter, delta_comm, ist=ist, ost=ost)
                newc = newc.astype(np.float64 if self._precision > 0 else np.float32)

                if np.all(newc == 0) and self._offset is not None:
                    newc[:len(self._offset)] += self._offset
                    print("WARNING (IIRCONTROL): newc is a null vector, applying offset.")

                if self._doExtraPol:
                    print("doing extrapolation")
                    thr_chisqr = 0.9
                    LPF = True
                    if np.all(self._extraPolMinMax == 0):
                        self._extraPolMinMax = [0, self._iirfilter.nfilter]
                    deltaCommHist = self._deltaCommHistEx if self._deltaCommHistEx is not None else np.zeros((0, 0))
                    commHist = self._commHistEx if self._commHistEx is not None else np.zeros((0, 0))
                    deltaCommFutureHist = self._deltaCommFutureHistEx if self._deltaCommFutureHistEx is not None else np.zeros((0, 0))
                    if LPF:
                        ostMat = self._ostMatEx if self._ostMatEx is not None else np.zeros((0, 0))
                        istMat = self._istMatEx if self._istMatEx is not None else np.zeros((0, 0))
                    newcExtrapol = compute_extrapol_comm(newc, self._delay + 1, self._nPastStepsEx, thr_chisqr, 
                                                         deltaCommHist=deltaCommHist, commHist=commHist, 
                                                         deltaCommFutureHist=deltaCommFutureHist, gainFuture=None, 
                                                         nMinMaxMode=self._extraPolMinMax, LPF=LPF, filterGain=self._gainEx, 
                                                         ostMat=ostMat, istMat=istMat)
                    self._deltaCommHistEx = deltaCommHist
                    self._commHistEx = commHist
                    self._deltaCommFutureHistEx = deltaCommFutureHist
                    if LPF:
                        self._ostMatEx = ostMat
                        self._istMatEx = istMat

                if self._verbose:
                    print(f"first {min(6, len(delta_comm))} delta_comm values: {delta_comm[:min(6, len(delta_comm))]}")
                    print(f"first {min(6, len(newc))} comm values: {newc[:min(6, len(newc))]}")
        else:
            if self._verbose:
                print(f"delta comm generation time: {self._delta_comm.generation_time} is not equal to {t}")
            newc = self.get_last_state()

        self._ist = ist
        self._ost = ost
        self.state_update(newc)

        self._out_comm.value = self.get_comm()
        self._out_comm.generation_time = t

        if self._verbose:
            print(f"first {min(6, len(self._out_comm.value))} output comm values: {self._out_comm.value[:min(6, len(self._out_comm.value))]}")

    def cleanup(self):
        del self._ist
        del self._ost
        del self._offset
        del self._deltaCommHistEx
        del self._commHistEx
        del self._deltaCommFutureHistEx
        del self._ostMatEx
        del self._istMatEx
        del self._bootstrap_ptr
        del self._og_shaper
        del self._time_gmt_imm
        del self._gain_gmt_imm
        del self._modal_start_time
        self._delta_comm.cleanup()
        self._out_comm.cleanup()
        self._iirfilter.cleanup()
        self._opticalgain.cleanup()
        super(TimeControl, self).cleanup()
        super(BaseProcessingObj, self).cleanup()

    @staticmethod
    def revision_track():
        return "$Rev$"

    def run_check(self, time_step, errmsg=""):
        return True
