
from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.base_value import BaseValue
from pyssata.connections import InputValue
from pyssata.data_objects.pixels import Pixels
from pyssata.data_objects.slopes import Slopes


class Slopec(BaseProcessingObj):
    def __init__(self, sn: Slopes=None, cm=None, total_counts=None, subap_counts=None, 
                 use_sn=False, accumulate=False, weight_from_accumulated=False, 
                 weight_from_acc_with_window=False, remove_mean=False, return0=False, 
                 update_slope_high_speed=False, do_rec=False, do_filter_modes=False, 
                 gain_slope_high_speed=0.0, ff_slope_high_speed=0.0, store_s=None, 
                 store_c=None, sn_scale_fact=None, command_list=None, intmat=None, 
                 recmat=None, filt_recmat=None, filt_intmat=None, accumulation_dt=0, 
                 accumulated_pixels=(0,0),
                 target_device_idx=None, 
                 precision=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self._slopes = Slopes(2)  # TODO resized in derived class
        self._slopes_ave = BaseValue()
        self._sn = sn
        self._cm = cm
        self._total_counts = total_counts
        self._subap_counts = subap_counts
        self._flux_per_subaperture_vector = BaseValue()
        self._max_flux_per_subaperture_vector = BaseValue()
        self._use_sn = use_sn
        self._accumulate = accumulate
        self._weight_from_accumulated = weight_from_accumulated
        self._weight_from_acc_with_window = weight_from_acc_with_window
        self._remove_mean = remove_mean
        self._return0 = return0
        self._update_slope_high_speed = update_slope_high_speed
        self._do_rec = do_rec
        self._do_filter_modes = do_filter_modes
        self._gain_slope_high_speed = gain_slope_high_speed
        self._ff_slope_high_speed = ff_slope_high_speed
        self._store_s = store_s
        self._store_c = store_c
        self._sn_scale_fact = sn_scale_fact
        self._command_list = command_list
        self._intmat = intmat
        self._recmat = recmat
        self._filt_recmat = filt_recmat
        self._filt_intmat = filt_intmat
        self._accumulation_dt = accumulation_dt
        self._accumulated_pixels = self.xp.array(accumulated_pixels, dtype=self.dtype)
        self._accumulated_slopes = Slopes(2)   # TODO resized in derived class.
        self._accumulated_pixels_ptr = None   # TODO, see do_accumulation() method

        self.inputs['in_pixels'] = InputValue(type=Pixels)
        self.outputs['out_slopes'] = self._slopes

    @property
    def in_sn(self):
        return self._sn

    @in_sn.setter
    def in_sn(self, value):
        self._sn = value

    @property
    def use_sn(self):
        return self._use_sn

    @use_sn.setter
    def use_sn(self, value):
        self._use_sn = value

    @property
    def sn_tag(self):
        return self._sn_tag

    @sn_tag.setter
    def sn_tag(self, value):
        self.load_sn(value)

    @property
    def cm(self):
        return self._cm

    @cm.setter
    def cm(self, value):
        self._cm = value

    @property
    def out_slopes(self):
        return self._slopes

    @property
    def weight_from_accumulated(self):
        return self._weight_from_accumulated

    @weight_from_accumulated.setter
    def weight_from_accumulated(self, value):
        self._weight_from_accumulated = value

    @property
    def weight_from_acc_with_window(self):
        return self._weight_from_acc_with_window

    @weight_from_acc_with_window.setter
    def weight_from_acc_with_window(self, value):
        self._weight_from_acc_with_window = value

    @property
    def accumulate(self):
        return self._accumulate

    @accumulate.setter
    def accumulate(self, value):
        self._accumulate = value

    @property
    def accumulation_dt(self):
        return self._accumulation_dt

    @accumulation_dt.setter
    def accumulation_dt(self, value):
        self._accumulation_dt = self.seconds_to_t(value)

    @property
    def remove_mean(self):
        return self._remove_mean

    @remove_mean.setter
    def remove_mean(self, value):
        self._remove_mean = value

    @property
    def return0(self):
        return self._return0

    @return0.setter
    def return0(self, value):
        self._return0 = value

    @property
    def update_slope_high_speed(self):
        return self._update_slope_high_speed

    @update_slope_high_speed.setter
    def update_slope_high_speed(self, value):
        self._update_slope_high_speed = value

    @property
    def command_list(self):
        return self._command_list

    @command_list.setter
    def command_list(self, value):
        self._command_list = value

    @property
    def intmat(self):
        return self._intmat

    @intmat.setter
    def intmat(self, value):
        self._intmat = value

    @property
    def recmat(self):
        return self._recmat

    @recmat.setter
    def recmat(self, value):
        self._recmat = value

    @property
    def gain_slope_high_speed(self):
        return self._gain_slope_high_speed

    @gain_slope_high_speed.setter
    def gain_slope_high_speed(self, value):
        self._gain_slope_high_speed = value

    @property
    def ff_slope_high_speed(self):
        return self._ff_slope_high_speed

    @ff_slope_high_speed.setter
    def ff_slope_high_speed(self, value):
        self._ff_slope_high_speed = value

    @property
    def do_rec(self):
        return self._do_rec

    @do_rec.setter
    def do_rec(self, value):
        self._do_rec = value

    @property
    def filtmat(self):
        return self._filtmat

    @filtmat.setter
    def filtmat(self, value):
        self.set_filtmat(value)

    @property
    def in_sn_scale_fact(self):
        return self._sn_scale_fact

    @in_sn_scale_fact.setter
    def in_sn_scale_fact(self, value):
        self._sn_scale_fact = value

    def set_filtmat(self, filtmat):
        self._filt_intmat = filtmat[:, :, 0]
        self._filt_recmat = self.xp.transpose(filtmat[:, :, 1])
        self._do_filter_modes = True

    def remove_filtmat(self):
        self._filt_intmat = None
        self._filt_recmat = None
        self._do_filter_modes = False
        print('doFilterModes set to 0')

    def build_and_save_filtmat(self, intmat, recmat, nmodes, filename):
        im = intmat[:nmodes, :]
        rm = recmat[:, :nmodes]

        output = self.xp.stack((im, self.xp.transpose(rm)), axis=-1)
        self.writefits(filename, output)
        print(f'saved {filename}')

    def _compute_flux_per_subaperture(self):
        raise NotImplementedError('abstract method must be implemented')

    def _compute_max_flux_per_subaperture(self):
        raise NotImplementedError('abstract method must be implemented')

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

    def trigger_code(self):
        raise NotImplementedError(f'{self.repr()} Please implement calc_slopes in your derived class!')
        
    def post_trigger(self):
        super().post_trigger()
        if self._do_rec:
            m = self.xp.dot(self._slopes.ptr_slopes, self._recmat.ptr_recmat)
            self._slopes.slopes = m

    def run_check(self, time_step, errmsg=''):
        self.prepare_trigger(0)
        if self._use_sn and not self._sn:
            errmsg += 'Slopes null are not valid'
        if self._weight_from_accumulated and self._accumulate:
            errmsg += 'weightFromAccumulated and accumulate must not be set together'
        if errmsg != '':
            print(errmsg)
        return not (self._weight_from_accumulated and self._accumulate) and self.local_inputs['in_pixels'] and self._slopes and ((not self._use_sn) or (self._use_sn and self._sn))

