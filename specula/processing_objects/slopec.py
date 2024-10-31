
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes


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

        self.slopes = Slopes(2)  # TODO resized in derived class
        self.slopes_ave = BaseValue()
        self.sn = sn
        self.cm = cm
        self.total_counts = total_counts
        self.subap_counts = subap_counts
        self.flux_per_subaperture_vector = BaseValue()
        self.max_flux_per_subaperture_vector = BaseValue()
        self.use_sn = use_sn
        self.accumulate = accumulate
        self.weight_from_accumulated = weight_from_accumulated
        self.weight_from_acc_with_window = weight_from_acc_with_window
        self.remove_mean = remove_mean
        self.return0 = return0
        self.update_slope_high_speed = update_slope_high_speed
        self.do_rec = do_rec
        self.do_filter_modes = do_filter_modes
        self.gain_slope_high_speed = gain_slope_high_speed
        self.ff_slope_high_speed = ff_slope_high_speed
        self.store_s = store_s
        self.store_c = store_c
        self.sn_scale_fact = sn_scale_fact
        self.command_list = command_list
        self.intmat = intmat
        self.recmat = recmat
        self.filt_recmat = filt_recmat
        self.filt_intmat = filt_intmat
        self._accumulation_dt = accumulation_dt
        self.accumulated_pixels = self.xp.array(accumulated_pixels, dtype=self.dtype)
        self.accumulated_slopes = Slopes(2)   # TODO resized in derived class.
        self.accumulated_pixels_ptr = None   # TODO, see do_accumulation() method

        self.inputs['in_pixels'] = InputValue(type=Pixels)
        self.outputs['out_slopes'] = self.slopes


    @property
    def sn_tag(self):
        return self._sn_tag

    @sn_tag.setter
    def sn_tag(self, value):
        self.load_sn(value)


    @property
    def accumulation_dt(self):
        return self._accumulation_dt

    @accumulation_dt.setter
    def accumulation_dt(self, value):
        self._accumulation_dt = self.seconds_to_t(value)

    def set_filtmat(self, filtmat):
        self.filt_intmat = filtmat[:, :, 0]
        self.filt_recmat = self.xp.transpose(filtmat[:, :, 1])
        self.do_filter_modes = True

    def remove_filtmat(self):
        self.filt_intmat = None
        self.filt_recmat = None
        self.do_filter_modes = False
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
        raise NotImplementedError(f'{self.__class__.__name__}: please implement trigger_code() in your derived class!')
        
    def post_trigger(self):
        # super().post_trigger()
        if self.do_rec:
            m = self.xp.dot(self.slopes.ptr_slopes, self.recmat.ptr_recmat)
            self.slopes.slopes = m

    def run_check(self, time_step, errmsg=''):
        self.prepare_trigger(0)
        if self.use_sn and not self.sn:
            errmsg += 'Slopes null are not valid'
        if self.weight_from_accumulated and self.accumulate:
            errmsg += 'weightFromAccumulated and accumulate must not be set together'
        if errmsg != '':
            print(errmsg)
        return not (self.weight_from_accumulated and self.accumulate) and self.local_inputs['in_pixels'] and self.slopes and ((not self.use_sn) or (self.use_sn and self.sn))

