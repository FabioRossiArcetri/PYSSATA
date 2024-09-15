import numpy as np
from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.base_value import BaseValue
from pyssata.data_objects.pixels import Pixels
from pyssata.data_objects.slopes import Slopes


class Slopec(BaseProcessingObj):
    def __init__(self, pixels=None, sn: Slopes=None, cm=None, total_counts=None, subap_counts=None, 
                 use_sn=False, accumulate=False, weight_from_accumulated=False, 
                 weight_from_acc_with_window=False, remove_mean=False, return0=False, 
                 update_slope_high_speed=False, do_rec=False, do_filter_modes=False, 
                 gain_slope_high_speed=0.0, ff_slope_high_speed=0.0, store_s=None, 
                 store_c=None, sn_scale_fact=None, command_list=None, intmat=None, 
                 recmat=None, filt_recmat=None, filt_intmat=None, accumulation_dt=0, 
                 accumulated_pixels=(0,0)):
 
        super().__init__()
        self._pixels = pixels
        self._slopes = Slopes(2)
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
        self._accumulated_pixels = np.array(accumulated_pixels)
        self._accumulated_slopes = Slopes(2)
        self._accumulated_pixels_ptr = None   # TODO, see do_accumulation() method

    @property
    def in_pixels(self):
        return self._pixels

    @in_pixels.setter
    def in_pixels(self, value):
        self._pixels = value

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
        self._filt_recmat = np.transpose(filtmat[:, :, 1])
        self._do_filter_modes = True

    def remove_filtmat(self):
        self._filt_intmat = None
        self._filt_recmat = None
        self._do_filter_modes = False
        print('doFilterModes set to 0')

    def build_and_save_filtmat(self, intmat, recmat, nmodes, filename):
        im = intmat[:nmodes, :]
        rm = recmat[:, :nmodes]

        output = np.stack((im, np.transpose(rm)), axis=-1)
        self.writefits(filename, output)
        print(f'saved {filename}')

    def _compute_flux_per_subaperture(self):
        raise NotImplementedError('abstract method must be implemented')

    def _compute_max_flux_per_subaperture(self):
        raise NotImplementedError('abstract method must be implemented')

    def run_check(self, time_step, errmsg=''):
        if self._use_sn and not self._sn:
            errmsg += 'Slopes null are not valid'
        if self._weight_from_accumulated and self._accumulate:
            errmsg += 'weightFromAccumulated and accumulate must not be set together'
        return not (self._weight_from_accumulated and self._accumulate) and self._pixels and self._slopes and ((not self._use_sn) or (self._use_sn and self._sn))

    def calc_slopes(self, t, accumulated=False):
        raise NotImplementedError(f'{self.repr()} Please implement calc_slopes in your derived class!')

    def revision_track(self):
        return '$Rev$'

    def do_accumulation(self, t):
        factor = float(self._loop_dt) / float(self._accumulation_dt)
        if not self._accumulated_pixels:
            self._accumulated_pixels = Pixels(self._pixels.size[0], self._pixels.size[1])
        if (t % self._accumulation_dt) == 0:
            self._accumulated_pixels.pixels = self._pixels.pixels * factor
        else:
            self._accumulated_pixels.pixels += self._pixels.pixels * factor
        self._accumulated_pixels.generation_time = t

        # TODO what is "accumulated_pixels_ptr" used for exacly?            
        if self._accumulated_pixels_ptr is None:
            acc_pixels = 0
        else:
            acc_pixels = self._accumulated_pixels_ptr
        if t >= self._accumulation_dt:
            self._accumulated_pixels_ptr = acc_pixels * (1 - factor) + self._pixels.pixels * factor
        else:
            self._accumulated_pixels_ptr = acc_pixels + self._pixels.pixels * factor
        if self._verbose:
            print(f'accumulation factor is: {factor}')

    def cleanup(self):
        if self._store_s:
            del self._store_s
        if self._store_c:
            del self._store_c
        if self._intmat:
            del self._intmat
        if self._filt_recmat:
            del self._filt_recmat
        if self._filt_intmat:
            del self._filt_intmat
        if self._accumulated_pixels_ptr:
            del self._accumulated_pixels_ptr
        if self._total_counts:
            del self._total_counts
        if self._subap_counts:
            del self._subap_counts
        self._flux_per_subaperture_vector.cleanup()
        self._max_flux_per_subaperture_vector.cleanup()
        self._slopes.cleanup()
        self._accumulated_pixels.cleanup()
        self._accumulated_slopes.cleanup()

    def trigger(self, t):
        if self._accumulate:
            self.do_accumulation(t)
            if (t + self._loop_dt) % self._accumulation_dt == 0:
                self.calc_slopes(t, accumulated=True)

        if self._weight_from_accumulated:
            self.do_accumulation(t)

        if self._pixels.generation_time == t:
            self.calc_slopes(t)
            if not np.isfinite(self._slopes.slopes).all():
                raise ValueError('slopes have non-finite elements')
            if self._sn is not None and self._use_sn:
                if self._verbose:
                    print('removing slope null')
                if self._sn_scale_fact:
                    temp_sn = BaseValue()
                    if self._sn_scale_fact.generation_time >= 0:
                        temp_sn.value = self._sn.slopes * self._sn_scale_fact.value
                    else:
                        temp_sn.value = self._sn.slopes
                    if self._verbose:
                        print('ATTENTION: slope null scaled by a factor')
                        print(f' Value: {self._sn_scale_fact.value}')
                        print(f' Is it applied? {self._sn_scale_fact.generation_time >= 0}')
                    self._slopes.subtract(temp_sn)
                else:
                    self._slopes.subtract(self._sn)

            self._slopes_ave.value = [np.mean(self._slopes.xslopes), np.mean(self._slopes.yslopes)]
            self._slopes_ave.generation_time = t

            if self._remove_mean:
                sx = self._slopes.xslopes - self._slopes_ave.value[0]
                sy = self._slopes.yslopes - self._slopes_ave.value[1]
                self._slopes.xslopes = sx
                self._slopes.yslopes = sy
                if self._verbose:
                    print('mean value of x and y slope was removed')

        else:
            if self._return0:
                self._slopes.xslopes = np.zeros_like(self._slopes.xslopes)
                self._slopes.yslopes = np.zeros_like(self._slopes.yslopes)

        if self._update_slope_high_speed:
            if self._gain_slope_high_speed == 0.0:
                self._gain_slope_high_speed = 1.0
            if self._ff_slope_high_speed == 0.0:
                self._ff_slope_high_speed = 1.0
            commands = []
            for comm in self._command_list:
                if len(comm.value) > 0:
                    commands.append(comm.value)
            if self._pixels.generation_time == t:
                self._store_s = np.array([self._gain_slope_high_speed * self._slopes.xslopes, self._gain_slope_high_speed * self._slopes.yslopes])
                self._store_c = np.array(commands)
            else:
                if len(commands) > 0 and self._store_s is not None and self._store_c is not None:
                    temp = np.dot(np.array(commands) - self._store_c, self._intmat)
                    self._store_s *= self._ff_slope_high_speed
                    self._slopes.xslopes = self._store_s[0] - temp[:len(temp)//2]
                    self._slopes.yslopes = self._store_s[1] - temp[len(temp)//2:]
                    self._slopes.generation_time = t

        if self._do_filter_modes:
            m = np.dot(self._slopes.ptr_slopes, self._filt_recmat)
            sl0 = np.dot(m, self._filt_intmat)
            sl = self._slopes.slopes
            if len(sl) != len(sl0):
                raise ValueError(f'mode filtering goes wrong: original slopes size is: {len(sl)} while filtered slopes size is: {len(sl0)}')
            self._slopes.slopes -= sl0
            if self._verbose:
                print(f'Slopes have been filtered. New slopes min, max and rms : {np.min(self._slopes.slopes)}, {np.max(self._slopes.slopes)}  //  {np.sqrt(np.mean(self._slopes.slopes**2))}')
            if not np.isfinite(self._slopes.slopes).all():
                raise ValueError('slopes have non-finite elements')

        if self._do_rec:
            m = np.dot(self._slopes.ptr_slopes, self._recmat.ptr_recmat)
            self._slopes.slopes = m

