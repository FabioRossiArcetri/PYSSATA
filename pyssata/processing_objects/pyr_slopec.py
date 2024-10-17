import numpy as np
from pyssata import fuse
from pyssata.processing_objects.slopec import Slopec
from pyssata.data_objects.slopes import Slopes
from  pyssata.base_value import BaseValue
from pyssata.data_objects.pupdata import PupData

@fuse(kernel_name='clamp_generic_less')
def clamp_generic_less(x, c, y, xp):
    y[:] = xp.where(y < x, c, y)

@fuse(kernel_name='clamp_generic_less1')
def clamp_generic_less1(x, c, y, xp):
    y = xp.where(y < x, c, y)


@fuse(kernel_name='clamp_generic_more')
def clamp_generic_more(x, c, y, xp):
    y[:] = xp.where(y > x, c, y)


@fuse(kernel_name='clamp_generic_more1')
def clamp_generic_more1(x, c, y, xp):
    y = xp.where(y > x, c, y)

class PyrSlopec(Slopec):
    def __init__(self, pupdata: PupData=None, shlike=False, norm_factor=None, thr_value=0, slopes_from_intensity=False, filtmat_tag='', 
                 target_device_idx=None, 
                 precision=None,
                **kwargs):
        super().__init__(target_device_idx=target_device_idx, precision=precision, **kwargs)
        self._shlike = shlike
        self._norm_factor = norm_factor
        self._thr_value = int(thr_value)
        self.threshold = self._thr_value if self._thr_value != -1 else None
        self._slopes_from_intensity = slopes_from_intensity
        if pupdata is not None:
            self.pupdata = pupdata  # Property set
            self.pup_idx  = self._pupdata.ind_pup.flatten().astype(self.xp.int64)
            self.pup_idx0 = self._pupdata.ind_pup[:, 0]
            self.pup_idx1 = self._pupdata.ind_pup[:, 1]
            self.pup_idx2 = self._pupdata.ind_pup[:, 2]
            self.pup_idx3 = self._pupdata.ind_pup[:, 3]
            self.n_pup = self._pupdata.ind_pup.shape[1]
            self.n_subap = self._pupdata.ind_pup.shape[0]

        if filtmat_tag:
            self.set_filtmat(self._cm.read_data(filtmat_tag))   # TODO

        self._total_counts = BaseValue()
        self._subap_counts = BaseValue()

    @property
    def shlike(self):
        return self._shlike

    @shlike.setter
    def shlike(self, value):
        self._shlike = value

    @property
    def norm_factor(self):
        return self._norm_factor

    @norm_factor.setter
    def norm_factor(self, value):
        self._norm_factor = value

    @property
    def thr_value(self):
        return self._thr_value

    @thr_value.setter
    def thr_value(self, value):
        self._thr_value = int(value)

    @property
    def slopes_from_intensity(self):
        return self._slopes_from_intensity

    @slopes_from_intensity.setter
    def slopes_from_intensity(self, value):
        self._slopes_from_intensity = value

    @property
    def pupdata(self):
        return self._pupdata

    @pupdata.setter
    def pupdata(self, p):
        if p is not None:
            self._pupdata = p
            # TODO replace this resize with an earlier initialization
            if self._slopes_from_intensity:
                self._slopes.resize(len(self._pupdata.ind_pup) * 4)
            else:
                self._slopes.resize(len(self._pupdata.ind_pup) * 2)
            self._accumulated_slopes.resize(len(self._pupdata.ind_pup) * 2)

    def run_check(self, time_step, errmsg=''):
        self.prepare_trigger(0)
        #super().build_stream()
        if self._use_sn and not self._sn:
            errmsg += 'Slopes null are not valid'
        if self._weight_from_accumulated and self._accumulate:
            errmsg += 'weightFromAccumulated and accumulate must not be set together'
        if errmsg != '':
            print(errmsg)
        return not (self._weight_from_accumulated and self._accumulate) and self.local_inputs['in_pixels'] and self._slopes and ((not self._use_sn) or (self._use_sn and self._sn))

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.flat_pixels = self.local_inputs['in_pixels'].pixels.flatten()

    def trigger_code(self):
        # outpus:
        # total_counts : computed here
        # subap_counts : computed in post_trigger
        # slopes : computed here

#        if not self._pupdata:
#            return
#        if self._verbose:
#            print('Average pixel counts:', self.xp.sum(pixels) / len(self._pupdata.ind_pup))

        self._total_counts.value = self.xp.sum(self.flat_pixels[self.pup_idx])

        if self.threshold is not None:
            self.flat_pixels -= self.threshold

        clamp_generic_less(0,0,self.flat_pixels, xp=self.xp)
        A = self.flat_pixels[self.pup_idx0]
        B = self.flat_pixels[self.pup_idx1]
        C = self.flat_pixels[self.pup_idx2]
        D = self.flat_pixels[self.pup_idx3]

        # Compute total intensity
        self.total_intensity = self.xp.sum(self.flat_pixels[self.pup_idx])

        if self._slopes_from_intensity:
            inv_factor = self.total_intensity / (4 * self.n_subap)
            factor = 1.0 / inv_factor
            self.sx = factor * self.xp.concatenate([A, B])
            self.sy = factor * self.xp.concatenate([C, D])
        else:
            if self._norm_factor is not None:
                inv_factor = self._norm_factor
                factor = 1.0 / inv_factor
            elif not self._shlike:
                inv_factor = self.total_intensity /  self.n_subap
                factor = 1.0 / inv_factor
            else:
                inv_factor = self.xp.sum(self.flat_pixels[self.pup_idx])
                factor = 1.0 / inv_factor

            # self.sx = (A+B-C-D).astype(self.dtype) * factor
            # self.sy = (B+C-A-D).astype(self.dtype) * factor
            self.sx = (A+B-C-D) * factor
            self.sy = (B+C-A-D) * factor

        clamp_generic_more(0, 1, inv_factor, xp=self.xp)
        self.sx *= inv_factor
        self.sy *= inv_factor

        self._slopes.xslopes = self.sx
        self._slopes.yslopes = self.sy 

        
    def post_trigger(self):
        # super().post_trigger()
        self._subap_counts.value = self._total_counts.value / self._pupdata.n_subap
        self._total_counts.generation_time = self.current_time
        self._subap_counts.generation_time = self.current_time
        self._slopes.generation_time = self.current_time


    def run_check(self, time_step, errmsg=''):
        if self._shlike and self._slopes_from_intensity:
            errmsg += 'Both SHLIKE and SLOPES_FROM_INTENSITY parameters are set. Only one of these should be used.'
            return False

        if self._shlike and self._norm_factor != 0:
            errmsg += 'Both SHLIKE and NORM_FACTOR parameters are set. Only one of these should be used.'
            return False

        if not self._pupdata:
            errmsg += 'Pupil data is not valid'
            return False

        return super().run_check(time_step, errmsg=errmsg)

