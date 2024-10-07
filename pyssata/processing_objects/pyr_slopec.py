import numpy as np
from pyssata import cpuArray
from pyssata.processing_objects.slopec import Slopec
from pyssata.data_objects.slopes import Slopes
from pyssata.lib.pyr_compute_slopes import pyr_compute_slopes
from  pyssata.base_value import BaseValue
from pyssata.data_objects.pupdata import PupData

    
class PyrSlopec(Slopec):
    def __init__(self, pupdata: PupData=None, shlike=False, norm_factor=None, thr_value=0, slopes_from_intensity=False, filtmat_tag='', 
                 target_device_idx=None, 
                 precision=None,
                **kwargs):
        super().__init__(target_device_idx=target_device_idx, precision=precision, **kwargs)
        self._shlike = shlike
        self._norm_factor = norm_factor
        self._thr_value = int(thr_value)
        self._slopes_from_intensity = slopes_from_intensity
        if pupdata is not None:
            self.pupdata = pupdata  # Property set
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

    def calc_slopes(self):
        if not self._pupdata:
            return

        pixels = self.inputs['in_pixels'].get(self._target_device_idx).pixels

        if self._verbose:
            print('Average pixel counts:', self.xp.sum(pixels) / len(self._pupdata.ind_pup))

        threshold = self._thr_value if self._thr_value != -1 else None
        sx, sy, flux = pyr_compute_slopes(pixels, self._pupdata.ind_pup, self._shlike, self._slopes_from_intensity, self._norm_factor, threshold, xp=self.xp)

        self._flux_per_subaperture_vector.value = flux
        self._flux_per_subaperture_vector.generation_time = self.current_time

        idx  = self._pupdata.ind_pup.flatten().astype(self.xp.int64)
        v = pixels.flatten()
        px = v[idx].ravel()
        self._total_counts.value = self.xp.sum(px)
        self._subap_counts.value = self.xp.sum(px) / self._pupdata.n_subap

        if self._slopes_from_intensity:
            self._slopes.slopes = [sx, sy]
        else:
            self._slopes.xslopes = sx
            self._slopes.yslopes = sy
        self._slopes.generation_time = self.current_time
        self._total_counts.generation_time = self.current_time
        self._subap_counts.generation_time = self.current_time

#        if 1:#if self._verbose:  # Verbose?
#            print(f'Slopes min, max and rms: {self.xp.min(self.xp.array([sx, sy]))}, {self.xp.max(self.xp.array([sx, sy]))}  //  {self.xp.sqrt(self.xp.mean(self.xp.array([sx**2, sy**2])))}')

    def _compute_flux_per_subaperture(self):
        return self._flux_per_subaperture_vector

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

