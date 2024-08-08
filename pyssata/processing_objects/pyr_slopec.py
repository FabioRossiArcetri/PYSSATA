import numpy as np
from dataclasses import dataclass, field

from pyssata.processing_objects.slopec import Slopec
from pyssata.data_objects.slopes import Slopes

@dataclass
class PyrSlopec(Slopec):
    _pupdata: any = field(default_factory=lambda: None)
    _shlike: bool = False
    _slopes_from_intensity: bool = False
    _norm_factor: float = 0.0
    _thr_value: float = -1

    def __post_init__(self):
        super().__post_init__()

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
    def pupdata_tag(self):
        return self._pupdata_tag

    @pupdata_tag.setter
    def pupdata_tag(self, value):
        self.load_pupdata(value)

    @property
    def thr_value(self):
        return self._thr_value

    @thr_value.setter
    def thr_value(self, value):
        self._thr_value = value

    @property
    def slopes_from_intensity(self):
        return self._slopes_from_intensity

    @slopes_from_intensity.setter
    def slopes_from_intensity(self, value):
        self._slopes_from_intensity = value

    def load_pupdata(self, pupdata_tag):
        p = self._cm.read_pupils(pupdata_tag)
        if p is not None:
            self._pupdata = p
            if self._slopes_from_intensity:
                self._slopes = Slopes(len(self._pupdata.ind_pup))
            else:
                self._slopes = Slopes(len(self._pupdata.ind_pup) // 2)
            self._accumulated_slopes = Slopes(len(self._pupdata.ind_pup) // 2)
            self._slopes.pupdata_tag = pupdata_tag

    def calc_slopes(self, t, accumulated=False):
        if not self._pupdata:
            return

        pixels = self._accumulated_pixels.pixels if accumulated else self._pixels.pixels

        if self._verbose:
            print('Average pixel counts:', np.sum(pixels) / len(self._pupdata.ind_pup))

        threshold = self._thr_value if self._thr_value != -1 else None
        sx, sy, flux = pyr_compute_slopes(pixels, self._pupdata.ind_pup, self._shlike, self._slopes_from_intensity, self._norm_factor, threshold)

        if accumulated:
            if self._slopes_from_intensity:
                self._accumulated_slopes.slopes = [sx, sy]
            else:
                self._accumulated_slopes.xslopes = sx
                self._accumulated_slopes.yslopes = sy
            self._accumulated_slopes.generation_time = t
        else:
            self._flux_per_subaperture_vector.value = flux
            self._flux_per_subaperture_vector.generation_time = t

            px = pixels[self._pupdata.ind_pup]
            self._total_counts.value = np.sum(px)
            self._subap_counts.value = np.sum(px) / self._pupdata.n_subap

            if self._slopes_from_intensity:
                self._slopes.slopes = [sx, sy]
            else:
                self._slopes.xslopes = sx
                self._slopes.yslopes = sy
            self._slopes.generation_time = t
            self._total_counts.generation_time = t
            self._subap_counts.generation_time = t

        if self._verbose:
            print(f'Slopes min, max and rms: {np.min([sx, sy])}, {np.max([sx, sy])}  //  {np.sqrt(np.mean([sx, sy]**2))}')

    def _compute_flux_per_subaperture(self):
        return self._flux_per_subaperture_vector

    def revision_track(self):
        return '$Rev$'

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

    def cleanup(self):
        if self._pupdata:
            del self._pupdata
        super().cleanup()

    @staticmethod
    def pyr_compute_slopes(pixels, ind_pup, shlike, intensity_based, norm_factor, threshold):
        # Dummy implementation
        sx = np.random.random(len(ind_pup))
        sy = np.random.random(len(ind_pup))
        flux = np.random.random(len(ind_pup))
        return sx, sy, flux
