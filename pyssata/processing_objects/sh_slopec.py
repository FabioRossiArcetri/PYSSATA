from pyssata.processing_objects.slopec import Slopec

class ShSlopec(Slopec):
    def __init__(self                 
                 target_device_idx: int = None, 
                 precision: int = None ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self._thr_value = -1
        self._exp_weight = 1.0
        self._thr_mask_cube = BaseValue()  
        self._total_counts = BaseValue()
        self._subap_counts = BaseValue()
        self._detailed_debug = -1
        self._subapdata = None
        self._subap_idx = None
        self._xweights = None
        self._yweights = None
        self._xcweights = None
        self._ycweights = None
        self._mask_weighted = None
        self._corr_template = None
        self._winMatWindowed = None
        self._vecWeiPixRadT = None
        self._weightedPixRad = 0.0
        self._windowing = False
        self._correlation = False
        self._corrWindowSidePix = 0
        self._thr_ratio_value = 0.0
        self._thr_pedestal = False
        self._mult_factor = 0.0
        self._quadcell_mode = False
        self._two_steps_cog = False
        self._cog_2ndstep_size = 0
        self._dotemplate = False
        self._store_thr_mask_cube = False
        self._window = 0
        self._wsize = self.xp.zeros(2, dtype=int)
        self._display2s = False

    @property
    def thr_value(self):
        return self._thr_value

    @thr_value.setter
    def thr_value(self, value):
        self._thr_value = value

    @property
    def exp_weight(self):
        return self._exp_weight

    @exp_weight.setter
    def exp_weight(self, value):
        self._exp_weight = value
        self.set_xy_weights()

    def set_xy_weights(self):
        if self._subapdata:
            out = self.compute_xy_weights(self._subapdata.np_sub, self._exp_weight, self._weightedPixRad, 
                                          self._quadcell_mode, self._windowing)
            self._mask_weighted = self.xp.array(out['mask_weighted'], copy=False)
            self._xweights = self.xp.array(out['x'], copy=False)
            self._yweights = self.xp.array(out['y'], copy=False)
            self._xcweights = self.xp.array(out['xc'], copy=False)
            self._ycweights = self.xp.array(out['yc'], copy=False)

    def compute_xy_weights(self, np_sub, exp_weight, weighted_pix_rad, quadcell_mode=False, windowing=False):
        x, y = self.make_xy(np_sub, 1.0)
        if quadcell_mode:
            x[x > 0] = 1.0
            x[x < 0] = -1.0
            y[y > 0] = 1.0
            y[y < 0] = -1.0
            xc, yc = x, y
        else:
            xc, yc = x, y
            x[x > 0] = self.xp.power(x[x > 0], exp_weight)
            x[x < 0] = -self.xp.abs(x[x < 0]) ** exp_weight
            y[y > 0] = self.xp.power(y[y > 0], exp_weight)
            y[y < 0] = -self.xp.abs(y[y < 0]) ** exp_weight

        if weighted_pix_rad != 0:
            if windowing:
                mask_weighted = self.make_mask(np_sub, diaratio=(2 * weighted_pix_rad / np_sub))
            else:
                mask_weighted = self.psf_gaussian(np_sub, fwhm=2 * [weighted_pix_rad, weighted_pix_rad])
                mask_weighted /= self.xp.max(mask_weighted)
            x *= mask_weighted
            y *= mask_weighted
        else:
            mask_weighted = self.xp.ones((np_sub, np_sub))

        return {'x': x, 'y': y, 'xc': xc, 'yc': yc, 'mask_weighted': mask_weighted}

    def load_subapdata(self, subapdata_tag):
        p = self._cm.read_subaps(subapdata_tag)
        if p is not None:
            self._subapdata = p
            self._slopes = Slopes(self._subapdata.n_subap * 2)
            self._accumulated_slopes = Slopes(self._subapdata.n_subap * 2)
            self._slopes.pupdata_tag = subapdata_tag

            n_subaps = self._subapdata.n_subaps
            subap_idx = self.xp.array([self._subapdata.subap_idx(i) for i in range(n_subaps)])
            self._subap_idx = self.xp.copy(subap_idx)

            self.set_xy_weights()
        else:
            print(f'subapdata_tag: {subapdata_tag} is not valid')

    # Method to handle property settings, replacing setproperty from IDL
    def set_property(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            if key == 'exp_weight':
                self.set_xy_weights()
            if key == 'subapdata_tag':
                self.load_subapdata(value)

    def calc_slopes(self, t, accumulated=False):
        if self._vecWeiPixRadT is not None:
            time = self.t_to_seconds(t)
            idxW = self.xp.where(time > self._vecWeiPixRadT[:, 1])[-1]
            if len(idxW) > 0:
                self._weightedPixRad = self._vecWeiPixRadT[idxW, 0]
                if self._verbose:
                    print(f'self._weightedPixRad: {self._weightedPixRad}')
                self.set_xy_weights()

        if self._dotemplate or self._correlation or self._two_steps_cog or self._detailed_debug > 0:
            self.calc_slopes_for(t, accumulated=accumulated)
        else:
            self.calc_slopes_nofor(t, accumulated=accumulated)

    # Methods for slope computation (For-loop and No-For-loop approaches)
    def calc_slopes_for(self, t, accumulated=False):
        pass

    def calc_slopes_nofor(self, t, accumulated=False):
        pass

    # Cleanup method, analogous to IDL cleanup
    def cleanup(self):
        if self._subapdata:
            self._subapdata.cleanup()
        self._thr_mask_cube = None
        self._subap_idx = None
        self._xweights = None
        self._yweights = None
        self._xcweights = None
        self._ycweights = None
        self._mask_weighted = None
        self._corr_template = None
        self._winMatWindowed = None
        self._vecWeiPixRadT = None
        super().cleanup()

    @staticmethod
    def make_xy(np_sub, scale=1.0):
        grid = self.xp.linspace(-scale / 2, scale / 2, np_sub)
        x, y = self.xp.meshgrid(grid, grid)
        return x, y

    @staticmethod
    def make_mask(np_sub, diaratio):
        center = np_sub / 2
        y, x = self.xp.ogrid[:np_sub, :np_sub]
        dist_from_center = self.xp.sqrt((x - center) ** 2 + (y - center) ** 2)
        mask = dist_from_center <= (diaratio * center)
        return mask

    @staticmethod
    def psf_gaussian(np_sub, fwhm):
        x = self.xp.linspace(-1, 1, np_sub)
        y = self.xp.linspace(-1, 1, np_sub)
        x, y = self.xp.meshgrid(x, y)
        gaussian = self.xp.exp(-4 * self.xp.log(2) * (x ** 2 + y ** 2) / fwhm[0] ** 2)
        return gaussian

# Additional helper classes to mimic behavior in IDL such as Slopes, BaseValue, etc.
class Slopes:
    def __init__(self, size):
        self.slopes = self.xp.zeros(size)
        self.generation_time = None

class BaseValue:
    def __init__(self):
        self.value = 0
        self.generation_time = None
