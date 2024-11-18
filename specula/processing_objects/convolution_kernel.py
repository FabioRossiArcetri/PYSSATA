import numpy as np

class ConvolutionKernel(BaseProcessingObj):
    def __init__(self, cm=None, airmass=1.0):
        super().__init__('kernel', 'ConvolutionKernel processing object')
        
        self._kernels = BaseValue()
        self._lenslet = None
        self._ef = None
        self._seeing = None
        self._zlayer = None
        self._zprofile = None
        self._zfocus = 0.0
        self._theta = np.array([0.0, 0.0])
        self._last_zfocus = 0.0
        self._last_theta = np.array([0.0, 0.0])
        self._dimension = 0
        self._pxscale = 0.0
        self._recalc = True
        self._returnFft = False
        self._launcher_size = 0.0
        self._last_seeing = -1.0
        self._oversampling = 1
        self._launcher_pos = np.zeros(3)
        self._last_zlayer = -1
        self._last_zprofile = -1
        self._last_recalc = 0
        self._cm = cm
        self._airmass = airmass
        self._positiveShiftTT = False

    def set_property(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, f"_{key}", value)
        self._recalc = True

    def get_property(self, **kwargs):
        return {key: getattr(self, f"_{key}", None) for key in kwargs}

    def set_lenslet(self, lenslet):
        self._lenslet = lenslet
        self.make_grid()

    def set_ef(self, ef):
        self._ef = ef
        self.make_grid()

    def set_launcher_pos(self, launcher_pos):
        if len(launcher_pos) != 3:
            raise ValueError("Launcher position must be a three-elements vector [m]")
        self._launcher_pos = np.array(launcher_pos)
        self._recalc = True

    def set_dimension(self, dimension):
        if dimension != self._dimension:
            self._dimension = dimension
            self._recalc = True

    def set_launcher_size(self, launcher_size):
        if launcher_size != self._launcher_size:
            self._launcher_size = launcher_size
            self._recalc = True

    def set_pxscale(self, pxscale):
        if pxscale != self._pxscale:
            self._pxscale = pxscale
            self._recalc = True

    def set_oversampling(self, oversampling):
        if oversampling != self._oversampling:
            self._oversampling = oversampling
            self._recalc = True

    def set_positiveShiftTT(self, positiveShiftTT):
        if positiveShiftTT != self._positiveShiftTT:
            self._positiveShiftTT = positiveShiftTT
            self._recalc = True

    def set_returnFft(self, returnFft):
        if returnFft != self._returnFft:
            self._returnFft = returnFft
            self._recalc = True

    def make_grid(self):
        if self._lenslet is None or self._ef is None:
            return
        dimx = max(self._lenslet.dimx, 2)
        pixel_pitch = self._ef.pixel_pitch * (self._ef.size[0] / 2.0)
        x, y = self.generate_xy_grid(dimx, pixel_pitch)
        self._xgrid = x
        self._ygrid = y
        self._recalc = True

    def generate_xy_grid(self, dimx, pitch):
        x = np.linspace(-pitch, pitch, dimx)
        y = np.linspace(-pitch, pitch, dimx)
        return np.meshgrid(x, y)

    def trigger(self, t):
        if self._recalc:
            self.recalc()
            self._recalc = False
            self._last_recalc = t
        self._kernels.generation_time = t

    def recalc(self):
        if len(self._zlayer) != len(self._zprofile):
            raise ValueError("Number of elements of zlayer and zprofile must be the same")

        zfocus = self._zfocus if self._zfocus != -1 else self.calculate_focus()
        layHeights = np.array(self._zlayer) * self._airmass
        zfocus *= self._airmass

        pupil_size_m = self._ef.pixel_pitch * self._ef.size[0]
        spotsize = np.sqrt(self._seeing**2 + self._launcher_size**2)
        LGS_TT = (np.array([-0.5, -0.5]) if not self._positiveShiftTT else np.array([0.5, 0.5])) * self._pxscale + self._theta

        if self._cm is not None:
            arr = [self._lenslet.dimx, pupil_size_m, zfocus, spotsize, self._pxscale, self._dimension, self._oversampling, LGS_TT]
            sha1 = self.generate_hash(arr)
            kk = self._cm.read_kernel(sha1)
            if kk is None:
                kk = self.calculate_lgs_map(zfocus, layHeights)
                self._cm.write_kernel(sha1, kk)
            else:
                self._kernels.value = kk

    def calculate_focus(self):
        return np.sum(np.array(self._zlayer) * np.array(self._zprofile)) / np.sum(self._zprofile)

    def calculate_lgs_map(self, zfocus, layHeights):
        # Placeholder for actual LGS map calculation
        return np.random.rand(self._dimension, self._dimension, self._lenslet.dimx * self._lenslet.dimy)

    def generate_hash(self, arr):
        # Placeholder function to compute SHA1 hash
        import hashlib, json
        sha1 = hashlib.sha1()
        sha1.update(json.dumps(arr).encode('utf-8'))
        return sha1.hexdigest()

    def run_check(self, time_step):
        errmsg = []
        if not self._seeing:
            errmsg.append("ConvolutionKernel seeing object is not valid")
        if not self._zlayer:
            errmsg.append("ConvolutionKernel zlayer object is not valid")
        if not self._zprofile:
            errmsg.append("ConvolutionKernel zprofile object is not valid")
        if not self._lenslet:
            errmsg.append("ConvolutionKernel lenslet object is not valid")
        if not self._ef:
            errmsg.append("ConvolutionKernel ef object is not valid")
        return len(errmsg) == 0, errmsg

    def cleanup(self):
        pass

    def revision_track(self):
        return "$Rev$"
