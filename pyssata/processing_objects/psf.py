class PSF:
    def __init__(self, wavelengthInNm, nd=1):
        self._wavelengthInNm = wavelengthInNm
        self._nd = nd

        self._sr = BaseValue()
        self._int_sr = BaseValue()
        self._psf = BaseValue()
        self._int_psf = BaseValue()
        self._in_ef = None
        self._ref = None
        self._intsr = 0.0
        self._count = 0
        self._start_time = 0.0

        self.reset_integration()

    @property
    def in_ef(self):
        return self._in_ef

    @in_ef.setter
    def in_ef(self, in_ef):
        self._in_ef = in_ef
        s = [dim * self._nd for dim in in_ef.size]
        self._psf.value = np.zeros(s)
        self._int_psf.value = np.zeros(s)
        self._ref = None

    @property
    def wavelengthInNm(self):
        return self._wavelengthInNm

    @wavelengthInNm.setter
    def wavelengthInNm(self, wavelengthInNm):
        self._wavelengthInNm = wavelengthInNm

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, start_time):
        self._start_time = start_time

    @property
    def out_psf(self):
        return self._psf

    @property
    def out_int_psf(self):
        return self._int_psf

    @property
    def out_sr(self):
        return self._sr

    @property
    def out_int_sr(self):
        return self._int_sr

    @property
    def nd(self):
        return self._nd

    @nd.setter
    def nd(self, nd):
        self._nd = nd

    @property
    def size(self):
        return self._in_ef.size if self._in_ef else None

    @property
    def out_ref(self):
        return self._ref

    @property
    def out_count(self):
        return self._count

    def run_check(self, time_step, errmsg=''):
        if not self._in_ef:
            errmsg += ' Input intensity object has not been set'
        if self._wavelengthInNm == 0:
            errmsg += ' PSF wavelength is zero'
        return bool(self._in_ef) and (self._wavelengthInNm > 0)

    def reset_integration(self):
        self._count = 0
        if self._in_ef:
            self._int_psf.value *= 0
        self._intsr = 0

    def trigger(self, t):
        if self._in_ef and self._in_ef.generation_time == t:
            if self.t_to_seconds(t) >= self._start_time:
                self._count += 1

            s = [round(dim * self._nd) for dim in self._in_ef.size]

            if not self._ref:
                self._ref = I(s[0], s[1])
                self._ref.i = calc_psf(np.zeros((s[0], s[1])), self._in_ef.A, imwidth=s[0], norm=True)

            self._psf.value = calc_psf(self._in_ef.phi_at_lambda(self._wavelengthInNm), self._in_ef.A, imwidth=s[0], norm=True)
            if self.t_to_seconds(t) >= self._start_time:
                self._int_psf.value += self._psf.value

            self._sr.value = self._psf.value[s[0] // 2, s[1] // 2] / self._ref.i[s[0] // 2, s[1] // 2]
            if self.t_to_seconds(t) >= self._start_time:
                self._intsr += self._sr.value
                self._int_sr.value = self._intsr / self._count

            self._psf.generation_time = t
            if self.t_to_seconds(t) >= self._start_time:
                self._int_psf.generation_time = t
            self._sr.generation_time = t
            if self.t_to_seconds(t) >= self._start_time:
                self._int_sr.generation_time = t

    @staticmethod
    def revision_track():
        return '$Rev$'

    def cleanup(self):
        pass
