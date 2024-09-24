from astropy.io import fits

from pyssata.data_objects.base_data_obj import BaseDataObj

class ElectricField(BaseDataObj):
    '''Electric field'''

    def __init__(self, dimx, dimy, pixel_pitch, target_device_idx=None, precision=None):
        super().__init__(precision=precision, target_device_idx=target_device_idx)

        dimx = int(dimx)
        dimy = int(dimy)
        self.pixel_pitch = pixel_pitch        
        self._S0 = 0.0

        self._A = self.xp.ones((dimx, dimy), dtype=self.dtype)
        self._phaseInNm = self.xp.zeros((dimx, dimy), dtype=self.dtype)

    def reset(self):
        self._A = self.xp.ones_like(self._A)
        self._phaseInNm = self.xp.zeros_like(self._phaseInNm)

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, new_A):
        self._A = new_A

    @property
    def phaseInNm(self):
        return self._phaseInNm

    @phaseInNm.setter
    def phaseInNm(self, new_phaseInNm):
        self._phaseInNm = new_phaseInNm

    @property
    def pixel_pitch(self):
        return self._pixel_pitch

    @pixel_pitch.setter
    def pixel_pitch(self, new_pixel_pitch):
        self._pixel_pitch = new_pixel_pitch

    @property
    def S0(self):
        return self._S0

    @S0.setter
    def S0(self, new_S0):
        self._S0 = new_S0
 
    @property
    def size(self):
        return self._A.shape

    def checkOther(self, ef2, subrect=None):
        if not isinstance(ef2, ElectricField):
            raise ValueError(f'{ef2} is not an ElectricField instance')
        if subrect is None:
            subrect = [0, 0]
        sz1 = self.xp.array(self.size) - self.xp.array(subrect)
        sz2 = self.xp.array(ef2.size)
        if any(sz1 != sz2):
            raise ValueError(f'{ef2} has size {sz2} instead of the required {sz1}')
        return subrect
        
    def phi_at_lambda(self, wavelengthInNm):
        return self._phaseInNm * ((2 * self.xp.pi) / wavelengthInNm)

    def ef_at_lambda(self, wavelengthInNm):
        phi = self.phi_at_lambda(wavelengthInNm)
        return self._A * self.xp.exp(1j * phi, dtype=self.complex_dtype)

    def product(self, ef2, subrect=None):
        subrect = self.checkOther(ef2, subrect=subrect)
        x2 = subrect[0] + self.size[0]
        y2 = subrect[1] + self.size[1]
        self._A *= ef2._A[subrect[0] : x2, subrect[1] : y2]
        self._phaseInNm += ef2._phaseInNm[subrect[0] : x2, subrect[1] : y2]

    def area(self):
        return self._A.size * (self.pixel_pitch ** 2)

    def masked_area(self):
        tot = self.xp.sum(self._A)
        return (self.pixel_pitch ** 2) * tot

    def square_modulus(self, wavelengthInNm):
        ef = self.ef_at_lambda(wavelengthInNm)
        return self.xp.abs(ef) ** 2

    def sub_ef(self, xfrom, xto, yfrom, yto, idx=None):
        if idx is not None:
            idx = self.xp.unravel_index(idx, self._A.shape)
            xfrom, xto = self.xp.min(idx[0]), self.xp.max(idx[0])
            yfrom, yto = self.xp.min(idx[1]), self.xp.max(idx[1])
        sub_ef = ElectricField(xto - xfrom + 1, yto - yfrom + 1, self.pixel_pitch)
        sub_ef.A = self._A[xfrom:xto+1, yfrom:yto+1]
        sub_ef.phaseInNm = self._phaseInNm[xfrom:xto+1, yfrom:yto+1]
        sub_ef.S0 = self._s0
        return sub_ef

    def compare(self, ef2):
        return not (self.xp.array_equal(self._A, ef2._A) and self.xp.array_equal(self._phaseInNm, ef2._phaseInNm))

    def save(self, filename):
        A = self._A
        phaseInNm = self._phaseInNm
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['DIMX'] = A.shape[0]
        hdr['DIMY'] = A.shape[1]
        hdr['PIXPITCH'] = self.pixel_pitch
        hdr['S0'] = self._S0

        hdu_A = fits.PrimaryHDU(A, header=hdr)
        hdu_phase = fits.ImageHDU(phaseInNm)
        hdul = fits.HDUList([hdu_A, hdu_phase])
        hdul.writeto(filename, overwrite=True)

    @staticmethod
    def restore(filename):
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            version = hdr['VERSION']
            if version != 1:
                raise ValueError(f"Error: unknown version {version} in file {filename}")
            dimx = hdr['DIMX']
            dimy = hdr['DIMY']
            pitch = hdr['PIXPITCH']
            S0 = hdr['S0']

            ef = ElectricField(dimx, dimy, pitch)
            ef.set_property(A=hdul[0].data, phaseInNm=hdul[1].data, S0=S0)
            return ef

