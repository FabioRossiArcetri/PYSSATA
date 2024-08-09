import numpy as np
from astropy.io import fits

from pyssata.data_objects.base_data_obj import BaseDataObj

class ElectricField(BaseDataObj):
    '''Electric field'''

    def __init__(self, dimx, dimy, pixel_pitch, precision=0, dtype=None):
        super().__init__(precision)

        dimx = int(dimx)
        dimy = int(dimy)
        self.pixel_pitch = pixel_pitch
        self.precision = precision
        self.dtype = dtype if dtype else np.float32 if precision == 0 else np.float64
        self._S0 = 0.0

        self._A = np.ones((dimx, dimy), dtype=self.dtype)
        self._phaseInNm = np.zeros((dimx, dimy), dtype=self.dtype)

    def reset(self):
        self._A = np.ones_like(self._A)
        self._phaseInNm = np.zeros_like(self._phaseInNm)

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

    def phi_at_lambda(self, wavelengthInNm):
        return self._phaseInNm * ((2 * np.pi) / wavelengthInNm)

    def ef_at_lambda(self, wavelengthInNm):
        phi = self.phi_at_lambda(wavelengthInNm)
        return self._A * np.exp(1j * phi)

    def product(self, ef2):
        self._A *= ef2._A
        self._phaseInNm += ef2._phaseInNm

    def area(self):
        return self._A.size * (self.pixel_pitch ** 2)

    def masked_area(self):
        tot = np.sum(self._A)
        return (self.pixel_pitch ** 2) * tot

    def square_modulus(self, wavelengthInNm):
        ef = self.ef_at_lambda(wavelengthInNm)
        return np.abs(ef) ** 2

    def copy_to(self, ef2):
        ef2.set_property(A=self._A, phaseInNm=self._phaseInNm, S0=self._S0, pixel_pitch=self.pixel_pitch)

    def sub_ef(self, xfrom, xto, yfrom, yto, idx=None):
        if idx is not None:
            idx = np.unravel_index(idx, self._A.shape)
            xfrom, xto = np.min(idx[0]), np.max(idx[0])
            yfrom, yto = np.min(idx[1]), np.max(idx[1])
        sub_ef = ElectricField(xto - xfrom + 1, yto - yfrom + 1, self.pixel_pitch)
        sub_ef.A = self._A[xfrom:xto+1, yfrom:yto+1]
        sub_ef.phaseInNm = self._phaseInNm[xfrom:xto+1, yfrom:yto+1]
        sub_ef.S0 = self._s0
        return sub_ef

    def compare(self, ef2):
        return not (np.array_equal(self._A, ef2._A) and np.array_equal(self._phaseInNm, ef2._phaseInNm))

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

    def cleanup(self):
        self._A = None
        self._phaseInNm = None
        self._S0 = 0.0
        self.pixel_pitch = 0.0

    def revision_track(self):
        return '$Rev$'
