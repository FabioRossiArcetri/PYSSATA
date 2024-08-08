import numpy as np
from astropy.io import fits

from pyssata.data_objects.layer import Layer

class PupilStop(Layer):
    '''Pupil stop'''

    def __init__(self, dimx, dimy, pixel_pitch, height, input_mask=None, mask_diam=None, obs_diam=None, PRECISION=0, TYPE=None):
        self._pixel_pitch = pixel_pitch
        self._height = height
        self._input_mask = input_mask
        self._mask_diam = mask_diam
        self._obs_diam = obs_diam

        if not super().__init__(dimx, dimy, pixel_pitch, height, PRECISION=PRECISION, TYPE=TYPE):
            return

        if input_mask is not None:
            self._mask_amp = input_mask
        else:
            self._mask_amp = self.make_mask(dimx, mask_diam, obs_diam)

        self.A = np.float32(self._mask_amp)
        
        if not BaseDataObj.__init__(self):
            return

    def make_mask(self, dimx, diaratio, obsratio):
        # Implement the mask creation logic here
        pass

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        self._A = value

    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = fits.Header()
        hdr['VERSION'] = 1

        super().save(filename, hdr)

        fits.append(filename, self._A)
        fits.append(filename, self._A.shape)
        fits.append(filename, [self._pixel_pitch])

    @staticmethod
    def restore(filename):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])

        if version != 1:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        input_mask = fits.getdata(filename, ext=1)
        dim = fits.getdata(filename, ext=2)
        pixel_pitch = fits.getdata(filename, ext=3)[0]

        pupilstop = PupilStop(dim[0], dim[1], pixel_pitch, 0, input_mask=input_mask)
        return pupilstop

    def revision_track(self):
        return '$Rev$'

    def cleanup(self):
        self._A = None
        super().cleanup()
