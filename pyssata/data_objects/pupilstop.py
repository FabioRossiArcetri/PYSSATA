import numpy as np

from pyssata import xp

from astropy.io import fits

from pyssata.data_objects.layer import Layer
from pyssata.lib.make_mask import make_mask


class Pupilstop(Layer):
    '''Pupil stop'''

    def __init__(self,
                 pixel_pupil: int,
                 pixel_pitch: float,
                 input_mask: xp.ndarray=None,
                 mask_diam: float=1.0,
                 obs_diam: float=None,
                 shiftXYinPixel=(0.0, 0.0),
                 rotInDeg: float=0.0,
                 magnification: float=1.0,
                 precision: int=None):

        super().__init__(pixel_pupil, pixel_pupil, pixel_pitch, height=0, precision=precision,
                         shiftXYinPixel=shiftXYinPixel, rotInDeg=rotInDeg, magnification=magnification)

        self._input_mask = input_mask
        self._mask_diam = mask_diam
        self._obs_diam = obs_diam

        if input_mask is not None:
            mask_amp = input_mask
        else:
            mask_amp = make_mask(pixel_pupil, obs_diam, mask_diam)

        self.A = mask_amp

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

