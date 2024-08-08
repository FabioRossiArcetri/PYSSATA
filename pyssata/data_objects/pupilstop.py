import numpy as np
from astropy.io import fits

from pyssata.data_objects.layer import Layer

class PupilStop(Layer):
    def __init__(self, dimx, dimy, pixel_pitch, height, GPU=False, input_mask=None, mask_diam=None, obs_diam=None, objname="pupilstop", objdescr="pupilstop object", PRECISION=0, TYPE=None):
        self._pixel_pitch = pixel_pitch
        self._height = height
        self._GPU = GPU
        self._input_mask = input_mask
        self._mask_diam = mask_diam
        self._obs_diam = obs_diam

        if not super().__init__(dimx, dimy, pixel_pitch, height, GPU=GPU, objname=objname + ' layer', objdescr=objname + ' layer object', PRECISION=PRECISION, TYPE=TYPE):
            return

        if input_mask is not None:
            self._mask_amp = input_mask
        else:
            self._mask_amp = self.make_mask(dimx, mask_diam, obs_diam)

        self.A = np.float32(self._mask_amp)
        
        if not BaseDataObj.__init__(self, objname, objdescr):
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
    def restore(filename, GPU=False):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])

        if version != 1:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        input_mask = fits.getdata(filename, ext=1)
        dim = fits.getdata(filename, ext=2)
        pixel_pitch = fits.getdata(filename, ext=3)[0]

        pupilstop = PupilStop(dim[0], dim[1], pixel_pitch, 0, input_mask=input_mask, GPU=GPU)
        return pupilstop

    def revision_track(self):
        return '$Rev$'

    def cleanup(self):
        self._A = None
        super().cleanup()

# Example definition of Layer and BaseDataObj if not available.
class BaseDataObj:
    def __init__(self, objname, objdescr):
        self.objname = objname
        self.objdescr = objdescr

    def save(self, filename, hdr):
        pass

    def read(self, filename, hdr=None, exten=0):
        pass

    def cleanup(self):
        pass

class Layer(BaseDataObj):
    def __init__(self, dimx, dimy, pixel_pitch, height, GPU=False, objname="layer", objdescr="layer object", PRECISION=0, TYPE=None):
        self._dimx = dimx
        self._dimy = dimy
        self._pixel_pitch = pixel_pitch
        self._height = height
        self._GPU = GPU
        self._PRECISION = PRECISION
        self._TYPE = TYPE
        super().__init__(objname, objdescr)
        # Additional initialization for Layer

    def save(self, filename, hdr=None):
        super().save(filename, hdr)
        # Additional save logic for Layer

    def read(self, filename, hdr=None, exten=0):
        super().read(filename, hdr, exten)
        # Additional read logic for Layer

    def cleanup(self):
        super().cleanup()
        # Additional cleanup logic for Layer
