import numpy as np

from pyssata.base_parameter_obj import BaseParameterObj
from pyssata.data_objects.base_data_obj import BaseDataObj

class Source(BaseDataObj, BaseParameterObj):
    '''source'''

    def __init__(self, polar_coordinate, height, magnitude, wavelengthInNm, band='', zeroPoint=0):
        self._polar_coordinate = np.array(polar_coordinate)
        self._height = height
        self._magnitude = magnitude
        self._wavelengthInNm = wavelengthInNm
        self._zeroPoint = zeroPoint
        self._band = band
        self._verbose = False

        if not super().__init__():
            return

    @property
    def polar_coordinate(self):
        return self._polar_coordinate

    @polar_coordinate.setter
    def polar_coordinate(self, value):
        self._polar_coordinate = np.array(value)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def magnitude(self):
        return self._magnitude

    @magnitude.setter
    def magnitude(self, value):
        self._magnitude = value

    @property
    def wavelengthInNm(self):
        return self._wavelengthInNm

    @wavelengthInNm.setter
    def wavelengthInNm(self, value):
        self._wavelengthInNm = value

    @property
    def x_coord(self):
        alpha = self._polar_coordinate[0] * 4.848e-6
        d = self._height * np.sin(alpha)
        return np.cos(np.radians(self._polar_coordinate[1])) * d

    @property
    def y_coord(self):
        alpha = self._polar_coordinate[0] * 4.848e-6
        d = self._height * np.sin(alpha)
        return np.sin(np.radians(self._polar_coordinate[1])) * d

    @property
    def band(self):
        return self._band

    @band.setter
    def band(self, value):
        self._band = value

    @property
    def zeroPoint(self):
        return self._zeroPoint

    @zeroPoint.setter
    def zeroPoint(self, value):
        self._zeroPoint = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    def phot_density(self):
        if self._zeroPoint > 0:
            e0 = self._zeroPoint
        else:
            e0 = None
        if self._band:
            band = self._band
        else:
            band = None

        res = n_phot(self._magnitude, band=band, LAMBDA=self._wavelengthInNm/1e9, width=1e-9, e0=e0)
        if self._verbose:
            print(f'source.phot_density: magnitude is {self._magnitude}, and flux (output of n_phot with width=1e-9, surf=1) is {res[0]}')
        return res[0]

    def revision_track(self):
        return '$Rev$'

    def cleanup(self):
        pass

# Placeholder for n_phot function
def n_phot(magnitude, band=None, LAMBDA=None, width=None, e0=None):
    # Dummy implementation
    return [0.0]
