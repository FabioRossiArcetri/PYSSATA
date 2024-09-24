from astropy.io import fits

from pyssata.data_objects.ef import ElectricField

class Layer(ElectricField):
    '''Layer'''

    def __init__(self, dimx, dimy, pixel_pitch, height, target_device_idx=None, precision=None, shiftXYinPixel=(0.0, 0.0), rotInDeg=0.0, magnification=1.0):
        super().__init__(dimx, dimy, pixel_pitch, target_device_idx=target_device_idx, precision=precision)
        self._height = height
        self._shiftXYinPixel = self.xp.array(shiftXYinPixel, dtype=self.dtype)
        self._rotInDeg = rotInDeg
        self._magnification = magnification

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def shiftXYinPixel(self):
        return self._shiftXYinPixel

    @shiftXYinPixel.setter
    def shiftXYinPixel(self, value):
        self._shiftXYinPixel = value

    @property
    def rotInDeg(self):
        return self._rotInDeg

    @rotInDeg.setter
    def rotInDeg(self, value):
        self._rotInDeg = value

    @property
    def magnification(self):
        return self._magnification

    @magnification.setter
    def magnification(self, value):
        self._magnification = value

    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = fits.Header()
        hdr['HEIGHT'] = self._height
        super().save(filename, hdr)

    def read(self, filename, hdr=None, exten=0):
        super().read(filename, hdr, exten)

    @staticmethod
    def restore(filename):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])

        if version != 1:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        dimx = int(hdr['DIMX'])
        dimy = int(hdr['DIMY'])
        height = float(hdr['HEIGHT'])
        pitch = float(hdr['PIXPITCH'])

        layer = Layer(dimx, dimy, pitch, height)
        layer.read(filename, hdr)
        return layer

