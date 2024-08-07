import numpy as np
from astropy.io import fits

class Pixels(BaseDataObj):
    def __init__(self, dimx, dimy, bits=16, signed=0):
        self._validate_bits(bits)
        self._signed = signed
        self._type = self._get_type(bits, signed)
        self._pixels = np.zeros((dimx, dimy), dtype=self._type)
        self._bpp = bits
        self._bytespp = (bits + 7) // 8  # bits rounded to the next multiple of 8

        super().__init__("pixels", "Pixels object")

    def _validate_bits(self, bits):
        if bits > 64:
            raise ValueError("Cannot create pixel object with more than 64 bits per pixel")

    def _get_type(self, bits, signed):
        type_matrix = [
            [np.uint8, np.int8],
            [np.uint16, np.int16],
            [np.uint32, np.int32],
            [np.uint32, np.int32],
            [np.uint64, np.int64],
            [np.uint64, np.int64],
            [np.uint64, np.int64],
            [np.uint64, np.int64]
        ]
        return type_matrix[(bits - 1) // 8][signed]

    @property
    def pixels(self):
        return self._pixels

    @pixels.setter
    def pixels(self, value):
        self._pixels = value

    @property
    def type(self):
        return self._type

    @property
    def bpp(self):
        return self._bpp

    @property
    def bytespp(self):
        return self._bytespp

    @property
    def signed(self):
        return self._signed

    @property
    def size(self):
        return self._pixels.shape

    def multiply(self, factor):
        self._pixels *= factor

    def set_size(self, size):
        self._pixels = np.zeros(size, dtype=self._type)

    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['TYPE'] = self._type.name
        hdr['BPP'] = self._bpp
        hdr['BYTESPP'] = self._bytespp
        hdr['SIGNED'] = self._signed
        hdr['DIMX'] = self.size[0]
        hdr['DIMY'] = self.size[1]

        super().save(filename, hdr)
        fits.append(filename, self._pixels, hdr)

    def read(self, filename, hdr=None, exten=0):
        super().read(filename, hdr, exten)
        self._pixels = fits.getdata(filename, ext=exten)

    @staticmethod
    def restore(filename):
        hdr = fits.getheader(filename)
        version = hdr['VERSION']

        if version != 1:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        dimx = hdr['DIMX']
        dimy = hdr['DIMY']
        bits = hdr['BPP']
        signed = hdr['SIGNED']

        pixels = Pixels(dimx, dimy, bits=bits, signed=signed)
        pixels.read(filename, hdr)
        return pixels

    def revision_track(self):
        return '$Rev$'

    def cleanup(self):
        self._pixels = None
        super().cleanup()

# Example definition of BaseDataObj if not available.
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
