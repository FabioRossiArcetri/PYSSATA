from astropy.io import fits

from specula.base_data_obj import BaseDataObj


class Pixels(BaseDataObj):
    '''Pixels'''

    def __init__(self, dimx, dimy, bits=16, signed=0, target_device_idx=None, precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self._validate_bits(bits)
        self._signed = signed
        self._type = self._get_type(bits, signed)
        self._pixels = self.xp.zeros((dimx, dimy), dtype=self.dtype)
        self._bpp = bits
        self._bytespp = (bits + 7) // 8  # bits self.xp.arounded to the next multiple of 8

    def _validate_bits(self, bits):
        if bits > 64:
            raise ValueError("Cannot create pixel object with more than 64 bits per pixel")

    def _get_type(self, bits, signed):
        type_matrix = [
            [self.xp.uint8, self.xp.int8],
            [self.xp.uint16, self.xp.int16],
            [self.xp.uint32, self.xp.int32],
            [self.xp.uint32, self.xp.int32],
            [self.xp.uint64, self.xp.int64],
            [self.xp.uint64, self.xp.int64],
            [self.xp.uint64, self.xp.int64],
            [self.xp.uint64, self.xp.int64]
        ]
        return type_matrix[(bits - 1) // 8][signed]

    def set_value(self, v):
        self._pixels = v

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
        self._pixels = self.xp.zeros(size, dtype=self.dtype)

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'Pixels'
        hdr['TYPE'] = str(self._type)
        hdr['BPP'] = self._bpp
        hdr['BYTESPP'] = self._bytespp
        hdr['SIGNED'] = self._signed
        hdr['DIMX'] = self.size[0]
        hdr['DIMY'] = self.size[1]
        return hdr

    def save(self, filename):
        hdr = self.get_fits_header()            
        super().save(filename, hdr)
        fits.append(filename, self._pixels, hdr)

    def read(self, filename, hdr=None, exten=0):
        super().read(filename, hdr, exten)
        self._pixels = fits.getdata(filename, ext=exten)


    @staticmethod
    def from_header(hdr):    
        version = hdr['VERSION']
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in file {filename}")
        dimx = hdr['DIMX']
        dimy = hdr['DIMY']
        bits = hdr['BPP']
        signed = hdr['SIGNED']

        pixels = Pixels(dimx, dimy, bits=bits, signed=signed)
        return pixels

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


