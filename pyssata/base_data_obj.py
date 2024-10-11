from astropy.io import fits

from pyssata.base_time_obj import BaseTimeObj

def get_properties(cls):
    result = []
    classlist = cls.__mro__
    for cc in classlist:
        result.extend([attr for attr, value in vars(cc).items() if isinstance(value, property) ]) 
    return result
    # return [attr for attr, value in vars(cls).items() if isinstance(value, property) ]

class BaseDataObj(BaseTimeObj):
    def __init__(self, target_device_idx=None, precision=None):
        """
        Initialize the base data object.

        Parameters:
        precision (int, optional):if None will use the global_precision, otherwise pass 0 for double, 1 for single
        """
        super().__init__(target_device_idx, precision)
        self._generation_time = -1

    @property
    def generation_time(self):
        return self._generation_time

    @generation_time.setter
    def generation_time(self, value):
        self._generation_time = value

    def save(self, filename):
        hdr = fits.Header()
        hdr['GEN_TIME'] = self._generation_time
        hdr['TIME_RES'] = self._time_resolution

        primary_hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(filename, overwrite=True)

    def read(self, filename):
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            self._generation_time = int(hdr.get('GEN_TIME', 0))
            self._time_resolution = int(hdr.get('TIME_RES', 0))


    def copyTo(self, target_device_idx):
        cloned = self
        excluded = ['_tag']
        if target_device_idx==self._target_device_idx:
            return self
        else:
            pp = get_properties(type(self))
            cloned = copy(self)
            for attr in dir(self):
                if attr not in excluded and attr not in pp:
                    aType = type(getattr(self, attr))
                    if target_device_idx==-1:
                        if aType==cp.ndarray:
                            setattr(cloned, attr, cp.asnumpy( getattr(cloned, attr) ) )
                            # print('Member', attr, 'of class', type(cloned).__name__, 'is now on CPU')
                    elif self._target_device_idx==-1:
                        if aType==np.ndarray:
                            setattr(cloned, attr, cp.asarray( getattr(cloned, attr) ) )
                            # print('Member', attr, 'of class', type(cloned).__name__, 'is now on GPU')
            if target_device_idx >= 0:
                cloned.xp = cp
            else:
                cloned.xp = np
            cloned._target_device_idx = target_device_idx
            return cloned
