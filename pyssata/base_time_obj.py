from astropy.io import fits
from pyssata import np, cp, xp, global_precision, default_device, default_device_idx, cpuArray
from pyssata import cpu_float_dtype_list, gpu_float_dtype_list
from pyssata import cpu_complex_dtype_list, gpu_complex_dtype_list
from copy import copy, deepcopy

def get_properties(cls):
    result = []
    classlist = cls.__mro__
    for cc in classlist:
        result.extend([attr for attr, value in vars(cc).items() if isinstance(value, property) ]) 
    return result
    # return [attr for attr, value in vars(cls).items() if isinstance(value, property) ]

class BaseTimeObj:
    def __init__(self, device_idx=None, precision=None):
        """
        Creates a new base_time object.

        Parameters:
        precision (int, optional): if None will use the global_precision, otherwise pass 0 for double, 1 for single
        device_idx (int, optional): if None will use the default_device_idx, otherwise pass -1 for cpu, i for GPU of index i

        """
        self._time_resolution = int(1e9)
        self._generation_time = -1

        #print(self.__class__.__name__)
        #print('device_idx', device_idx)
        #print('precision', precision)

        if precision is None:
            self._precision = global_precision
        else:
            self._precision = precision

        if device_idx is None:
            self._device_idx = default_device_idx
        else:
            self._device_idx = device_idx

        if not self._device_idx==-1:
            self._device = cp.cuda.Device(self._device_idx)      # GPU case
            self.dtype = gpu_float_dtype_list[self._precision]
            self.complex_dtype = gpu_complex_dtype_list[self._precision]
            self.xp = cp
        else:
            self._device = default_device                # CPU case
            self.dtype = cpu_float_dtype_list[self._precision]
            self.complex_dtype = cpu_complex_dtype_list[self._precision]
            self.xp = np
        
    def copyTo(self, device_idx):
        cloned = self
        excluded = ['_tag']
        if device_idx==self._device_idx:
            return self
        else:
            pp = get_properties(type(self))
            cloned = copy(self)
            for attr in dir(self):
                if attr not in excluded and attr not in pp:
                    aType = type(getattr(self, attr))
                    if device_idx==-1:
                        if aType==cp.ndarray:
                            setattr(cloned, attr, cp.asnumpy( getattr(cloned, attr) ) )
                            # print('Member', attr, 'of class', type(cloned).__name__, 'is now on CPU')
                    elif self._device_idx==-1:
                        if aType==np.ndarray:
                            setattr(cloned, attr, cp.asarray( getattr(cloned, attr) ) )
                            # print('Member', attr, 'of class', type(cloned).__name__, 'is now on GPU')
            if device_idx >= 0:
                cloned.xp = cp
            else:
                cloned.xp = np
            cloned._device_idx = device_idx
            return cloned

    @property
    def generation_time(self):
        return self._generation_time

    @generation_time.setter
    def generation_time(self, value):
        self._generation_time = value

    @property
    def time_resolution(self):
        return self._time_resolution

    @time_resolution.setter
    def time_resolution(self, value):
        self._time_resolution = value

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, value):
        self._precision = value
        if not self._device_idx==-1:
            self._device = cp.cuda.Device(self._device_idx)      # GPU case
            self.dtype = gpu_float_dtype_list[self._precision]
            self.complex_dtype = gpu_complex_dtype_list[self._precision]
            self.xp = cp
        else:
            self._device = default_device                # CPU case
            self.dtype = cpu_float_dtype_list[self._precision]
            self.complex_dtype = cpu_complex_dtype_list[self._precision]
            self.xp = np

    def trigger(self, t):
        # if the device is not the CPU and it is different from the default one, 
        # then put in in use
        if not self.device_idx==-1 and not self.device_idx==default_device_idx:           
            self._device.use()

    def t_to_seconds(self, t):
        return float(t) / float(self._time_resolution)

    def seconds_to_t(self, seconds):
        if self._time_resolution == 0:
            return 0

        ss = f"{float(seconds):.9f}".rstrip('0').rstrip('.')
        if '.' not in ss:
            ss += '.0'

        dotpos = ss.find('.')
        intpart = ss[:dotpos]
        fracpart = ss[dotpos + 1:]

        return (int(intpart) * self._time_resolution +
                int(fracpart) * (self._time_resolution // (10 ** len(fracpart))))

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

    def get_properties_list(self):
        return vars(self)

