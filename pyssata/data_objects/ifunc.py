import numpy as np
from astropy.io import fits

class Ifunc:
    def __init__(self):
        self._influence_function = None
        self._mask_inf_func = None
        self._idx_inf_func = None
        self._doZeroPad = False
        self._precision = np.float32
        self.init()

    def init(self):
        self._influence_function = np.array([])
        self._mask_inf_func = np.array([])
        self._idx_inf_func = np.array([])
        return True

    @property
    def influence_function(self):
        return self._influence_function

    @influence_function.setter
    def influence_function(self, ifunc):
        if self._doZeroPad:
            raise ValueError("zeroPad is not working.")
            if self._mask_inf_func is None:
                raise ValueError("if doZeroPad is set, mask_inf_func must be set before setting ifunc.")
            sIfunc = ifunc.shape
            tIfunc = ifunc.dtype

            if tIfunc == np.float32:
                if sIfunc[0] < sIfunc[1]:
                    ifuncPad = np.zeros((sIfunc[0], len(self._mask_inf_func)), dtype=np.float32)
                else:
                    ifuncPad = np.zeros((len(self._mask_inf_func), sIfunc[1]), dtype=np.float32)
            elif tIfunc == np.float64:
                if sIfunc[0] < sIfunc[1]:
                    ifuncPad = np.zeros((sIfunc[0], len(self._mask_inf_func)), dtype=np.float64)
                else:
                    ifuncPad = np.zeros((len(self._mask_inf_func), sIfunc[1]), dtype=np.float64)

            if sIfunc[0] < sIfunc[1]:
                ifuncPad[:, self._idx_inf_func] = ifunc
            else:
                ifuncPad[self._idx_inf_func, :] = ifunc
            ifunc = ifuncPad

        self._influence_function = np.array(ifunc)

    @property
    def mask_inf_func(self):
        return self._mask_inf_func

    @mask_inf_func.setter
    def mask_inf_func(self, mask_inf_func):
        self._mask_inf_func = np.array(mask_inf_func)
        self._idx_inf_func = np.where(mask_inf_func)[0]

    @property
    def idx_inf_func(self):
        return self._idx_inf_func

    @property
    def ptr_ifunc(self):
        return self._influence_function

    @property
    def size(self):
        return self._influence_function.shape

    @property
    def type(self):
        return self._influence_function.dtype

    @property
    def zeroPad(self):
        return self._doZeroPad

    @zeroPad.setter
    def zeroPad(self, zeroPad):
        self._doZeroPad = zeroPad

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, precision):
        if self._influence_function.dtype == precision:
            return

        self._precision = precision
        old_if = self._influence_function
        if precision == np.float32:
            self.influence_function = old_if.astype(np.float32)
        elif precision == np.float64:
            self.influence_function = old_if.astype(np.float64)

    def cleanup(self):
        self.free()

    def free(self):
        del self._influence_function
        del self._mask_inf_func
        del self._idx_inf_func

    def save(self, filename, hdr=None):
        hdr = hdr if hdr is not None else fits.Header()
        hdr['VERSION'] = 1

        hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=self._influence_function, name='INFLUENCE_FUNCTION'))
        hdul.append(fits.ImageHDU(data=self._mask_inf_func, name='MASK_INF_FUNC'))
        hdul.writeto(filename, overwrite=True)

    def read(self, filename, exten=0, start_mode=None, nmodes=None, idx_modes=None, zeroPad=False):
        self._doZeroPad = zeroPad
        hdul = fits.open(filename)

        temp = hdul[exten].data
        if self._precision == np.float32:
            temp = temp.astype(np.float32)
        elif self._precision == np.float64:
            temp = temp.astype(np.float64)

        stemp = temp.shape
        if start_mode is None:
            start_mode = 0
        if nmodes is None:
            nmodes = stemp[0] if stemp[1] > stemp[0] else stemp[1]

        self.mask_inf_func = hdul[exten + 1].data

        if idx_modes is not None:
            if stemp[1] > stemp[0]:
                self.influence_function = temp[idx_modes, :]
            else:
                self.influence_function = temp[:, idx_modes]
        else:
            if stemp[1] > stemp[0]:
                self.influence_function = temp[start_mode:nmodes, :]
            else:
                self.influence_function = temp[:, start_mode:nmodes]

    def restore(self, filename, start_mode=None, nmodes=None, zeroPad=False, idx_modes=None, precision=None):
        if idx_modes is not None:
            if start_mode is not None:
                start_mode = None
                print('ifunc.restore: start_mode cannot be set together with idx_modes. Setting to None start_mode.')
            if nmodes is not None:
                nmodes = None
                print('ifunc.restore: nmodes cannot be set together with idx_modes. Setting to None nmodes.')

        p = Ifunc()
        if precision is not None:
            p.precision = precision

        p.read(filename, start_mode=start_mode, nmodes=nmodes, idx_modes=idx_modes, zeroPad=zeroPad)
        return p

    def revision_track(self):
        return '$Rev$'

