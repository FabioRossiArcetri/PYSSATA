import numpy as np
from astropy.io import fits

from pyssata.lib.compute_zern_ifunc import compute_zern_ifunc

def compute_kl_ifunc(*args, **kwargs):
    raise NotImplementedError

def compute_mixed_ifunc(*args, **kwargs):
    raise NotImplementedError


class IFunc:
    def __init__(self,
                 ifunc: np.array=None,
                 type: str=None,
                 mask: np.array=None,
                 npixels: int=None,
                 nzern: int=None,
                 obsratio: float=None,
                 diaratio: float=None,
                 start_mode: int=None,
                 nmodes: int=None,
                 idx_modes=None,
                ):
        self._doZeroPad = False
        self._precision = np.float32
        
        if ifunc is None:
            if type is None:
                raise ValueError('At least one of ifunc and type must be set')
            if mask is not None:
                mask = (np.array(mask) > 0).astype(float)
            if npixels is None:
                raise ValueError("If ifunc is not set, then npixels must be set!")
            
            type_lower = type.lower()
            if type_lower == 'kl':
                ifunc, mask = compute_kl_ifunc(npixels, nmodes=nmodes, obsratio=obsratio, diaratio=diaratio, mask=mask)
            elif type_lower in ['zern', 'zernike']:
                ifunc, mask = compute_zern_ifunc(npixels, nzern=nmodes, obsratio=obsratio, diaratio=diaratio, mask=mask)
            elif type_lower == 'mixed':
                ifunc, mask = compute_mixed_ifunc(npixels, nzern=nzern, nmodes=nmodes, obsratio=obsratio, diaratio=diaratio, mask=mask)
            else:
                raise ValueError(f'Invalid ifunc type {type}')
        
        self._influence_function = ifunc
        self._mask_inf_func = mask
        self._idx_inf_func = np.where(mask)
        self.cut(start_mode=start_mode, nmodes=nmodes, idx_modes=idx_modes)

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
        self._idx_inf_func = np.where(mask_inf_func)

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

    def inverse(self):
        return np.linalg.pinv(self._influence_function)

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

    def save(self, filename, hdr=None):
        hdr = hdr if hdr is not None else fits.Header()
        hdr['VERSION'] = 1

        hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=self._influence_function, name='INFLUENCE_FUNCTION'))
        hdul.append(fits.ImageHDU(data=self._mask_inf_func, name='MASK_INF_FUNC'))
        hdul.writeto(filename, overwrite=True)

    def cut(self, start_mode=None, nmodes=None, idx_modes=None):

        if idx_modes is not None:
            if start_mode is not None:
                start_mode = None
                print('ifunc.cut: start_mode cannot be set together with idx_modes. Setting to None start_mode.')
            if nmodes is not None:
                nmodes = None
                print('ifunc.cut: nmodes cannot be set together with idx_modes. Setting to None nmodes.')
                        
        nrows, ncols = self.influence_function.shape

        if start_mode is None:
            start_mode = 0
        if nmodes is None:
            nmodes = nrows if ncols > nrows else ncols
            
        if idx_modes is not None:
            if ncols > nrows:
                self._influence_function = self._influence_function[idx_modes, :]
            else:
                self._influence_function = self._influence_function[:, idx_modes]
        else:
            if ncols > nrows:
                self._influence_function = self._influence_function[start_mode:nmodes, :]
            else:
                self._influence_function = self._influence_function[:, start_mode:nmodes] 
      
    def read(self, filename, exten=0):
        hdul = fits.open(filename)
        self.influence_function = hdul[exten].data
        self.mask_inf_func = hdul[exten+1].data
        
    def restore(self, filename):
        p = IFunc()
        p.read(filename)
        return p


