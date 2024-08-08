import numpy as np
from astropy.io import fits

class Ifunc:
    def __init__(self):
        self.mask_inf_func = None
        self.zeroPad = None
        self.ifunc = None

    def set_ifunc(self, ifunc_data, doNotPutOnGpu=False):
        self.ifunc = ifunc_data
        # Placeholder for GPU handling, if needed

    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = fits.Header()
        hdu = fits.PrimaryHDU(data=self.ifunc, header=hdr)
        hdu.writeto(filename, overwrite=True)

def make_mask(dim, obsratio=0.0, diaratio=1.0):
    y, x = np.ogrid[-dim // 2: dim // 2, -dim // 2: dim // 2]
    mask = x**2 + y**2 <= (dim // 2 * diaratio)**2
    if obsratio > 0:
        inner_mask = x**2 + y**2 <= (dim // 2 * obsratio)**2
        mask = mask & ~inner_mask
    return mask.astype(float), np.where(mask)

def zern2phi(dim, nzern, mask=None):
    # Placeholder function for zernike polynomials generation
    zernike_cube = np.random.rand(nzern, dim, dim)  # Replace with actual Zernike calculation
    return zernike_cube

def pseudo_invert(matrix):
    return np.linalg.pinv(matrix)

def compute_zern_ifunc(dim, nzern, obsratio=0.0, diaratio=1.0, start_mode=0, fits_filename=None, 
                       make_inv=False, inv_fits_filename=None, return_inv=False, mask=None, zeroPad=None, doNotPutOnGpu=False):

    if mask is None:
        mask, idx = make_mask(dim, obsratio, diaratio)
    else:
        mask = mask.astype(float)
        idx = np.where(mask)

    zern_phase_3d = zern2phi(dim, nzern, mask=mask)
    zern_phase_3d = zern_phase_3d[start_mode:]
    nzern -= start_mode

    zern_phase_2d = np.array([zern_phase_3d[i][idx] for i in range(nzern)])
    
    zern_phase_2d = zern_phase_2d / np.std(zern_phase_2d, axis=1, keepdims=True)

    ifunc_inv = None
    if make_inv:
        zern_phase_2d_inv = pseudo_invert(zern_phase_2d).astype(float)
        ifunc_inv = Ifunc()
        if zeroPad:
            ifunc_inv.zeroPad = zeroPad
        ifunc_inv.mask_inf_func = mask
        ifunc_inv.set_ifunc(zern_phase_2d_inv, doNotPutOnGpu=doNotPutOnGpu)
        if inv_fits_filename:
            hdr = fits.Header()
            hdr['INF_FUNC'] = 'ZERNIKES'
            hdr['N_MODES'] = str(nzern)
            hdr['X_Y_SIDE'] = str(dim)
            ifunc_inv.save(inv_fits_filename, hdr)

    ifunc = Ifunc()
    if zeroPad:
        ifunc.zeroPad = zeroPad
    ifunc.mask_inf_func = mask
    if return_inv:
        ifunc.set_ifunc(pseudo_invert(zern_phase_2d).astype(float), doNotPutOnGpu=doNotPutOnGpu)
    else:
        ifunc.set_ifunc(zern_phase_2d.astype(float), doNotPutOnGpu=doNotPutOnGpu)

    if fits_filename:
        hdr = fits.Header()
        hdr['INF_FUNC'] = 'ZERNIKES'
        hdr['N_MODES'] = str(nzern)
        hdr['X_Y_SIDE'] = str(dim)
        ifunc.save(fits_filename, hdr)

    return ifunc, ifunc_inv if make_inv else ifunc

