import numpy as np
from astropy.io import fits

from pyssata.data_objects.ifunc import IFunc
from pyssata.lib.make_mask import make_mask
from pyssata.lib.zernike_generator import ZernikeGenerator

def pseudo_invert(matrix):
    return np.linalg.pinv(matrix)

def compute_zern_ifunc(dim, nzern, obsratio=0.0, diaratio=1.0, start_mode=0, fits_filename=None, 
                       make_inv=False, inv_fits_filename=None, return_inv=False, mask=None, zeroPad=None):

    if mask is None:
        mask, idx = make_mask(dim, obsratio, diaratio, get_idx=True)
    else:
        mask = mask.astype(float)
        idx = np.where(mask)[0]

    zg = ZernikeGenerator(dim)
    zern_phase_3d = np.stack([zg.getZernike(z) for z in range(2, nzern + 2)])
    zern_phase_3d = zern_phase_3d[start_mode:]
    nzern -= start_mode

    zern_phase_2d = np.array([zern_phase_3d[i][idx] for i in range(nzern)])
    
    zern_phase_2d = zern_phase_2d / np.std(zern_phase_2d, axis=1, keepdims=True)

    ifunc_inv = None
    if make_inv:
        zern_phase_2d_inv = pseudo_invert(zern_phase_2d).astype(float)
        ifunc_inv = IFunc()
        if zeroPad:
            ifunc_inv.zeroPad = zeroPad
        ifunc_inv.mask_inf_func = mask
        ifunc_inv.influence_function = zern_phase_2d_inv
        if inv_fits_filename:
            hdr = fits.Header()
            hdr['INF_FUNC'] = 'ZERNIKES'
            hdr['N_MODES'] = str(nzern)
            hdr['X_Y_SIDE'] = str(dim)
            ifunc_inv.save(inv_fits_filename, hdr)

    ifunc = IFunc()
    if zeroPad:
        ifunc.zeroPad = zeroPad
    ifunc.mask_inf_func = mask
    if return_inv:
        ifunc.influence_function = pseudo_invert(zern_phase_2d).astype(float)
    else:
        ifunc.influence_function = zern_phase_2d.astype(float)

    if fits_filename:
        hdr = fits.Header()
        hdr['INF_FUNC'] = 'ZERNIKES'
        hdr['N_MODES'] = str(nzern)
        hdr['X_Y_SIDE'] = str(dim)
        ifunc.save(fits_filename, hdr)

    return ifunc

