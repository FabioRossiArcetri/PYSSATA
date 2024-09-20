import numpy as np

from pyssata import xp
from pyssata import float_dtype

from pyssata.lib.make_mask import make_mask
from pyssata.lib.zernike_generator import ZernikeGenerator


def compute_zern_ifunc(dim, nzern, obsratio=0.0, diaratio=1.0, start_mode=0, mask=None):

    if mask is None:
        mask, idx = make_mask(dim, obsratio, diaratio, get_idx=True)
    else:
        mask = mask.astype(float)
        idx = xp.where(mask)[0]

    zg = ZernikeGenerator(dim)
    zern_phase_3d = xp.stack([zg.getZernike(z) for z in range(2, nzern + 2)])
    zern_phase_3d = zern_phase_3d[start_mode:]
    nzern -= start_mode

    zern_phase_2d = xp.array([zern_phase_3d[i][idx] for i in range(nzern)], dtype=float_dtype)
    
    zern_phase_2d = zern_phase_2d / xp.std(zern_phase_2d, axis=1, keepdims=True)

    return zern_phase_2d.astype(float), mask

