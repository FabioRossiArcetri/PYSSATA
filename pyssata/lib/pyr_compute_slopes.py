import numpy as np

from pyssata import cp

clamp_generic = cp.ElementwiseKernel(
        'T x, T c',
        'T y',
        'y = (y < x)?c:y',
        'clamp_generic')

def pyr_compute_slopes(frame, ind_pup, SHLIKE=False, INTENSITY_BASED=False, norm_fact=None, threshold=None, xp=None, float_dtype=None):
    """
    Computes the pyramid signals from a CCD frame.
    
    Parameters:
        frame (numpy.ndarray): CCD frame containing the four sub-pupils.
        ind_pup (numpy.ndarray): 2D array with pupil indices.
        SHLIKE (bool): If True, normalize signals to each subaperture's flux.
        INTENSITY_BASED (bool): If True, compute slopes based on intensity.
        norm_fact (float): Normalize signals to this constant value if provided.
        threshold (float): Subtract this threshold from all pixel values.
    
    Returns:
        sx (numpy.ndarray): x-signal vector.
        sy (numpy.ndarray): y-signal vector.
        flux (float): Total intensity.
    """
    
    if INTENSITY_BASED and SHLIKE:
        raise ValueError('INTENSITY_BASED and SHLIKE keywords cannot be set together.')

    A = frame.flatten()[ind_pup[:, 0]]
    B = frame.flatten()[ind_pup[:, 1]]
    C = frame.flatten()[ind_pup[:, 2]]
    D = frame.flatten()[ind_pup[:, 3]]
    # Extract intensity arrays for each sub-pupil
    
    # Compute total intensity
    flux = xp.sum(A+B+C+D)

    if threshold is not None:
        A -= threshold
        clamp_generic(0,0,A)
        B -= threshold
        clamp_generic(0,0,B)
        C -= threshold
        clamp_generic(0,0,C)
        D -= threshold
        clamp_generic(0,0,D)

    summed = A+B+C+D
    total_intensity = xp.sum(summed)
    n_subap = ind_pup.shape[0]
    sx = (A+B-C-D).astype(float_dtype)
    sy = (B+C-A-D).astype(float_dtype)
    inv_factor = summed
    clamp_generic(0,1e-6, inv_factor)
    factor = 1.0 / inv_factor
    clamp_generic(0,0, factor)
    sx *= factor
    sy *= factor

    # if total_intensity > 0:
    #     if norm_fact is not None:
    #         factor = 1.0 / norm_fact
    #     elif INTENSITY_BASED:
    #         factor = 4 * n_subap / total_intensity
    #         sx = factor * xp.concatenate([A, B])
    #         sy = factor * xp.concatenate([C, D])
    #     else:
    #         if not SHLIKE:
    #             factor = n_subap / total_intensity
    #         else:
    #             inv_factor = summed
    #             clamp_generic(0,1e-6, inv_factor)
    #             factor = 1.0 / inv_factor
    #             clamp_generic(0,0, factor)
    #         sx *= factor
    #         sy *= factor
    # else:
    #     if INTENSITY_BASED:
    #         sx = xp.zeros(2 * n_subap, dtype=float_dtype)
    #         sy = xp.zeros(2 * n_subap, dtype=float_dtype)
    #     else:
    #         sx = xp.zeros(n_subap, dtype=float_dtype)
    #         sy = xp.zeros(n_subap, dtype=float_dtype)

    return sx, sy, flux
