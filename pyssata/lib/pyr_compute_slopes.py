import numpy as np

def pyr_compute_slopes(frame, ind_pup, SHLIKE=False, INTENSITY_BASED=False, norm_fact=None, threshold=None):
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

    # Extract intensity arrays for each sub-pupil
    intensity = [frame.flat[ind_pup[:, i]].reshape(-1) for i in range(4)]

    # Compute total intensity
    flux = np.sum([np.sum(arr) for arr in intensity])
    
    if threshold is not None:
        # Apply thresholding
        intensity = [np.maximum(arr - threshold, 0) for arr in intensity]
    
    total_intensity = np.sum([np.sum(arr) for arr in intensity])
    n_subap = ind_pup.shape[0]

    if total_intensity > 0:
        if norm_fact is not None:
            factor = 1.0 / norm_fact
        elif INTENSITY_BASED:
            factor = 4 * n_subap / total_intensity
            sx = factor * np.concatenate([intensity[0], intensity[1]])
            sy = factor * np.concatenate([intensity[2], intensity[3]])
        else:
            if not SHLIKE:
                factor = n_subap / total_intensity
            else:
                inv_factor = np.array([np.sum(arr) for arr in intensity])
                inv_factor[inv_factor <= 0] = 1e-6
                factor = 1.0 / inv_factor
                factor[inv_factor <= 0] = 0.0
                
            sx = (intensity[0] + intensity[1] - intensity[2] - intensity[3]) * factor
            sy = (intensity[1] + intensity[2] - intensity[3] - intensity[0]) * factor
    else:
        if INTENSITY_BASED:
            sx = np.zeros(2 * n_subap)
            sy = np.zeros(2 * n_subap)
        else:
            sx = np.zeros(n_subap)
            sy = np.zeros(n_subap)

    return sx, sy, flux