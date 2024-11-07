

def extrapolate_edge_pixel(phase, sum_1pix_extra, sum_2pix_extra, xp):
    """
    Extrapolates the phase array at the edges based on neighboring pixel values.

    Parameters:
        phase (ndarray): Input phase array.
        sum_pix_extra (ndarray): Array of matrices for extrapolation routine.

    Returns:
        ndarray: Updated phase array with extrapolated values.
    """
    
    # Find indices where extrapolation is needed (for 1-pixel extrapolation)
    idx_1pix = xp.where(sum_1pix_extra >= 0)
    
    # Extract extrapolated values from the phase array using indices
    vectExtraPol = phase.flat[sum_1pix_extra[idx_1pix]]
    vectExtraPol2 = phase.flat[sum_2pix_extra[idx_1pix]]
    
    # Find indices for 2-pixel extrapolation where value is defined
    idxExtraPol2 = xp.where(sum_2pix_extra[idx_1pix] >= 0)
    
    # Update vectExtraPol based on 2-pixel extrapolation conditions
    vectExtraPol[idxExtraPol2] = 2 * vectExtraPol[idxExtraPol2] - vectExtraPol2[idxExtraPol2]
    
    # Assign extrapolated values back to the phase array at specified indices
    phase[idx_1pix] = vectExtraPol
    
    return phase