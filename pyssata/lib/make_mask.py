from pyssata.lib.make_xy import make_xy

def closest(value, array, xp):
    """Find the closest value in the array and return its index."""
    return (xp.abs(array - value)).argmin()


def make_mask(np_size, obsratio=0.0, diaratio=1.0, xc=0.0, yc=0.0, 
              square=False, inverse=False, centeronpixel=False, get_idx=False, xp=None):
    """
    Create a mask array.

    Parameters:
        np_size (int): Frame size [px].
        obsratio (float): Diameter of the obstruction (fraction of pupil diameter). Default is 0.0.
        diaratio (float): Diameter of the pupil (fraction of frame size). Default is 1.0.
        xc (float): X-center of the pupil (fraction of frame size). Default is 0.0.
        yc (float): Y-center of the pupil (fraction of frame size). Default is 0.0.
        square (bool): If True, make a square mask.
        inverse (bool): If True, invert the mask (1->0, 0->1).
        centeronpixel (bool): If True, move the center of the pupil to the nearest pixel.
        get_idx: if True, return a tuple with (mask, idx)
    
    Returns:
        mask (numpy.ndarray): Array representing the mask with the specified properties.
        idx (numpy.ndarray): Array of indices inside the pupil. (only if get_idx is True)
    """
    if diaratio is None:
        diaratio = 1.0
    if obsratio is None:
        obsratio = 0.0

    # Generate coordinate grids
    x, y = make_xy(sampling=np_size, ratio=1.0, xp=xp)

    # Adjust center if centeronpixel is set
    if centeronpixel:
        idx_x = closest(xc, x[:, 0], xp=xp)
        neighbours_x = [abs(x[idx_x-1, 0] - xc), abs(x[idx_x+1, 0] - xc)]
        idxneigh_x = xp.argmin(neighbours_x)
        kx = -0.5 if idxneigh_x == 0 else 0.5
        xc = x[idx_x, 0] + kx * (x[1, 0] - x[0, 0])

        idx_y = closest(yc, y[0, :], xp=xp)
        neighbours_y = [abs(y[0, idx_y-1] - yc), abs(y[0, idx_y+1] - yc)]
        idxneigh_y = xp.argmin(neighbours_y)
        ky = -0.5 if idxneigh_y == 0 else 0.5
        yc = y[0, idx_y] + ky * (y[0, 1] - y[0, 0])

    ir = obsratio

    # Generate mask based on the square or circular option
    if square:
        idx = xp.where(
            (xp.abs(x - xc) <= diaratio) & (xp.abs(y - yc) <= diaratio) &
            ((xp.abs(x - xc) >= diaratio * ir) | (xp.abs(y - yc) >= diaratio * ir))
        )
    else:
        idx = xp.where(
            ((x - xc) ** 2 + (y - yc) ** 2 < diaratio ** 2) &
            ((x - xc) ** 2 + (y - yc) ** 2 >= (diaratio * ir) ** 2)
        )

    # Create the mask
    mask = xp.zeros((np_size, np_size), dtype=xp.uint8)
    mask[idx] = 1

    # Invert the mask if the inverse keyword is set
    if inverse:
        mask = 1 - mask

    if get_idx:
        return mask, idx
    else:
        return mask
