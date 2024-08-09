import numpy as np
from pyssata.lib.make_xy import make_xy

def closest(value, array):
    """Find the closest value in the array and return its index."""
    return (np.abs(array - value)).argmin()

def make_mask(np_size, obsratio=0.0, diaratio=1.0, xc=0.0, yc=0.0, 
              square=False, inverse=False, centeronpixel=False):
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
    
    Returns:
        mask (numpy.ndarray): Array representing the mask with the specified properties.
        idx (numpy.ndarray): Array of indices inside the pupil.
    """

    # Generate coordinate grids
    x, y = make_xy(sampling=np_size, ratio=1.0)

    # Adjust center if centeronpixel is set
    if centeronpixel:
        idx_x = closest(xc, x[:, 0])
        neighbours_x = [abs(x[idx_x-1, 0] - xc), abs(x[idx_x+1, 0] - xc)]
        idxneigh_x = np.argmin(neighbours_x)
        kx = -0.5 if idxneigh_x == 0 else 0.5
        xc = x[idx_x, 0] + kx * (x[1, 0] - x[0, 0])

        idx_y = closest(yc, y[0, :])
        neighbours_y = [abs(y[0, idx_y-1] - yc), abs(y[0, idx_y+1] - yc)]
        idxneigh_y = np.argmin(neighbours_y)
        ky = -0.5 if idxneigh_y == 0 else 0.5
        yc = y[0, idx_y] + ky * (y[0, 1] - y[0, 0])

    ir = obsratio

    # Generate mask based on the square or circular option
    if square:
        idx = np.where(
            (np.abs(x - xc) <= diaratio) & (np.abs(y - yc) <= diaratio) &
            ((np.abs(x - xc) >= diaratio * ir) | (np.abs(y - yc) >= diaratio * ir))
        )[0]
    else:
        idx = np.where(
            ((x - xc) ** 2 + (y - yc) ** 2 < diaratio ** 2) &
            ((x - xc) ** 2 + (y - yc) ** 2 >= (diaratio * ir) ** 2)
        )[0]

    # Create the mask
    mask = np.zeros((np_size, np_size), dtype=np.uint8)
    mask[idx] = 1

    # Invert the mask if the inverse keyword is set
    if inverse:
        mask = 1 - mask

    return mask, idx
