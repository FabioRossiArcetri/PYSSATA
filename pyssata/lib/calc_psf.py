import numpy as np

from pyssata import xp
from pyssata import standard_dtype
from pyssata import complex_dtype
from numpy.fft import fft2, fftshift

def calc_psf(phase, amp, imwidth=None, normalize=False, nocenter=False):
    """
    Calculates a PSF from an electrical field phase and amplitude.

    Parameters:
    phase : ndarray
        2D phase array.
    amp : ndarray
        2D amplitude array (same dimensions as phase).
    imwidth : int, optional
        Width of the output image. If provided, the output will be of shape (imwidth, imwidth).
    normalize : bool, optional
        If set, the PSF is normalized to total(psf).
    nocenter : bool, optional
        If set, avoids centering the PSF and leaves the maximum pixel at [0,0].

    Returns:
    psf : ndarray
        2D PSF (same dimensions as phase).
    """

    # Set up the complex array based on input dimensions and data type
    if imwidth is not None:        
        u_ef = xp.zeros((imwidth, imwidth), dtype=complex_dtype)
        result = amp * xp.exp(1j * phase)
        s = result.shape
        u_ef[:s[0], :s[1]] = result
    else:
        u_ef = amp * xp.exp(1j * phase)
    # Compute FFT (forward)
    u_fp = fft2(u_ef)
    # Center the PSF if required
    if not nocenter:
        u_fp = fftshift(u_fp)
    # Compute the PSF as the square modulus of the Fourier transform
    psf = xp.abs(u_fp)**2
    # Normalize if required
    if normalize:
        psf /= xp.sum(psf)

    return psf
