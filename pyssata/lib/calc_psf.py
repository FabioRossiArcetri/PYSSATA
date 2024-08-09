import numpy as np
from numpy.fft import fft2, fftshift

def calc_psf(phase, amp, imwidth=None, normalize=False, nocenter=False, GPU=True):
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
    GPU : bool, optional
        If set, uses GPU routines to compute FFT. Default is True (1B in IDL).

    Returns:
    psf : ndarray
        2D PSF (same dimensions as phase).
    """

    # Set up the complex array based on input dimensions and data type
    if imwidth is not None:
        if phase.dtype == np.float64:
            u_ef = np.zeros((imwidth, imwidth), dtype=np.complex128)
            u_ef[0, 0] = amp * np.exp(1j * phase)
        else:
            u_ef = np.zeros((imwidth, imwidth), dtype=np.complex64)
            u_ef[0, 0] = amp * np.exp(1j * phase)
    else:
        if phase.dtype == np.float64:
            u_ef = amp * np.exp(1j * phase)
        else:
            u_ef = amp * np.exp(1j * phase)

    # Compute FFT (forward)
    if GPU:
        u_fp = fft2(u_ef)  # Assuming GPU support is not available, using numpy FFT
    else:
        u_fp = fft2(u_ef)

    # Center the PSF if required
    if not nocenter:
        u_fp = fftshift(u_fp)

    # Compute the PSF as the square modulus of the Fourier transform
    psf = np.abs(u_fp)**2

    # Normalize if required
    if normalize:
        psf /= np.sum(psf)

    return psf
