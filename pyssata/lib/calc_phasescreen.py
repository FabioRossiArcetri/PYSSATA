import numpy as np

def calc_phasescreen(L0, dimension, pixel_pitch, seed=0, precision=False, verbose=False):
    if verbose:
        print("Phase-screen computation")

    # Ensure that the dimension is a multiple of 2
    n = int(np.ceil(np.log2(float(dimension))))
    if dimension != 2**n:
        # Force dimension to be a multiple of 2^n
        dimension = 2**n
        if verbose:
            print(f"Dimension is not a multiple of 2, it has been set to {dimension}")

    # Data type based on precision
    dtype = np.complex128 if precision else np.complex64

    # Dimension in meters
    m_dimension = dimension * pixel_pitch

    # Create random Gaussian matrices for the real and imaginary parts
    half_dim = dimension // 2

    if verbose:
        print("Compute random matrices")

    np.random.seed(seed)
    re_gauss = np.random.standard_normal((half_dim + 1, 2 * half_dim + 1))
    im_gauss = np.random.standard_normal((half_dim + 1, 2 * half_dim + 1))

    # Check for non-finite elements and handle them
    if not np.isfinite(re_gauss).all():
        temp = np.isfinite(re_gauss)
        idx_inf = np.where(~temp)
        idx_fin = np.where(temp)
        if len(idx_inf[0]) > 0.01 * temp.size:
            print("Not finite elements are more than 1% of the total!")
            return None
        print(f"Not finite elements: {len(idx_inf[0])}")
        re_gauss[idx_inf] = np.mean(re_gauss[idx_fin])

    if not np.isfinite(im_gauss).all():
        temp = np.isfinite(im_gauss)
        idx_inf = np.where(~temp)
        idx_fin = np.where(temp)
        if len(idx_inf[0]) > 0.01 * temp.size:
            print("Not finite elements are more than 1% of the total!")
            return None
        print(f"Not finite elements: {len(idx_inf[0])}")
        im_gauss[idx_inf] = np.mean(im_gauss[idx_fin])

    # Initialize the phasescreen
    phasescreen = np.zeros((dimension, dimension), dtype=dtype)

    if verbose:
        print("Compute noise matrix")

    # Fill in the noise matrix
    phasescreen[half_dim:2 * half_dim, 0:2 * half_dim] = re_gauss[1:half_dim + 1, 1:2 * half_dim + 1] + 1j * im_gauss[1:half_dim + 1, 1:2 * half_dim + 1]
    phasescreen[0:half_dim, 0:2 * half_dim] = np.flipud(re_gauss[1:half_dim + 1, 1:2 * half_dim + 1]) - 1j * np.flipud(im_gauss[1:half_dim + 1, 1:2 * half_dim + 1])
    phasescreen[half_dim, 0:half_dim] = re_gauss[0, 1:half_dim+1] + 1j * im_gauss[0, 1:half_dim+1]
    phasescreen[half_dim, half_dim:2 * half_dim] = np.flipud(re_gauss[0, 0:half_dim]) - 1j * np.flipud(im_gauss[0, 0:half_dim])

    if verbose:
        print("Compute spatial frequency matrix")

    # Compute spatial frequency matrix
    spatial_frequency = calc_spatialfrequency(dimension, precision=precision)
    spatial_frequency = spatial_frequency / m_dimension**2

    # Check for non-finite elements and handle them
    if not np.isfinite(phasescreen).all():
        temp = np.isfinite(phasescreen)
        idx_inf = np.where(~temp)
        idx_fin = np.where(temp)
        if len(idx_inf[0]) > 0.01 * temp.size:
            print("Not finite elements are more than 1% of the total!")
            return None
        print(f"Not finite elements: {len(idx_inf[0])}")
        phasescreen[idx_inf] = np.mean(phasescreen[idx_fin])

    # Apply spatial frequency
    phasescreen *= (spatial_frequency + 1. / L0**2)**(-11./12.)
    phasescreen *= np.sqrt(0.033/2./m_dimension**2) * (2 * np.pi)**(2./3.) * np.sqrt(0.06) * (1 / pixel_pitch)**(5./6.)

    # Perform the inverse FFT
    phasescreen = np.fft.ifftshift(phasescreen)
    phasescreen = np.fft.ifft2(phasescreen)
    phasescreen = np.fft.fftshift(phasescreen)

    return np.real(phasescreen)

def calc_spatialfrequency(dimension, precision=False):
    """Helper function to compute the spatial frequency matrix."""
    half_dim = dimension // 2
    freq_range = np.fft.fftfreq(dimension)

    fx, fy = np.meshgrid(freq_range, freq_range)
    spatial_frequency = np.sqrt(fx**2 + fy**2)

    return spatial_frequency if precision else np.float32(spatial_frequency)
