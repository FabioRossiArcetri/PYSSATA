def calc_loop_delay(fs, dm_set=1e-3, type=None, emccd=False, bin=1, comp_time=0.32e-3, CCD_speed=None):
    """
    Calculate the total AO loop delay in seconds.

    Parameters:
    fs : float
        Sampling frequency [Hz].
    dm_set : float, optional
        Deformable mirror settling time (default is 1 ms).
    type : str, optional
        Type of CCD ('CCD39', 'OCAM2', 'CCD220', 'CCID75'). Default is 'CCD39' for emccd=False and 'OCAM2' for emccd=True.
    emccd : bool, optional
        If True, use 'OCAM2'.
    bin : float, optional
        Binning mode (only for CCD39, when emccd is False).
    comp_time : float, optional
        Total computation time (default is 300 ms).
    CCD_speed : float, optional
        CCD read-out time. If not set, uses tables.

    Returns:
    delay : float
        Delay in seconds.
    RON : float
        Read-Out-Noise (optional).
    dark : float
        Dark current (optional).
    """

    if type is None:
        type = 'OCAM2' if emccd else 'CCD39'

    # Integration time
    T = 1.0 / fs

    # CCD Read Out Speed and noise parameters
    RON, dark = calc_detector_noise(fs, type, bin, CCD_speed=CCD_speed)

    # Calculate total delay
    delay = T / 2.0 + CCD_speed + comp_time + dm_set / 2.0 + T / 2.0

    return delay, RON, dark
