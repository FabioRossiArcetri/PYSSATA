import numpy as np

main = {
 'root_dir':          './calib/SCAO',               # Root directory for calibration manager
 'pixel_pupil':       160,                    # Linear dimension of pupil phase array
 'pixel_pitch':       0.05,                   # [m] Pitch of the pupil phase array
 'total_time':        1.000,                 # [s] Total simulation running time
 'time_step':         0.001                  # [s] Simulation time step
}

dm = {
 'type':              'zernike',              # modes type
 'nmodes':            54,                     # number of modes
 'npixels':           160,                    # linear dimension of DM phase array
 'obsratio':          0.1,                    # obstruction dimension ratio w.r.t. diameter
 'height':            0                       # DM height [m]
}

pyramid = {
 'pup_diam':          30.,                     # Pupil diameter in subaps.
 'pup_dist':          36.,                     # Separation between pupil centers in subaps.
 'fov':               2.0,                     # Requested field-of-view [arcsec]
 'mod_amp':           3.0,                     # Modulation radius (in lambda/D units)
 'output_resolution': 80,                      # Output sampling [usually corresponding to
 										    # CCD pixels]
 'wavelengthInNm':    750,                     # [nm] Pyramid wavelength
}

slopec = {
 'pupdata_tag' :      'scao_pup',             # tag of the pyramid WFS pupils
 'sn_tag':            'scao_sn'               # tag of the slope reference vector
}

control = {
 'delay':             2,                      # Total temporal delay in time steps
 'type':              'INT',                  # type of control 
 'int_gain':          0.5 * np.ones(54)       # Integrator gain (for 'INT' control)
}

detector = {
 'size':              [80,80],                # Detector size in pixels
 'dt':                0.001,                 # [s] Detector integration time
 'bandw':             300,                    # [nm] Sensor bandwidth
 'photon_noise':      True,                     # activate photon noise
 'readout_noise':     True,                     # activate readout noise
 'readout_level':     1.0,                    # readout noise in [e-/pix/frame]
 'quantum_eff':       0.32                    # quantum efficiency * total transmission
}

wfs_source = [
    {
 'polar_coordinate':  [0.0, 0.0],           # [arcsec, degrees] source polar coordinates
 'magnitude':         8,                    # source magnitude
 'wavelengthInNm':    750                   # [nm] wavelength
}
]

camera = {
 'wavelengthInNm':    1650,                 # [nm] Imaging wavelength
 'nd':                8,                    # padding coefficient for PSF computation
 'start_time':        0.05                 # PSF integration start time
}

atmo = {
 'L0':                40,                   # [m] Outer scale
 'heights':           np.array([119.,837,3045,12780]), # [m] layer heights at 0 zenith angle
 'Cn2':               np.array([0.70,0.06,0.14,0.10]), # Cn2 weights (total must be eq 1)
}

seeing = {
 'constant':          0.8,                  # ["] seeing value
 'func_type':         'SIN'                 # TODO necessary for factory.py line 217
}

wind_speed = {
 'constant':          [5.,10.,20.,10.]      # [m/s] Wind speed value
}

wind_direction = {
 'constant':          [90.,270.,270.,90.]   # [degrees] Wind direction value
}

modalrec = {
 'recmat_tag':        'scao_recmat'         # reconstruction matrix tag
}

pupil_stop = {
 'obs_diam':          0.1                   # pupil stop mask obstruction size     
}
