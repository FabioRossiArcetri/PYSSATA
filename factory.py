import numpy as np
import os

class CalibManager:
    # Placeholder for the CalibManager class
    def __init__(self, root_dir):
        self.root_dir = root_dir

class Factory:
    def __init__(self, params, GPU=False, NOCM=False, SINGLEGPU=False):
        """
        Initialize the factory object.

        Parameters:
        params (dict): Dictionary or struct with main simulation parameters
        GPU (bool, optional): If set, GPU-accelerated objects will be used when available
        NOCM (bool, optional): If set, no calibration manager will be created inside the factory
        SINGLEGPU (bool, optional): If set, only the first GPU in the system will be used
        """
        self._main = self.ensure_dictionary(params)

        if GPU:
            if not SINGLEGPU:
                print('    ***    ------->>>>>> factory will use all GPUs <<<<<<-------    ***')
            else:
                print('    ***    ------->>>>>> factory will use 1 GPU  <<<<<<-------    ***')
        
        if (GPU or SINGLEGPU) and self.has_gpu():
            self._gpu = True
        
        if SINGLEGPU and self.has_gpu():
            required_version = 1.48
            actual_version = self.cuda_version()
            if actual_version < required_version:
                raise Exception(f'SINGLEGPU keyword needs the GPU_SIMUL module version {required_version} or better, found version {actual_version} instead')
            self.cuda_set_device_count(1)

        self._global_params = ['verbose', 'precision']
        self._main['precision'] = self._main.get('precision', 0)

        if not NOCM:
            self._cm = self.get_calib_manager()

    def ensure_dictionary(self, params):
        """
        Ensure that params is a dictionary.

        Parameters:
        params (dict): Dictionary of parameters

        Returns:
        dict: Ensured dictionary of parameters
        """
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary")
        return params

    def remove_keywords(self, dictionary, keywords):
        """
        Remove specified keywords from the dictionary.

        Parameters:
        dictionary (dict): Dictionary of parameters
        keywords (list): List of keywords to remove

        Returns:
        dict: Dictionary with keywords removed
        """
        for key in keywords:
            dictionary.pop(key, None)
        return dictionary

    def get_calib_manager(self):
        """
        Get the calibration manager.

        Returns:
        CalibManager: Calibration manager object
        """
        root_dir = self._main.get('root_dir', '/default/path')
        return CalibManager(root_dir)

    def has_gpu(self):
        """
        Check if the system has GPU capabilities.

        Returns:
        bool: True if GPU is available, False otherwise
        """
        # Placeholder for actual GPU check
        return True

    def cuda_version(self):
        """
        Get the CUDA version.

        Returns:
        float: CUDA version
        """
        # Placeholder for actual CUDA version check
        return 1.5

    def cuda_set_device_count(self, count):
        """
        Set the number of CUDA devices.

        Parameters:
        count (int): Number of CUDA devices to set
        """
        # Placeholder for setting CUDA device count
        pass

def ensure_dictionary(self, arg):
    """
    Converts structs to dictionaries, and gives error for all other types.
    Dictionaries are copied to allow removal of elements without
    touching the original object.

    Parameters:
    arg (dict or object): Parameter to be converted.

    Returns:
    dict: Converted dictionary.
    """
    if isinstance(arg, dict):
        return dict(arg)
    if not isinstance(arg, object):
        raise ValueError("Cannot convert to dictionary")
    
    keys = arg.__dict__.keys()
    if len(keys) == 1 and list(keys)[0] == '':
        return arg
    
    if isinstance(arg, dict):
        return dict(arg)
    else:
        raise ValueError("Cannot convert to dictionary")

def extract(self, dictionary, key, default=None):
    """
    Gets a keyword and remove it from the dictionary.
    Similar to the "remove" function of a dictionary, but allows
    for a default value to be specified.

    Parameters:
    dictionary (dict): Dictionary from which the keyword must be extracted.
    key (str): Keyword to extract.
    default: Default value to return if the key does not exist.

    Returns:
    Value of the extracted key or the default value.
    """
    exist = key in dictionary
    if not exist and default is None:
        raise KeyError(f"Error: missing key: {key}")
    
    return dictionary.pop(key, default)

def apply_global_params(self, obj):
    """
    Applies global parameters (verbose, precision, etc) to the specified object.

    Parameters:
    obj (object): Object reference.
    """
    if not isinstance(obj, BaseProcessingObj):
        return
    
    for p in self._global_params:
        if p in self._main:
            setattr(obj, p, self._main[p])
            obj.apply_properties({p: self._main[p]}, ignore=True)

def ifunc_restore(self, tag=None, type=None, npixels=None, nmodes=None, nzern=None, 
                  obsratio=None, diaratio=None, start_mode=None, mask=None, 
                  return_inv=None, doNotPutOnGpu=None, zeroPad=None, idx_modes=None):
    """
    Restore ifunc object from disk.

    Parameters:
    Various optional parameters to specify the ifunc object.

    Returns:
    ifunc: Restored ifunc object.
    """
    precision = self._main['precision']
    ifunc = None
    if tag:
        ifunc = self._cm.read_ifunc(tag, start_mode=start_mode, nmodes=nmodes, 
                                    doNotPutOnGpu=doNotPutOnGpu, zeroPad=zeroPad, 
                                    precision=precision, idx_modes=idx_modes)
    if tag and not ifunc:
        print(f"ifunc {tag} not present on disk!")
    
    if not ifunc and type:
        if mask is not None:
            mask = (np.array(mask) > 0).astype(float)
        if npixels is None:
            raise ValueError("factory::ifunc_restore --> npixels must be set!")
        
        type_lower = type.lower()
        if type_lower == 'kl':
            ifunc = compute_kl_ifunc(npixels, nmodes, obsratio=obsratio, diaratio=diaratio, 
                                     start_mode=start_mode, mask=mask, point4radius=point4radius, 
                                     zeroPad=zeroPad, return_inv=return_inv)
        elif type_lower in ['zern', 'zernike']:
            ifunc = compute_zern_ifunc(npixels, nmodes, obsratio=obsratio, diaratio=diaratio, 
                                       start_mode=start_mode, mask=mask, zeroPad=zeroPad, return_inv=return_inv)
        elif type_lower == 'mixed':
            ifunc = compute_mixed_ifunc(npixels, nzern, nmodes, obsratio=obsratio, diaratio=diaratio, 
                                        start_mode=start_mode, mask=mask, point4radius=point4radius, 
                                        zeroPad=zeroPad, return_inv=return_inv)

    return ifunc

def get_atmo_container(self, source_list, params_atmo, params_seeing, params_windspeed, params_winddirection, GPU=None):
    """
    Gets a processing container with a full complement of atmospheric objects.

    Parameters:
    source_list (list): List of source objects
    params_atmo (dict): Parameter dictionary for the atmo_evolution object
    params_seeing (dict): Parameter dictionary for the seeing func_generator object
    params_windspeed (dict): Parameter dictionary for the wind speed func_generator object
    params_winddirection (dict): Parameter dictionary for the wind direction func_generator object
    GPU (bool, optional): Flag for using GPU

    Returns:
    ProcessingContainer: Processing container with atmospheric objects
    """
    useGPU = GPU if GPU is not None else self._gpu

    container = ProcessingContainer()

    # the following lines are used to reduce the layers to 1 when seeing is 0
    params_atmo_copy = params_atmo.copy()
    params_windspeed_copy = params_windspeed.copy()
    params_winddirection_copy = params_winddirection.copy()

    if params_seeing['func_type'] == 'SIN' and 'amp' not in params_seeing and 'constant' in params_seeing:
        if np.sum(np.abs(params_seeing['constant'])) == 0:
            print('WARNING: seeing is 0, change the atmo profile to 1 layer.')
            params_atmo_copy['heights'] = [0.0]
            params_atmo_copy['cn2'] = [1.0]
            if 'pixel_phasescreens' in params_atmo_copy:
                params_atmo_copy.pop('pixel_phasescreens')
            params_windspeed_copy['constant'] = [0.0]
            params_winddirection_copy['constant'] = [0.0]

    atmo = self.get_atmo_evolution(params_atmo_copy, source_list, GPU=useGPU)
    seeing = self.get_func_generator(params_seeing)
    wind_speed = self.get_func_generator(params_windspeed_copy)
    wind_dir = self.get_func_generator(params_winddirection_copy)

    atmo.seeing = seeing.output
    atmo.wind_speed = wind_speed.output
    atmo.wind_dir = wind_dir.output

    container.add(seeing, name='seeing')
    container.add(wind_speed, name='wind_speed')
    container.add(wind_dir, name='wind_dir')
    container.add(atmo, name='atmo', output='layer_list')

    return container

def get_atmo_cube_container(self, source_list, params_atmo, params_seeing, GPU=None):
    """
    Gets a processing container with a full complement of atmospheric objects for reading cubes.

    Parameters:
    source_list (list): List of source objects
    params_atmo (dict): Parameter dictionary for the atmo_evolution object
    params_seeing (dict): Parameter dictionary for the seeing func_generator object
    GPU (bool, optional): Flag for using GPU

    Returns:
    ProcessingContainer: Processing container with atmospheric objects
    """
    useGPU = GPU if GPU is not None else self._gpu

    container = ProcessingContainer()

    atmo = self.get_atmo_readcube(params_atmo, source_list, GPU=useGPU)
    seeing = self.get_func_generator(params_seeing)

    atmo.seeing = seeing.output

    container.add(seeing, name='seeing')
    container.add(atmo, name='atmo', output='layer_list')

    return container

def get_kernel_full(self, sh, params_launcher, params_seeing, params_zlayer, params_zprofile):
    """
    Gets a processing container with a full complement of kernel objects.

    Parameters:
    sh (object): SH object that has received its own EF object
    params_launcher (dict): Parameter dictionary for the launcher
    params_seeing (dict): Parameter dictionary for the seeing func_generator object
    params_zlayer (dict): Parameter dictionary for the zlayer func_generator object
    params_zprofile (dict): Parameter dictionary for the zprofile func_generator object

    Returns:
    ProcessingContainer: Processing container with kernel objects
    """
    container = ProcessingContainer()

    params_launcher = self.ensure_dictionary(params_launcher)
    params_seeing = self.ensure_dictionary(params_seeing)
    params_zlayer = self.ensure_dictionary(params_zlayer)
    params_zprofile = self.ensure_dictionary(params_zprofile)

    zfocus_temp = self.extract(params_zlayer, 'zfocus', default=None)
    theta_temp = self.extract(params_zlayer, 'theta', default=None)

    if not zfocus_temp:
        zfocus_temp = self.extract(params_launcher, 'zfocus', default=-1)
    if not theta_temp:
        theta_temp = self.extract(params_launcher, 'theta', default=[0.0, 0.0])

    if isinstance(zfocus_temp, (list, np.ndarray)) and len(zfocus_temp) > 1:
        params_zfocus = {'func_type': 'TIME_HIST', 'time_hist': zfocus_temp}
        params_theta = {'func_type': 'TIME_HIST', 'time_hist': theta_temp}
    else:
        params_zfocus = {'func_type': 'SIN', 'constant': zfocus_temp}
        params_theta = {'func_type': 'SIN', 'constant': theta_temp}

    zfocus = self.get_func_generator(params_zfocus)
    theta = self.get_func_generator(params_theta)

    kernel = self.get_kernel({})
    kernel.zfocus = zfocus.output
    kernel.theta = theta.output
    kernel.lenslet = sh.lenslet

    seeing = self.get_func_generator(params_seeing)
    zlayer = self.get_func_generator(params_zlayer)
    zprofile = self.get_func_generator(params_zprofile)

    launcher_sizeArcsec = self.extract(params_launcher, 'sizeArcsec', default=0)
    launcher_pos = self.extract(params_launcher, 'position', default=[0, 0, 0])

    kernel.in_seeing = seeing.output
    kernel.in_zlayer = zlayer.output
    kernel.in_zprofile = zprofile.output
    kernel.launcher_pos = launcher_pos
    kernel.launcher_size = launcher_sizeArcsec

    kernel.apply_properties(params_launcher)

    container.add(seeing, name='seeing')
    container.add(zfocus, name='zfocus')
    container.add(theta, name='theta')
    container.add(zlayer, name='zlayer')
    container.add(zprofile, name='zprofile')
    container.add(kernel, name='kernel', output='out_kernels')

    container.add_input('kernel', 'ef')
    container.add_input('kernel', 'pxscale')
    container.add_input('kernel', 'dimension')
    container.add_input('kernel', 'oversampling')
    container.add_input('kernel', 'positiveShiftTT')
    container.add_input('kernel', 'returnFft')
    container.add_output('kernel', 'last_recalc')
    container.add_output('kernel', 'dimension')
    container.add_output('kernel', 'pxscale')
    container.add_output('kernel', 'returnFft')

    return container

def get_avc(self, params):
    """
    Create an Adaptive Vibration Cancellation object.

    Parameters:
    params (dict): Dictionary of parameters

    Returns:
    AVC: Adaptive Vibration Cancellation object
    """
    T = self._main['time_step']
    freq = params.pop('freq')
    sinusSingleFreq = params.pop('sinusSingleFreq')
    x = params.pop('x')
    oversampling = self.extract(params, 'oversampling', default=1.0)
    estTFparams = self.extract(params, 'estTFparams', default=False)

    omega = 2 * np.pi * freq
    alpha = 0.0
    gx = 10.0 if sinusSingleFreq else 40.0
    gomega = 0
    k = 0.5
    c = 1.0 if estTFparams else 0.0
    N = oversampling

    print(f'AVC gx and c: {gx}, {c}')

    avc = AVC(T, x, omega, alpha, gx, gomega, k, c, N)

    self.apply_global_params(avc)
    avc.apply_properties(params)

    return avc

def get_aveslopes(self, params):
    """
    Create an average slopes object.

    Parameters:
    params (dict): Dictionary of parameters

    Returns:
    AveSlopes: Average slopes object
    """
    n_iter = params.pop('n_iter')
    aveslopes = AveSlopes(n_iter)

    self.apply_global_params(aveslopes)
    aveslopes.apply_properties(params)

    return aveslopes

def get_sh_with_kernel(self, params_sh, source, params_seeing, params_zlayer, params_zprofile):
    """
    Create a SH object with a kernel.

    Parameters:
    params_sh (dict): Dictionary of SH parameters
    source (Source): Source object
    params_seeing (dict): Dictionary of seeing parameters
    params_zlayer (dict): Dictionary of zlayer parameters
    params_zprofile (dict): Dictionary of zprofile parameters

    Returns:
    ProcessingContainer: Processing container with SH and kernel objects
    """
    container = ProcessingContainer()

    zfocus_temp = self.extract(params_zlayer, 'zfocus', default=-1)
    theta_temp = self.extract(params_zlayer, 'theta', default=[0.0, 0.0])

    if isinstance(zfocus_temp, (list, np.ndarray)) and len(zfocus_temp) > 1:
        params_zfocus = {'func_type': 'TIME_HIST', 'time_hist': zfocus_temp}
        params_theta = {'func_type': 'TIME_HIST', 'time_hist': theta_temp}
    else:
        params_zfocus = {'func_type': 'SIN', 'constant': zfocus_temp}
        params_theta = {'func_type': 'SIN', 'constant': theta_temp}

    zfocus = self.get_func_generator(params_zfocus)
    theta = self.get_func_generator(params_theta)

    sh = self.get_sh(params_sh)
    kernel = self.get_kernel()
    kernel.zfocus = zfocus.output
    kernel.theta = theta.output
    kernel.source = source
    kernel.lenslet = sh.lenslet
    kernel.ef = sh.in_ef
    seeing = self.get_func_generator(params_seeing)
    zlayer = self.get_func_generator(params_zlayer)
    zprofile = self.get_func_generator(params_zprofile)

    kernel.seeing = seeing.output
    kernel.zlayer = zlayer.output
    kernel.zprofile = zprofile.output
    sh.in_ef = kernel.out_ef

    container.add(seeing, name='seeing')
    container.add(zfocus, name='zfocus')
    container.add(theta, name='theta')
    container.add(zlayer, name='zlayer')
    container.add(zprofile, name='zprofile')
    container.add(kernel, name='kernel', input='in_ef')
    container.add(sh, output='out_i')

    return container

def get_atmo_evolution(self, params, source_list, GPU=None):
    """
    Create an atmo_evolution processing object.

    Parameters:
    params (dict): Dictionary of parameters
    source_list (list): List of source objects
    GPU (bool, optional): Flag for using GPU

    Returns:
    AtmoEvolution: AtmoEvolution processing object
    """
    useGPU = GPU if GPU is not None else self._gpu

    params = self.ensure_dictionary(params)

    pixel_pup = self._main['pixel_pupil']
    pixel_pitch = self._main['pixel_pitch']
    precision = self._main['precision']
    zenithAngleInDeg = self._main.get('zenithAngleInDeg')

    L0 = params.pop('L0')
    wavelengthInNm = self.extract(params, 'wavelengthInNm', default=500)
    heights = params.pop('heights')
    Cn2 = params.pop('Cn2')
    mcao_fov = self.extract(params, 'mcao_fov', default=None)
    fov_in_m = self.extract(params, 'fov_in_m', default=None)
    pixel_phasescreens = self.extract(params, 'pixel_phasescreens', default=None)
    seed = self.extract(params, 'seed', default=1)
    pupil_position = self.extract(params, 'pupil_position', default=None)
    
    directory = self._cm.root_subdir('phasescreen')

    user_defined_phasescreen = self.extract(params, 'user_defined_phasescreen', default='')
    
    force_mcao_fov = self.extract(params, 'force_mcao_fov', default=None)
    make_cycle = self.extract(params, 'make_cycle', default=None)
    doFresnel = self.extract(params, 'doFresnel', default=None)

    atmo_evolution = AtmoEvolution(L0, wavelengthInNm, pixel_pitch, heights, Cn2,
                                   pixel_pup, directory, source_list,
                                   zenithAngleInDeg=zenithAngleInDeg, mcao_fov=mcao_fov,
                                   pixel_phasescreens=pixel_phasescreens,
                                   precision=precision, seed=seed, GPU=useGPU,
                                   user_defined_phasescreen=user_defined_phasescreen,
                                   force_mcao_fov=force_mcao_fov, make_cycle=make_cycle,
                                   fov_in_m=fov_in_m, pupil_position=pupil_position)

    self.apply_global_params(atmo_evolution)
    atmo_evolution.apply_properties(params)

    return atmo_evolution

def get_atmo_readcube(self, params, source_list, GPU=None):
    """
    Create an atmo_readcube processing object.

    Parameters:
    params (dict): Dictionary of parameters
    source_list (list): List of source objects
    GPU (bool, optional): Flag for using GPU

    Returns:
    AtmoReadCube: AtmoReadCube processing object
    """
    useGPU = GPU if GPU is not None else self._gpu

    params = self.ensure_dictionary(params)

    pixel_pitch = self._main['pixel_pitch']
    precision = self._main['precision']
    filename = params.pop('filename')
    filename_ol = self.extract(params, 'filename_ol', default=None)
    wavelengthInNm = params.pop('wavelengthInNm')
    r0data = self.extract(params, 'r0data', default=None)
    seeingData = self.extract(params, 'seeingData', default=None)
    outPhaseSize = self.extract(params, 'outPhaseSize', default=None)
    rad = self.extract(params, 'rad', default=None)
    startZero = self.extract(params, 'startZero', default=None)
    onlyLo = self.extract(params, 'onlyLo', default=None)
    onlyHo = self.extract(params, 'onlyHo', default=None)
    startingPosition = self.extract(params, 'startingPosition', default=None)
    cut_modes = self.extract(params, 'cutModes', default=None)
    cut_coeff = self.extract(params, 'cut_coeff', default=None)
    cut_from_OL = self.extract(params, 'cut_from_OL', default=None)
    firstDimNiter = self.extract(params, 'firstDimNiter', default=None)

    ifunc_tag = self.extract(params, 'ifunc_tag', default=None)
    type = self.extract(params, 'type', default=None)
    nmodes = self.extract(params, 'nmodes', default=None)
    nzern = self.extract(params, 'nzern', default=None)
    start_mode = self.extract(params, 'start_mode', default=None)
    npixels = self.extract(params, 'npixels', default=None)
    obsratio = self.extract(params, 'obsratio', default=None)
    diaratio = self.extract(params, 'diaratio', default=None)

    phase2modes_tag = self.extract(params, 'phase2modes_tag', default=None)
    pupil_mask_tag = self.extract(params, 'pupil_mask_tag', default='')
    zeroPad = self.extract(params, 'zeroPadp2m', default='')

    mask = None
    if pupil_mask_tag:
        if phase2modes_tag:
            print('if phase2modes_tag is defined then pupil_mask_tag will not be used!')
        pupilstop = self._cm.read_pupilstop(pupil_mask_tag, GPU=useGPU)
        if not pupilstop:
            raise ValueError(f'Pupil mask tag {pupil_mask_tag} not found.')
        mask = pupilstop.A
        if not npixels:
            npixels = mask.shape[0]

    ifunc = self.ifunc_restore(tag=ifunc_tag, type=type, npixels=npixels, nmodes=nmodes, nzern=nzern,
                               obsratio=obsratio, diaratio=diaratio, start_mode=start_mode)

    phase2modes = self.ifunc_restore(tag=phase2modes_tag, type=type, npixels=npixels, nmodes=nmodes, nzern=nzern,
                                     obsratio=obsratio, diaratio=diaratio, start_mode=start_mode,
                                     mask=mask, return_inv=True, zeroPad=zeroPad)
  
    zenithAngleInDeg = self._main.get('zenithAngleInDeg')

    atmo_readcube = AtmoReadCube(filename, wavelengthInNm, pixel_pitch,
                                 filename_ol=filename_ol, r0data=r0data,
                                 seeingData=seeingData, outPhaseSize=outPhaseSize,
                                 rad=rad, ifunc=ifunc, phase2modes=phase2modes,
                                 startZero=startZero, onlyLo=onlyLo, onlyHo=onlyHo,
                                 startingPosition=startingPosition,
                                 GPU=useGPU, precision=precision, cut_modes=cut_modes,
                                 cut_coeff=cut_coeff, cut_from_OL=cut_from_OL,
                                 firstDimNiter=firstDimNiter, zenithAngleInDeg=zenithAngleInDeg)

    # self.apply_global_params(atmo_readcube)
    # atmo_readcube.apply_properties(params)

    return atmo_readcube

def get_atmo_propagation(self, params, source_list, GPU=None):
    """
    Create an atmo_propagation processing object.

    Parameters:
    params (dict): Dictionary of atmo parameters
    source_list (list): List of source objects
    GPU (bool, optional): Flag for using GPU

    Returns:
    AtmoPropagation: AtmoPropagation processing object
    """
    useGPU = GPU if GPU is not None else self._gpu

    params = self.ensure_dictionary(params)

    pixel_pupil = self._main['pixel_pupil']
    pixel_pitch = self._main['pixel_pitch']

    atmo_propagation = AtmoPropagation(source_list, pixel_pupil, pixel_pitch, GPU=useGPU)

    doFresnel = self.extract(params, 'doFresnel', default=None)
    wavelengthInNm = self.extract(params, 'wavelengthInNm', default=None)
    if doFresnel is not None:
        atmo_propagation.doFresnel = doFresnel
    if doFresnel is not None and wavelengthInNm is None:
        raise ValueError('get_atmo_propagation: wavelengthInNm is required when doFresnel key is set to correctly simulate physical propagation.')
    if wavelengthInNm is not None:
        atmo_propagation.wavelengthInNm = wavelengthInNm

    self.apply_global_params(atmo_propagation)

    pupil_position = self.extract(params, 'pupil_position', default=None)
    atmo_propagation.pupil_position = pupil_position

    return atmo_propagation

def get_atmo_propagation2(self, source_list=None, GPU=None):
    """
    Create an atmo_propagation processing object.

    Parameters:
    source_list (list, optional): List of source objects. If not given, an empty list is initialized.
    GPU (int, optional): Flag for using GPU processing

    Returns:
    AtmoPropagation: AtmoPropagation processing object
    """
    useGPU = GPU if GPU is not None else self._gpu
    if source_list is None:
        source_list = []

    pixel_pupil = self._main['pixel_pupil']
    pixel_pitch = self._main['pixel_pitch']

    atmo_propagation = AtmoPropagation(source_list, pixel_pupil, pixel_pitch, GPU=useGPU)

    self.apply_global_params(atmo_propagation)

    return atmo_propagation

def get_calib_manager(self, params):
    """
    Create a calibration manager object.

    Parameters:
    params (dict): Dictionary of parameters

    Returns:
    CalibManager: Calibration manager object
    """
    root_dir = self._main['root_dir']
    cm = CalibManager(root_dir)

    self.apply_global_params(cm)

    if params:
        params = self.ensure_dictionary(params)
        cm.apply_properties(params)

    return cm

def get_ccd(self, ccd_params, wfs_params=None):
    """
    Create a CCD processing object.

    Parameters:
    ccd_params (dict): Dictionary of parameters
    wfs_params (dict, optional): Dictionary of pyramid/SH parameters. Required for certain 'auto' keywords

    Returns:
    CCD: CCD processing object
    """
    params = self.ensure_dictionary(ccd_params)

    if wfs_params:
        params = CCD.auto_params_management(self._main, wfs_params, ccd_params)

    name = self.extract(params, 'name', default=None)
    sky_bg_norm = self.extract(params, 'sky_bg_norm', default=None)
    pixelGains_tag = self.extract(params, 'pixelGains_tag', default=None)
    charge_diffusion = self.extract(params, 'charge_diffusion', default=None)
    charge_diffusion_fwhm = self.extract(params, 'charge_diffusion_fwhm', default=None)

    sz = params.pop('size')

    ccd = CCD(sz)
    if charge_diffusion is not None:
        ccd.charge_diffusion = charge_diffusion
    if charge_diffusion_fwhm is not None:
        ccd.charge_diff_fwhm = charge_diffusion_fwhm

    if pixelGains_tag is not None:
        pixelGains = self._cm.read_data(pixelGains_tag)
        ccd.pixelGains = pixelGains

    self.apply_global_params(ccd)
    ccd.apply_properties(params)
    return ccd

def get_ch2ndcontrol(self, params_pyr1, params_pyr2):
    """
    Create a ch2ndcontrol processing object.

    Parameters:
    params_pyr1 (dict): Dictionary of parameters for the first channel
    params_pyr2 (dict): Dictionary of parameters for the second channel

    Returns:
    Ch2ndControl: Ch2ndControl processing object
    """
    params_pyr1 = self.ensure_dictionary(params_pyr1)
    params_pyr2 = self.ensure_dictionary(params_pyr2)

    mod_amp = self.extract(params_pyr2, 'mod_amp', default=None)
    wavelengthInNm = self.extract(params_pyr1, 'wavelengthInNm', default=None)

    ch2ndcontrol = Ch2ndControl(mod_amp, wavelengthInNm, verbose=self._main.get('verbose', 0))

    self.apply_global_params(ch2ndcontrol)

    return ch2ndcontrol

def get_ch2ndcontrol(self, params_pyr1, params_pyr2):
    """
    Create a ch2ndcontrol processing object.

    Parameters:
    params_pyr1 (dict): Dictionary of parameters for the first channel
    params_pyr2 (dict): Dictionary of parameters for the second channel

    Returns:
    Ch2ndControl: Ch2ndControl processing object
    """
    params_pyr1 = self.ensure_dictionary(params_pyr1)
    params_pyr2 = self.ensure_dictionary(params_pyr2)

    mod_amp = self.extract(params_pyr2, 'mod_amp', default=None)
    wavelengthInNm = self.extract(params_pyr1, 'wavelengthInNm', default=None)

    ch2ndcontrol = Ch2ndControl(mod_amp, wavelengthInNm, verbose=self._main.get('verbose', 0))

    self.apply_global_params(ch2ndcontrol)

    return ch2ndcontrol

def get_control(self, params):
    """
    Create an intcontrol or iircontrol processing object.

    Parameters:
    params (dict): Dictionary of parameters

    Returns:
    Control: Control processing object
    """
    params = self.ensure_dictionary(params)
    control_type = params.pop('type')

    offset_tag = self.extract(params, 'offset_tag', default=None)
    offset = self._cm.read_data(offset_tag) if offset_tag else None

    offset_gain = self.extract(params, 'offset_gain', default=None)

    if control_type == 'INT':
        return self.get_int_control(params, offset=offset)
    elif control_type == 'INT_STATE':
        return self.get_int_control_state(params, offset=offset)
    elif control_type == 'INT_AUTO':
        return self.get_int_control_autogain(params, offset=offset)
    elif control_type == 'IIR':
        return self.get_iir_control(params, offset=offset)
    elif control_type == 'IIR_STATE':
        return self.get_iir_control_state(params, offset=offset)
    elif control_type == 'DER':
        return self.get_derpre_control(params)
    elif control_type == 'MAT':
        return self.get_mat_control(params, offset=offset)
    elif control_type == 'LUT':
        return self.get_lut_control(params, offset=offset)
    else:
        raise ValueError(f'Unknown control type: {control_type}')

def get_datastore(self):
    """
    Create a datastore processing object.

    Returns:
    Datastore: Datastore processing object
    """
    return Datastore()

def get_demodulate(self, demodulate_params):
    """
    Create a demodulate processing object.

    Parameters:
    demodulate_params (dict): Dictionary of demodulate parameters

    Returns:
    Demodulate: Demodulate processing object
    """
    params = self.ensure_dictionary(demodulate_params)

    modeNumber = params.pop('modeNumber')
    carrierFrequency = params.pop('carrierFrequency')
    dt = params.pop('dt')

    return Demodulate(modeNumber, carrierFrequency, dt)

def get_derpre_control(self, derpre_params):
    """
    Create a derprecontrol processing object.

    Parameters:
    derpre_params (dict): Dictionary of parameters

    Returns:
    DerPreControl: DerPreControl processing object
    """
    params = self.ensure_dictionary(derpre_params)

    nModes = params.pop('nModes')
    nStep = params.pop('nStep')
    delay = params.pop('delay')

    return DerPreControl(nModes, nStep, delay=delay)

def get_disturbance(self, disturbance_params, GPU=None):
    """
    Create a disturbance processing object.

    Parameters:
    disturbance_params (dict): Dictionary of parameters
    GPU (bool, optional): Flag for using GPU

    Returns:
    Disturbance: Disturbance processing object
    """
    useGPU = GPU if GPU is not None else self._gpu

    params = self.ensure_dictionary(disturbance_params)

    verbose = self.extract(params, 'verbose', default=None)
    map_tag = self.extract(params, 'map_tag', default='')
    map_data = self.extract(params, 'map', default=None)
    dataPackageDir = self.extract(params, 'dataPackageDir', default='')
    func_type = params.pop('func_type', None) if map_tag == '' and map_data is None and dataPackageDir == '' else None
    height = params.pop('height', None) if dataPackageDir == '' else None
    nmodes = self.extract(params, 'nmodes', default=None)
    influence_function_tag = self.extract(params, 'influence_function', default=None)
    if not influence_function_tag:
        influence_function_tag = self.extract(params, 'ifunc_tag', default=None)
    dm_type = self.extract(params, 'dm_type', default=None)
    dm_nzern = self.extract(params, 'dm_nzern', default=None)
    dm_start_mode = self.extract(params, 'dm_start_mode', default=None)
    dm_idx_modes = self.extract(params, 'dm_idx_modes', default=None)
    dm_npixels = self.extract(params, 'dm_npixels', default=None)
    dm_obsratio = self.extract(params, 'dm_obsratio', default=None)
    dm_diaratio = self.extract(params, 'dm_diaratio', default=None)
    dm_shiftXYinPixel = self.extract(params, 'dm_shiftXYinPixel', default=None)
    dm_rotInDeg = self.extract(params, 'dm_rotInDeg', default=None)
    dm_magnification = self.extract(params, 'dm_magnification', default=None)
    doNotPutOnGpu = self.extract(params, 'doNotPutOnGpu', default=None)
    dt = self.extract(params, 'dt', default=None)
    pupil_mask_tag = self.extract(params, 'pupil_mask_tag', default='')
    m2c_tag = params.get('m2c') or params.get('m2c_tag')
    m2c = self._cm.read_m2c(m2c_tag) if m2c_tag else None
    time_hist_tag = params.pop('time_hist_tag', None)
    time_hist = self._cm.read_data(time_hist_tag) if time_hist_tag else self.extract(params, 'time_hist', default=None)
    vect_amplitude = self.extract(params, 'vect_amplitude', default=None)
    amp = self.extract(params, 'amp', default=None)
    repeat_amp_mode = self.extract(params, 'repeat_amp_mode', default=None)
    vib_tag = self.extract(params, 'vib_data', default='')
    amp_factor = self.extract(params, 'amp_factor', default=1.0)
    if vib_tag:
        vib = self._cm.readfits('vibrations', vib_tag)
        if vib is None:
            raise ValueError(f'Error reading vibration spectrum: {vib_tag}')
        fr_psd = vib[:, 0]
        psd = vib[:, 1:] * amp_factor
    else:
        fr_psd = self.extract(params, 'fr_psd', default=None)
        psd = self.extract(params, 'psd', default=None)
    continuous_psd = self.extract(params, 'continuous_psd', default=False)
    pixel_pitch = self.extract(params, 'pixel_pitch', default=self._main.pixel_pitch)
    precision = self.extract(params, 'precision', default=self._main.precision)
    seed = self.extract(params, 'seed', default=1)
    linear = self.extract(params, 'linear', default=None)
    min_amplitude = self.extract(params, 'min_amplitude', default=None)
    amp3D = self.extract(params, 'amp3D', default=None)
    resetAmpAtMode = self.extract(params, 'resetAmpAtMode', default=None)
    resetAmp = self.extract(params, 'resetAmp', default=None)
    map_cycle = self.extract(params, 'map_cycle', default=None)
    ncycles = self.extract(params, 'ncycles', default=None)

    dynamic = self.extract(params, 'dataPackageDynamic', default=None)
    hardpact = self.extract(params, 'dataPackageHardpact', default=None)
    softpact = self.extract(params, 'dataPackageSoftpact', default=None)
    t0 = self.extract(params, 'dataPackageT0', default=None)

    modeNumber = self.extract(params, 'modeNumber', default=0)
    carrierAmplitude = self.extract(params, 'carrierAmplitudeInNm', default=0.0)
    carrierFrequency = self.extract(params, 'carrierFrequency', default=0.0)

    mask = None
    if pupil_mask_tag:
        if phase2modes_tag:
            print('if phase2modes_tag is defined then pupil_mask_tag will not be used!')
        pupilstop = self._cm.read_pupilstop(pupil_mask_tag, GPU=useGPU)
        if pupilstop is None:
            raise ValueError(f'Pupil mask tag {pupil_mask_tag} not found.')
        mask = pupilstop.A
        if not dm_npixels:
            dm_npixels = mask.shape[0]

    if map_tag or map_data is not None:
        if map_tag:
            map_data = self._cm.read_data(map_tag)
        if isinstance(map_data, tuple) and len(map_data) == 2:
            mask = map_data[1] if not mask else mask
            map_data = map_data[0]
        if map_data.ndim == 3:
            map_temp = map_data
            idx_mask = np.where(mask)
            map_data = np.array([map_temp[i, :, :].ravel()[idx_mask] for i in range(map_temp.shape[0])])
        disturbance = DisturbanceMap(map_data.shape, pixel_pitch, height, mask, map_data, GPU=useGPU, cycle=map_cycle)
    elif dataPackageDir:
        disturbance = DisturbanceM1Elt(dataPackageDir, dm_npixels, pixel_pitch, dynamic=dynamic, 
                                       hardpact=hardpact, softpact=softpact, t0=t0, GPU=useGPU)
    else:
        if not m2c:
            nmodes_temp = nmodes
        influence_function = self.ifunc_restore(tag=influence_function_tag, type=dm_type, npixels=dm_npixels, 
                                                nmodes=nmodes_temp, nzern=dm_nzern, obsratio=dm_obsratio, 
                                                diaratio=dm_diaratio, start_mode=dm_start_mode, idx_modes=dm_idx_modes, 
                                                mask=mask, doNotPutOnGpu=doNotPutOnGpu)
        if influence_function is None:
            raise ValueError(f'Error reading influence function: {influence_function_tag}')

        if height == 0 and influence_function.mask_inf_func.shape[0] != self._main.pixel_pupil:
            print('ATTENTION: size of disturbance is not equal to main.pixel_pupil!')
            ans = input("Do you want to continue [y/n]? ").lower()
            if ans == 'n':
                raise ValueError('Simulation interrupted by user.')
        else:
            print('Size of disturbance is equal to main.pixel_pupil!')

        if nmodes and dm_start_mode:
            nmodes -= dm_start_mode

        if linear or (min_amplitude and amp):
            modal_pushpull_signal(nmodes, amplitude=amp, vect_amplitude=vect_amplitude, linear=linear, 
                                  min_amplitude=min_amplitude, ncycles=ncycles)
            amp = None

        if amp3D:
            modal_pushpull_signal(nmodes, amplitude=amp, vect_amplitude=vect_amplitude, ncycles=ncycles)
            amp = None
            vect_amplitude[3:5] *= amp3D

        if resetAmpAtMode:
            modal_pushpull_signal(nmodes, amplitude=amp, vect_amplitude=vect_amplitude, linear=linear, 
                                  min_amplitude=min_amplitude, ncycles=ncycles)
            vect_amplitude[resetAmpAtMode:] = resetAmp
            amp = None

        if repeat_amp_mode:
            modal_pushpull_signal(nmodes, amplitude=amp, vect_amplitude=vect_amplitude, linear=linear, 
                                  min_amplitude=min_amplitude, ncycles=ncycles)
            vect_amplitude0 = vect_amplitude[:repeat_amp_mode]
            for i in range(ceil(nmodes / repeat_amp_mode)):
                vect_amplitude[i*repeat_amp_mode:min((i+1)*repeat_amp_mode, nmodes)] = vect_amplitude0[:min(repeat_amp_mode, nmodes - i*repeat_amp_mode)]
            amp = None

        disturbance = Disturbance(func_type, nmodes, pixel_pitch, height, influence_function, time_hist=time_hist, amp=amp, 
                                  vect_amplitude=vect_amplitude, m2c=m2c, precision=precision, psd=psd, fr_psd=fr_psd, 
                                  continuous_psd=continuous_psd, seed=seed, ncycles=ncycles, shiftXYinPixel=dm_shiftXYinPixel, 
                                  rotInDeg=dm_rotInDeg, magnification=dm_magnification, verbose=verbose, GPU=useGPU)
        if amp_factor == 0.0:
            disturbance.active = False

    self.apply_global_params(disturbance)
    disturbance.apply_properties(params)
    return disturbance

def get_dm(self, params, GPU=None, ifunc=None, m2c=None):
    """
    Create a DM processing object.

    Parameters:
    params (dict): Dictionary of parameters
    GPU (bool, optional): Flag for using GPU
    ifunc: Influence function object
    m2c: M2C matrix object

    Returns:
    DM: DM processing object
    """
    useGPU = GPU if GPU is not None else self._gpu
    params = self.ensure_dictionary(params)

    if 'm2c_tag' in params:
        return self.get_dm_m2c(params, GPU=useGPU, ifunc=ifunc, m2c=m2c)
    else:
        settling_time = self.extract(params, 'settling_time', default=None)

        pixel_pitch = self._main['pixel_pitch']
        height = params.pop('height')
        ifunc_tag = self.extract(params, 'ifunc_tag', default=None)
        dm_type = self.extract(params, 'type', default=None)
        nmodes = self.extract(params, 'nmodes', default=None)
        nzern = self.extract(params, 'nzern', default=None)
        start_mode = self.extract(params, 'start_mode', default=None)
        idx_modes = self.extract(params, 'idx_modes', default=None)
        npixels = self.extract(params, 'npixels', default=None)
        obsratio = self.extract(params, 'obsratio', default=None)
        diaratio = self.extract(params, 'diaratio', default=None)
        doNotPutOnGpu = self.extract(params, 'doNotPutOnGpu', default=None)
        pupil_mask_tag = self.extract(params, 'pupil_mask_tag', default='')

        doNotBuildRecProp = self.extract(params, 'doNotBuildRecProp', default=None)
        notSeenByLgs = self.extract(params, 'notSeenByLgs', default=None)

        mask = None
        if pupil_mask_tag:
            if phase2modes_tag:
                print('if phase2modes_tag is defined then pupil_mask_tag will not be used!')
            pupilstop = self._cm.read_pupilstop(pupil_mask_tag, GPU=useGPU)
            if pupilstop is None:
                raise ValueError(f'Pupil mask tag {pupil_mask_tag} not found.')
            mask = pupilstop.A
            if not npixels:
                npixels = mask.shape[0]

        doRestore = False
        if not ifunc or not isinstance(ifunc, SomeValidClassType):
            doRestore = True
        if doRestore:
            ifunc = self.ifunc_restore(tag=ifunc_tag, type=dm_type, npixels=npixels, nmodes=nmodes, nzern=nzern, 
                                       obsratio=obsratio, diaratio=diaratio, mask=mask, start_mode=start_mode, 
                                       idx_modes=idx_modes, doNotPutOnGpu=doNotPutOnGpu)
        if ifunc is None:
            raise ValueError(f'Error reading influence function: {ifunc_tag}')

        dm = DM(pixel_pitch, height, ifunc, GPU=useGPU)
        self.apply_global_params(dm)
        dm.apply_properties(params)
        return dm

def get_dm_m2c(self, params, GPU=None, ifunc=None, m2c=None):
    """
    Create a DM_M2C processing object.

    Parameters:
    params (dict): Dictionary of parameters
    GPU (bool, optional): Flag for using GPU
    ifunc: Influence function object
    m2c: M2C matrix object

    Returns:
    DM_M2C: DM_M2C processing object
    """
    useGPU = GPU if GPU is not None else self._gpu

    params = self.ensure_dictionary(params)
    settling_time = self.extract(params, 'settling_time', default=None)

    pixel_pitch = self._main['pixel_pitch']
    height = params.pop('height')
    ifunc_tag = self.extract(params, 'ifunc_tag', default=None)
    dm_type = self.extract(params, 'type', default=None)
    nmodes = self.extract(params, 'nmodes', default=None)
    nzern = self.extract(params, 'nzern', default=None)
    start_mode = self.extract(params, 'start_mode', default=None)
    idx_modes = self.extract(params, 'idx_modes', default=None)
    npixels = self.extract(params, 'npixels', default=None)
    obsratio = self.extract(params, 'obsratio', default=None)
    diaratio = self.extract(params, 'diaratio', default=None)
    doNotPutOnGpu = self.extract(params, 'doNotPutOnGpu', default=None)
    pupil_mask_tag = self.extract(params, 'pupil_mask_tag', default='')

    doNotBuildRecProp = self.extract(params, 'doNotBuildRecProp', default=None)
    notSeenByLgs = self.extract(params, 'notSeenByLgs', default=None)

    mask = None
    if pupil_mask_tag:
        if phase2modes_tag:
            print('if phase2modes_tag is defined then pupil_mask_tag will not be used!')
        pupilstop = self._cm.read_pupilstop(pupil_mask_tag, GPU=useGPU)
        if pupilstop is None:
            raise ValueError(f'Pupil mask tag {pupil_mask_tag} not found.')
        mask = pupilstop.A
        if not npixels:
            npixels = mask.shape[0]

    if 'm2c_tag' in params:
        if not ifunc:
            ifunc = self.ifunc_restore(tag=ifunc_tag, type=dm_type, npixels=npixels, nzern=nzern, 
                                       obsratio=obsratio, diaratio=diaratio, mask=mask, 
                                       idx_modes=idx_modes, doNotPutOnGpu=doNotPutOnGpu)
        if ifunc is None:
            raise ValueError(f'Error reading influence function: {ifunc_tag}')
        if not m2c:
            m2c_tag = params.pop('m2c_tag')
            m2c = self._cm.read_m2c(m2c_tag)
        m2c_mat = m2c.m2c
        nmodes_temp = nmodes if nmodes else m2c_mat.shape[0]
        m2c.m2c = m2c_mat[start_mode:nmodes_temp] if start_mode else m2c_mat[:nmodes_temp]
    else:
        if not ifunc:
            ifunc = self.ifunc_restore(tag=ifunc_tag, type=dm_type, npixels=npixels, nmodes=nmodes, nzern=nzern, 
                                       obsratio=obsratio, diaratio=diaratio, mask=mask, start_mode=start_mode, 
                                       idx_modes=idx_modes, doNotPutOnGpu=doNotPutOnGpu)
        if ifunc is None:
            raise ValueError(f'Error reading influence function: {ifunc_tag}')
        m2c = M2C()
        nmodes_m2c = nmodes - start_mode if start_mode else nmodes
        m2c.set_m2c(np.identity(nmodes_m2c))

    dm_m2c = DM_M2C(pixel_pitch, height, ifunc, m2c, GPU=useGPU)
    self.apply_global_params(dm_m2c)
    dm_m2c.apply_properties(params)
    return dm_m2c

def get_ef_generator(self, ef_params, zero=False, ignore_extra_keywords=False):
    """
    Create an EF generator processing object.

    Parameters:
    ef_params (dict): Dictionary of parameters
    zero (bool, optional): If set, the generated EF will have a zero phase everywhere
    ignore_extra_keywords (bool, optional): Flag to ignore extra keywords

    Returns:
    EFGenerator: EF generator processing object
    """
    ef_generator = EFGenerator(zero=zero)
    ef_generator.apply_properties(ef_params, ignore_extra_keywords=ignore_extra_keywords)
    self.apply_global_params(ef_generator)
    return ef_generator

def get_ef_product(self):
    """
    Create an EF product processing object.

    Returns:
    EFProduct: EF product processing object
    """
    efprod = EFProduct()
    self.apply_global_params(efprod)
    return efprod

def get_ef_resize(self, params):
    """
    Create an EF resize processing object.

    Parameters:
    params (dict): Dictionary of parameters

    Returns:
    EFResize: EF resize processing object
    """
    params = self.ensure_dictionary(params)

    efresize = EFResize()
    self.apply_global_params(efresize)
    efresize.apply_properties(params)

    return efresize

def get_ef_shift(self, params):
    """
    Create an EF shift processing object.

    Parameters:
    params (dict): Dictionary of parameters

    Returns:
    EFShift: EF shift processing object
    """
    params = self.ensure_dictionary(params)

    qe_factor = self.extract(params, 'qe_factor', default=None)
    shiftwavelengthInNm = self.extract(params, 'shiftWavelengthInNm', default=None)

    efshift = EFShift()
    self.apply_global_params(efshift)
    efshift.apply_properties(params)

    return efshift

def get_ef_spatial_filter(self, params):
    """
    Create an EF spatial filter processing object.

    Parameters:
    params (dict): Dictionary of parameters

    Returns:
    EFSpatialFilter: EF spatial filter processing object
    """
    params = self.ensure_dictionary(params)

    DpupPix = self._main['pixel_pupil']
    pixel_pitch = self._main['pixel_pitch']
    wavelengthInNm = params.pop('wavelengthInNm')
    FoV = params.pop('FoV')
    pup_diam = 30
    ccd_side = 80
    fov_errinf = self.extract(params, 'fov_errinf', default=0.01)
    fov_errsup = self.extract(params, 'fov_errsup', default=100.0)
    fft_res = self.extract(params, 'fft_res', default=3.0)
    fp_obs = self.extract(params, 'fp_obs', default=None)

    result = modulated_pyramid.calc_geometry(DpupPix, pixel_pitch, wavelengthInNm, FoV, pup_diam, ccd_side, 
                                             fov_errinf=fov_errinf, fov_errsup=fov_errsup, /NOTEST)

    wavelengthInNm = result['wavelengthInNm']
    fov_res = result['fov_res']
    fp_masking = result['fp_masking']
    fft_res = result['fft_res']
    tilt_scale = result['tilt_scale']
    fft_sampling = result['fft_sampling']
    fft_padding = result['fft_padding']
    fft_totsize = result['fft_totsize']
    toccd_side = result['toccd_side']
    final_ccd_side = result['final_ccd_side']

    spat_filter = EFSpatialFilter(wavelengthInNm, fov_res, fp_masking, fft_res, tilt_scale, 
                                  fft_sampling, fft_padding, fft_totsize, toccd_side, final_ccd_side, 
                                  fp_obs=fp_obs)

    self.apply_global_params(spat_filter)
    spat_filter.apply_properties(params)

    return spat_filter

def get_ef_variance(self):
    """
    Create an EF variance processing object.

    Returns:
    EFVariance: EF variance processing object
    """
    efvar = EFVariance()
    self.apply_global_params(efvar)
    return efvar

def get_ef_zoom(self, params):
    """
    Create an EF zoom processing object.

    Parameters:
    params (dict): Dictionary of parameters

    Returns:
    EFZoom: EF zoom processing object
    """
    params = self.ensure_dictionary(params)

    efzoom = EFZoom()
    self.apply_global_params(efzoom)
    efzoom.apply_properties(params)

    return efzoom

def get_extended_source(self, params):
    """
    Create an extended source data object.

    Parameters:
    params (dict): Dictionary of parameters

    Returns:
    ExtendedSource: Extended source data object
    """
    params = self.ensure_dictionary(params)

    polar_coordinate = params.pop('polar_coordinate')
    height = params.pop('height')
    magnitude = params.pop('magnitude')
    wavelengthInNm = params.pop('wavelengthInNm')
    d_tel = self._main['pixel_pupil'] * self._main['pixel_pitch']

    band = self.extract(params, 'band', default='')
    zeroPoint = self.extract(params, 'zeroPoint', default=0)
    multiples_fwhm = self.extract(params, 'multiples_fwhm', default=1)
    source_type = self.extract(params, 'type', default='POINT_SOURCE')
    size_obj = self.extract(params, 'size_obj', default=None)
    xy_array = self.extract(params, 'xy_array', default=None)
    sampling_type = self.extract(params, 'sampling_type', default='CARTESIAN')
    layerHeight = self.extract(params, 'layerHeight', default=None)
    intensityProfile = self.extract(params, 'intensityProfile', default=None)
    ttProfile = self.extract(params, 'ttProfile', default=None)
    focusHeight = self.extract(params, 'focusHeight', default=height)
    n_rings = self.extract(params, 'n_rings', default=None)
    show_source = self.extract(params, 'show_source', default=False)
    show_3Dsource = self.extract(params, 'show_3Dsource', default=False)
    fluxThreshold = self.extract(params, 'fluxThreshold', default=0.0)
    npoints = self.extract(params, 'npoints', default=0)
    error_coord = self.extract(params, 'error_coord', default=[0.0, 0.0])
    pixelScalePSF = self.extract(params, 'pixelScalePSF', default=None)
    PSF = self.extract(params, 'PSF', default=None)
    kernel4PSF_conv = self.extract(params, 'kernel4PSF_conv', default=None)
    PSF_tag = self.extract(params, 'PSF_tag', default=None)
    kernel4PSF_conv_tag = self.extract(params, 'kernel4PSF_conv_tag', default=None)
    sampl_dist_step4PSF = self.extract(params, 'sampl_dist_step4PSF', default=None)
    if PSF_tag and PSF is None:
        PSF = self._cm.read_data(PSF_tag)
    if kernel4PSF_conv_tag and kernel4PSF_conv is None:
        kernel4PSF_conv = self._cm.read_data(kernel4PSF_conv_tag)

    polar_coordinate += error_coord

    if 'zenithAngleInDeg' in self._main:
        airmass = 1.0 / np.cos(np.radians(self._main['zenithAngleInDeg']))
    else:
        airmass = 1.0
    height *= airmass
    if layerHeight is not None:
        layerHeight *= airmass
    if focusHeight is not None:
        focusHeight *= airmass

    extended_source = ExtendedSource(polar_coordinate, height, magnitude, wavelengthInNm, multiples_fwhm, d_tel, source_type, 
                                     band=band, zeroPoint=zeroPoint, size_obj=size_obj, xy_array=xy_array, 
                                     sampling_type=sampling_type, layerHeight=layerHeight, intensityProfile=intensityProfile, 
                                     ttProfile=ttProfile, focusHeight=focusHeight, n_rings=n_rings, show_source=show_source, 
                                     show_3Dsource=show_3Dsource, fluxThreshold=fluxThreshold, npoints=npoints, 
                                     PSF=PSF, pixelScalePSF=pixelScalePSF, kernel4PSF_conv=kernel4PSF_conv, 
                                     sampl_dist_step4PSF=sampl_dist_step4PSF)

    self.apply_global_params(extended_source)
    # extended_source.apply_properties(params)

    return extended_source

def get_func_generator(self, params):
    """
    Create a func_generator processing object.

    Parameters:
    params (dict): Dictionary of parameters

    Returns:
    FuncGenerator: FuncGenerator processing object
    """
    params = self.ensure_dictionary(params)
    func_type = self.extract(params, 'func_type', default='SIN')
    nmodes = self.extract(params, 'nmodes', default=None)

    if 'constant_tag' in params:
        tag = params.pop('constant_tag')
        params['constant'] = self._cm.read_data(tag)
    if 'amp_tag' in params:
        tag = params.pop('amp_tag')
        params['amp'] = self._cm.read_data(tag)
    if 'freq_tag' in params:
        tag = params.pop('freq_tag')
        params['freq'] = self._cm.read_data(tag)
    if 'offset_tag' in params:
        tag = params.pop('offset_tag')
        params['offset'] = self._cm.read_data(tag)
    if 'vect_amplitude_tag' in params:
        tag = params.pop('vect_amplitude_tag')
        params['vect_amplitude'] = self._cm.read_data(tag)
    if 'time_hist_tag' in params:
        tag = params.pop('time_hist_tag')
        params['time_hist'] = self._cm.read_data(tag)
    else:
        time_hist = self.extract(params, 'time_hist', default=None)

    funcgenerator = FuncGenerator(func_type, time_hist=time_hist, nmodes=nmodes)

    self.apply_global_params(funcgenerator)
    funcgenerator.apply_properties(params)
    return funcgenerator
