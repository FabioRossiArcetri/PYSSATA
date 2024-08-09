import numpy as np
import os


from pyssata.loop_control import LoopControl
from pyssata.calib_manager import CalibManager
from pyssata.base_processing_obj import BaseProcessingObj

from pyssata.processing_objects.ccd import CCD
from pyssata.processing_objects.atmo_propagation import AtmoPropagation
from pyssata.processing_objects.modulated_pyramid import ModulatedPyramid
from pyssata.processing_objects.processing_container import ProcessingContainer
from pyssata.processing_objects.pyr_slopec import PyrSlopec
from pyssata.processing_objects.datastore import Datastore
from pyssata.processing_objects.modalrec import ModalRec
from pyssata.processing_objects.intcontrol import IntControl
from pyssata.processing_objects.dm import DM

from pyssata.data_objects.source import Source
from pyssata.data_objects.ef import ElectricField

from pyssata.lib.compute_zern_ifunc import compute_zern_ifunc

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
        self._gpu = False
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

    def extract(self, dictionary, key, default=None, optional=False):
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
        if key not in dictionary and default is None and optional is False:
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
                obj.apply_properties({p: self._main[p]}, ignore_extra_keywords=True)

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
                                        zeroPad=zeroPad, 
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

    def get_atmo_container(self, source_list, params_atmo, params_seeing, params_windspeed, params_winddirection):
        """
        Gets a processing container with a full complement of atmospheric objects.

        Parameters:
        source_list (list): List of source objects
        params_atmo (dict): Parameter dictionary for the atmo_evolution object
        params_seeing (dict): Parameter dictionary for the seeing func_generator object
        params_windspeed (dict): Parameter dictionary for the wind speed func_generator object
        params_winddirection (dict): Parameter dictionary for the wind direction func_generator object

        Returns:
        ProcessingContainer: Processing container with atmospheric objects
        """
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

        atmo = self.get_atmo_evolution(params_atmo_copy, source_list)
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

    def get_atmo_cube_container(self, source_list, params_atmo, params_seeing):
        """
        Gets a processing container with a full complement of atmospheric objects for reading cubes.

        Parameters:
        source_list (list): List of source objects
        params_atmo (dict): Parameter dictionary for the atmo_evolution object
        params_seeing (dict): Parameter dictionary for the seeing func_generator object

        Returns:
        ProcessingContainer: Processing container with atmospheric objects
        """

        container = ProcessingContainer()

        atmo = self.get_atmo_readcube(params_atmo, source_list)
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

    def get_atmo_evolution(self, params, source_list):
        """
        Create an atmo_evolution processing object.

        Parameters:
        params (dict): Dictionary of parameters
        source_list (list): List of source objects

        Returns:
        AtmoEvolution: AtmoEvolution processing object
        """
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
                                    precision=precision, seed=seed,
                                    user_defined_phasescreen=user_defined_phasescreen,
                                    force_mcao_fov=force_mcao_fov, make_cycle=make_cycle,
                                    fov_in_m=fov_in_m, pupil_position=pupil_position)

        self.apply_global_params(atmo_evolution)
        atmo_evolution.apply_properties(params)

        return atmo_evolution

    def get_atmo_readcube(self, params, source_list):
        """
        Create an atmo_readcube processing object.

        Parameters:
        params (dict): Dictionary of parameters
        source_list (list): List of source objects

        Returns:
        AtmoReadCube: AtmoReadCube processing object
        """

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

    def get_atmo_propagation(self, params, source_list):
        """
        Create an atmo_propagation processing object.

        Parameters:
        params (dict): Dictionary of atmo parameters
        source_list (list): List of source objects

        Returns:
        AtmoPropagation: AtmoPropagation processing object
        """

        params = self.ensure_dictionary(params)

        pixel_pupil = self._main['pixel_pupil']
        pixel_pitch = self._main['pixel_pitch']

        atmo_propagation = AtmoPropagation(source_list, pixel_pupil, pixel_pitch)

        doFresnel = self.extract(params, 'doFresnel', default=False, optional=True)
        wavelengthInNm = self.extract(params, 'wavelengthInNm', default=False, optional=True)
        if doFresnel:
            atmo_propagation.doFresnel = doFresnel
        if doFresnel and wavelengthInNm is None:
            raise ValueError('get_atmo_propagation: wavelengthInNm is required when doFresnel key is set to correctly simulate physical propagation.')
        if wavelengthInNm:
            atmo_propagation.wavelengthInNm = wavelengthInNm

        self.apply_global_params(atmo_propagation)

        pupil_position = self.extract(params, 'pupil_position', default=[0., 0.], optional=True)
        atmo_propagation.pupil_position = pupil_position

        return atmo_propagation

    def get_atmo_propagation2(self, source_list=None):
        """
        Create an atmo_propagation processing object.

        Parameters:
        source_list (list, optional): List of source objects. If not given, an empty list is initialized.

        Returns:
        AtmoPropagation: AtmoPropagation processing object
        """
        if source_list is None:
            source_list = []

        pixel_pupil = self._main['pixel_pupil']
        pixel_pitch = self._main['pixel_pitch']

        atmo_propagation = AtmoPropagation(source_list, pixel_pupil, pixel_pitch)

        self.apply_global_params(atmo_propagation)

        return atmo_propagation

    def get_calib_manager(self, params=None):
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

        name = self.extract(params, 'name', default=None, optional=True)
        sky_bg_norm = self.extract(params, 'sky_bg_norm', default=None, optional=True)
        pixelGains_tag = self.extract(params, 'pixelGains_tag', default=None, optional=True)
        charge_diffusion = self.extract(params, 'charge_diffusion', default=None, optional=True)
        charge_diffusion_fwhm = self.extract(params, 'charge_diffusion_fwhm', default=None, optional=True)

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

        offset_tag = self.extract(params, 'offset_tag', default=None, optional=True)
        offset = self._cm.read_data(offset_tag) if offset_tag else None

        offset_gain = self.extract(params, 'offset_gain', default=None, optional=True)

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
            settling_time = self.extract(params, 'settling_time', default=None, optional=True)

            pixel_pitch = self._main['pixel_pitch']
            height = params.pop('height')
            ifunc_tag = self.extract(params, 'ifunc_tag', default=None, optional=True)
            dm_type = self.extract(params, 'type', default=None)
            nmodes = self.extract(params, 'nmodes', default=None)
            nzern = self.extract(params, 'nzern', default=None, optional=True)
            start_mode = self.extract(params, 'start_mode', default=0, optional=True)
            idx_modes = self.extract(params, 'idx_modes', default=None, optional=True)
            npixels = self.extract(params, 'npixels', default=None)
            obsratio = self.extract(params, 'obsratio', default=None)
            diaratio = self.extract(params, 'diaratio', default=1.0, optional=True)
            doNotPutOnGpu = self.extract(params, 'doNotPutOnGpu', default=None, optional=True)
            pupil_mask_tag = self.extract(params, 'pupil_mask_tag', default='', optional=True)

            doNotBuildRecProp = self.extract(params, 'doNotBuildRecProp', default=None, optional=True)
            notSeenByLgs = self.extract(params, 'notSeenByLgs', default=None, optional=True)

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

            dm = DM(pixel_pitch, height, ifunc)
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

        result = ModulatedPyramid.calc_geometry(DpupPix, pixel_pitch, wavelengthInNm, FoV, pup_diam, ccd_side, 
                                                fov_errinf=fov_errinf, fov_errsup=fov_errsup, NOTEST=True)

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

    def get_ideal_wfs(self, params):
        """
        Create an Ideal WFS processing object.

        Parameters:
        params (dict): Dictionary of parameters

        Returns:
        IdealWFS: Ideal WFS processing object
        """
        params = self.ensure_dictionary(params)

        n_subap_on_diameter = params.pop('subap_on_diameter')
        sensor_npx = params.pop('sensor_npx')
        sensor_fov = params.pop('sensor_fov')
        subapdata_tag = self.extract(params, 'subapdata_tag', default=None)
        energy_th = params.pop('energy_th')

        lenslet = Lenslet(n_subap_on_diameter)
        ideal_wfs = IdealWFS(lenslet)

        self.apply_global_params(ideal_wfs)
        ideal_wfs.apply_properties(params)

        return ideal_wfs

    def get_ideal_wfs_slopec(self, params):
        """
        Create an Ideal WFS Slopec processing object.

        Parameters:
        params (dict): Dictionary of parameters

        Returns:
        IdealWFSSlopec: Ideal WFS Slopec processing object
        """
        params = self.ensure_dictionary(params)

        computation_time = self.extract(params, 'computation_time', default=None)
        FoV = params.pop('sensor_fov')
        obs = self.extract(params, 'obs', default=None)
        pup_mask_tag = self.extract(params, 'pup_mask_tag', default=None)
        if not pup_mask_tag:
            pup_mask_tag = self.extract(params, 'pupil_mask_tag', default=None)
        thr_value = self.extract(params, 'thr_value', default=None)
        quadcell_mode = self.extract(params, 'quadcell_mode', default=None)
        filtmat_tag = self.extract(params, 'filtmat_tag', default='')

        good_pixels = None
        if pup_mask_tag:
            pupilstop = self._cm.read_pupilstop(pup_mask_tag)
            if pupilstop is None:
                raise ValueError(f'Pupil mask tag {pup_mask_tag} not found for ideal_wfs_slopec.')
            good_pixels = pupilstop.A

        sc = IdealWFSSlopec(self._main.pixel_pitch, FoV, obs=obs, good_pixels=good_pixels)

        if filtmat_tag:
            filtmat = self._cm.read_data(filtmat_tag)
            sc.filtmat = filtmat

        sc.set_property(cm=self._cm)

        self.apply_global_params(sc)
        sc.apply_properties(params)

        return sc

    def get_int_control(self, params, offset=None):
        """
        Create an Int Control (Integrator) processing object.

        Parameters:
        params (dict): Dictionary of parameters
        offset: Offset value

        Returns:
        IntControl: Int Control processing object
        """
        params = self.ensure_dictionary(params)

        gain = params.pop('int_gain')
        ff = self.extract(params, 'ff', default=None, optional=True)
        delay = self.extract(params, 'delay', default=None, optional=True)
        og_shaper_tag = self.extract(params, 'og_shaper_tag', default=None, optional=True)
        og_shaper = self._cm.read_data(og_shaper_tag) if og_shaper_tag else None

        if params.get('opt_dt', 0) == 1:
            intc = IntControlOpt(gain, ff=ff, delay=delay)
        else:
            intc = IntControl(gain, ff=ff, delay=delay)

        if offset is not None:
            intc.offset = offset
        if og_shaper is not None:
            intc.og_shaper = og_shaper

        self.apply_global_params(intc)
        intc.apply_properties(params)

        return intc

    def get_int_control_autogain(self, params, offset=None):
        """
        Create an Int Control AutoGain (Integrator) processing object.

        Parameters:
        params (dict): Dictionary of parameters
        offset: Offset value

        Returns:
        IntControlAutoGain: Int Control AutoGain processing object
        """
        params = self.ensure_dictionary(params)

        gain_vect = params.pop('gain_vect')
        ff = self.extract(params, 'ff', default=None)
        delay = self.extract(params, 'delay', default=None)
        og_shaper_tag = self.extract(params, 'og_shaper_tag', default=None)
        og_shaper = self._cm.read_data(og_shaper_tag) if og_shaper_tag else None
        stepsBeforeChange = self.extract(params, 'stepsBeforeChange', default=None)
        gainLength = self.extract(params, 'gainLength', default=None)

        intc = IntControlAutoGain(gain_vect, gainLength, stepsBeforeChange, ff=ff, delay=delay)

        if offset is not None:
            intc.offset = offset
        if og_shaper is not None:
            intc.og_shaper = og_shaper

        self.apply_global_params(intc)
        intc.apply_properties(params)

        return intc

    def get_int_control_state(self, params, offset=None):
        """
        Create an Int Control State (Integrator) processing object.

        Parameters:
        params (dict): Dictionary of parameters
        offset: Offset value

        Returns:
        IntControlState: Int Control State processing object
        """
        params = self.ensure_dictionary(params)

        gain = params.pop('int_gain')
        ff = self.extract(params, 'ff', default=None)
        delay = self.extract(params, 'delay', default=None)
        og_shaper_tag = self.extract(params, 'og_shaper_tag', default=None)
        og_shaper = self._cm.read_data(og_shaper_tag) if og_shaper_tag else None

        intc = IntControlState(gain, ff=ff, delay=delay)

        if offset is not None:
            intc.offset = offset
        if og_shaper is not None:
            intc.og_shaper = og_shaper

        self.apply_global_params(intc)
        intc.apply_properties(params)

        return intc

    def get_iir_control(self, params, offset=None):
        """
        Create an IIR Control processing object.

        Parameters:
        params (dict): Dictionary of parameters
        offset: Offset value

        Returns:
        IIRControl: IIR Control processing object
        """
        params = self.ensure_dictionary(params)

        delay = self.extract(params, 'delay', default=None)
        og_shaper_tag = self.extract(params, 'og_shaper_tag', default=None)
        og_shaper = self._cm.read_data(og_shaper_tag) if og_shaper_tag else None
        iir_tag = params.pop('iir_tag')
        iirfilter = self._cm.read_iirfilter(iir_tag)

        iirc = IIRControl(iirfilter, delay=delay)

        if offset is not None:
            iirc.offset = offset
        if og_shaper is not None:
            iirc.og_shaper = og_shaper

        self.apply_global_params(iirc)
        iirc.apply_properties(params)

        return iirc

    def get_iir_control_state(self, params, offset=None):
        """
        Create an IIR Control State processing object.

        Parameters:
        params (dict): Dictionary of parameters
        offset: Offset value

        Returns:
        IIRControlState: IIR Control State processing object
        """
        params = self.ensure_dictionary(params)

        delay = self.extract(params, 'delay', default=None)
        og_shaper_tag = self.extract(params, 'og_shaper_tag', default=None)
        og_shaper = self._cm.read_data(og_shaper_tag) if og_shaper_tag else None
        iir_tag = params.pop('iir_tag')
        iirfilter = self._cm.read_iirfilter(iir_tag)

        iirc = IIRControlState(iirfilter, delay=delay)

        if offset is not None:
            iirc.offset = offset
        if og_shaper is not None:
            iirc.og_shaper = og_shaper

        self.apply_global_params(iirc)
        iirc.apply_properties(params)

        return iirc

    def get_lut_control(self, params, offset=None):
        """
        Create a LUT Control (Look-Up Table) processing object.

        Parameters:
        params (dict): Dictionary of parameters
        offset: Offset value

        Returns:
        LUTControl: LUT Control processing object
        """
        params = self.ensure_dictionary(params)

        nmodes = params.pop('nmodes')
        xLut_tag = self.extract(params, 'xLut_tag', default=None)
        yLut_tag = self.extract(params, 'yLut_tag', default=None)
        xLut = self.extract(params, 'xLut', default=None)
        yLut = self.extract(params, 'yLut', default=None)
        delay = self.extract(params, 'delay', default=None)

        if xLut_tag:
            xLut = self._cm.read_data(xLut_tag)
        if yLut_tag:
            yLut = self._cm.read_data(yLut_tag)

        lutc = LUTControl(nmodes, xLut, yLut, delay=delay)

        if offset is not None:
            lutc.offset = offset

        self.apply_global_params(lutc)
        lutc.apply_properties(params)

        return lutc

    def get_mat_control(self, params, A=None, B=None, offset=None):
        """
        Create an Int Control Mat (Integrator) processing object.

        Parameters:
        params (dict): Dictionary of parameters
        A: A matrix
        B: B matrix
        offset: Offset value

        Returns:
        IntControlMat: Int Control Mat processing object
        """
        params = self.ensure_dictionary(params)

        A_tag = self.extract(params, 'A_tag', default=None)
        B_tag = self.extract(params, 'B_tag', default=None)
        gain = self.extract(params, 'int_gain', default=None)
        ff = self.extract(params, 'ff', default=None)
        delay = self.extract(params, 'delay', default=None)
        og_shaper_tag = self.extract(params, 'og_shaper_tag', default=None)
        og_shaper = self._cm.read_data(og_shaper_tag) if og_shaper_tag else None

        if A_tag:
            A = self._cm.read_data(A_tag)
        if B_tag:
            B = self._cm.read_data(B_tag)

        if A is None and ff is not None:
            A = np.diag(ff)
        if B is None and gain is not None:
            B = np.diag(gain)

        intc = IntControlMat(A, B, delay=delay)

        if offset is not None:
            intc.offset = offset
        if og_shaper is not None:
            intc.og_shaper = og_shaper

        self.apply_global_params(intc)
        intc.apply_properties(params)

        return intc

    def get_mat_filter(self, params):
        """
        Create a Mat Filter processing object.

        Parameters:
        params (dict): Dictionary of parameters

        Returns:
        MatFilter: Mat Filter processing object
        """
        params = self.ensure_dictionary(params)

        apply2comm = self.extract(params, 'apply2comm', default=None)
        estimator_tag = params.pop('estimator_tag')
        A_tag = params.pop('A_tag')

        estimator = self._cm.read_data(estimator_tag)
        A = self._cm.read_data(A_tag)

        mat_filter = MatFilter(estimator, A)

        self.apply_global_params(mat_filter)
        mat_filter.apply_properties(params)

        return mat_filter

    def get_kernel(self, params):
        """
        Create a Kernel processing object.

        Parameters:
        params (dict): Dictionary of parameters

        Returns:
        Kernel: Kernel processing object
        """
        if 'zenithAngleInDeg' in self._main:
            airmass = 1.0 / np.cos(np.radians(self._main['zenithAngleInDeg']))
        else:
            airmass = 1.0

        params = self.ensure_dictionary(params)

        kernel = Kernel(cm=self._cm, airmass=airmass)
        self.apply_global_params(kernel)
        kernel.apply_properties(params)

        return kernel

    def get_lgsfocus_container(self, pupil, dm_commands, dm_params, ifunc=None):
        """
        Create a processing container for LGS Focus correction.

        Parameters:
        pupil: Pupil EF
        dm_commands: DM commands
        dm_params (dict): Parameter dictionary for the DM object
        ifunc: Influence function

        Returns:
        ProcessingContainer: Processing container with LGS Focus correction
        """
        container = ProcessingContainer()

        dm = self.get_dm(dm_params, ifunc=ifunc)
        dm.sign = 1

        ef_prod = self.get_ef_product()

        dm.in_command = dm_commands
        ef_prod.in_ef1 = pupil
        ef_prod.in_ef2 = dm.out_layer

        container.add(ef_prod, name='ef_prod', output='out_ef')
        container.add(dm, name='dm')

        return container

    def get_lgstt_container(self, pupil, mod_params, dm_params, res_params=None, ifunc=None, phase2modes=None):
        """
        Create a processing container for LGS Tip-Tilt correction and residual.

        Parameters:
        pupil: Pupil EF
        mod_params (dict): Parameter dictionary for the Modal Analysis object
        dm_params (dict): Parameter dictionary for the DM object
        res_params (dict, optional): Parameter dictionary for the Disturbance object
        ifunc: Influence function
        phase2modes: Phase to modes object

        Returns:
        ProcessingContainer: Processing container with LGS Tip-Tilt correction and residual
        """
        container = ProcessingContainer()

        modalAnalysis = self.get_modalanalysis(mod_params, phase2modes=phase2modes)
        ef_prod1 = self.get_ef_product()
        dm = self.get_dm(dm_params, ifunc=ifunc)

        if res_params:
            ef_prod2 = self.get_ef_product()
            res = self.get_disturbance(res_params)

        modalAnalysis.in_ef = pupil
        dm.in_command = modalAnalysis.out_modes
        ef_prod1.in_ef1 = pupil
        ef_prod1.in_ef2 = dm.out_layer

        if res_params:
            ef_prod2.in_ef1 = ef_prod1.out_ef
            ef_prod2.in_ef2 = res.out_layer

        container.add(modalAnalysis, name='modalAnalysis')
        container.add(dm, name='dm')
        if res_params:
            container.add(ef_prod1, name='ef_prod1')
            container.add(ef_prod2, name='ef_prod2', output='out_ef')
            container.add(res, name='res')
            container.add_output('res', 'output')
        else:
            container.add(ef_prod1, name='ef_prod1', output='out_ef')

        return container

    def get_lift(self, params, params_lens, params_ref, params_control, mode_basis=None, pup_mask=None, GPU=None, display=None):
        """
        Create a LIFT processing object.

        Parameters:
        params (dict): Dictionary of parameters
        params_lens (dict): Dictionary of parameters for the single lens SH object
        params_ref (dict): Dictionary of parameters for the reference aberration
        params_control (dict): Dictionary of control parameters
        mode_basis (array, optional): Array with modal basis for LIFT modal estimation
        pup_mask (array, optional): Array with pupil mask
        GPU (bool, optional): Use GPU if available
        display (bool, optional): Display settings

        Returns:
        LIFT: LIFT processing object
        """
        useGPU = GPU if GPU is not None else self._gpu

        params = self.ensure_dictionary(params)
        params_lens = self.ensure_dictionary(params_lens)
        params_ref = self.ensure_dictionary(params_ref)
        params_control = self.ensure_dictionary(params_control)

        nmodes_est = params.pop('nmodes_est')
        computation_time = self.extract(params, 'computation_time', default=None)
        subapdata_tag = self.extract(params, 'subapdata_tag', default='')
        chrom_lambdas = self.extract(params, 'chrom_lambdas', default=None)
        chrom_lambda0 = self.extract(params, 'chrom_lambda0', default=None)
        chrom_photo = self.extract(params, 'chrom_photo', default=None)
        regul = self.extract(params, 'regul', default=None)
        ron = self.extract(params, 'ron', default=None)
        nophotnoise = self.extract(params, 'nophotnoise', default=None)
        n_iter = self.extract(params, 'n_iter', default=None)
        ncrop = self.extract(params, 'ncrop', default=None)
        ftccd = self.extract(params, 'ftccd', default=None)
        thresh = self.extract(params, 'thresh', default=None)
        display = self.extract(params, 'display', default=None)
        silent = self.extract(params, 'silent', default=None)
        fixed = self.extract(params, 'fixed', default=None)
        flux = self.extract(params, 'flux', default=None)
        noconv = self.extract(params, 'noconv', default=None)
        estim_flux = self.extract(params, 'estim_flux', default=None)
        padding = self.extract(params, 'padding', default=None)
        corr = self.extract(params, 'corr', default=None)
        lowsamp = self.extract(params, 'lowsamp', default=None)
        gauss = self.extract(params, 'gauss', default=None)
        tfdir = self.extract(params, 'tfdir', default=None)
        pup_mask_tag = self.extract(params, 'pup_mask_tag', default=None)
        modes_tag = self.extract(params, 'modes_tag', default=None)
        bootstrap = self.extract(params, 'bootstrap', default=None)
        gain = self.extract(params_control, 'int_gain', default=None)
        ncutmodes = self.extract(params, 'ncutmodes', default=None)
        fft_oversample = self.extract(params, 'fft_oversample', default=None)
        norm = self.extract(params, 'norm', default=None)

        if pup_mask_tag and pup_mask is not None:
            print('WARNING: pup_mask_tag will be ignored in factory.get_lift, because pup_mask input is defined')
        if modes_tag and mode_basis is not None:
            print('WARNING: modes_tag will be ignored in factory.get_lift, because mode_basis input is defined')

        if regul is not None:
            regul = np.diag(regul)

        if isinstance(chrom_lambdas, (list, np.ndarray)) and len(chrom_lambdas) == 1:
            chrom_lambdas = chrom_lambdas[0]
        if isinstance(chrom_lambda0, (list, np.ndarray)) and len(chrom_lambda0) == 1:
            chrom_lambda0 = chrom_lambda0[0]
        if isinstance(chrom_photo, (list, np.ndarray)) and len(chrom_photo) == 1:
            chrom_photo = chrom_photo[0]
        chrom = {'_lambdas': chrom_lambdas, '_lambda0': chrom_lambda0, '_photo': chrom_photo}

        rad2asec = 3600.0 * 360.0 / (2 * np.pi)
        wavelengthInM = params_lens['wavelengthInNm'] * 1e-9
        if isinstance(wavelengthInM, (list, np.ndarray)) and len(wavelengthInM) == 1:
            wavelengthInM = wavelengthInM[0]
        diameterInM = self._main['pixel_pupil'] * self._main['pixel_pitch'] / params_lens['subap_on_diameter']
        oversamp = (wavelengthInM / diameterInM * rad2asec) / (2 * params_lens['sensor_fov'] / params_lens['sensor_npx'])
        nmodes_mb = max(nmodes_est, len(params_ref['constant']))

        if pup_mask is None:
            if pup_mask_tag:
                pupilstop = self._cm.read_pupilstop(pup_mask_tag)
                if pupilstop is None:
                    raise ValueError(f'Pupil mask tag {pup_mask_tag} not found in factory.get_lift.')
                pup_mask = pupilstop.A
            else:
                if 'dm_obsratio' in params_ref:
                    diaratio = self.extract(params_ref, 'dm_diaratio', default=None)
                    pup_mask = make_mask(self._main['pixel_pupil'], obs=params_ref['dm_obsratio'], dia=diaratio)
                elif 'pupil_mask_tag' in params_ref:
                    pup_mask = self._cm.read_pupilstop(params_ref['pupil_mask_tag'])
                    if pup_mask is None:
                        raise ValueError(f'Pupil mask tag {params_ref["pupil_mask_tag"]} not found in factory.get_lift.')
                else:
                    raise ValueError('factory.get_lift does not know how to build mask')

        if mode_basis is None:
            if not pup_mask_tag or not modes_tag:
                mode_basis = zern2phi(self._main['pixel_pupil'], nmodes_mb, mask=pup_mask, no_round_mask=True)
                mode_basis = mode_basis.reshape((nmodes_mb, self._main['pixel_pupil'] ** 2))
            else:
                pupdiam = pup_mask.shape[1]
                ifunc = self._cm.read_ifunc(modes_tag)
                ifuncMat = ifunc.influence_function
                ifuncMask = ifunc.mask_inf_func
                ifuncIdx = np.where(ifuncMask)
                mode_basis = np.zeros((nmodes_mb, pupdiam ** 2), dtype=float)
                for i in range(nmodes_mb):
                    mode_basis[i, ifuncIdx] = ifuncMat[i, :]

        airefInRad = params_ref['constant'] / (wavelengthInM * 1e9) * 2 * np.pi
        if gain is not None:
            gain = gain[0]

        lift = LIFT(
            precision=self._main['precision'], oversamp=oversamp, airef=airefInRad,
            ron=ron, chrom=chrom, nophotnoise=nophotnoise, nmodes_est=nmodes_est,
            n_iter=n_iter, ncrop=ncrop, ftccd=ftccd, thresh=thresh,
            display=display, silent=silent, fixed=fixed, flux=flux, noconv=noconv,
            estim_flux=estim_flux, regul=regul, padding=padding, pup_mask=pup_mask,
            mode_basis=mode_basis, corr=corr, lowsamp=lowsamp, gauss=gauss, tfdir=tfdir,
            bootstrap=bootstrap, gain=gain, ncutmodes=ncutmodes, fft_oversample=fft_oversample,
            pixel_pitch_mode_basis=self._main['pixel_pitch']
        )

        if subapdata_tag:
            subapdata = self._cm.read_subaps(subapdata_tag)
            lift.subapdata = subapdata

        self.apply_global_params(lift)
        lift.apply_properties(params)

        return lift

    def get_lift_sh_slopec(self, params, mode_basis=None, pup_mask=None, GPU=None):
        """
        Create a LIFT SH Slopec processing object.

        Parameters:
        params (dict): Dictionary of parameters
        mode_basis (array, optional): Array with modal basis for LIFT modal estimation
        pup_mask (array, optional): Array with pupil mask
        GPU (bool, optional): Use GPU if available

        Returns:
        LIFT_SH_Slopec: LIFT SH Slopec processing object
        """
        params = self.ensure_dictionary(params)

        computation_time = self.extract(params, 'computation_time', default=None)
        filtName = self.extract(params, 'filtName', default='')
        display = self.extract(params, 'display', default=False)

        single_modal_basis = self.extract(params, 'single_modal_basis', default=False)
        substitute_nzern = self.extract(params, 'substitute_nzern', default=0)

        sc = LIFT_SH_Slopec()

        lift_list = []

        modes_tag = self.extract(params, 'lift_modes_tag', default=None)
        lift_params = {k[5:]: params.pop(k) for k in list(params.keys()) if k.startswith('lift_') or k.startswith('LIFT_')}
        control_params = {'int_gain': params.pop('control_int_gain')[0]}
        sh_params = {
            'wavelengthInNm': params.pop('sh_wavelengthInNm')[0],
            'subap_on_diameter': params.pop('sh_subap_on_diameter')[0],
            'sensor_fov': params.pop('sh_sensor_fov')[0],
            'sensor_npx': params.pop('sh_sensor_npx')[0]
        }

        if pup_mask is None:
            pup_mask_tag = self.extract(lift_params, 'pup_mask_tag', default=None)
            pupilstop = self._cm.read_pupilstop(pup_mask_tag)
            if pupilstop is None:
                raise ValueError(f'Pupil mask tag {pup_mask_tag} not found in factory.get_lift.')
            pup_mask = pupilstop.A

        pup_mask_diameter = pup_mask.shape[0]

        aberr_map_tag = self.extract(params, 'aberr_map_tag', default='')
        aberr_coeff = self.extract(params, 'aberr_constant', default='')
        aberr_ifunc_tag = self.extract(params, 'aberr_ifunc_tag', default=None)
        aberr_type = self.extract(params, 'aberr_type', default=None)
        aberr_nzern = self.extract(params, 'aberr_nzern', default=None)
        aberr_start_mode = self.extract(params, 'aberr_start_mode', default=None)
        aberr_npixels = self.extract(params, 'aberr_npixels', default=None)
        aberr_obsratio = self.extract(params, 'aberr_obsratio', default=None)
        aberr_diaratio = self.extract(params, 'aberr_diaratio', default=None)
        doNotPutOnGpu = self.extract(params, 'aberr_doNotPutOnGpu', default=None)

        if aberr_map_tag:
            map_aberr = self._cm.read_data(aberr_map_tag)
        else:
            nmodes_aberr = max([lift_params['nmodes_est'], len(aberr_coeff)])
            ifunc_aberr = self.ifunc_restore(
                tag=aberr_ifunc_tag, type=aberr_type, npixels=aberr_npixels, nmodes=nmodes_aberr,
                nzern=aberr_nzern, obsratio=aberr_obsratio, diaratio=aberr_diaratio,
                start_mode=aberr_start_mode, mask=pup_mask, doNotPutOnGpu=doNotPutOnGpu
            )
            ifuncMat_aberr = ifunc_aberr.influence_function
            ifuncMask_aberr = ifunc_aberr.mask_inf_func
            sMaskAber = ifuncMask_aberr.shape
            ifuncIdx_aberr = np.where(ifuncMask_aberr)
            map_aberr_2d = np.dot(aberr_coeff, ifuncMat_aberr)
            map_aberr = np.zeros(sMaskAber, dtype=aberr_coeff.dtype)
            map_aberr[ifuncIdx_aberr] = map_aberr_2d

        subapdata = self._cm.read_subaps(params['subapdata_tag'])
        if subapdata is None:
            print(f'subapdata_tag: {subapdata_tag} is not valid in factory.get_lift.')

        idxModalBaseAll = []
        projMat = None

        for i in range(subapdata['n_subaps']):
            iSaMap = subapdata['map'][i] % subapdata['nx']
            jSaMap = subapdata['map'][i] // subapdata['nx']
            dimSaPup = self._main['pixel_pupil'] // subapdata['nx']
            range_x = [iSaMap * dimSaPup, (iSaMap + 1) * dimSaPup]
            range_y = [jSaMap * dimSaPup, (jSaMap + 1) * dimSaPup]

            pup_mask_ith = pup_mask[range_x[0]:range_x[1], range_y[0]:range_y[1]]
            idx_pup_mask_ith = np.where(pup_mask_ith)
            if len(idx_pup_mask_ith[0]) == 0:
                print(f'factory.get_lift_sh_slopec: mask is zero for i={i}')
                continue

            aberr_ith = map_aberr[range_x[0]:range_x[1], range_y[0]:range_y[1]]

            if single_modal_basis or mode_basis is None:
                if not modes_tag:
                    mode_basis_ith = zern2phi(dimSaPup, lift_params['nmodes_est'], mask=pup_mask_ith, no_round_mask=True)
                    mode_basis_ith = mode_basis_ith.reshape((lift_params['nmodes_est'], dimSaPup ** 2))
                    ifuncMask_diameter = dimSaPup
                else:
                    ifunc = self._cm.read_ifunc(modes_tag)
                    ifuncMat = ifunc.influence_function
                    ifuncMask = ifunc.mask_inf_func
                    ifuncIdx = np.where(ifuncMask)
                    ifuncMask_diameter = ifuncMask.shape[0]
                    mode_basis_ith = np.zeros((lift_params['nmodes_est'], ifuncMask_diameter ** 2), dtype=ifuncMat.dtype)
                    for j in range(lift_params['nmodes_est']):
                        mode_basis_ith[j, ifuncIdx] = ifuncMat[j, :]
            else:
                mode_basis_ith = mode_basis
                ifuncMask = pup_mask
                ifuncIdx = np.where(ifuncMask)
                ifuncMask_diameter = ifuncMask.shape[0]

            lift_params_ith = lift_params.copy()

            if ifuncMask_diameter == self._main['pixel_pupil']:
                ifuncMask = ifuncMask[range_x[0]:range_x[1], range_y[0]:range_y[1]]
                ifuncIdx = np.where(ifuncMask)
                mode_basis_ith = mode_basis_ith.reshape((lift_params['nmodes_est'], self._main['pixel_pupil'], self._main['pixel_pupil']))
                mode_basis_ith = mode_basis_ith[:, range_x[0]:range_x[1], range_y[0]:range_y[1]]
                mode_basis_temp = mode_basis_ith.reshape((lift_params['nmodes_est'], dimSaPup ** 2))

                isItPiston = np.zeros(lift_params['nmodes_est'], dtype=int)
                isItGood = np.zeros(lift_params['nmodes_est'], dtype=int)
                for j in range(lift_params['nmodes_est']):
                    temp = mode_basis_ith[j, :, :][ifuncIdx]
                    if np.max(temp) != np.min(temp):
                        yhist, xhist = np.histogram(temp, bins=int((np.max(temp) - np.min(temp)) * 0.01))
                        idxYhistTemp = np.where(yhist > np.sum(yhist) * 1e-3)[0]
                        if len(idxYhistTemp) == 1:
                            isItPiston[j] = 2
                        elif len(idxYhistTemp) == 2:
                            isItPiston[j] = 1
                    else:
                        isItPiston[j] = 2

                idxPiston = np.where(isItPiston == 1)[0]
                isItGood = (isItPiston <= 1).astype(int)
                if len(idxPiston) == 2:
                    isItBad = np.zeros(len(idxPiston), dtype=bool)
                    for j in range(len(idxPiston)):
                        for k in range(len(idxPiston)):
                            if k > j:
                                continue
                            tempj = mode_basis_ith[idxPiston[j], :, :][ifuncIdx]
                            tempk = mode_basis_ith[idxPiston[k], :, :][ifuncIdx]
                            temp = tempj + tempk
                            yhist, xhist = np.histogram(temp, bins=int((np.max(temp) - np.min(temp)) * 0.01))
                            idxYhistTemp = np.where(yhist > np.max(yhist) * 1e-3)[0]
                            if len(idxYhistTemp) == 1:
                                if np.sum(tempj) > np.sum(tempk):
                                    isItBad[j] = True
                                else:
                                    isItBad[k] = True
                    idxIsItBad = np.where(isItBad)[0]
                    if len(idxIsItBad) > 0:
                        isItGood[idxPiston[idxIsItBad]] = 0

                if len(idxPiston) == 3:
                    isItBad = np.zeros(len(idxPiston), dtype=bool)
                    for j in range(len(idxPiston)):
                        for k in range(len(idxPiston)):
                            for l in range(len(idxPiston)):
                                if k > j or l > k:
                                    continue
                                tempj = mode_basis_ith[idxPiston[j], :, :][ifuncIdx]
                                tempk = mode_basis_ith[idxPiston[k], :, :][ifuncIdx]
                                templ = mode_basis_ith[idxPiston[l], :, :][ifuncIdx]
                                temp = tempj + tempk + templ
                                if np.max(temp) == np.min(temp):
                                    isItBad[j] = True
                    idxIsItBad = np.where(isItBad)[0]
                    if len(idxIsItBad) > 0:
                        isItGood[idxPiston[idxIsItBad]] = 0

                idxTestMB = np.where(isItGood)[0]
                print('countTestMB', len(idxTestMB))

                lift_params_ith['nmodes_est'] = len(idxTestMB)
                if len(idxTestMB) > 0:
                    idxModalBaseAll.append(idxTestMB)
                    mode_basis_ith = mode_basis_ith[idxModalBaseAll[i], :, :]
                else:
                    print(f'factory.get_lift_sh_slopec: all modes are zero for i={i}')
                    continue

                mode_basis_ith = mode_basis_ith.reshape((mode_basis_ith.shape[0], dimSaPup ** 2))

            if substitute_nzern > 0:
                mode_basis_temp = zern2phi(dimSaPup, substitute_nzern, mask=pup_mask_ith, no_round_mask=True)
                mode_basis_ith[:substitute_nzern, :] = mode_basis_temp.reshape((substitute_nzern, dimSaPup ** 2))

            n_mode_basis_ith = mode_basis_ith.shape[0]
            mode_basis_ith_3d = mode_basis_ith.reshape((n_mode_basis_ith, dimSaPup, dimSaPup))
            mode_basis_ith_2d_idx = np.zeros((n_mode_basis_ith, len(idx_pup_mask_ith[0])), dtype=mode_basis_ith.dtype)
            for j in range(n_mode_basis_ith):
                mode_basis_ith_2d_idx[j, :] = mode_basis_ith_3d[j, :, :].flatten()[idx_pup_mask_ith]

            inv_mode_basis_ith = np.linalg.pinv(mode_basis_ith_2d_idx)

            if mode_basis is not None:
                projMatTemp = np.dot(mode_basis_ith_2d_idx, inv_mode_basis_ith)
                if ifuncMask_diameter == self._main['pixel_pupil']:
                    if projMatTemp.shape[0] != lift_params['nmodes_est']:
                        projMatIth = np.zeros((projMatTemp.shape[0], lift_params['nmodes_est']), dtype=projMatTemp.dtype)
                        projMatIth[:, idxModalBaseAll[i]] = projMatTemp
                    else:
                        projMatIth = projMatTemp
                else:
                    projMatIth = projMatTemp
                projMat = np.hstack((projMat, projMatIth)) if projMat is not None else projMatIth

            aberr_coeff_ith = np.dot(inv_mode_basis_ith, aberr_ith[idx_pup_mask_ith])
            wfs_aberr_params = {'constant': aberr_coeff_ith}

            lift_list.append(
                self.get_lift(
                    lift_params_ith, sh_params.copy(),
                    wfs_aberr_params.copy(), control_params.copy(),
                    mode_basis=mode_basis_ith.astype(float), pup_mask=pup_mask_ith.astype(float),
                    display=display
                )
            )

        for i in range(lift_params['nmodes_est']):
            normFact = sum(1 for j in idxModalBaseAll if np.min(np.abs(j - i)) == 0)
            if normFact > 0:
                projMat[:, i] /= normFact

        sc.lift_list = lift_list
        sc.projMat = projMat

        sc.setproperty(cm=self._cm)
        self.apply_global_params(sc)
        sc.apply_properties(params)

        return sc

    def get_loop_control(self):
        """
        Create a loop_control object.

        Returns:
        LoopControl: loop_control object
        """
        return LoopControl(run_time=self._main['total_time'], dt=self._main['time_step'])

    def get_m2crec(self, params):
        """
        Create a m2crec processing object.

        Parameters:
        params (dict): Dictionary of parameters

        Returns:
        M2CRec: m2crec processing object
        """
        params = self.ensure_dictionary(params)

        if 'm2c_tag' in params:
            m2c_tag = params.pop('m2c_tag')
            m2c = self._cm.read_m2c(m2c_tag)
        else:
            nmodes = params.pop('nmodes')
            m2c = M2C()
            m2c.set_m2c(np.identity(nmodes))

        m2crec = M2CRec(m2c)

        self.apply_global_params(m2crec)
        m2crec.apply_properties(params)
        return m2crec

    def get_modalanalysis(self, params, phase2modes=None):
        """
        Create a modalanalysis processing object.

        Parameters:
        params (dict): Dictionary of parameters
        phase2modes (optional): Phase to modes object

        Returns:
        ModalAnalysis: modalanalysis processing object
        """
        params = self.ensure_dictionary(params)

        phase2modes_tag = self.extract(params, 'phase2modes_tag', default=None)
        type_ = self.extract(params, 'type', default=None)
        nmodes = self.extract(params, 'nmodes', default=None)
        start_mode = self.extract(params, 'start_mode', default=None)
        idx_modes = self.extract(params, 'idx_modes', default=None)
        nzern = self.extract(params, 'nzern', default=None)
        npixels = self.extract(params, 'npixels', default=None)
        pupil_mask_tag = self.extract(params, 'pupil_mask_tag', default='')
        obsratio = self.extract(params, 'obsratio', default=None)
        diaratio = self.extract(params, 'diaratio', default=None)
        doNotPutOnGpu = self.extract(params, 'doNotPutOnGpu', default=None)
        zeroPad = self.extract(params, 'zeroPadp2m', default='')

        if pupil_mask_tag:
            if phase2modes_tag:
                print('if phase2modes_tag is defined then pupil_mask_tag will not be used!')
            pupilstop = self._cm.read_pupilstop(pupil_mask_tag, GPU=GPU)
            if pupilstop is None:
                raise ValueError(f'Pupil mask tag {pupil_mask_tag} not found.')
            mask = pupilstop.A
            if not npixels:
                npixels = mask.shape[0]

        if not phase2modes:
            phase2modes = self.ifunc_restore(
                tag=phase2modes_tag, type=type_, npixels=npixels, nmodes=nmodes,
                nzern=nzern, obsratio=obsratio, diaratio=diaratio, start_mode=start_mode,
                idx_modes=idx_modes, mask=mask, doNotPutOnGpu=doNotPutOnGpu,
                return_inv=True, zeroPad=zeroPad
            )
            if phase2modes is None:
                raise ValueError(f'Error reading influence function: {phase2modes_tag}')
            print('phase2modes restored!')
        else:
            print('phase2modes already defined!')

        modalanalysis = ModalAnalysis(phase2modes)

        self.apply_global_params(modalanalysis)
        modalanalysis.apply_properties(params)
        return modalanalysis

    def get_modalanalysis_slopec(self):
        """
        Create a modalanalysis_slopec processing object.

        Returns:
        ModalAnalysisSlopec: modalanalysis_slopec object
        """
        return ModalAnalysisSlopec()

    def get_modalanalysis_wfs(self, params, phase2modes=None):
        """
        Create a modalanalysis_wfs processing object.

        Parameters:
        params (dict): Dictionary of parameters
        phase2modes (object, optional): ifunc object

        Returns:
        ModalAnalysisWFS: modalanalysis_wfs object
        """
        params = self.ensure_dictionary(params)

        phase2modes_tag = self.extract(params, 'phase2modes_tag', default=None)
        type_ = self.extract(params, 'type', default=None)
        nmodes = self.extract(params, 'nmodes', default=None)
        nzern = self.extract(params, 'nzern', default=None)
        npixels = self.extract(params, 'npixels', default=None)
        pupil_mask_tag = self.extract(params, 'pupil_mask_tag', default='')
        obsratio = self.extract(params, 'obsratio', default=None)
        diaratio = self.extract(params, 'diaratio', default=None)
        zeroPad = self.extract(params, 'zeroPadp2m', default='')

        if pupil_mask_tag:
            if phase2modes_tag:
                print('if phase2modes_tag is defined then pupil_mask_tag will not be used!')
            pupilstop = self._cm.read_pupilstop(pupil_mask_tag, GPU=useGPU)
            if pupilstop is None:
                raise ValueError(f'Pupil mask tag {pupil_mask_tag} not found.')
            mask = pupilstop.A
            if npixels is None:
                npixels = mask.shape[0]

        if not phase2modes:
            phase2modes = self.ifunc_restore(tag=phase2modes_tag, type=type_, npixels=npixels, nmodes=nmodes,
                                            nzern=nzern, obsratio=obsratio, diaratio=diaratio,
                                            mask=mask, return_inv=True, zeroPad=zeroPad)
            print('phase2modes restored!')
        else:
            print('phase2modes already defined!')

        if phase2modes is None:
            raise ValueError(f'Error reading influence function: {phase2modes_tag}')

        modalanalysis_wfs = ModalAnalysisWFS(phase2modes)
        self.apply_global_params(modalanalysis_wfs)
        modalanalysis_wfs.apply_properties(params)
        return modalanalysis_wfs

    def get_modalrec(self, params, recmat=None, intmat=None):
        """
        Create a modalrec processing object.

        Parameters:
        params (dict): Dictionary of parameters
        recmat (optional): Reconstruction matrix
        intmat (optional): Interaction matrix

        Returns:
        ModalRec: modalrec object
        """
        params = self.ensure_dictionary(params)

        intmat_tag = self.extract(params, 'intmat_tag', default=None, optional=True)
        nmodes = self.extract(params, 'nmodes', default=None, optional=True)
        recmat_tag = self.extract(params, 'recmat_tag', default=None, optional=True)
        projmat_tag = self.extract(params, 'projmat_tag', default=None, optional=True)
        filtmat_tag = self.extract(params, 'filtmat_tag', default=None, optional=True)

        identity = self.extract(params, 'identity', default=False, optional=True)
        ncutmodes = self.extract(params, 'ncutmodes', default=None, optional=True)
        nSlopesToBeDiscarded = self.extract(params, 'nSlopesToBeDiscarded', default=None, optional=True)
        polc = self.extract(params, 'polc', default=False)
        dmNumber = self.extract(params, 'dmNumber', default=None, optional=True)
        noProj = self.extract(params, 'noProj', default=False)
        projmat = None

        if params.get('mPCuRed_tag'):
            return self.get_modalrec_cured(params)
        if params.get('nn_python'):
            return self.get_modalrec_nn_python(params)
        if params.get('WeightsBiases_tag'):
            return self.get_modalrec_nn(params)
        if params.get('WeightsBiases1_tag'):
            return self.get_modalrec_nn_multi(params)

        if polc:
            if identity:
                raise ValueError('identity cannot be set with POLC.')
            if ncutmodes is not None:
                raise ValueError('ncutmodes cannot be set with POLC.')
            if recmat is None:
                recmat = self._cm.read_rec(recmat_tag, doNotPutOnGpu=doNotPutOnGpu)
            if intmat is None:
                intmat = self._cm.read_im(intmat_tag, doNotPutOnGpu=doNotPutOnGpu)
            if intmat is None:
                raise ValueError(f'WARNING: intmat is null. intmat_tag is: {intmat_tag}')
        else:
            if recmat is None:
                if identity:
                    recmat = RecMat()
                    if nmodes is None:
                        raise ValueError('modalrec nmodes key must be set!')
                    recmat.recmat = np.identity(nmodes)
                else:
                    if recmat_tag and intmat_tag:
                        intmat = self._cm.read_im(intmat_tag, doNotPutOnGpu=doNotPutOnGpu)
                        if nmodes:
                            nmodes_intmat = intmat.size[0]
                            intmat.reduce_size(nmodes_intmat - nmodes)
                        if nSlopesToBeDiscarded:
                            intmat.reduce_slopes(nSlopesToBeDiscarded)
                        recmat = RecMat()
                        recmat.recmat = intmat.intmat
                    else:
                        recmat = self._cm.read_rec(recmat_tag)

            if ncutmodes:
                if recmat is not None:
                    recmat.reduce_size(ncutmodes)
                else:
                    print('recmat cannot be reduced because it is null.')

        if projmat_tag and not noProj:
            projmat = self._cm.read_rec(projmat_tag, doNotPutOnGpu=doNotPutOnGpu)

        if recmat is None:
            if projmat and recmat.proj_list and not noProj:
                if dmNumber is not None:
                    if dmNumber <= 0:
                        raise ValueError('dmNumber must be > 0')
                    projmat = RecMat()
                    projmat.recmat = recmat.proj_list[dmNumber - 1]
                else:
                    raise ValueError('dmNumber (>0) must be defined if projmat_tag is not defined!')

        if filtmat_tag:
            filtmat = self._cm.read_data(filtmat_tag)
            recmat_orig = recmat.recmat
            recmat_new = np.matmul(recmat_orig, filtmat)
            recmat.recmat = recmat_new
            print('recmat updated with filmat!')
            stop

        modalrec = ModalRec(recmat, intmat=intmat, projmat=projmat, polc=polc)
        self.apply_global_params(modalrec)
        modalrec.apply_properties(params)
        return modalrec

    def get_modalrec_nn(self, params):
        """
        Create a modalrec_nn processing object.

        Parameters:
        params (dict): Dictionary of parameters

        Returns:
        ModalRecNN: modalrec_nn object
        """
        params = self.ensure_dictionary(params)

        WeightsBiases_tag = params.pop('WeightsBiases_tag')
        noBias = self.extract(params, 'noBias', default=False)
        nnFunc = self.extract(params, 'nnFunc', default='lin')
        WeightsBiases_list = self._cm.read_data_ext(WeightsBiases_tag)
        nEle = WeightsBiases_list.count() if noBias else WeightsBiases_list.count() // 2

        layerWeights = []
        layerBiases = [] if not noBias else None
        for i in range(nEle):
            if noBias:
                layerWeights.append(WeightsBiases_list[i])
            else:
                layerWeights.append(WeightsBiases_list[2 * i])
                layerBiases.append(WeightsBiases_list[2 * i + 1])

        modalrec_nn = ModalRecNN(layerWeights, layerBiases, nnFunc, verbose=verbose)
        self.apply_global_params(modalrec_nn)
        modalrec_nn.apply_properties(params)
        return modalrec_nn

    def get_modalrec_nn_multi(self, params):
        """
        Create a modalrec_nn_multi processing object.

        Parameters:
        params (dict): Dictionary of parameters

        Returns:
        ModalRecNNMulti: modalrec_nn_multi object
        """
        params = self.ensure_dictionary(params)

        n = 1
        while params.get(f'WeightsBiases{n}_tag'):
            n += 1
        n -= 1

        layerWeights_list = []
        layerBiases_list = []
        nnFunc_list = []

        for j in range(n):
            str_j = str(j + 1)
            WeightsBiases_tag = params.pop(f'WeightsBiases{str_j}_tag')
            noBias = self.extract(params, f'noBias{str_j}', default=False)
            nnFunc = self.extract(params, f'nnFunc{str_j}', default='lin')
            WeightsBiases_list = self._cm.read_data_ext(WeightsBiases_tag)
            nEle = WeightsBiases_list.count() if noBias else WeightsBiases_list.count() // 2

            layerWeights = []
            layerBiases = [] if not noBias else None
            for i in range(nEle):
                if noBias:
                    layerWeights.append(WeightsBiases_list[i])
                else:
                    layerWeights.append(WeightsBiases_list[2 * i])
                    layerBiases.append(WeightsBiases_list[2 * i + 1])

            layerWeights_list.append(layerWeights)
            layerBiases_list.append(layerBiases)
            nnFunc_list.append(nnFunc)

        modalrec_nn_multi = ModalRecNNMulti(layerWeights_list, layerBiases_list, nnFunc_list, verbose=verbose)
        self.apply_global_params(modalrec_nn_multi)
        modalrec_nn_multi.apply_properties(params)
        return modalrec_nn_multi

    def get_modalrec_nn_python(self, params):
        """
        Create a modalrec_nn_python processing object.

        Parameters:
        params (dict): Dictionary of parameters

        Returns:
        ModalRecNNPython: modalrec_nn_python object
        """
        params = self.ensure_dictionary(params)
        params.pop('nn_python')

        modalrec_nn_python = ModalRecNNPython(verbose=verbose)
        self.apply_global_params(modalrec_nn_python)
        modalrec_nn_python.apply_properties(params)
        return modalrec_nn_python

    def get_modalrec_cured(self, params):
        """
        Create a modalrec_cured processing object.

        Parameters:
        params (dict): Dictionary of parameters

        Returns:
        ModalRecCured: modalrec_cured object
        """
        params = self.ensure_dictionary(params)

        mPCuReD_tag = params.pop('mPCuReD_tag')
        ifunc_tag = params.pop('ifunc_tag')
        wfsLambda = params.pop('wfsLambda')
        I_fried_tag = self.extract(params, 'I_fried_tag', default=None)
        pupdata_tag = self.extract(params, 'pupdata_tag', default=None)
        myGain = self.extract(params, 'myGain', default=None)
        verbose = self.extract(params, 'verbose', default=None)

        mPCuReD = self._cm.read_data(mPCuReD_tag)

        if I_fried_tag is None:
            pupdata = self._cm.read_pupils(pupdata_tag)

            # Compute IFried from pupdata
            pup2Dfull = np.zeros(pupdata.framesize, dtype=float)
            pup2Dfull[pupdata.ind_pup[0, :]] = 1
            idx1 = np.where(np.sum(pup2Dfull, axis=1) > 0)
            idx2 = np.where(np.sum(pup2Dfull, axis=0) > 0)

            pup2D = pup2Dfull[idx2[0][0]:idx2[0][-1] + 1, idx1[0][0]:idx1[0][-1] + 1]
            sPup2D = pup2D.shape

            I_fried = np.zeros((sPup2D[0] + 1, sPup2D[1] + 1), dtype=float)
            I_fried[:-1, :-1] += pup2D
            I_fried[1:, :-1] += pup2D
            I_fried[:-1, 1:] += pup2D
            I_fried[1:, 1:] += pup2D

            I_fried = np.transpose(I_fried > 0)
        else:
            I_fried = self._cm.read_data(I_fried_tag)

        ifunc_obj = self.ifunc_restore(tag=ifunc_tag)
        ifunc = ifunc_obj.influence_function
        ifunc_mask = ifunc_obj.mask_inf_func

        diameter = self._main.pixel_pitch * self._main.pixel_pupil

        modalrec_cured = ModalRecCured(I_fried, mPCuReD, ifunc, ifunc_mask,
                                    wfsLambda, diameter, myGain, verbose=verbose)
        self.apply_global_params(modalrec_cured)
        modalrec_cured.apply_properties(params)
        return modalrec_cured

    def get_modalrec_display(self, modalrec, window=None):
        """
        Create a modalrec_display processing object.

        Parameters:
        modalrec (objref): `modalrec` object to display
        window (int, optional): Window number to use, will be incremented in output

        Returns:
        ModalRecDisplay: modalrec_display object
        """
        disp = ModalRecDisplay(modalrec=modalrec)
        if window is not None:
            disp.window = window
            window += 1
        self.apply_global_params(disp)
        return disp

    def get_modes_display(self, modes, window=None):
        """
        Create a modes_display processing object.

        Parameters:
        modes (objref): A `base_value` object with the mode vector
        window (int, optional): Window number to use, will be incremented in output

        Returns:
        ModesDisplay: modes_display object
        """
        disp = ModesDisplay(modes=modes)
        if window is not None:
            disp.window = window
            window += 1
        self.apply_global_params(disp)
        return disp

    def get_removes_highfreq(self, pupil, mod_params, dm_params, phase2modes=None, ifunc=None):
        """
        Gets a processing container which removes high spatial frequency

        Parameters:
        pupil (objref): The pupil object
        mod_params (dict): Dictionary of modal parameters
        dm_params (dict): Dictionary of deformable mirror parameters
        phase2modes (objref, optional): Phase to modes object
        ifunc (objref, optional): Influence function object

        Returns:
        ProcessingContainer: A processing container
        """
        container = ProcessingContainer()

        gain = self.extract(mod_params, 'gain', default=None)

        modal_analysis = self.get_modalanalysis(mod_params, phase2modes=phase2modes)
        dm = self.get_dm(dm_params, ifunc=ifunc)
        dm.sign = 1

        modal_analysis.in_ef = pupil
        if gain is not None:
            mult = BaseOperation(constant_mult=gain)
            mult.in_value1 = modal_analysis.out_modes
            dm.in_command = mult.out_value
        else:
            dm.in_command = modal_analysis.out_modes
        
        dm.out_layer.S0 = pupil.S0

        container.add(modal_analysis, name='modalAnalysis')
        if gain is not None:
            container.add(mult, name='mult')
        container.add(dm, name='dm', output='out_layer')

        return container

    def get_sh(self, params, GPU=None):
        """
        Builds a `sh` or `sh_gpu` processing object.

        Parameters:
        params (dict): Dictionary of parameters
        GPU (bool, optional): Use GPU if available

        Returns:
        Sh or ShGpu: A new `sh` or `sh_gpu` processing object
        """
        useGPU = GPU if GPU is not None else self._gpu
        params = self.ensure_dictionary(params)

        convolGaussSpotSize = self.extract(params, 'convolGaussSpotSize', default=None)
        useGPUfromParams = self.extract(params, 'useGPU', default=None)
        if useGPUfromParams is not None:
            useGPU = useGPUfromParams

        if 'xyshift' in params:
            sh = self.get_sh_shift(params, GPU=GPU)
            return sh
        if 'xytilt' in params:
            sh = self.get_sh_tilt(params, GPU=GPU)
            return sh

        wavelengthInNm = params.pop('wavelengthInNm')
        sensor_fov = params.pop('sensor_fov')
        sensor_pxscale = params.pop('sensor_pxscale')
        sensor_npx = params.pop('sensor_npx')
        n_subap_on_diameter = params.pop('subap_on_diameter')
        FoVres30mas = self.extract(params, 'FoVres30mas', default=None)
        gkern = self.extract(params, 'gauss_kern', default=None)

        subapdata_tag = self.extract(params, 'subapdata_tag', default=None)
        energy_th = params.pop('energy_th')

        lenslet = Lenslet(n_subap_on_diameter)

        if useGPU:
            sh = ShGpu(wavelengthInNm, lenslet, sensor_fov, sensor_pxscale, sensor_npx, FoVres30mas=FoVres30mas)
        else:
            sh = Sh(wavelengthInNm, lenslet, sensor_fov, sensor_pxscale, sensor_npx, FoVres30mas=FoVres30mas, gkern=gkern)

        self.apply_global_params(sh)

        xShiftPhInPixel = self.extract(params, 'xShiftPhInPixel', default=None)
        yShiftPhInPixel = self.extract(params, 'yShiftPhInPixel', default=None)
        aXShiftPhInPixel = self.extract(params, 'aXShiftPhInPixel', default=None)
        aYShiftPhInPixel = self.extract(params, 'aYShiftPhInPixel', default=None)
        rotAnglePhInDeg = self.extract(params, 'rotAnglePhInDeg', default=None)
        aRotAnglePhInDeg = self.extract(params, 'aRotAnglePhInDeg', default=None)

        if xShiftPhInPixel is not None:
            sh.xShiftPhInPixel = xShiftPhInPixel
        if yShiftPhInPixel is not None:
            sh.yShiftPhInPixel = yShiftPhInPixel
        if aXShiftPhInPixel is not None:
            sh.aXShiftPhInPixel = aXShiftPhInPixel
        if aYShiftPhInPixel is not None:
            sh.aYShiftPhInPixel = aYShiftPhInPixel
        if rotAnglePhInDeg is not None:
            sh.rotAnglePhInDeg = rotAnglePhInDeg
        if aRotAnglePhInDeg is not None:
            sh.aRotAnglePhInDeg = aRotAnglePhInDeg

        sh.apply_properties(params)

        if convolGaussSpotSize is not None:
            kernelobj = KernelGauss()
            kernelobj.spotsize = convolGaussSpotSize
            kernelobj.lenslet = sh.lenslet
            kernelobj.cm = self._cm
            sh.kernelobj = kernelobj

        return sh

    def get_sh_shift(self, params, GPU=None):
        """
        Builds a `sh_shift` processing object.

        Parameters:
        params (dict): Dictionary of parameters
        GPU (bool, optional): Use GPU if available

        Returns:
        ShShift: A new `sh_shift` processing object
        """
        GPU = GPU if GPU is not None else self._gpu
        params = self.ensure_dictionary(params)

        shiftWavelengthInNm = params.pop('shiftWavelengthInNm')
        xyshift = params.pop('xyshift')
        qe_factor = params.pop('qe_factor_shift')
        resize_fact = params.pop('resize_fact')

        sh_shift = ShShift(params, self._main, shiftWavelengthInNm, xyshift, qe_factor, resize_fact, GPU=self._GPU)
        return sh_shift

    def get_sh_tilt(self, params, GPU=None):
        """
        Builds a `sh_tilt` processing object.

        Parameters:
        params (dict): Dictionary of parameters
        GPU (bool, optional): Use GPU if available

        Returns:
        ShTilt: A new `sh_tilt` processing object
        """
        GPU = GPU if GPU is not None else self._gpu
        params = self.ensure_dictionary(params)
        params_tilt = {}

        if 'dm_type' in params:
            params_tilt['dm_type'] = params.pop('dm_type')
            params_tilt['dm_npixels'] = params.pop('dm_npixels')
            params_tilt['dm_obsratio'] = params.pop('dm_obsratio')
            params_tilt['precision'] = self.extract(params, 'precision', default=None)

        if 'ifunc_tag' in params:
            params_tilt['influence_function'] = params.pop('ifunc_tag')
        params_tilt['func_type'] = params.pop('func_type')
        params_tilt['nmodes'] = self.extract(params, 'nmodes', default=2)
        params_tilt['height'] = self.extract(params, 'height', default=0)

        tiltWavelengthInNm = params.pop('tiltWavelengthInNm')
        xyTilt = params.pop('xyTilt')
        qe_factor = params.pop('qe_factor_tilt')

        sh_tilt = ShTilt(params, params_tilt, self._main, tiltWavelengthInNm, xyTilt, qe_factor, GPU=self._GPU)
        return sh_tilt

    def get_modulated_pyramid(self, params, GPU=None):
        """
        Builds a `modulated_pyramid` or `modulated_pyramid_gpu` processing object.

        Parameters:
        params (dict): Dictionary of parameters
        GPU (bool, optional): Use GPU if available

        Returns:
        ModulatedPyramid or ModulatedPyramidGpu: A new `modulated_pyramid` or `modulated_pyramid_gpu` processing object
        """
        if 'xyTilt' in params:
            pyr = self.get_pyr_tilt(params, GPU=GPU)
            return pyr

        useGPU = GPU if GPU is not None else self._gpu
        params = self.ensure_dictionary(params)

        DpupPix = self._main['pixel_pupil']
        pixel_pitch = self._main['pixel_pitch']
        wavelengthInNm = params.pop('wavelengthInNm')

        FoV = params.pop('fov')
        pup_diam = params.pop('pup_diam')
        ccd_side = params.pop('output_resolution')
        fov_errinf = self.extract(params, 'fov_errinf', default=0.5, optional=True)
        fov_errsup = self.extract(params, 'fov_errsup', default=2, optional=True)
        pup_dist = self.extract(params, 'pup_dist', default=None)
        pup_margin = self.extract(params, 'pup_margin', default=2, optional=True)
        fft_res = self.extract(params, 'fft_res', default=3.0, optional=True)
        fp_obs = self.extract(params, 'fp_obs', default=None, optional=True)
        pyr_tlt_coeff = self.extract(params, 'pyr_tlt_coeff', default=None, optional=True)
        pyr_edge_def_ld = self.extract(params, 'pyrEdgeDefLd', default=0.0, optional=True)
        pyr_tip_def_ld = self.extract(params, 'pyrTipDefLd', default=0.0, optional=True)
        pyr_tip_maya_ld = self.extract(params, 'pyrTipMayaLd', default=0.0, optional=True)
        min_pup_dist = self.extract(params, 'min_pup_dist', default=None, optional=True)

        result = ModulatedPyramid.calc_geometry(DpupPix, pixel_pitch, wavelengthInNm, FoV, pup_diam, ccd_side,
                                                fov_errinf=fov_errinf, fov_errsup=fov_errsup, pup_dist=pup_dist, pup_margin=pup_margin,
                                                fft_res=fft_res, min_pup_dist=min_pup_dist, NOTEST=True)

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

        if useGPU:
            pyr = ModulatedPyramidGpu(wavelengthInNm, fov_res, fp_masking, fft_res, tilt_scale,
                                    fft_sampling, fft_padding, fft_totsize, toccd_side, final_ccd_side,
                                    fp_obs=fp_obs, pyr_tlt_coeff=pyr_tlt_coeff,
                                    pyr_edge_def_ld=pyr_edge_def_ld, pyr_tip_def_ld=pyr_tip_def_ld,
                                    pyr_tip_maya_ld=pyr_tip_maya_ld)
        else:
            pyr = ModulatedPyramid(wavelengthInNm, fov_res, fp_masking, fft_res, tilt_scale,
                                fft_sampling, fft_padding, fft_totsize, toccd_side, final_ccd_side,
                                fp_obs=fp_obs, pyr_tlt_coeff=pyr_tlt_coeff,
                                pyr_edge_def_ld=pyr_edge_def_ld, pyr_tip_def_ld=pyr_tip_def_ld,
                                pyr_tip_maya_ld=pyr_tip_maya_ld)

        pup_shifts_std = self.extract(params, 'pup_shifts_std', default=[0, 0], optional=True)
        pup_shifts_seed = self.extract(params, 'pup_shifts_seed', default=None, optional=True)
        pup_shifts_cons = self.extract(params, 'pup_shifts_cons', default=None, optional=True)
        pup_shifts = self.extract(params, 'pup_shifts', default=[0, 0], optional=True)

        if pup_shifts_std and any(pup_shifts_std):
            pyr.pup_shifts = FuncGenerator('RANDOM', amp=pup_shifts_std, constant=pup_shifts_cons, seed=pup_shifts_seed)
        elif pup_shifts and any(pup_shifts):
            pyr.pup_shifts = FuncGenerator('RANDOM', amp=[0., 0.], constant=pup_shifts)

        mod_amp = self.extract(params, 'mod_amp', default=3, optional=True)
        mod_step_original = self.extract(params, 'mod_step', default=None, optional=True)
        if mod_step_original:
            mod_step = mod_step_original
        else:
            mod_step = round(max([1., mod_amp / 2. * 8.])) * 2.

        if mod_step_original and mod_step_original < mod_step:
            print(f' Attention mod_step={mod_step_original} is too low!')
            print(f' Would you like to change it to {mod_step}? [y,n]')
            ans = input()
            if ans.lower() == 'y':
                print(' mod_step changed.')
            else:
                mod_step = mod_step_original

        pyr.mod_amp = mod_amp
        pyr.mod_step = mod_step

        self.apply_global_params(pyr)
        pyr.apply_properties(params)

        return pyr

    def get_pyr_tilt(self, params, GPU=None):
        """
        Builds a `pyr_tilt` processing object.

        Parameters:
        params (dict): Dictionary of parameters
        GPU (bool, optional): Use GPU if available

        Returns:
        PyrTilt: A new `pyr_tilt` processing object
        """
        GPU = GPU if GPU is not None else self._gpu
        params = self.ensure_dictionary(params)
        params_tilt = {}

        if 'dm_type' in params:
            params_tilt['dm_type'] = params.pop('dm_type')
            params_tilt['dm_npixels'] = params.pop('dm_npixels')
            params_tilt['dm_obsratio'] = params.pop('dm_obsratio')
            params_tilt['precision'] = self.extract(params, 'precision', default=None)

        if 'ifunc_tag' in params:
            params_tilt['influence_function'] = params.pop('ifunc_tag')
        params_tilt['func_type'] = params.pop('func_type')
        params_tilt['nmodes'] = params.pop('nmodes')
        params_tilt['height'] = params.pop('height')

        tiltWavelengthInNm = params.pop('tiltWavelengthInNm')
        xyTilt = params.pop('xyTilt')
        qe_factor = params.pop('qe_factor_tilt')
        pup_shifts = self.extract(params, 'pup_shifts', default=None)
        delta_pup_dist = self.extract(params, 'delta_pup_dist', default=[0] * len(qe_factor))
        pyr_tlt_coeff = self.extract(params, 'pyr_tlt_coeff', default=None)

        pyr = PyrTilt(params, params_tilt, self._main, tiltWavelengthInNm,
                    xyTilt, qe_factor, delta_pup_dist=delta_pup_dist,
                    pup_shifts=pup_shifts, pyr_tlt_coeff=pyr_tlt_coeff, GPU=GPU)
        return pyr

    def get_optgaincontrol(self, params):
        """
        Builds a `optgaincontrol` processing object.

        Parameters:
        params (dict): Dictionary of parameters

        Returns:
        OptGainControl: A new `optgaincontrol` processing object
        """
        params = self.ensure_dictionary(params)
        gain = params.pop('gain')
        optgaincontrol = OptGainControl(gain)
        optgaincontrol.apply_properties(params)
        return optgaincontrol

    def get_phase_display(self, phase, window=None):
        """
        Builds a `phase_display` processing object.

        Parameters:
        phase (objref): The `phase` object to display
        window (int, optional): Window number to use, will be incremented in output

        Returns:
        PhaseDisplay: A new `phase_display` processing object
        """
        disp = PhaseDisplay(phase=phase)
        if window is not None:
            disp.window = window
            window += 1
        self.apply_global_params(disp)
        return disp

    def get_sh_display(self, sh, pyr_style=None, window=None):
        """
        Builds a `sh_display` processing object.

        Parameters:
        sh (objref): The `sh` object to display
        pyr_style (optional): Pyramid style
        window (int, optional): Window number to use, will be incremented in output

        Returns:
        ShDisplay: A new `sh_display` processing object
        """
        disp = ShDisplay(sh=sh, pyr_style=pyr_style)
        if window is not None:
            disp.window = window
            window += 1
        self.apply_global_params(disp)
        return disp

    def get_plot_display(self, value, window=None):
        """
        Builds a `plot_display` processing object.

        Parameters:
        value (objref): Object of type `base_value` to display
        window (int, optional): Window number to use, will be incremented in output

        Returns:
        PlotDisplay: A new `plot_display` processing object
        """
        disp = PlotDisplay(value=value)
        if window is not None:
            disp.window = window
            window += 1
        self.apply_global_params(disp)
        return disp

    def get_psf(self, params, GPU=None):
        """
        Builds a `psf` processing object.

        Parameters:
        params (dict): Dictionary of parameters
        GPU (bool, optional): Use GPU if available

        Returns:
        Psf or PsfGpu: A new `psf` processing object
        """
        useGPU = GPU if GPU is not None else self._gpu
        params = self.ensure_dictionary(params)

        wavelengthInNm = params.pop('wavelengthInNm')
        nd = self.extract(params, 'nd', default=None)

        if useGPU:
            psf = PsfGpu(wavelengthInNm, nd=nd)
        else:
            psf = Psf(wavelengthInNm, nd=nd)
        
        self.apply_global_params(psf)
        psf.apply_properties(params)
        return psf

    def get_psf_display(self, psf, window=None):
        """
        Builds a `psf_display` processing object.

        Parameters:
        psf (objref): The `psf` object to display
        window (int, optional): Window number to use, will be incremented in output

        Returns:
        PsfDisplay: A new `psf_display` processing object
        """
        disp = PsfDisplay(psf=psf)
        if window is not None:
            disp.window = window
            window += 1
        self.apply_global_params(disp)
        return disp

    def get_pupilstop(self, params, GPU=None):
        """
        Builds a `pupilstop` processing object.

        Parameters:
        params (dict): Dictionary of parameters
        GPU (bool, optional): Use GPU if available

        Returns:
        PupilStop: A new `pupilstop` processing object
        """
        useGPU = GPU if GPU is not None else self._gpu
        params = self.ensure_dictionary(params)

        mask_diam = self.extract(params, 'mask_diam', default=1.)
        obs_diam = self.extract(params, 'obs_diam', default=None)
        pupil_mask_tag = self.extract(params, 'pupil_mask_tag', default='')

        shiftXYinPixel = self.extract(params, 'shiftXYinPixel', default=None)
        rotInDeg = self.extract(params, 'rotInDeg', default=None)
        magnification = self.extract(params, 'magnification', default=None)

        if pupil_mask_tag:
            pupilstop = self._cm.read_pupilstop(pupil_mask_tag, GPU=useGPU)
            if pupilstop is None:
                raise ValueError(f'Pupil mask tag {pupil_mask_tag} not found.')
        else:
            dim = self._main.pixel_pupil
            pixel_pitch = self._main.pixel_pitch
            pupilstop = PupilStop(dim, dim, pixel_pitch, 0, GPU=useGPU, mask_diam=mask_diam, obs_diam=obs_diam)

        if shiftXYinPixel is not None:
            pupilstop.shiftXYinPixel = shiftXYinPixel
        if rotInDeg is not None:
            pupilstop.rotInDeg = rotInDeg
        if magnification is not None:
            pupilstop.magnification = magnification

        return pupilstop

    def get_pyr_slopec(self, params):
        """
        Builds a `pyr_slopec` processing object.

        Parameters:
        params (dict): Dictionary of parameters

        Returns:
        PyrSlopec: A new `pyr_slopec` processing object
        """
        params = self.ensure_dictionary(params)
        computation_time = self.extract(params, 'computation_time', default=None, optional=True)
        use_optg_sn = self.extract(params, 'use_optg_sn', default=None, optional=True)

        shlike = self.extract(params, 'shlike', default=None, optional=True)
        slopes_from_intensity = self.extract(params, 'slopes_from_intensity', default=None, optional=True)
        norm_factor = self.extract(params, 'norm_factor', default=None, optional=True)
        thr1 = self.extract(params, 'thr1', default=None, optional=True)
        thr2 = self.extract(params, 'thr2', default=None, optional=True)
        subap_norm = self.extract(params, 'subap_norm', default=None, optional=True)    # old keyword to be removed!
        filtmat_tag = self.extract(params, 'filtmat_tag', default='', optional=True)

        sc = PyrSlopec(shlike=shlike, norm_factor=norm_factor, slopes_from_intensity=slopes_from_intensity)

        if filtmat_tag:
            filtmat = self._cm.read_data(filtmat_tag)
            sc.filtmat = filtmat

        sc.setproperty(cm=self._cm)
        self.apply_global_params(sc)
        sc.apply_properties(params)

        return sc

    def get_sh_slopec(self, params, GPU=None, recmat=None, device=None, mode_basis=None, pup_mask=None):
        """
        Builds a `sh_slopec` or `sh_slopec_gpu` processing object.

        Parameters:
        params (dict): Dictionary of parameters
        GPU (bool, optional): Use GPU if available
        recmat (objref, optional): Reconstruction matrix
        device (optional): Device to use for GPU
        mode_basis (objref, optional): Mode basis
        pup_mask (objref, optional): Pupil mask

        Returns:
        ShSlopec or ShSlopecGpu: A new `sh_slopec` or `sh_slopec_gpu` processing object
        """
        params = self.ensure_dictionary(params)
        lifted_sh = self.extract(params, 'lifted_sh', default=False)
        
        if lifted_sh:
            sc = self.get_lift_sh_slopec(params, mode_basis=mode_basis, pup_mask=pup_mask, GPU=GPU)
            return sc

        computation_time = self.extract(params, 'computation_time', default=None)
        template_tag = self.extract(params, 'template_tag', default=None)
        intmat_tag = self.extract(params, 'intmat_tag', default='')
        recmat_tag = self.extract(params, 'recmat_tag', default='')
        filtmat_tag = self.extract(params, 'filtmat_tag', default='')
        matched_tag = self.extract(params, 'matched_tag', default='')

        filtName = self.extract(params, 'filtName', default='')

        if matched_tag:
            sc = ShMatchedSlopec()
            sc.matched_filter = self._cm.read_data(matched_tag)
        else:
            useGPU = GPU if GPU is not None else False
            if useGPU:
                sc = ShSlopecGpu(device=device)
            else:
                sc = ShSlopec()

        if intmat_tag:
            intmat = self._cm.read_data(intmat_tag)
            sc.intmat = intmat

        if recmat is None and recmat_tag:
            recmat = self._cm.read_rec(recmat_tag, doNotPutOnGpu=doNotPutOnGpu)
            sc.recmat = recmat

        if filtmat_tag:
            filtmat = self._cm.read_data(filtmat_tag)
            sc.filtmat = filtmat

        sc.setproperty(cm=self._cm)
        self.apply_global_params(sc)
        if 'windowing' in params:
            print('ATTENTION: SH SLOPEC windowing keyword is set!')
        sc.apply_properties(params)

        if template_tag:
            sc.corr_template = self._cm.read_data(template_tag)

        return sc

    def get_slopec_display(self, slopec, window=None):
        """
        Builds a `slopec_display` processing object.
        Can display both Pyramid and SH slope computers.

        Parameters:
        slopec (objref): The `slopec` object to display
        window (int, optional): Window number to use, will be incremented in output

        Returns:
        SlopecDisplay: A new `slopec_display` processing object
        """
        sc_disp = SlopecDisplay(slopec=slopec)
        if window is not None:
            sc_disp.window = window
            window += 1
        self.apply_global_params(sc_disp)
        return sc_disp

    def get_source_field(self, params):
        """
        Builds a list of `source` objects arranged on a regular grid.

        Parameters:
        params (dict): Dictionary of parameters

        Returns:
        list: A list of `source` objects.
        """
        params = self.ensure_dictionary(params)

        field_width = params.pop('field_width')
        sources_per_side = params.pop('sources_per_side')
        field_center = params.pop('field_center')

        height = self.extract(params, 'height', default=float('inf'))
        if hasattr(self._main, 'zenithAngleInDeg'):
            airmass = 1. / cos(self._main.zenithAngleInDeg / 180. * pi)
        else:
            airmass = 1.
        height *= airmass
        if 'verbose' in params and params['verbose']:
            if airmass != 1.:
                print(f'get_source_field: changing source height by airmass value ({airmass})')

        magnitude = params.pop('magnitude')
        wavelengthInNm = self.extract(params, 'wavelengthInNm', default=750)

        x, y = make_xy(sources_per_side, field_width)
        x += field_center[0]
        y += field_center[1]

        source_list = []
        phi = np.degrees(np.arctan2(y, x))
        r = np.sqrt(x**2 + y**2)

        for i in range(len(x)):
            p = {
                'polar_coordinate': [r[i], phi[i]],
                'height': height,
                'magnitude': magnitude,
                'wavelengthInNm': wavelengthInNm
            }
            source_list.append(self.get_source(p))

        return source_list

    def get_source(self, params):
        """
        Builds a `source` object.

        Parameters:
        params (dict): Dictionary of parameters

        Returns:
        Source: A `source` object.
        """
        params = self.ensure_dictionary(params)

        polar_coordinate = params.pop('polar_coordinate')
        height = self.extract(params, 'height', default=float('inf'), optional=True)
        error_coord = self.extract(params, 'error_coord', default=[0., 0.], optional=True)

        polar_coordinate = np.add(polar_coordinate, error_coord)

        if any(error_coord):
            print(f'there is a desired error ({error_coord[0]},{error_coord[1]}) on source coordinates.')
            print(f'final coordinates are: {polar_coordinate[0]},{polar_coordinate[1]}')

        if hasattr(self._main, 'zenithAngleInDeg'):
            airmass = 1. / cos(self._main.zenithAngleInDeg / 180. * pi)
        else:
            airmass = 1.
        height *= airmass
        if 'verbose' in params and params['verbose']:
            if airmass != 1.:
                print(f'get_source: changing source height by airmass value ({airmass})')

        magnitude = params.pop('magnitude')
        wavelengthInNm = params.pop('wavelengthInNm')
#        optSource = self.extract(params, 'optSource', default=None)

        band = self.extract(params, 'band', default='', optional=True)
        zeroPoint = self.extract(params, 'zeroPoint', default=0, optional=True)

        source = Source(polar_coordinate, height, magnitude, wavelengthInNm,
                        band=band, zeroPoint=zeroPoint)
        self.apply_global_params(source)
        source.apply_properties(params)
        return source

    def get_cm(self):
        """
        Returns the calibration manager

        Returns:
        CalibrationManager: The calibration manager
        """
        return self._cm

    def gpu(self):
        """
        Returns the GPU acceleration flag.
        If this flag is non-zero, GPU versions will be automatically returned
        for objects of type `sh`, `modulated_pyramid` and `sh_slopec`.

        Returns:
        bool: GPU acceleration flag
        """
        return self._gpu

    def revision_track(self):
        """
        Returns the revision of the SVN

        Returns:
        str: SVN revision
        """
        return '$Rev$'
