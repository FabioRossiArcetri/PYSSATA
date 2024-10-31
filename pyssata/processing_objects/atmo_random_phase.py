import numpy as np
from pyssata import xp

from astropy.io import fits

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.data_objects.ef import ElectricField
from pyssata.base_value import BaseValue
from pyssata.base_list import BaseList
from pyssata.data_objects.layer import Layer
from pyssata.data_objects.pupilstop import Pupilstop
from pyssata.lib.phasescreen_manager import phasescreens_manager
from pyssata.connections import InputValue
from pyssata import cpuArray

class AtmoRandomPhase(BaseProcessingObj):
    def __init__(self,
                 L0,
                 pixel_pitch,
                 pixel_pupil,
                 data_dir, 
                 source_dict,
                 wavelengthInNm: float=500.0,
                 zenithAngleInDeg=None,
                 pixel_phasescreens=None,
                 seed: int=1,
                 target_device_idx=None,
                 precision=None,
                 verbose=None):


        super().__init__(target_device_idx=target_device_idx, precision=precision)
                
        self.source_dict = source_dict
        self.last_position = 0
        self.seeing = 1
        self.airmass = 1
        self.wavelengthInNm = wavelengthInNm
        self.pixel_pitch = pixel_pitch         
        
        self.inputs['seeing'] = InputValue(type=BaseValue)
        
        if zenithAngleInDeg is not None:
            self.airmass = 1.0 / np.cos(np.radians(zenithAngleInDeg))
            print(f'AtmoRandomPhase: zenith angle is defined as: {zenithAngleInDeg} deg')
            print(f'AtmoRandomPhase: airmass is: {self.airmass}')
        else:
            self.airmass = 1.0

        # Conversion coefficient from arcseconds to radians
        sec2rad = 4.848e-6
               
        # Compute layers dimension in pixels
        self.pixel_layer = pixel_pupil

        self.L0 = L0
        self.pixel_pupil = pixel_pupil
        self.data_dir = data_dir
        self.seeing = None

        if pixel_phasescreens is None:
            self.pixel_square_phasescreens = 8192
        else:
            self.pixel_square_phasescreens = pixel_phasescreens

        # Error if phase-screens dimension is smaller than maximum layer dimension
        if self.pixel_square_phasescreens < self.pixel_layer:
            raise ValueError('Error: phase-screens dimension must be greater than layer dimension!')
        
        self.verbose = verbose if verbose is not None else False
        
        # Initialize layer list with correct heights
        self.layer_list = BaseList(target_device_idx=self.target_device_idx)
        layer = Layer(self.pixel_pupil, self.pixel_pupil, pixel_pitch, 0, precision=self.precision, target_device_idx=self.target_device_idx)
        self.layer_list.append(layer)
        
        for name, source in source_dict.items():
            ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, target_device_idx=self.target_device_idx)
            ef.S0 = source.phot_density()
            self.outputs['out_'+name+'_ef'] = ef

        if seed is not None:
            self.seed = seed

        self.inputs['pupilstop'] = InputValue(type=Pupilstop)
    
    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self.compute()


    def compute(self):

        # Seed
        seed = np.array([self.seed])

        # Square phasescreens
        square_phasescreens = phasescreens_manager(np.array([self.L0]), self.pixel_square_phasescreens,
                                                    self.pixel_pitch, self.data_dir,
                                                    seed=seed, precision=self.precision,
                                                    verbose=self.verbose)

        # number of slices to be cut from the 2D array
        num_slices = (self.pixel_square_phasescreens // self.pixel_pupil)

        # it cuts the array to have dimensions multiple of pixel_pupil
        input_array = square_phasescreens[0][0:num_slices*self.pixel_pupil,0:num_slices*self.pixel_pupil]

        # it makes a 3D array stacking neighbouring squares of the 2D array
        temp_screen = input_array.reshape(num_slices, self.pixel_pupil,num_slices, self.pixel_pupil).swapaxes(1, 2).reshape(-1, self.pixel_pupil, self.pixel_pupil)

        # phase in rad
        temp_screen *= self.wavelengthInNm / (2 * np.pi)

        temp_screen = self.xp.array(temp_screen, dtype=self.dtype)
        
        self.phasescreens = temp_screen

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.pupilstop = self.local_inputs['pupilstop']
    
    def trigger_code(self):
        r0 = 0.9759 * 0.5 / (self.local_inputs['seeing'].value * 4.848) * self.airmass**(-3./5.) # if seeing > 0 else 0.0
        r0wavelength = r0 * (self.wavelengthInNm / 500.0)**(6./5.)
        scale_coeff = (self.pixel_pitch / r0wavelength)**(5./6.) # if seeing > 0 else 0.0

        new_position = self.last_position
        if new_position+1 > self.phasescreens.shape[0]:
            self.seed += 1
            self.compute()
            new_position = 0

        for name, source in self.source_dict.items():
            self.outputs['out_'+name+'_ef'].phaseInNm = self.phasescreens[new_position,:,:] * scale_coeff
            self.outputs['out_'+name+'_ef'].A = self.pupilstop.A
            self.outputs['out_'+name+'_ef'].generation_time = self.current_time

        # Update position output
        self.last_position = new_position + 1
        
    def save(self, filename):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['INTRLVD'] = int(self.interleave)
        hdr['PUPD_TAG'] = self.pupdata_tag
        super().save(filename, hdr)

        with fits.open(filename, mode='append') as hdul:
            hdul.append(fits.ImageHDU(data=self.phasescreens))

    def read(self, filename):
        super().read(filename)
        self.phasescreens = fits.getdata(filename, ext=1)

    def set_last_position(self, last_position):
        self.last_position = last_position

    def set_last_t(self, last_t):
        self.last_t = last_t

    def run_check(self, time_step):
        self.prepare_trigger(0)

        errmsg = ''
        if not (self.seed > 0):
            errmsg += ' Seed <= 0.'
        if not isinstance(self.seeing, BaseValue):
            errmsg += ' Missing input seeing.'

        seeing = self.inputs['seeing'].get(self.target_device_idx)
                
        check = self.seed > 0 and isinstance(seeing, BaseValue)
        if not check:
            raise ValueError(errmsg)
          
        # super().build_stream()
        return check

