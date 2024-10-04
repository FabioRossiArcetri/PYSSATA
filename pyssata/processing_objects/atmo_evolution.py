import numpy as np

from astropy.io import fits

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.base_value import BaseValue
from pyssata.base_list import BaseList
from pyssata.data_objects.layer import Layer
from pyssata.lib.cv_coord import cv_coord
from pyssata.lib.phasescreen_manager import phasescreens_manager
from pyssata.connections import InputValue


class AtmoEvolution(BaseProcessingObj):
    def __init__(self, L0, pixel_pitch, heights, Cn2, pixel_pupil, data_dir, source_list, wavelengthInNm: float=500.0,
                 zenithAngleInDeg=None, mcao_fov=None, pixel_phasescreens=None, seed: int=1, target_device_idx=None, precision=None,
                 verbose=None, user_defined_phasescreen: str='', force_mcao_fov=False, make_cycle=None,
                 fov_in_m=None, pupil_position=None):
        
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        
        self._last_position = None
        self._last_t = 0
        self._extra_delta_time = 0
        self._cycle_screens = True

        if pupil_position is None:
            pupil_position = [0, 0]
        
        if zenithAngleInDeg is not None:
            self._airmass = 1.0 / np.cos(np.radians(zenithAngleInDeg))
            print(f'Atmo_Evolution: zenith angle is defined as: {zenithAngleInDeg} deg')
            print(f'Atmo_Evolution: airmass is: {self._airmass}')
        else:
            self._airmass = 1.0
        heights = self.xp.array(heights, dtype=self.dtype) * self._airmass

        # Conversion coefficient from arcseconds to radians
        sec2rad = 4.848e-6
        
        if force_mcao_fov:
            print(f'\nATTENTION: MCAO FoV is forced to diameter={mcao_fov} arcsec\n')
            alpha_fov = mcao_fov / 2.0
        else:
            alpha_fov = 0.0
            for element in source_list:
                alpha_fov = max(alpha_fov, *abs(cv_coord(from_polar=[element.polar_coordinate[1], element.polar_coordinate[0]],
                                                       to_rect=True, degrees=True)))
            if mcao_fov is not None:
                alpha_fov = max(alpha_fov, mcao_fov / 2.0)
        
        # Max star angle from arcseconds to radians
        rad_alpha_fov = alpha_fov * sec2rad

        # Compute layers dimension in pixels
        self._pixel_layer = self.xp.ceil((pixel_pupil + 2 * self.xp.sqrt(self.xp.sum(self.xp.array(pupil_position, dtype=self.dtype) * 2)) / pixel_pitch + 
                               2.0 * abs(heights) / pixel_pitch * rad_alpha_fov) / 2.0) * 2.0
        if fov_in_m is not None:
            self._pixel_layer = self.xp.full_like(heights, long(fov_in_m / pixel_pitch / 2.0) * 2)
        
        self._L0 = L0
        self._wavelengthInNm = wavelengthInNm
        self._pixel_pitch = pixel_pitch
        self._n_phasescreens = len(heights)
        self._heights = heights
        self._Cn2 = self.xp.array(Cn2, dtype=self.dtype)
        self._pixel_pupil = pixel_pupil
        self._data_dir = data_dir
        self._make_cycle = make_cycle
        self._seeing = None
        self._wind_speed = None
        self._wind_direction = None

        if pixel_phasescreens is None:
            self._pixel_square_phasescreens = 8192
        else:
            self._pixel_square_phasescreens = pixel_phasescreens

        # Error if phase-screens dimension is smaller than maximum layer dimension
        if self._pixel_square_phasescreens < max(self._pixel_layer):
            raise ValueError('Error: phase-screens dimension must be greater than layer dimension!')
        
        self.verbose = verbose if verbose is not None else False

        # Use a specific user-defined phase screen if provided
        if user_defined_phasescreen is not None:
            self._user_defined_phasescreen = user_defined_phasescreen
        
        # Initialize layer list with correct heights
        self._layer_list = BaseList(target_device_idx=self._target_device_idx)

        for i in range(self._n_phasescreens):
            layer = Layer(self._pixel_layer[i], self._pixel_layer[i], pixel_pitch, heights[i], precision=self._precision, target_device_idx=self._target_device_idx)
            self._layer_list.append(layer)
        
        if seed is not None:
            self.seed = seed

        self.inputs['seeing'] = InputValue(type=BaseValue)
        self.inputs['wind_speed'] = InputValue(type=BaseValue)
        self.inputs['wind_direction'] = InputValue(type=BaseValue)
        self.outputs['layer_list'] = self.layer_list
        self._last_position = self.xp.zeros(self._n_phasescreens, dtype=self.dtype)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self.compute()

    @property
    def layer_list(self):
        return self._layer_list

    @property
    def L0(self):
        return self._L0

    @property
    def wavelengthInNm(self):
        return self._wavelengthInNm

    @property
    def pixel_pitch(self):
        return self._pixel_pitch

    @property
    def n_phasescreens(self):
        return self._n_phasescreens

    @property
    def heights(self):
        return self._heights

    @property
    def Cn2(self):
        return self._Cn2

    @property
    def pixel_pupil(self):
        return self._pixel_pupil

    @property
    def pixel_layer(self):
        return self._pixel_layer

    @property
    def data_dir(self):
        return self._data_dir

    def compute(self):
        # Phase screens list
        self._phasescreens = []

        if self._user_defined_phasescreen:
            temp_screen = fits.getdata(self._user_defined_phasescreen)

            if len(self._Cn2) > 1:
                raise ValueError('The user-defined phasescreen works only if the total phasescreens are 1.')

            if temp_screen.shape[0] < temp_screen.shape[1]:
                temp_screen = temp_screen.T

            temp_screen -= self.xp.mean(temp_screen)
            # Convert to nm
            temp_screen *= self._wavelengthInNm / (2 * self.xp.pi)
            
            self._phasescreens.append(temp_screen)

        else:
            self._pixel_phasescreens = self.xp.max(self._pixel_layer)

            if len(self.xp.unique(self._L0)) == 1:
                # Number of rectangular phase screens from a single square phasescreen
                n_ps_from_square_ps = self.xp.floor(self._pixel_square_phasescreens / self._pixel_phasescreens)
                # Number of square phasescreens
                n_ps = self.xp.ceil(float(self._n_phasescreens) / n_ps_from_square_ps)

                # Seed vector
                seed = self.xp.arange(self._seed, self._seed + int(n_ps))

                # Square phasescreens
                if self._make_cycle:
                    pixel_square_phasescreens = self._pixel_square_phasescreens - self._pixel_pupil
                    ps_cycle = get_layers(1, pixel_square_phasescreens, pixel_square_phasescreens * self._pixel_pitch,
                                          500e-9, 1, L0=self._L0[0], par=par, START=start, SEED=seed, DIR=self._data_dir,
                                          FILE=filename, no_sha=True, verbose=self._verbose)
                    ps_cycle = self.xp.vstack([ps_cycle, ps_cycle[:, :self._pixel_pupil]])
                    ps_cycle = self.xp.hstack([ps_cycle, ps_cycle[:self._pixel_pupil, :]])

                    square_phasescreens = [ps_cycle * 4 * self.xp.pi]  # 4 * π is added to get the correct amplitude
                else:
                    if hasattr(self._L0, '__len__'):
                        L0 = self._L0[0]
                    else:
                        L0 = self._L0
                    L0 = np.array([L0])
                    square_phasescreens = phasescreens_manager(L0, self._pixel_square_phasescreens,
                                                               self._pixel_pitch, self._data_dir,
                                                               seed=seed, precision=self._precision,
                                                               verbose=self._verbose)

                square_ps_index = -1
                ps_index = 0

                for i in range(self._n_phasescreens):
                    # Increase square phase-screen index
                    if i % n_ps_from_square_ps == 0:
                        square_ps_index += 1
                        ps_index = 0

                    temp_screen = self.xp.array(square_phasescreens[square_ps_index][int(self._pixel_phasescreens) * ps_index:
                                                                       int(self._pixel_phasescreens) * (ps_index + 1), :], dtype=self.dtype)
                    # print('self._Cn2[i]', self._Cn2[i], type(self._Cn2[i]), type(self._Cn2))  # Verbose?
                    # print('temp_screen', temp_screen, type(temp_screen))  # Verbose?

                    temp_screen *= self.xp.sqrt(self._Cn2[i])
                    temp_screen -= self.xp.mean(temp_screen)
                    # Convert to nm
                    temp_screen *= self._wavelengthInNm / (2 * np.pi)

                    temp_screen = self.xp.array(temp_screen, dtype=self.dtype)

                    # Flip x-axis for each odd phase-screen
                    if i % 2 != 0:
                        temp_screen = self.xp.flip(temp_screen, axis=1)

                    ps_index += 1

                    self._phasescreens.append(temp_screen)

            else:
                seed = self._seed + self.xp.arange(self._n_phasescreens)

                if len(seed) != len(self._L0):
                    raise ValueError('Number of elements in seed and L0 must be the same!')

                # Square phasescreens
                square_phasescreens = phasescreens_manager(self._L0, self._pixel_square_phasescreens,
                                                           self._pixel_pitch, self._data_dir,
                                                           seed=seed, precision=self._precision,
                                                           verbose=self._verbose)

                for i in range(self._n_phasescreens):
                    temp_screen = square_phasescreens[i][:, :self._pixel_phasescreens]
                    temp_screen *= self.xp.sqrt(self._Cn2[i])
                    temp_screen -= self.xp.mean(temp_screen)
                    # Convert to nm
                    temp_screen *= self._wavelengthInNm / (2 * self.xp.pi)

                    self._phasescreens.append(temp_screen)


    def shift_screens(self, t):
        seeing = self.inputs['seeing'].get(self._target_device_idx).value
        wind_speed = self.inputs['wind_speed'].get(self._target_device_idx).value
        wind_direction = self.inputs['wind_direction'].get(self._target_device_idx).value
        # if len(self._phasescreens) != len(wind_speed) or len(self._phasescreens) != len(wind_direction):
        #     raise ValueError('Error: number of elements of wind speed and/or direction does not match the number of phasescreens')
        delta_time = self.t_to_seconds(t - self._last_t) + self._extra_delta_time        
        r0 = 0.9759 * 0.5 / (seeing * 4.848) * self._airmass**(-3./5.) # if seeing > 0 else 0.0
        r0wavelength = r0 * (self._wavelengthInNm / 500.0)**(6./5.)
        scale_coeff = (self._pixel_pitch / r0wavelength)**(5./6.) # if seeing > 0 else 0.0

        # Compute the delta position in pixels
        delta_position = wind_speed * delta_time / self._pixel_pitch  # [pixel]
        new_position = self._last_position + delta_position
        # Get quotient and remainder
        new_position_quo = self.xp.floor(new_position).astype(int)
        new_position_rem = new_position - new_position_quo

        wdf, wdi = self.xp.modf(wind_direction/90.0)
        wdf_full, wdi_full = self.xp.modf(wind_direction)

        for ii, p in enumerate(self._phasescreens):
            # Check if we need to cycle the screens
            # print(ii, new_position[ii], self._pixel_layer[ii], p.shape[1]) # Verbose?
            if self._cycle_screens:
                if new_position[ii] + self._pixel_layer[ii] > p.shape[1]:
                    new_position[ii] = 0.            
            #if new_position[ii] + self._pixel_layer[ii] > p.shape[1]:
            #    print(f'phasescreens size: {self.xp.around(p.shape[0], 2)}')
            #    print(f'requested position: {self.xp.around(new_position[ii], 2)}')
            #    raise ValueError(f'phasescreens_shift cannot go out of the {ii}-th phasescreen!')            
            pos = new_position_quo[ii]
            # print(pos, self._pixel_layer) # Verbose?
            ipli = int(self._pixel_layer[ii])
            ipli_p = int(pos + self._pixel_layer[ii])
            layer = (1.0 - new_position_rem[ii]) * p[0: ipli, pos: ipli_p] + new_position_rem[ii] * p[0: ipli, pos + 1: ipli_p + 1]

            layer = self.xp.rot90(layer, wdi[ii])
            # is the rotate function is already checking for 0 angles? should we set very small rotations to 0?
            # this looks fast on the default example
            if not wdf_full[ii]==0:
                layer = self.rotate(layer, wdf_full[ii], reshape=False, order=1)

            self._layer_list[ii].phaseInNm = layer * scale_coeff
            self._layer_list[ii].generation_time = t
        # print(f'Phasescreen_shift: {new_position=}') # Verbose?
        # Update position output
        self._last_position = new_position
        self._layer_list.generation_time = t
        self._last_t = t
        
    def save(self, filename):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['INTRLVD'] = int(self._interleave)
        hdr['PUPD_TAG'] = self._pupdata_tag
        super().save(filename, hdr)

        with fits.open(filename, mode='append') as hdul:
            hdul.append(fits.ImageHDU(data=self._phasescreens))

    def read(self, filename):
        super().read(filename)
        self._phasescreens = fits.getdata(filename, ext=1)

    def set_last_position(self, last_position):
        self._last_position = last_position

    def set_last_t(self, last_t):
        self._last_t = last_t

    def run_check(self, time_step):
        errmsg = ''
        if not (self._seed > 0):
            errmsg += ' Seed <= 0.'
        if not isinstance(self._seeing, BaseValue):
            errmsg += ' Missing input seeing.'
        if not isinstance(self._wind_direction, BaseValue):
            errmsg += ' Missing input wind direction.'
        if not isinstance(self._wind_speed, BaseValue):
            errmsg += ' Missing input speed.'
        if not self.xp.isclose(self.xp.sum(self._Cn2), 1.0, atol=1e-6):
            errmsg += f' Cn2 total must be 1. Instead is: {self.xp.sum(self._Cn2)}.'

        seeing = self.inputs['seeing'].get(self._target_device_idx)
        wind_speed = self.inputs['wind_speed'].get(self._target_device_idx)
        wind_direction = self.inputs['wind_direction'].get(self._target_device_idx)

        check = self._seed > 0 and isinstance(seeing, BaseValue) and isinstance(wind_direction, BaseValue) and isinstance(wind_speed, BaseValue)
        if not check:
            raise ValueError(errmsg)
        return check

    def trigger(self, t):
        self.shift_screens(t)

