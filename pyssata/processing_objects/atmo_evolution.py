import numpy as np
from astropy.io import fits

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.data_objects.layer import Layer

class AtmoEvolution(BaseProcessingObj):
    '''layers obj from statistics and time'''

    def __init__(self, L0, wavelengthInNm, pixel_pitch, heights, Cn2, pixel_pupil, directory, source_list, 
                 zenithAngleInDeg=None, mcao_fov=None, pixel_phasescreens=None, seed=None, precision=0, verbose=False,
                 user_defined_phasescreen=None, force_mcao_fov=False, make_cycle=False, fov_in_m=None, pupil_position=None):
        super().__init__(precision=precision)

        if pupil_position is None:
            pupil_position = [0, 0]

        if zenithAngleInDeg:
            self._airmass = 1. / np.cos(np.radians(zenithAngleInDeg))
            print(f'Atmo_Evolution: zenith angle is defined as: {zenithAngleInDeg} deg')
            print(f'Atmo_Evolution: airmass is: {self._airmass}')
        else:
            self._airmass = 1.0
        heights = heights * self._airmass

        sec2rad = 4.848e-6

        self._L0 = L0
        self._wavelengthInNm = wavelengthInNm
        self._pixel_pitch = pixel_pitch
        self._n_phasescreens = len(heights)
        self._heights = heights
        self._Cn2 = Cn2
        self._pixel_pupil = pixel_pupil
        self._pixel_layer = self.calculate_pixel_layer(pixel_pupil, pupil_position, heights, sec2rad, mcao_fov, fov_in_m)
        self._directory = directory
        self._make_cycle = make_cycle
        self._pixel_square_phasescreens = pixel_phasescreens if pixel_phasescreens else 8192

        if self._pixel_square_phasescreens < max(self._pixel_layer):
            raise ValueError('Error: phase-screens dimension must be greater than layer dimension!')

        self._user_defined_phasescreen = user_defined_phasescreen
        self._layer_list = [Layer(px, py, pixel_pitch, h, PRECISION=precision) for px, py, h in zip(self._pixel_layer, self._pixel_layer, heights)]
        self._seed = seed
        self._verbose = verbose

        if seed is not None:
            self.compute()

    def calculate_pixel_layer(self, pixel_pupil, pupil_position, heights, sec2rad, mcao_fov, fov_in_m):
        rad_alpha_fov = (mcao_fov / 2. if mcao_fov else max(heights) / 2.) * sec2rad
        pixel_layer = np.ceil((pixel_pupil + 2 * np.sqrt(np.sum(np.square(pupil_position))) / self._pixel_pitch + 
                               2. * np.abs(heights) / self._pixel_pitch * rad_alpha_fov) / 2.).astype(int) * 2
        if fov_in_m:
            pixel_layer = np.full_like(pixel_layer, (fov_in_m / self._pixel_pitch / 2.).astype(int) * 2)
        return pixel_layer

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self.compute()

    @property
    def seeing(self):
        return self._seeing

    @seeing.setter
    def seeing(self, value):
        self._seeing = value

    @property
    def wind_speed(self):
        return self._wind_speed

    @wind_speed.setter
    def wind_speed(self, value):
        self._wind_speed = value

    @property
    def wind_direction(self):
        return self._wind_direction

    @wind_direction.setter
    def wind_direction(self, value):
        self._wind_direction = value

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
    def directory(self):
        return self._directory

    def compute(self):
        # Phase screens list
        self._phasescreens = []

        if self._user_defined_phasescreen:
            temp_screen = fits.getdata(self._user_defined_phasescreen)

            if len(self._Cn2) > 1:
                raise ValueError('The user-defined phasescreen works only if the total phasescreens are 1.')

            if temp_screen.shape[0] < temp_screen.shape[1]:
                temp_screen = temp_screen.T

            temp_screen -= np.mean(temp_screen)
            # Convert to nm
            temp_screen *= self._wavelengthInNm / (2 * np.pi)

            if self._gpu:
                ptr = GPUarray(temp_screen)
            else:
                ptr = np.array(temp_screen, copy=False)

            self._phasescreens.append(ptr)

        else:
            self._pixel_phasescreens = np.max(self._pixel_layer)

            if len(np.unique(self._L0)) == 1:
                # Number of rectangular phase screens from a single square phasescreen
                n_ps_from_square_ps = np.floor(self._pixel_square_phasescreens / self._pixel_phasescreens)
                # Number of square phasescreens
                n_ps = np.ceil(float(self._n_phasescreens) / n_ps_from_square_ps)

                # Seed vector
                seed = np.arange(self._seed, self._seed + int(n_ps))

                # Square phasescreens
                if self._make_cycle:
                    pixel_square_phasescreens = self._pixel_square_phasescreens - self._pixel_pupil
                    ps_cycle = get_layers(1, pixel_square_phasescreens, pixel_square_phasescreens * self._pixel_pitch,
                                          500e-9, 1, L0=self._L0[0], par=par, START=start, SEED=seed, DIR=self._directory,
                                          FILE=filename, no_sha=True, verbose=self._verbose)
                    ps_cycle = np.vstack([ps_cycle, ps_cycle[:, :self._pixel_pupil]])
                    ps_cycle = np.hstack([ps_cycle, ps_cycle[:self._pixel_pupil, :]])

                    square_phasescreens = [ps_cycle * 4 * np.pi]  # 4 * π is added to get the correct amplitude
                else:
                    square_phasescreens = phasescreens_manager(self._L0[0], self._pixel_square_phasescreens,
                                                               self._pixel_pitch, self._directory,
                                                               seed=seed, precision=self._precision,
                                                               verbose=self._verbose)

                square_ps_index = -1
                ps_index = 0

                for i in range(self._n_phasescreens):
                    # Increase square phase-screen index
                    if i % n_ps_from_square_ps == 0:
                        square_ps_index += 1
                        ps_index = 0

                    temp_screen = square_phasescreens[square_ps_index][:, self._pixel_phasescreens * ps_index:
                                                                           self._pixel_phasescreens * (ps_index + 1)]
                    temp_screen *= np.sqrt(self._Cn2[i])
                    temp_screen -= np.mean(temp_screen)
                    # Convert to nm
                    temp_screen *= self._wavelengthInNm / (2 * np.pi)

                    # Flip x-axis for each odd phase-screen
                    if i % 2 != 0:
                        temp_screen = np.flip(temp_screen, axis=1)

                    ps_index += 1

                    if self._gpu:
                        ptr = GPUarray(temp_screen)
                    else:
                        ptr = np.array(temp_screen, copy=False)

                    self._phasescreens.append(ptr)

            else:
                seed = self._seed + np.arange(self._n_phasescreens)

                if len(seed) != len(self._L0):
                    raise ValueError('Number of elements in seed and L0 must be the same!')

                # Square phasescreens
                square_phasescreens = phasescreens_manager(self._L0, self._pixel_square_phasescreens,
                                                           self._pixel_pitch, self._directory,
                                                           seed=seed, precision=self._precision,
                                                           verbose=self._verbose)

                for i in range(self._n_phasescreens):
                    temp_screen = square_phasescreens[i][:, :self._pixel_phasescreens]
                    temp_screen *= np.sqrt(self._Cn2[i])
                    temp_screen -= np.mean(temp_screen)
                    # Convert to nm
                    temp_screen *= self._wavelengthInNm / (2 * np.pi)

                    if self._gpu:
                        ptr = GPUarray(temp_screen)
                    else:
                        ptr = np.array(temp_screen, copy=False)

                    self._phasescreens.append(ptr)

    def shift_screens(self, t):
        seeing = self._seeing.value
        wind_speed = self._wind_speed.value
        wind_direction = self._wind_direction.value

        if len(self._phasescreens) != len(wind_speed) or len(self._phasescreens) != len(wind_direction):
            raise ValueError('Error: number of elements of wind speed and/or direction does not match the number of phasescreens')

        last_position = self._last_position if self._last_position else np.zeros_like(wind_speed)
        delta_time = self.t_to_seconds(t - self._last_t)
        if self._extra_delta_time and not self._last_position:
            delta_time += self._extra_delta_time
        delta_time = float(delta_time) if self._precision == 0 else delta_time

        r0 = 0.9759 * 0.5 / (seeing * 4.848) * self._airmass**(-3./5.) if seeing > 0 else 0.0
        r0wavelength = r0 * (self._wavelengthInNm / 500.0)**(6./5.)
        scale_coeff = (self._pixel_pitch / r0wavelength)**(5./6.) if seeing > 0 else 0.0

        self.phasescreens_shift(self._phasescreens, self._pixel_layer, wind_speed, wind_direction, delta_time,
                                self._pixel_pitch, scale_coeff, self._layer_list, position=last_position, cycle_screens=self._cycle_screens)

        for element in self._layer_list:
            element.generation_time = t
        self._layer_list.generation_time = t

        self._last_position = last_position
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
        if not np.isclose(np.sum(self._Cn2), 1.0, atol=1e-6):
            errmsg += f' Cn2 total must be 1. Instead is: {np.sum(self._Cn2)}.'

        return self._seed > 0 and isinstance(self._seeing, BaseValue) and isinstance(self._wind_direction, BaseValue) and isinstance(self._wind_speed, BaseValue)

    def trigger(self, t):
        self.shift_screens(t)

    def cleanup(self):
        super().cleanup()
        self._phasescreens.clear()
        del self._L0, self._heights, self._Cn2, self._pixel_layer, self._layer_list
        if self._verbose:
            print('Atmo_Propagation has been cleaned up.')
