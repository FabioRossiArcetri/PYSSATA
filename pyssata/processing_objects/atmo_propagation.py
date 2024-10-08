from astropy.io import fits

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.data_objects.ef import ElectricField
from pyssata.connections import InputList
from pyssata.data_objects.layer import Layer

import numpy as np
import warnings

class AtmoPropagation(BaseProcessingObj):
    '''Atmospheric propagation'''
    def __init__(self,
                 source_dict,
                 pixel_pupil: int,
                 pixel_pitch: float,
                 target_device_idx=None, 
                 precision=None,
                 doFresnel: bool=False,
                 wavelengthInNm: float=500.0,
                 pupil_position=(0., 0.)):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if doFresnel and wavelengthInNm is None:
            raise ValueError('get_atmo_propagation: wavelengthInNm is required when doFresnel key is set to correctly simulate physical propagation.')

        self._pixel_pupil = pixel_pupil
        self._pixel_pitch = pixel_pitch        
        self._source_dict = source_dict
        self._pupil_dict = {}
        self._layer_list = []
        self._shiftXY_list = []
        self._rotAnglePhInDeg_list = []
        self._magnification_list = []
        self._pupil_position = pupil_position
        self._doFresnel = doFresnel
        self._wavelengthInNm = wavelengthInNm
        self._propagators = None

        for name, source in source_dict.items():
            self.add_source(name, source)
            self.outputs[name] = self._pupil_dict[name]
            
        self.inputs['layer_list'] = InputList(type=Layer)
#       uncomment when the code is a stream
#        super().build_stream()

    def add_source(self, name, source):
        ef = ElectricField(self._pixel_pupil, self._pixel_pupil, self._pixel_pitch, target_device_idx=self._target_device_idx)
        ef.S0 = source.phot_density()
        self._pupil_dict[name] = ef

    @property
    def pupil_dict(self):
        return self._pupil_dict

    @property
    def wavelengthInNm(self):
        return self._wavelengthInNm

    @wavelengthInNm.setter
    def wavelengthInNm(self, value):
        self._wavelengthInNm = value
        self._propagators = None

    @property
    def doFresnel(self):
        return self._doFresnel

    @doFresnel.setter
    def doFresnel(self, value):
        self._doFresnel = value

    def doFresnel_setup(self):
   
        if not self._propagators:
                        
            self._layer_list = self.inputs['layer_list'].get(self._target_device_idx)
            
            nlayers = len(self._layer_list)
            self._propagators = []

            height_layers = np.array([layer.height for layer in self._layer_list], dtype=self.dtype)
            sorted_heights = np.sort(height_layers)
            if not (np.allclose(height_layers, sorted_heights) or np.allclose(height_layers, sorted_heights[::-1])):
                raise ValueError('Layers must be sorted from highest to lowest or from lowest to highest')

            for j in range(nlayers):
                if j < nlayers - 1:
                    diff_height_layer = self._layer_list[j].height - self._layer_list[j + 1].height
                else:
                    diff_height_layer = self._layer_list[j].height
                
                side = self._pixel_pupil
                diameter = self._pixel_pupil * self._pixel_pitch
                H = field_propagator(side, diameter, self._wavelengthInNm, diff_height_layer, do_shift=True)
                
                self._propagators.append(H)

    def trigger_code(self):
        if self._doFresnel:
            self.doFresnel_setup()

        pupil_position = np.array(self._pupil_position)
        self._layer_list = self.inputs['layer_list'].get(self._target_device_idx)
    
        for name, source in self._source_dict.items():
            height_star = source.height
            polar_coordinate = source.polar_coordinate
            pupil = self._pupil_dict[name]
            pupil.reset()
            update_ef = pupil
            pixel_pupil = update_ef.size[0]
            #if pupil_position is not None:
            #    warnings.warn('WARNING: pupil_position is not null')
            for i, layer in enumerate(self._layer_list):
                shiftXY = self._shiftXY_list[i]
                rotAnglePhInDeg = self._rotAnglePhInDeg_list[i]
                magnify = self._magnification_list[i]
                if self._propagators:
                    propagator = self._propagators[i]
                else:
                    propagator = None

                height_layer = layer.height
                pixel_pitch = layer.pixel_pitch
                diff_height = height_star - height_layer
                s_layer = layer.size

                if (height_layer == 0 or (np.isinf(height_star) and polar_coordinate[0] == 0)) and \
                ((shiftXY is None) or (not np.any(shiftXY))) and \
                ((not pupil_position.any())) and \
                ((rotAnglePhInDeg is None) or (rotAnglePhInDeg == 0)) and \
                ((magnify is None) or (magnify == 1)):

                    topleft = [(s_layer[0] - pixel_pupil) // 2, (s_layer[1] - pixel_pupil) // 2]
                    update_ef.product(layer, subrect=topleft)

                elif diff_height > 0:
                    sec2rad = 4.848e-6
                    degree2rad = np.pi / 180.
                    r_angle = polar_coordinate[0] * sec2rad
                    phi_angle = polar_coordinate[1] * degree2rad

                    pixel_layer = s_layer[0]
                    half_pixel_layer = np.array([(pixel_layer - 1) / 2., (pixel_layer - 1) / 2.]) 
                    if shiftXY is not None:
                        half_pixel_layer -= shiftXY

                    if pupil_position is not None and pixel_layer > pixel_pupil and np.isfinite(height_star):
                        pixel_position_s = r_angle * height_layer / pixel_pitch
                        pixel_position = pixel_position_s * np.array( [np.cos(phi_angle), np.sin(phi_angle)]) + pupil_position / pixel_pitch
                    elif pupil_position is not None and pixel_layer > pixel_pupil and not np.isfinite(height_star):
                        pixel_position_s = r_angle * height_star / pixel_pitch
                        sky_pixel_position = pixel_position_s * np.array( [np.cos(phi_angle), np.sin(phi_angle)])
                        pupil_pixel_position = pupil_position / pixel_pitch
                        pixel_position = (sky_pixel_position - pupil_pixel_position) * height_layer / height_star + pupil_pixel_position
                    else:
                        pixel_position_s = r_angle * height_layer / pixel_pitch
                        pixel_position = pixel_position_s * np.array( [np.cos(phi_angle), np.sin(phi_angle)])
                    if np.isfinite(height_star):
                        pixel_pupmeta = pixel_pupil
                    else:
                        cone_coeff = abs(height_star - abs(height_layer)) / height_star
                        pixel_pupmeta = pixel_pupil * cone_coeff
                    #if magnify is not None:
                    #    pixel_pupmeta /= magnify
                    #    tempA = layer.A
                    #    tempP = layer.phaseInNm
                    #    tempP[tempA == 0] = self.xp.mean(tempP[tempA != 0])
                    #    layer.phaseInNm = tempP

                    xx, yy = np.meshgrid(np.arange(pixel_pupil), np.arange(pixel_pupil))

                    if rotAnglePhInDeg is not None:
                        angle = (-rotAnglePhInDeg % 360) * np.pi / 180
                        x = np.cos(angle) * xx - np.sin(angle) * yy + half_pixel_layer + pixel_position[0]
                        y = np.sin(angle) * xx + np.cos(angle) * yy + half_pixel_layer + pixel_position[1]
                        GRID = 0
                    else:
                        x = xx + half_pixel_layer + pixel_position[0]
                        y = yy + half_pixel_layer + pixel_position[1]
                        GRID = 1

                    points = self.xp.asarray(np.vstack((x.ravel(), y.ravel())).T)
                    interpolator_A = self.RegularGridInterpolator((self.xp.arange(layer.size[0]), self.xp.arange(layer.size[1])), layer.A, bounds_error=False, fill_value=0)
                    interpolator_phase = self.RegularGridInterpolator((self.xp.arange(layer.size[0]), self.xp.arange(layer.size[1])), layer.phaseInNm, bounds_error=False, fill_value=0)
                    pupil_ampl_temp = interpolator_A(points).reshape(pixel_pupil, pixel_pupil)
                    pupil_phase_temp = interpolator_phase(points).reshape(pixel_pupil, pixel_pupil)


                    update_ef.A *= pupil_ampl_temp
                    update_ef.phaseInNm += pupil_phase_temp

                if self._doFresnel:
                    update_ef.physical_prop(self.wavelengthInNm, propagator, temp_array=None)

            pupil.generation_time = self.current_time


    def run_check(self, time_step):
        # TODO here for no better place, we need something like a "setup()" method called before the loop starts        
        self._layer_list = self.inputs['layer_list'].get(self._target_device_idx)        
        self._shiftXY_list = [layer.shiftXYinPixel if hasattr(layer, 'shiftXYinPixel') else np.array([0, 0]) for layer in self._layer_list]
        self._rotAnglePhInDeg_list = [layer.rotInDeg if hasattr(layer, 'rotInDeg') else 0 for layer in self._layer_list]
        self._magnification_list = [max(layer.magnification, 1.0) if hasattr(layer, 'magnification') else 1.0 for layer in self._layer_list]

        errmsg = ''
        if not (len(self._source_dict) > 0):
            errmsg += 'no source'
        if not (len(self._layer_list) > 0):
            errmsg += 'no layers'
        if not (self._pixel_pupil > 0):
            errmsg += 'pixel pupil <= 0'
        if not (self._pixel_pitch > 0):
            errmsg += 'pixel pitch <= 0'
        return (len(self._source_dict) > 0 and
                len(self._layer_list) > 0 and
                self._pixel_pupil > 0 and
                self._pixel_pitch > 0), errmsg

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


