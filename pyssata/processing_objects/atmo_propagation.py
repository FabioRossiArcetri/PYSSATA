from astropy.io import fits

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.data_objects.ef import ElectricField
#from pyssata.lib.layers2pupil_ef import layers2pupil_ef
from pyssata.connections import InputList
from pyssata.data_objects.layer import Layer

from scipy.ndimage import rotate
import warnings
from scipy.interpolate import RegularGridInterpolator

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

            height_layers = self.xp.array([layer.height for layer in self._layer_list], dtype=self.dtype)
            sorted_heights = self.xp.sort(height_layers)
            if not (self.xp.allclose(height_layers, sorted_heights) or self.xp.allclose(height_layers, sorted_heights[::-1])):
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

    def propagate(self, t):
        if self._doFresnel:
            self.doFresnel_setup()

        shiftXY_list = self._shiftXY_list if self._shiftXY_list else None
        rotAnglePhInDeg_list = self._rotAnglePhInDeg_list if self._rotAnglePhInDeg_list else None
        magnification_list = self._magnification_list if self._magnification_list else None
        pupil_position = self.xp.array(self._pupil_position, dtype=self.dtype) if self.xp.any(self.xp.array(self._pupil_position, dtype=self.dtype)) else None

        self._layer_list = self.inputs['layer_list'].get(self._target_device_idx)
    
        for name, source in self._source_dict.items():
            height_star = source.height
            polar_coordinate_star = source.polar_coordinate
            pupil = self._pupil_dict[name]

            pupil.reset()
            self.layers2pupil_ef(self._layer_list, height_star, polar_coordinate_star,
                            update_ef=pupil, shiftXY_list=shiftXY_list,
                            rotAnglePhInDeg_list=rotAnglePhInDeg_list, magnify_list=magnification_list,
                            pupil_position=pupil_position, doFresnel=self._doFresnel,
                            propagators=self._propagators, wavelengthInNm=self._wavelengthInNm)

            pupil.generation_time = t

    def trigger(self, t):
        self.propagate(t)

    def run_check(self, time_step):
        # TODO here for no better place, we need something like a "setup()" method called before the loop starts        
        self._layer_list = self.inputs['layer_list'].get(self._target_device_idx)        
        self._shiftXY_list = [layer.shiftXYinPixel if hasattr(layer, 'shiftXYinPixel') else self.xp.array([0, 0]) for layer in self._layer_list]
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



    def single_layer2pupil_ef(self, layer_ef, polar_coordinate, height_source, update_ef=None, shiftXY=None, rotAnglePhInDeg=None,
                            magnify=None, pupil_position=None, temp_ef=None, doFresnel=False, propagator=None,
                            wavelengthInNm=None, temp_array=None):
        
        height_layer = layer_ef.height
        pixel_pitch = layer_ef.pixel_pitch
        pixel_pupil = update_ef.size[0]

        diff_height = height_source - height_layer

        if (height_layer == 0 or (self.xp.isinf(height_source) and polar_coordinate[0] == 0)) and \
        ((shiftXY is None) or (self.xp.all(shiftXY == self.xp.array([0, 0], dtype=self.dtype)))) and \
        ((pupil_position is None) or (pupil_position == 0)) and \
        ((rotAnglePhInDeg is None) or (rotAnglePhInDeg == 0)) and \
        ((magnify is None) or (magnify == 1)):

            s_layer = layer_ef.size

            topleft = [(s_layer[0] - pixel_pupil) // 2, (s_layer[1] - pixel_pupil) // 2]
            update_ef.product(layer_ef, subrect=topleft)

        elif diff_height > 0:
            sec2rad = 4.848e-6
            degree2rad = self.xp.pi / 180.
            r_angle = polar_coordinate[0] * sec2rad
            phi_angle = polar_coordinate[1] * degree2rad

            pixel_layer = layer_ef.size[0]
            half_pixel_layer_x = (pixel_layer - 1) / 2.
            half_pixel_layer_y = (pixel_layer - 1) / 2.
            if shiftXY is not None:
                half_pixel_layer_x -= shiftXY[0]
                half_pixel_layer_y -= shiftXY[1]

            if pupil_position is not None and pixel_layer > pixel_pupil and self.xp.isfinite(height_source):
                pixel_position = r_angle * height_layer / pixel_pitch
                pixel_position_x = pixel_position * self.xp.cos(phi_angle) + pupil_position[0] / pixel_pitch
                pixel_position_y = pixel_position * self.xp.sin(phi_angle) + pupil_position[1] / pixel_pitch
            elif pupil_position is not None and pixel_layer > pixel_pupil and not self.xp.isfinite(height_source):
                pixel_position = r_angle * height_source / pixel_pitch
                sky_pixel_position_x = pixel_position * self.xp.cos(phi_angle)
                sky_pixel_position_y = pixel_position * self.xp.sin(phi_angle)

                pupil_pixel_position_x = pupil_position[0] / pixel_pitch
                pupil_pixel_position_y = pupil_position[1] / pixel_pitch

                pixel_position_x = (sky_pixel_position_x - pupil_pixel_position_x) * height_layer / height_source + pupil_pixel_position_x
                pixel_position_y = (sky_pixel_position_y - pupil_pixel_position_y) * height_layer / height_source + pupil_pixel_position_y
            else:
                pixel_position = r_angle * height_layer / pixel_pitch
                pixel_position_x = pixel_position * self.xp.cos(phi_angle)
                pixel_position_y = pixel_position * self.xp.sin(phi_angle)

            if self.xp.isfinite(height_source):
                pixel_pupmeta = pixel_pupil
            else:
                cone_coeff = abs(height_source - abs(height_layer)) / height_source
                pixel_pupmeta = pixel_pupil * cone_coeff

            if magnify is not None:
                pixel_pupmeta /= magnify
                tempA = layer_ef.A
                tempP = layer_ef.phaseInNm
                tempP[tempA == 0] = self.xp.mean(tempP[tempA != 0])
                layer_ef.phaseInNm = tempP

            xx, yy = self.xp.meshgrid(self.xp.arange(pixel_pupil), self.xp.arange(pixel_pupil))

            if rotAnglePhInDeg is not None:
                angle = (-rotAnglePhInDeg % 360) * self.xp.pi / 180
                x = self.xp.cos(angle) * xx - self.xp.sin(angle) * yy + half_pixel_layer_x + pixel_position_x
                y = self.xp.sin(angle) * xx + self.xp.cos(angle) * yy + half_pixel_layer_y + pixel_position_y
                GRID = 0
            else:
                x = xx + half_pixel_layer_x + pixel_position_x
                y = yy + half_pixel_layer_y + pixel_position_y
                GRID = 1

            points = self.xp.vstack((x.ravel(), y.ravel())).T
            interpolator_A = RegularGridInterpolator((self.xp.arange(layer_ef.size[0]), self.xp.arange(layer_ef.size[1])), layer_ef.A, bounds_error=False, fill_value=0)
            interpolator_phase = RegularGridInterpolator((self.xp.arange(layer_ef.size[0]), self.xp.arange(layer_ef.size[1])), layer_ef.phaseInNm, bounds_error=False, fill_value=0)
            pupil_ampl_temp = interpolator_A(points).reshape(pixel_pupil, pixel_pupil)
            pupil_phase_temp = interpolator_phase(points).reshape(pixel_pupil, pixel_pupil)


            update_ef.A *= pupil_ampl_temp
            update_ef.phaseInNm += pupil_phase_temp

        if doFresnel:
            update_ef.physical_prop(wavelengthInNm, propagator, temp_array=temp_array)

    def layers2pupil_ef(self, layers, height_source, polar_coordinate_source, update_ef=None, shiftXY_list=None, rotAnglePhInDeg_list=None,
                        magnify_list=None, pupil_position=None, temp_ef=None, doFresnel=False, propagators=None,
                        wavelengthInNm=None, temp_array=None):

        if pupil_position is not None:
            warnings.warn('WARNING: pupil_position is not null')

        for i, layer in enumerate(layers):
            shiftXY = shiftXY_list[i] if shiftXY_list is not None and len(shiftXY_list) > 0 else None
            rotAnglePhInDeg = rotAnglePhInDeg_list[i] if rotAnglePhInDeg_list is not None and len(rotAnglePhInDeg_list) > 0 else None
            magnify = magnify_list[i] if magnify_list is not None and len(magnify_list) > 0 else None
            propagator = propagators[i] if propagators is not None else None
            
            self.single_layer2pupil_ef(layer, polar_coordinate_source, height_source, update_ef=update_ef, shiftXY=shiftXY,
                                rotAnglePhInDeg=rotAnglePhInDeg, magnify=magnify, pupil_position=pupil_position,
                                temp_ef=temp_ef, doFresnel=doFresnel, propagator=propagator,
                                wavelengthInNm=wavelengthInNm, temp_array=temp_array)
