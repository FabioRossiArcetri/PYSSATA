from astropy.io import fits

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.data_objects.ef import ElectricField
from pyssata.connections import InputList
from pyssata.data_objects.layer import Layer
from pyssata import cp, fuse

import numpy as np
import warnings

sec2rad = 4.848e-6
degree2rad = np.pi / 180.

@fuse(kernel_name='rot_points')
def rot_points(angle , xx, yy, half_pixel_layer, p0, p1, xp):
    x = xp.cos(angle) * xx - xp.sin(angle) * yy + half_pixel_layer + p0
    y = xp.sin(angle) * xx + xp.cos(angle) * yy + half_pixel_layer + p1
    points = xp.vstack((x.ravel(), y.ravel())).T
    return points

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

        self._pixel_pupil_size = pixel_pupil
        self._pixel_pitch = pixel_pitch        
        self._source_dict = source_dict
        self._pupil_dict = {}
        self._layer_list = []
        self._shiftXY_list = []
        self._rotAnglePhInDeg_list = []
        self._magnification_list = []
        self._pupil_position = pupil_position
        self.pupil_position_arr = np.array(self._pupil_position)
        self.pupil_position_cond = self.pupil_position_arr.any()

        self._doFresnel = doFresnel
        self._wavelengthInNm = wavelengthInNm
        self._propagators = None

        for name, source in source_dict.items():
            self.add_source(name, source)
            self.outputs[name] = self._pupil_dict[name]
            
        self.inputs['layer_list'] = InputList(type=Layer)
        self.xx, self.yy = self.xp.meshgrid(self.xp.arange(self._pixel_pupil_size), self.xp.arange(self._pixel_pupil_size))

    def add_source(self, name, source):
        ef = ElectricField(self._pixel_pupil_size, self._pixel_pupil_size, self._pixel_pitch, target_device_idx=self._target_device_idx)
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
                        
            self._layer_list = self.local_inputs['layer_list']
            
            nlayers = len(self._layer_list)
            self._propagators = []

            height_layers = np.array([layer.height for layer in self._layer_list], dtype=self.dtype)
            sorted_heights = np.sort(height_layers)
            if not (np.allclose(height_layers, sorted_heights) or np.allclose(height_layers, sorted_heights[::-1])):
                raise ValueError('Layers must be sorted from highest to lowest or from lowest to highest')

            for j in range(nlayers):
                if j < nlayers - 1:
                    self.diff_height_layer = self._layer_list[j].height - self._layer_list[j + 1].height
                else:
                    self.diff_height_layer = self._layer_list[j].height
                
                diameter = self._pixel_pupil_size * self._pixel_pitch
                H = field_propagator(self._pixel_pupil_size, diameter, self._wavelengthInNm, self.diff_height_layer, do_shift=True)
                
                self._propagators.append(H)

    def trigger_code(self):
        #if self._doFresnel:
        #    self.doFresnel_setup()
        self._layer_list = self.local_inputs['layer_list']
        j=-1
        kk=len(self._layer_list)
        for name, source in self._source_dict.items():
            j+=1
            self.polar_coordinate = source.polar_coordinate
            r_angle = self.polar_coordinate[0] * sec2rad
            phi_angle = self.polar_coordinate[1] * degree2rad
            self.update_ef = self._pupil_dict[name]
            self.update_ef.reset()
            for i, layer in enumerate(self._layer_list):
#                if self._propagators:
#                    propagator = self._propagators[i]
#                else:
#                    propagator = None
                self.pixel_pitch = layer.pixel_pitch
                self.diff_height = self.height_star[j] - self.height_layer[i]

                if (self.height_layer[i] == 0 or (not self.height_star_cond[j] and self.polar_coordinate[0] == 0)) and \
                                not self._shiftXY_cond[i] and \
                                not self.pupil_position_cond and \
                                self._rotAnglePhInDeg_list[i] == 0 and \
                                self._magnification_list[i] == 1:
                    topleft = [(self._layer_sizes[i][0] - self._pixel_pupil_size) // 2, (self._layer_sizes[i][1] - self._pixel_pupil_size) // 2]
                    self.update_ef.product(layer, subrect=topleft)
                elif self.diff_height > 0:
                    pixel_layer = self._layer_sizes[i][0]
                    half_pixel_layer = np.array([(pixel_layer - 1) / 2., (pixel_layer - 1) / 2.]) 
                    if self._shiftXY_list[i] is not None:
                        half_pixel_layer -= self._shiftXY_list[i]

                    if pixel_layer > self._pixel_pupil_size and self.height_star_cond[j]:
                        pixel_position_s = r_angle * self.height_layer[i] / self.pixel_pitch
                        pixel_position = pixel_position_s * np.array( [np.cos(phi_angle), np.sin(phi_angle)]) + self.pupil_position_arr / self.pixel_pitch
                    elif pixel_layer > self._pixel_pupil_size and not self.height_star_cond[j]:
                        pixel_position_s = r_angle * self.height_star[j] / self.pixel_pitch
                        sky_pixel_position = pixel_position_s * np.array( [np.cos(phi_angle), np.sin(phi_angle)])
                        pupil_pixel_position = self.pupil_position_arr / self.pixel_pitch
                        pixel_position = (sky_pixel_position - pupil_pixel_position) * self.height_layer[i] / self.height_star[j] + pupil_pixel_position
                    else:
                        pixel_position_s = r_angle * self.height_layer[i] / self.pixel_pitch
                        pixel_position = pixel_position_s * np.array( [np.cos(phi_angle), np.sin(phi_angle)])
                    if self.height_star_cond[j]:
                        pixel_pupmeta = self._pixel_pupil_size
                    else:
                        cone_coeff = abs(self.height_star[j] - abs(self.height_layer[i])) / self.height_star[j]
                        pixel_pupmeta = self._pixel_pupil_size * cone_coeff
                    if self._magnification_list[i] is not None:
                        pixel_pupmeta /= self._magnification_list[i]
                        tempA = layer.A
                        tempP = layer.phaseInNm
                        tempP[tempA == 0] = self.xp.mean(tempP[tempA != 0])
                        layer.phaseInNm = tempP
                    with self.map_streams[kk*j+i]:
                        angle = self._rotAnglePhInDeg_list[i]
                        points = rot_points(angle, self.xx, self.yy, half_pixel_layer, pixel_position[0], pixel_position[1])
                        interpolator_A = self.RegularGridInterpolator(self.LL[i], layer.A, bounds_error=False, fill_value=0)
                        interpolator_phase = self.RegularGridInterpolator(self.LL[i], layer.phaseInNm, bounds_error=False, fill_value=0)
                        self.update_ef.A *= interpolator_A(points).reshape(self._pixel_pupil_size, self._pixel_pupil_size)
                        self.update_ef.phaseInNm += interpolator_phase(points).reshape(self._pixel_pupil_size, self._pixel_pupil_size)
        #self._target_device.synchronize()
#                if self._doFresnel:
#                    self.update_ef.physical_prop(self.wavelengthInNm, propagator, temp_array=None)
        for name, source in self._source_dict.items():
            self.update_ef = self._pupil_dict[name]
            self.update_ef.generation_time = self.current_time


    def run_check(self, time_step):
        # TODO here for no better place, we need something like a "setup()" method called before the loop starts        
        self._layer_list = self.inputs['layer_list'].get(self._target_device_idx)        
        self._shiftXY_list = [layer.shiftXYinPixel if hasattr(layer, 'shiftXYinPixel') else np.array([0, 0]) for layer in self._layer_list]
        self._shiftXY_cond = [np.any(layer.shiftXYinPixel) if hasattr(layer, 'shiftXYinPixel')  else False for layer in self._layer_list]
        self._rotAnglePhInDeg_list = [ (-layer.rotInDeg % 360) * degree2rad if hasattr(layer, 'rotInDeg') else 0 for layer in self._layer_list]
        self._magnification_list = [max(layer.magnification, 1.0) if hasattr(layer, 'magnification') else 1.0 for layer in self._layer_list]
        self._layer_sizes = [layer.size if hasattr(layer, 'size') else [0, 0] for layer in self._layer_list]
        self.height_layer = [layer.height for layer in self._layer_list]

        self.height_star = [source.height for source in self._source_dict.values() ]
        self.height_star_cond = [np.isfinite(source.height) for source in self._source_dict.values() ]

#        self._condition1 = [ for i in range(len(self._layer_list)) ]
        self.LL=[]
        for i, ls in enumerate(self._layer_sizes):
            self.LL.append( (self.xp.arange(self._layer_sizes[i][0]), self.xp.arange(self._layer_sizes[i][1])) )

        self.map_streams = []
        for i in range(len(self._layer_list)*len(self._source_dict)):
            self.map_streams.append(cp.cuda.stream.Stream(non_blocking=True))

        errmsg = ''
        if not (len(self._source_dict) > 0):
            errmsg += 'no source'
        if not (len(self._layer_list) > 0):
            errmsg += 'no layers'
        if not (self._pixel_pupil_size > 0):
            errmsg += 'pixel pupil <= 0'
        if not (self._pixel_pitch > 0):
            errmsg += 'pixel pitch <= 0'
        return (len(self._source_dict) > 0 and
                len(self._layer_list) > 0 and
                self._pixel_pupil_size > 0 and
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


