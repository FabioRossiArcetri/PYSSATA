import numpy as np
from astropy.io import fits

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.data_objects.ef import ElectricField
from pyssata.lib.layers2pupil_ef import layers2pupil_ef
from pyssata.connections import InputList, OutputValue
from pyssata.data_objects.layer import Layer

class AtmoPropagation(BaseProcessingObj):
    '''Atmospheric propagation'''
    def __init__(self,
                 source_dict,
                 pixel_pupil: int,
                 pixel_pitch: float,
                 precision=0,
                 doFresnel: bool=False,
                 wavelengthInNm: float=500.0,
                 pupil_position=(0., 0.)):
        super().__init__(precision=precision)

        if doFresnel and wavelengthInNm is None:
            raise ValueError('get_atmo_propagation: wavelengthInNm is required when doFresnel key is set to correctly simulate physical propagation.')

        self._pixel_pupil = pixel_pupil
        self._pixel_pitch = pixel_pitch
        self._precision = precision
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
            self.outputs[name] = OutputValue(object=self._pupil_dict[name], type=ElectricField)
            setattr(self, name, self._pupil_dict[name])   # TODO it will be removed when output get/set methods will be used
            
        self.inputs['layer_list'] = InputList(object=self.layer_list, type=Layer)

    def add_source(self, name, source):
        ef = ElectricField(self._pixel_pupil, self._pixel_pupil, self._pixel_pitch)
        ef.S0 = source.phot_density()
        self._pupil_dict[name] = ef

    @property
    def pupil_dict(self):
        return self._pupil_dict

    @property
    def layer_list(self):
        return self._layer_list

    @layer_list.setter
    def layer_list(self, layer_list):
        self._layer_list = layer_list
        self._propagators = None

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
            
            nlayers = len(self._layer_list)
            self._propagators = []

            height_layers = np.array([layer.height for layer in self._layer_list])
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

    def propagate(self, t):
        if self._doFresnel:
            self.doFresnel_setup()

        shiftXY_list = self._shiftXY_list if self._shiftXY_list else None
        rotAnglePhInDeg_list = self._rotAnglePhInDeg_list if self._rotAnglePhInDeg_list else None
        magnification_list = self._magnification_list if self._magnification_list else None
        pupil_position = self._pupil_position if np.any(self._pupil_position) else None

        for name, source in self._source_dict.items():
            height_star = source.height
            polar_coordinate_star = source.polar_coordinate
            pupil = self._pupil_dict[name]

            pupil.reset()
            layers2pupil_ef(self._layer_list, height_star, polar_coordinate_star,
                            update_ef=pupil, shiftXY_list=shiftXY_list,
                            rotAnglePhInDeg_list=rotAnglePhInDeg_list, magnify_list=magnification_list,
                            pupil_position=pupil_position, doFresnel=self._doFresnel,
                            propagators=self._propagators, wavelengthInNm=self._wavelengthInNm)

            pupil.generation_time = t

    def trigger(self, t):
        self.propagate(t)

    def add_layer_to_layer_list(self, layer):
        self._layer_list.append(layer)
        self._shiftXY_list.append(layer.shiftXYinPixel if hasattr(layer, 'shiftXYinPixel') else [0, 0])
        self._rotAnglePhInDeg_list.append(layer.rotInDeg if hasattr(layer, 'rotInDeg') else 0)
        self._magnification_list.append(max(layer.magnification, 1.0) if hasattr(layer, 'magnification') else 1.0)
        self._propagators = None

    def add_layer(self, layer):
        self.add_layer_to_layer_list(layer)

    def run_check(self, time_step):
        # TODO here for no better place, we need something like a "setup()" method called before the loop starts
        self._shiftXY_list = [layer.shiftXYinPixel if hasattr(layer, 'shiftXYinPixel') else [0, 0] for layer in self.layer_list]
        self._rotAnglePhInDeg_list = [layer.rotInDeg if hasattr(layer, 'rotInDeg') else 0 for layer in self.layer_list]
        self._magnification_list = [max(layer.magnification, 1.0) if hasattr(layer, 'magnification') else 1.0 for layer in self.layer_list]

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

    def cleanup(self):
        self._source_dict.clear()
        self._pupil_list.clear()
        self._layer_list.clear()
        self._shiftXY_list.clear()
        self._rotAnglePhInDeg_list.clear()
        self._magnification_list.clear()

        super().cleanup()
        if self._verbose:
            print('Atmo_Propagation has been cleaned up.')

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
