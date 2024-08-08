import numpy as np
from astropy.io import fits

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.data_objects.ef import ElectricField

class AtmoPropagation(BaseProcessingObj):
    '''Atmospheric propagation'''
    def __init__(self, source_list, pixel_pupil, pixel_pitch, precision=0):
        super().__init__(precision=precision)

        self._pixel_pupil = pixel_pupil
        self._pixel_pitch = pixel_pitch
        self._precision = precision
        self._wavelengthInNm = 500.0
        self._doFresnel = False
        self._pupil_list = []
        self._layer_list = []
        self._source_list = []
        self._shiftXY_list = []
        self._rotAnglePhInDeg_list = []
        self._magnification_list = []

        for source in source_list:
            self.add_source(source)

    @property
    def pupil_list(self):
        return self._pupil_list

    @property
    def layer_list(self):
        return self._layer_list

    @layer_list.setter
    def layer_list(self, value):
        self._layer_list = value
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

        for i, element in enumerate(self._source_list):
            height_star = element.height
            polar_coordinate_star = element.polar_coordinate

            self._pupil_list[i].reset()
            layers2pupil_ef(self._layer_list, height_star, polar_coordinate_star,
                            update_ef=self._pupil_list[i], shiftXY_list=shiftXY_list,
                            rotAnglePhInDeg_list=rotAnglePhInDeg_list, magnify_list=magnification_list,
                            pupil_position=pupil_position, doFresnel=self._doFresnel,
                            propagators=self._propagators, wavelengthInNm=self._wavelengthInNm)

            self._pupil_list[i].generation_time = t

        self._pupil_list.generation_time = t

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

    def add_source(self, source):
        self._source_list.append(source)
        ef = ElectricField(self._pixel_pupil, self._pixel_pupil, self._pixel_pitch)
        ef.S0 = source.phot_density()
        self._pupil_list.append(ef)

    def pupil(self, num):
        if num >= len(self._pupil_list):
            raise ValueError(f'Pupil #{num} does not exist')
        return self._pupil_list[num]

    def copy(self):
        prop = AtmoPropagation([], self._pixel_pupil, self._pixel_pitch, precision=self._precision)
        for layer in self._layer_list:
            prop.add_layer(layer)
        for source in self._source_list:
            prop.add_source(source)
        return prop

    def run_check(self, time_step):
        errmsg = ''
        if not (len(self._source_list) > 0):
            errmsg += 'no source'
        if not (len(self._layer_list) > 0):
            errmsg += 'no layers'
        if not (self._pixel_pupil > 0):
            errmsg += 'pixel pupil <= 0'
        if not (self._pixel_pitch > 0):
            errmsg += 'pixel pitch <= 0'
        return (len(self._source_list) > 0 and
                len(self._layer_list) > 0 and
                self._pixel_pupil > 0 and
                self._pixel_pitch > 0), errmsg

    def cleanup(self):
        self._source_list.clear()
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
