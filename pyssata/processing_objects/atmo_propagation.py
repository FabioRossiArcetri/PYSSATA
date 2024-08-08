import numpy as np
from astropy.io import fits

from pyssata.base_processing_obj import BaseProcessingObj


class AtmoPropagation(BaseProcessingObj):
    '''Atmospheric propagation'''
    def __init__(self, source_list, pixel_pupil, pixel_pitch, GPU=False, precision=0):
        super().__init__(precision=precision)

        self._pixel_pupil = pixel_pupil
        self._pixel_pitch = pixel_pitch
        self._GPU = GPU
        self._precision = precision
        self._wavelengthInNm = 500.0
        self._doFresnel = False
        self._pupil_list = []
        self._layer_list = []
        self._source_list = []
        self._shiftXY_list = []
        self._rotAnglePhInDeg_list = []
        self._magnification_list = []
        self._recording = None
        self._useCudaRecording = False

        if GPU:
            self._recording = CudaRecording()
            if self._recording.is_valid():
                self._useCudaRecording = True

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
        if self._useCudaRecording:
            self._recording.invalidate()

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
        if self._GPU:
            required_version = 1.59
            actual_version = cuda_version()
            if actual_version < required_version:
                raise RuntimeError(f'This version of the propagation code requires GPU_SIMUL module version {required_version}, found version {actual_version} instead')

        if not self._propagators:
            if self._GPU:
                before = cuda_memory()[0, :]
            
            nlayers = len(self._layer_list)
            self._propagators = np.empty(nlayers, dtype=object) if self._GPU else []

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
                
                if self._GPU:
                    self._propagators[j] = GPUArray(H)
                else:
                    self._propagators.append(H)

            if self._verbose and self._GPU:
                occupy = before - cuda_memory()[0, :]
                print('Atmo_propagation GPU memory occupation:')
                for i, mem in enumerate(occupy):
                    print(f'GPU #{i}: {int(mem / (1024 * 1024))} MB')

    def propagate(self, t):
        if self._doFresnel:
            self.doFresnel_setup()

        if self._useCudaRecording and self._recording.is_valid():
            count = self._recording.replay()
            if count == self._recording.count():
                for i in range(len(self._source_list)):
                    self._pupil_list[i].generation_time = t
                self._pupil_list.generation_time = t
                return
            else:
                raise RuntimeError(f'Cuda replay failed, return value={count}')

        if not self._rotAnglePhInDeg_list and self._useCudaRecording:
            self._recording.start()

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

        if not self._rotAnglePhInDeg_list and self._useCudaRecording:
            self._recording.stop()

        self._pupil_list.generation_time = t

    def trigger(self, t):
        self.propagate(t)

    def add_layer_to_layer_list(self, layer):
        self._layer_list.append(layer)
        self._shiftXY_list.append(layer.shiftXYinPixel if hasattr(layer, 'shiftXYinPixel') else [0, 0])
        self._rotAnglePhInDeg_list.append(layer.rotInDeg if hasattr(layer, 'rotInDeg') else 0)
        self._magnification_list.append(max(layer.magnification, 1.0) if hasattr(layer, 'magnification') else 1.0)
        if self._useCudaRecording:
            self._recording.invalidate()
        self._propagators = None

    def add_layer(self, layer):
        self.add_layer_to_layer_list(layer)
        if self._useCudaRecording:
            self._recording.invalidate()

    def add_source(self, source):
        self._source_list.append(source)
        ef = EF(self._pixel_pupil, self._pixel_pupil, self._pixel_pitch, GPU=self._GPU, PRECISION=self._precision)
        ef.S0 = source.phot_density()
        self._pupil_list.append(ef)
        if self._useCudaRecording:
            self._recording.invalidate()

    def pupil(self, num):
        if num >= len(self._pupil_list):
            raise ValueError(f'Pupil #{num} does not exist')
        return self._pupil_list[num]

    def copy(self):
        prop = AtmoPropagation([], self._pixel_pupil, self._pixel_pitch, GPU=self._GPU, precision=self._precision)
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
        if self._recording:
            del self._recording
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
