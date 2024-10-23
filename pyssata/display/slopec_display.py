from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.data_objects.pixels import Pixels
from pyssata.data_objects.pupdata import PupData
from pyssata.data_objects.subap_data import SubapData
from pyssata.data_objects.slopes import Slopes
from pyssata.processing_objects.pyr_slopec import PyrSlopec
from pyssata.processing_objects.sh_slopec import ShSlopec
from pyssata.display.pupil_display import pupil_display

from pyssata import cpuArray
from pyssata.connections import InputValue

class SlopecDisplay(BaseProcessingObj):
    def __init__(self, disp_factor=1):
        super().__init__(target_device_idx=-1)

        self._disp_factor = disp_factor
        self._windows = 21
        self._title = ''
        self._do_image_show = False
        self._circle_disp = 0
        self.fig = self.ax = None
        self.inputs['slopes'] = InputValue(type=Slopes)
        self.inputs['pupdata'] = InputValue(type=PupData)
        self.inputs['pixels'] = InputValue(type=Pixels)
        self.inputs['subapdata'] = InputValue(type=SubapData)

    def trigger_code(self):
        
        slopes_obj = self.local_inputs['slopes']
        pixels_obj = self.local_inputs['pixels']
        pupdata = self.local_inputs['pupdata']
        subapdata = self.local_inputs['subapdata']
        pixels = pixels_obj.pixels

        # TODO - ModalAnalysisSlopec not available yet
        # if isinstance(self._slopec, ModalAnalysisSlopec):
        #     if not plt.fignum_exists(self._windows):
        #         plt.figure(self._windows)
        #         plt.title('Modal Analysis Measurement')
        #     plt.clf()
        #     plt.plot(self._slopec.out_slopes.slopes)
        #     plt.xlabel('Mode Number')
        #     plt.ylabel('Mode Amplitude')
        #     plt.draw()
        #else:
        sx = slopes_obj.xslopes
        sy = slopes_obj.yslopes
        if pupdata:
            # Pyramid
            map_data = pupdata.ind_pup[:, 1]
        else:
            #SH
            nx = subapdata.nx
            map_data = subapdata.map
            np_sub = subapdata.np_sub
            map_data = (map_data // nx) * nx * np_sub + (map_data % nx)

        if self.fig is not None:
            TARGET = (self.fig, self.ax)
        else:
            TARGET = None
        title = self._title if self._title else 'Slope Display'
        _, _, _, self.fig, self.ax = pupil_display(pixels, sx, sy, 
                                                    map_data, pixels.shape[0], title=title, TARGET=TARGET,
                                                    do_image_show=True)

    def run_check(self, time_step):
        # TODO
        #slopec = self.inputs['slopec'].get(self.target_device_idx)
        #return slopec is not None
        return True
