import matplotlib.pyplot as plt

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.processing_objects.pyr_slopec import PyrSlopec
from pyssata.display.pupil_display import pupil_display
import numpy as np

from pyssata import xp
from pyssata import cpuArray
from pyssata.connections import InputValue
from pyssata.processing_objects.slopec import Slopec

class SlopecDisplay(BaseProcessingObj):
    def __init__(self, disp_factor=1):
        super().__init__()

        self._disp_factor = disp_factor
        self._windows = 21
        self._slopec = None
        self._title = ''
        self._do_image_show = False
        self._circle_disp = 0
        self.fig = self.ax = None
        self.inputs['slopec'] = InputValue(object=self._slopec, type=Slopec)

    @property
    def slopec(self):
        return self._slopec

    @slopec.setter
    def slopec(self, slopec):
        self._slopec = slopec

    @property
    def disp_factor(self):
        return self._disp_factor

    @disp_factor.setter
    def disp_factor(self, disp_factor):
        self._disp_factor = disp_factor

    @property
    def windows(self):
        return self._windows

    @windows.setter
    def windows(self, windows):
        self._windows = windows

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        self._title = title

    @property
    def do_image_show(self):
        return self._do_image_show

    @do_image_show.setter
    def do_image_show(self, do_image_show):
        self._do_image_show = do_image_show

    @property
    def circle_disp(self):
        return self._circle_disp

    @circle_disp.setter
    def circle_disp(self, circle_disp):
        self._circle_disp = circle_disp

    def trigger(self, t):
        s = self._slopec.out_slopes
        if s.generation_time == t:
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
                sx = s.xslopes
                sy = s.yslopes
                if isinstance(self._slopec, PyrSlopec):
                    map_data = self._slopec.pupdata.ind_pup[:, 1]
                    slope_side = None  # Auto scaled
#                TODO --- these SlopeC are not available yet
#                elif isinstance(self._slopec, (IdealWfsSlopec, ShSlopec, ShSlopec)):
#                    nx = self._slopec.subapdata.nx
#                    map_data = self._slopec.subapdata.map
#                    map_data = (map_data // nx) * nx * 2 + (map_data % nx)
#                    slope_side = nx * 2

                if self.fig is not None:
                    TARGET = (self.fig, self.ax)
                else:
                    TARGET = None
                title = self._title if self._title else 'Slope Display'
                _, _, _, self.fig, self.ax = pupil_display(cpuArray(self._slopec.in_pixels.pixels), cpuArray(sx), cpuArray(sy), 
                                                           cpuArray(map_data), self._slopec.in_pixels.pixels.shape[0], title=title, TARGET=TARGET,
                                                           do_image_show=True)

    def run_check(self, time_step):
        return self._slopec is not None

    def cleanup(self):
        plt.close(self._windows)

    @classmethod
    def from_dict(cls, params):
        return cls(**params)
