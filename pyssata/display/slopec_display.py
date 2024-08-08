import matplotlib.pyplot as plt
import numpy as np

from pyssata.base_processing_obj import BaseProcessingObj


class SlopecDisplay(BaseProcessingObj):
    def __init__(self, slopec=None):
        super().__init__()

        self._disp_factor = 1
        self._windows = 21
        self._slopec = slopec if slopec is not None else None
        self._title = ''
        self._do_image_show = False
        self._circle_disp = 0

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
            if isinstance(self._slopec, ModalAnalysisSlopec):
                if not plt.fignum_exists(self._windows):
                    plt.figure(self._windows)
                    plt.title('Modal Analysis Measurement')
                plt.clf()
                plt.plot(self._slopec.out_slopes.slopes)
                plt.xlabel('Mode Number')
                plt.ylabel('Mode Amplitude')
                plt.draw()
            else:
                sx = s.xslopes
                sy = s.yslopes
                if isinstance(self._slopec, PyrSlopec):
                    map_data = self._slopec.pupdata.ind_pup[1, :]
                    slope_side = None  # Auto scaled
                elif isinstance(self._slopec, (IdealWfsSlopec, ShSlopec, ShSlopecGpu)):
                    nx = self._slopec.subapdata.nx
                    map_data = self._slopec.subapdata.map
                    map_data = (map_data // nx) * nx * 2 + (map_data % nx)
                    slope_side = nx * 2

                title = self._title if self._title else 'Slope Display'
                self.pupil_display(self._slopec.in_pixels.pixels, sx, sy, map_data, slope_side=slope_side, title=title)

    def pupil_display(self, pixels, sx, sy, map_data, slope_side=None, title='', do_image_show=False, circle_disp=0):
        plt.figure(self._windows)
        plt.clf()
        plt.imshow(pixels, cmap='gray')
        plt.quiver(map_data[::2], map_data[1::2], sx, sy)
        plt.title(title)
        plt.draw()

    def run_check(self, time_step):
        return self._slopec is not None

    def cleanup(self):
        plt.close(self._windows)

    @classmethod
    def from_dict(cls, params):
        return cls(**params)
