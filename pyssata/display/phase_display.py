import numpy as np
from pyssata import gpuEnabled
from pyssata import xp
from pyssata import cpuArray

import matplotlib.pyplot as plt

from pyssata.base_processing_obj import BaseProcessingObj


class PhaseDisplay(BaseProcessingObj):
    def __init__(self, phase=None, doImage=False, window=23, disp_factor=1, title='phase'):
        super().__init__()

        self._phase = phase if phase is not None else None
        self._doImage = doImage
        self._window = window
        self._disp_factor = disp_factor
        self._title = title
        self._opened = False
        self._size_frame = (0, 0)
        self._first = True

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase

    @property
    def doImage(self):
        return self._doImage

    @doImage.setter
    def doImage(self, doImage):
        self._doImage = doImage

    @property
    def disp_factor(self):
        return self._disp_factor

    @disp_factor.setter
    def disp_factor(self, disp_factor):
        self._disp_factor = disp_factor

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, window):
        self._window = window

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        self._title = title

    @property
    def size_frame(self):
        return self._size_frame

    @size_frame.setter
    def size_frame(self, size_frame):
        self._size_frame = size_frame

    def set_w(self, size_frame):
        self.fig = plt.figure(self._window, figsize=(size_frame[0] * self._disp_factor / 100, size_frame[1] * self._disp_factor / 100))
        self.ax = self.fig.add_subplot(111)
#        plt.figure(self._window, figsize=(size_frame[0] * self._disp_factor / 100, size_frame[1] * self._disp_factor / 100))
#        plt.title(self._title)

    def trigger(self, t):
        phase = self._phase
        if phase.generation_time == t:
            frame = cpuArray(phase.phaseInNm * (phase.A > 0).astype(float))
            idx = np.where(cpuArray(phase.A) > 0)[0]
            frame[idx] -= np.mean(frame[idx])

            if self._verbose:
                print('removing average phase in phase_display')

            if np.sum(self._size_frame) == 0:
                size_frame = frame.shape
            else:
                size_frame = self._size_frame

            if not self._opened:
                self.set_w(size_frame)
                self._opened = True
            if self._first:
                self.img = self.ax.imshow(frame)
                self._first = False
            else:
                self.img.set_data(frame)
                self.img.set_clim(frame.min(), frame.max())
            self.fig.canvas.draw()
            plt.pause(0.001)

            # plt.figure(self._window)

            # if self._doImage:
            #     plt.imshow(frame, aspect='auto')
            # else:
            #     plt.imshow(np.repeat(np.repeat(frame, self._disp_factor, axis=0), self._disp_factor, axis=1), cmap='gray')
            # plt.draw()
            # plt.pause(0.01)

    def run_check(self, time_step):
        return self._phase is not None

    @classmethod
    def from_dict(cls, params):
        return cls(**params)
