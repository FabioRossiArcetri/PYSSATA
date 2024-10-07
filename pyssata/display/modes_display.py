import matplotlib.pyplot as plt

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.connections import InputValue
from pyssata.base_value import BaseValue

class ModesDisplay(BaseProcessingObj):
    def __init__(self, disp_factor=1, wsize=(600, 300), window=22, yrange=(-100, 100), oplot=False, color=1, psym=-4, title=''):
        super().__init__(target_device_idx=-1)

        self._modes = None
        self._wsize = wsize
        self._window = window
        self._yrange = yrange
        self._oplot = oplot
        self._color = color
        self._psym = psym
        self._title = title
        self._opened = False
        self._disp_factor = disp_factor
        self._first = True
        self.inputs['modes'] = InputValue(type=BaseValue)

    @property
    def modes(self):
        return self._modes

    @modes.setter
    def modes(self, modes):
        self._modes = modes

    @property
    def wsize(self):
        return self._wsize

    @wsize.setter
    def wsize(self, wsize):
        self._wsize = wsize

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, window):
        self._window = window

    @property
    def yrange(self):
        return self._yrange

    @yrange.setter
    def yrange(self, yrange):
        self._yrange = yrange

    @property
    def oplot(self):
        return self._oplot

    @oplot.setter
    def oplot(self, oplot):
        self._oplot = oplot

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    @property
    def psym(self):
        return self._psym

    @psym.setter
    def psym(self, psym):
        self._psym = psym

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        self._title = title

    def set_w(self):
        plt.figure(self._window, figsize=(self._wsize[0] / 100, self._wsize[1] / 100))
        plt.title(self._title if self._title != '' else 'modes')

    def trigger(self, t):
        m = self.inputs['modes'].get(self._target_device_idx)
        if m.generation_time == t:
            if not self._opened and not self._oplot:
                self.set_w()
                self._opened = True

            plt.figure(self._window)
            if self._first:
                self._line = plt.plot(m.value, '.-')
                plt.title(self._title)
#                plt.ylim(self._yrange)
                self._first = False
            else:
                self._line[0].set_ydata(m.value)
            plt.draw()
            plt.pause(0.01)

    def run_check(self, time_step):
        return self.inputs['modes'].get(self._target_device_idx) is not None

    @classmethod
    def from_dict(cls, params):
        return cls(**params)
