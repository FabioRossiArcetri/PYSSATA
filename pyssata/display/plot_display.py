import numpy as np

from pyssata import xp

import matplotlib.pyplot as plt
plt.ion()

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.connections import InputValue
from pyssata.base_value import BaseValue


class PlotDisplay(BaseProcessingObj):
    def __init__(self, disp_factor=1, histlen=200, wsize=(600, 400), window=23, yrange=(0, 0), oplot=False, color=1, psym=-4, title=''):
        super().__init__()
        
        self._wsize = wsize
        self._window = window
        self.set_histlen(histlen)
        self._yrange = yrange
        self._value = None
        self._oplot = oplot
        self._color = color
        self._psym = psym
        self._title = title
        self._opened = False
        self._first = True
        self._disp_factor = disp_factor
        self.inputs['value'] = InputValue(type=BaseValue)

    @property
    def histlen(self):
        return len(self._history)

    @histlen.setter
    def histlen(self, length):
        self.set_histlen(length)

    @property
    def wsize(self):
        return self._wsize

    @wsize.setter
    def wsize(self, size):
        self._wsize = size

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, num):
        self._window = num

    @property
    def yrange(self):
        return self._yrange

    @yrange.setter
    def yrange(self, rng):
        self._yrange = rng

    @property
    def oplot(self):
        return self._oplot

    @oplot.setter
    def oplot(self, op):
        self._oplot = op

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, col):
        self._color = col

    @property
    def psym(self):
        return self._psym

    @psym.setter
    def psym(self, sym):
        self._psym = sym

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, t):
        self._title = t

    def set_histlen(self, histlen):
        self._history = np.zeros(histlen)
        self._count = 0

    def set_w(self):
        self.fig = plt.figure(self._window, figsize=(self._wsize[0] / 100, self._wsize[1] / 100))
        self.ax = self.fig.add_subplot(111)
#        plt.figure(self._window, figsize=(self._wsize[0] / 100, self._wsize[1] / 100))
#        plt.title(self._title)

    def trigger(self, t):
        value = self.inputs['value'].get(self.target_device_idx)
        
        if value is not None and value.generation_time == t:
            if not self._opened:
                self.set_w()
                self._opened = True

            n = len(self._history)
            if self._count >= n:
                self._history[:-1] = self._history[1:]
                self._count = n - 1

            self._history[self._count] = value.value
            self._count += 1

            x = np.arange(self._count)
            y = self._history[:self._count]
            plt.figure(self._window)
            if self._first:
                self.fig.suptitle(self._title)
                self.line = self.ax.plot(x, y, marker='.')
                self._first = False
            else:
                self.line[0].set_xdata(x)
                self.line[0].set_ydata(y)
                self.ax.set_xlim(x.min(), x.max())
                if np.sum(np.abs(self._yrange)) > 0:
                    self.ax.set_ylim(self._yrange[0], self._yrange[1])
                else:
                    self.ax.set_ylim(y.min(), y.max())
            self.fig.canvas.draw()
            plt.pause(0.001)

            # if self._oplot:
            #     plt.plot(xp.arange(self._count), self._history[:self._count], marker='.', color=self._color)
            # else:
            #     plt.plot(xp.arange(self._count), self._history[:self._count], marker='.')
            #     plt.ylim(self._yrange)
            #     plt.title(self._title)
            # plt.draw()
            # plt.pause(0.01)

    def run_check(self, time_step, errmsg=None):
        value = self.inputs['value'].get(self.target_device_idx)
        return value is not None and self._history is not None

