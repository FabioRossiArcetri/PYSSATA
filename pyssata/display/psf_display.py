import numpy as np
import matplotlib.pyplot as plt

from pyssata.base_processing_obj import BaseProcessingObj


class PSFDisplay(BaseProcessingObj):
    def __init__(self, psf=None, wsize=[600, 600], window=23):
        super().__init__()
        self._psf = psf
        self._wsize = wsize
        self._window = window
        self._log = False
        self._image_p2v = 0.0
        self._title = 'PSF'
        self._opened = False
        self._first = True

    @property
    def psf(self):
        return self._psf

    @psf.setter
    def psf(self, psf):
        self._psf = psf

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
    def log(self):
        return self._log

    @log.setter
    def log(self, log):
        self._log = log

    @property
    def image_p2v(self):
        return self._image_p2v

    @image_p2v.setter
    def image_p2v(self, image_p2v):
        self._image_p2v = image_p2v

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        self._title = title

    def set_w(self):
#        plt.figure(self._window, figsize=(self._wsize[0] / 100, self._wsize[1] / 100))
#        plt.title(self._title)
        self.fig = plt.figure(self._window, figsize=(self._wsize[0] / 100, self._wsize[1] / 100))
        self.ax = self.fig.add_subplot(111)

    def trigger(self, t):
        psf = self._psf
        if psf.generation_time == t:

            image = psf.value

            if self._image_p2v > 0:
                image = np.maximum(image, self._image_p2v**(-1.) * np.max(image))
            
            if self._log:
                image = np.log10(image)

            if not self._opened:
                self.set_w()
                self._opened = True
            if self._first:
                self.img = self.ax.imshow(image)
                self._first = False
            else:
                self.img.set_data(image)
                self.img.set_clim(image.min(), image.max())
#            plt.colorbar()
            self.fig.canvas.draw()
            plt.pause(0.001)


    def run_check(self, time_step):
        return self._psf is not None

    def cleanup(self):
        plt.close(self._window)

    @classmethod
    def from_dict(cls, params):
        return cls(**params)
