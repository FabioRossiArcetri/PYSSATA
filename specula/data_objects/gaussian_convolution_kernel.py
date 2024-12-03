from specula.base_data_obj import BaseDataObj
from specula.data_objects.convolution_kernel import ConvolutionKernel, lgs_map_sh

import numpy as np

from astropy.io import fits

class GaussianConvolutionKernel(ConvolutionKernel):
    """
    Kernel processing object for Gaussian kernels.
    """
    
    def __init__(self, convolGaussSpotSize, dimx, dimy, target_device_idx=None, precision=None):
        super().__init__(dimx, dimy, target_device_idx=target_device_idx, precision=precision)        
        self.spotsize = convolGaussSpotSize

    def build(self):
        """
        Recalculates the Gaussian kernel based on current settings.
        """
        self.orig_dimx = self.dimx
        self.dimx = max(self.dimx, 2)        
        self.pupil_size_m = 2 * self.pixel_pitch
        self.lgs_tt = [-0.5, -0.5] if not self.positiveShiftTT else [0.5, 0.5]
        self.lgs_tt = [x * self.pxscale for x in self.lgs_tt]        
        self.hash_arr = [
            self.dimx, self.pupil_size_m, 90e3, self.spotsize,
            self.pxscale, self.dimension, 3, self.lgs_tt, [0, 0, 0], [90e3], [1.0]
        ]
        return 'ConvolutionKernel' + self.generate_hash()        
        # use calib manager to determine the file path?
        # if file exist ...
        # self = ConvolutionKernel.restore(filename)
        # else: ...

    def calculate_lgs_map(self):
        self.kernels = lgs_map_sh(
            self.dimx, self.pupil_size_m, 0, 90e3, [0], profz=[1.0], fwhmb=self.spotsize, ps=self.pxscale,
            ssp=self.dimension, overs=1, theta=self.lgs_tt, xp=self.xp )
        if self.dimx != self.orig_dimx:
            self.kernels = self.kernels[:, :, 0]
        for i in range(self.dimx):
            for j in range(self.dimy):
                subap_kern = self.xp.array(self.kernels[:, :, j * self.dimx + i])
                subap_kern /= self.xp.sum(subap_kern)
                subap_kern_fft = self.xp.fft.ifftshift(self.xp.fft.ifft2(subap_kern))
                self.kernels[j * self.dimx + i] = subap_kern_fft
        

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])

        if version > 2:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        c = GaussianConvolutionKernel(target_device_idx=target_device_idx)
        if version >= 2:
            c.pixel_pitch = hdr['PIXEL_PITCH']
            c.pxscale = hdr['PXSCALE']
            c.dimension = hdr['DIMENSION']
            c.oversampling = hdr['OVERSAMPLING']
            c.positiveShiftTT = hdr['POSITIVESHIFTTT']
            c.spotsize = hdr['SPOTSIZE']
            c.dimx = hdr['DIMX']
            c.dimy = hdr['DIMY']            

        c.read(filename, hdr)
        return c