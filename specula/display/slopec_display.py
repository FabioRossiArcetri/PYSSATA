from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.pixels import Pixels
from specula.data_objects.pupdata import PupData
from specula.data_objects.subap_data import SubapData
from specula.data_objects.slopes import Slopes
from specula.processing_objects.pyr_slopec import PyrSlopec
from specula.processing_objects.sh_slopec import ShSlopec
# from specula.display.pupil_display import pupil_display

from specula import cpuArray
from specula.connections import InputValue

import numpy as np

import matplotlib.pyplot as plt
from scipy.ndimage import zoom


class SlopecDisplay(BaseProcessingObj):
    def __init__(self, window=27, disp_factor=1):
        super().__init__(target_device_idx=-1)

        self._disp_factor = disp_factor
        self._window = window
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

        title = self._title if self._title else 'Slope Display'
        _, _, _ = self.pupil_display(pixels, sx, sy, map_data, pixels.shape[0], title=title, do_image_show=True)

        self.fig.canvas.draw()
        plt.pause(0.001)


    def run_check(self, time_step):
        # TODO
        #slopec = self.inputs['slopec'].get(self.target_device_idx)
        #return slopec is not None
        return True

    def remap_signals(self, sx_or_sy, slopemap, slope_side_x, slope_side_y):
        """
        Remap the signals based on the slopemap.

        Parameters:
            sx_or_sy (numpy.ndarray): Signal vector (sx or sy).
            slopemap (numpy.ndarray): Slopes remapping vector.
            slope_side_x (int): Width of the remapped signal.
            slope_side_y (int): Height of the remapped signal.

        Returns:
            numpy.ndarray: Remapped signal.
        """
        # Reshape and remap the signal based on the slopemap.
        remapped_signal = np.zeros((slope_side_x, slope_side_y))
        remapped_signal.flat[slopemap.ravel()] = sx_or_sy
        return remapped_signal


    def pupil_display(self, frame, sx, sy, slopemap, real_ccd_side, cirlceDisp=None, NEGATIVE=False,
                    NOFRAME=False, GET_SIZE=False, NOSHOW=False,
                    MAGNIFY=1, slope_side=None, title='pupil_display', do_image_show=False, plot_title=''):
        """
        Display a CCD frame and the corresponding signals.

        Parameters:
            frame (numpy.ndarray): CCD frame containing the four sub-pupils.
            sx (numpy.ndarray): x-signal vector.
            sy (numpy.ndarray): y-signal vector.
            slopemap (numpy.ndarray): Slopes remapping vector.
            real_ccd_side (int): CCD frame side (assuming square dimensions).
            cirlceDisp (int): Number of sub-apertures for circle display.
            NEGATIVE (bool): Extend display range to the frame's minimum value.
            NOFRAME (bool): Show only signals without the frame.
            GET_SIZE (bool): Return the display window size.
            NOSHOW (bool): If True, do not show anything.
            MAGNIFY (int): Magnification factor for the display.
            slope_side (int): Side length for remapping signals.
            title (str): Title for the display window.
            do_image_show (bool): If True, use image_show display method.
            plot_title (str): Title for the plot.

        Returns:
            frame_signalx, frame_signaly, GET_SIZE: Remapped signals and window size (if requested).
        """
        
        # Initialize or calculate some parameters
        if slope_side is None:
            slope_side = real_ccd_side
        
        dx = 0.5 if NOFRAME else 1.5
        GET_SIZE = [int(dx * real_ccd_side * MAGNIFY), int(real_ccd_side * MAGNIFY)]
        
        # Exit if NOSHOW is True
        if NOSHOW:
            return
        
        # Prepare the display size
        if do_image_show:
            if NOFRAME:
                xsize, ysize = 640 * MAGNIFY, 480 * MAGNIFY
            else:
                xsize, ysize = 800 * MAGNIFY, 400 * MAGNIFY
        else:
            xsize, ysize = dx * real_ccd_side * MAGNIFY, real_ccd_side * MAGNIFY
        
        if self.fig is None:
            self.fig = plt.figure(self._window, figsize=(xsize/100, ysize/100))
            self.ax = self.fig.add_subplot(111)
        
        # Display the frame and signals
        if not NOFRAME:
            if do_image_show:
                self.ax.imshow(frame, cmap='gray')
                self.ax.set_title(plot_title)
            else:
                if NEGATIVE:
                    display_frame = zoom(np.clip(frame, np.min(frame), np.max(frame)), MAGNIFY, order=0)
                else:
                    display_frame = zoom(np.clip(frame, 0, np.max(frame)), MAGNIFY, order=0)
                
                self.ax.imshow(display_frame, cmap='gray')
                self.ax.set_title(title)
                
                # Circle display if required
                if cirlceDisp:
                    for i in range(cirlceDisp):
                        for j in range(cirlceDisp):
                            circ = plt.Circle(((1 + 2 * i) / (2 * cirlceDisp) * real_ccd_side * MAGNIFY, 
                                            (1 + 2 * j) / (2 * cirlceDisp) * real_ccd_side * MAGNIFY), 
                                            10, color='red', fill=False)
                            ax.add_patch(circ)
        
        # Remap the signals
        if sx.size == slopemap.size:
            frame_signalx = self.remap_signals(sx, slopemap, slope_side, slope_side)
            frame_signaly = self.remap_signals(sy, slopemap, slope_side, slope_side)
        else:
            frame_signalx = self.remap_signals(np.zeros_like(slopemap), slopemap, slope_side, slope_side)
            frame_signaly = self.remap_signals(np.zeros_like(slopemap), slopemap, slope_side, slope_side)
        
        # Combine and display the signal
        dx = frame_signalx.shape[0]
        dy = frame_signalx.shape[1]
        signal_total = np.zeros((dx//2, dy))
        signal_total[:, :dy//2] = frame_signalx[:dx//2, :dy//2]
        signal_total[:, dy//2:] = frame_signaly[:dx//2, :dy//2]
        
        if do_image_show:
            if NOFRAME or signal_total.shape[1] != frame.shape[1]:
                self.ax.imshow(signal_total, cmap='gray')
            else:
                combined = np.vstack((frame/np.max(frame) * np.max(signal_total), signal_total))
                self.ax.imshow(combined, cmap='gray')
        else:
            signal_display = zoom(signal_total, [MAGNIFY/2, MAGNIFY], order=0)
            self.ax.imshow(signal_display, cmap='gray', vmin=-np.max(np.abs(signal_total)), vmax=np.max(np.abs(signal_total)))
        
        return frame_signalx, frame_signaly, GET_SIZE if GET_SIZE else None
