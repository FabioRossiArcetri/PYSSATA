
import numpy as np

from pyssata.data_objects.slopes import Slopes
from pyssata.data_objects.subap_data import SubapData
from pyssata.base_value import BaseValue
from pyssata.lib import make_mask

from pyssata.processing_objects.slopec import Slopec

    
class ShSlopec(Slopec):
    def __init__(self,
                 thr_value: float = -1,
                 exp_weight: float = 1.0,
                 subapdata: SubapData = None,
                 corr_template = None,                
                 target_device_idx: int = None, 
                 precision: int = None ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self._thr_value = thr_value
        self._thr_mask_cube = BaseValue()  
        self._total_counts = BaseValue()
        self._subap_counts = BaseValue()
        self._detailed_debug = -1
        self._subap_idx = None
        self._xweights = None
        self._yweights = None
        self._xcweights = None
        self._ycweights = None
        self._mask_weighted = None
        self._corr_template = corr_template
        self._winMatWindowed = None
        self._vecWeiPixRadT = None
        self._weightedPixRad = 0.0
        self._windowing = False
        self._correlation = False
        self._corrWindowSidePix = 0
        self._thr_ratio_value = 0.0
        self._thr_pedestal = False
        self._mult_factor = 0.0
        self._quadcell_mode = False
        self._two_steps_cog = False
        self._cog_2ndstep_size = 0
        self._dotemplate = False
        self._store_thr_mask_cube = False
        self._window = 0
        self._wsize = self.xp.zeros(2, dtype=int)
        self._display2s = False

        # Property settings
        self.subapdata = subapdata
        self.exp_weight = exp_weight

    @property
    def thr_value(self):
        return self._thr_value

    @thr_value.setter
    def thr_value(self, value):
        self._thr_value = value

    @property
    def exp_weight(self):
        return self._exp_weight

    @exp_weight.setter
    def exp_weight(self, value):
        self._exp_weight = value
        self.set_xy_weights()

    def set_xy_weights(self):
        if self._subapdata:
            out = self.computeXYweights(self._subapdata.np_sub, self._exp_weight, self._weightedPixRad, 
                                          self._quadcell_mode, self._windowing)
            self._mask_weighted = self.xp.array(out['mask_weighted'], copy=False)
            self._xweights = self.xp.array(out['x'], copy=False)
            self._yweights = self.xp.array(out['y'], copy=False)
            self._xcweights = self.xp.array(out['xc'], copy=False)
            self._ycweights = self.xp.array(out['yc'], copy=False)

    def computeXYweights(self, np_sub, exp_weight, weightedPixRad, quadcell_mode=False, windowing=False):
        """
        Compute XY weights for SH slope computation.

        Parameters:
        np_sub (int): Number of subapertures.
        exp_weight (float): Exponential weight factor.
        weightedPixRad (float): Radius for weighted pixels.
        quadcell_mode (bool): Whether to use quadcell mode.
        windowing (bool): Whether to apply windowing.
        """
        # Generate x, y coordinates
        x, y = np.meshgrid(np.linspace(-1, 1, np_sub), np.linspace(-1, 1, np_sub))
        
        # Compute weights in quadcell mode or otherwise
        if quadcell_mode:
            x = np.where(x > 0, 1.0, -1.0)
            y = np.where(y > 0, 1.0, -1.0)
            xc, yc = np.copy(x), np.copy(y)
        else:
            xc, yc = np.copy(x), np.copy(y)
            # Apply exponential weights if exp_weight is not 1
            x = np.where(x > 0, np.power(x, exp_weight), -np.power(np.abs(x), exp_weight))
            y = np.where(y > 0, np.power(y, exp_weight), -np.power(np.abs(y), exp_weight))

        # Adjust xc, yc for centroid calculations in two steps
        xc = np.where(x > 0, np.copy(xc), -np.abs(xc))
        yc = np.where(y > 0, np.copy(yc), -np.abs(yc))

        # Apply windowing or weighted pixel mask
        if weightedPixRad != 0:
            if windowing:
                # Windowing case (must be an integer)
                mask_weighted = make_mask(np_sub, diaratio=(2.0 * weightedPixRad / np_sub), xp=self.xp)
            else:
                # Weighted Center of Gravity (WCoG)
                mask_weighted = self.psf_gaussian(np_sub, 2, [weightedPixRad, weightedPixRad])
                mask_weighted /= self.xp.max(mask_weighted)

            mask_weighted[mask_weighted < 1e-6] = 0.0

            x *= mask_weighted.astype(float)
            y *= mask_weighted.astype(float)
        else:
            mask_weighted = self.xp.ones((np_sub, np_sub), dtype=float)

        return {"x": x, "y": y, "xc": xc, "yc": yc, "mask_weighted": mask_weighted}

    
    @property
    def subapdata(self):
        return self._subapdata
    
    @subapdata.setter
    def subapdata(self, p):
        self._subapdata = p
        self._slopes = Slopes(self._subapdata.n_subap * 2)
        self._accumulated_slopes = Slopes(self._subapdata.n_subap * 2)

        n_subaps = self._subapdata.n_subaps
        subap_idx = self.xp.array([self._subapdata.subap_idx(i) for i in range(n_subaps)])
        self._subap_idx = self.xp.copy(subap_idx)

        self.set_xy_weights()

    def calc_slopes(self, t, accumulated=False):
        if self._vecWeiPixRadT is not None:
            time = self.t_to_seconds(t)
            idxW = self.xp.where(time > self._vecWeiPixRadT[:, 1])[-1]
            if len(idxW) > 0:
                self._weightedPixRad = self._vecWeiPixRadT[idxW, 0]
                if self._verbose:
                    print(f'self._weightedPixRad: {self._weightedPixRad}')
                self.set_xy_weights()

        if self._dotemplate or self._correlation or self._two_steps_cog or self._detailed_debug > 0:
            self.calc_slopes_for(t, accumulated=accumulated)
        else:
            self.calc_slopes_nofor(t, accumulated=accumulated)


    def calc_slopes_for(self, t, accumulated=False):
        """
        Calculate slopes using a for loop over subapertures.

        Parameters:
        t (float): The time for which to calculate the slopes.
        accumulated (bool): If True, use accumulated pixels for slope calculation.
        """
        if self._verbose and self._subapdata is None:
            print('subapdata is not valid.')
            return

        n_subaps = self._subapdata.n_subaps
        np_sub = self._subapdata.np_sub
        pixels = self._accumulatedPixels.pixels if accumulated else self._pixels.pixels

        sx = self.xp.zeros(n_subaps, dtype=float)
        sy = self.xp.zeros(n_subaps, dtype=float)

        if self._store_thr_mask_cube:
            thr_mask_cube = self.xp.zeros((np_sub, np_sub, n_subaps), dtype=int)
            thr_mask = self.xp.zeros((np_sub, np_sub), dtype=int)

        flux_per_subaperture = self.xp.zeros(n_subaps, dtype=float)
        max_flux_per_subaperture = self.xp.zeros(n_subaps, dtype=float)

        if self._dotemplate:
            corr_template = self.xp.zeros((np_sub, np_sub, n_subaps), dtype=float)
        elif self._corr_template is not None:
            corr_template = self._corr_template

        if self._thr_value > 0 and self._thr_ratio_value > 0:
            raise ValueError('Only one between _thr_value and _thr_ratio_value can be set.')

        if self._weightFromAccumulated:
            n_weight_applied = 0

        for i in range(n_subaps):
            idx = self._subap_idx[i, :]
            subap = pixels[idx].reshape(np_sub, np_sub)

            if self._weightFromAccumulated and self._accumulatedPixelsPtr is not None and t >= self._accumulationDt:
                accumulated_pixels_weight = self._accumulatedPixelsPtr[idx].reshape(np_sub, np_sub)
                accumulated_pixels_weight -= self.xp.min(accumulated_pixels_weight)
                max_temp = self.xp.max(accumulated_pixels_weight)
                if max_temp > 0:
                    if self._weightFromAccWithWindow:
                        window_threshold = 0.05
                        over_threshold = self.xp.where(
                            (accumulated_pixels_weight >= max_temp * window_threshold) | 
                            (self.xp.rot90(accumulated_pixels_weight, 2) >= max_temp * window_threshold)
                        )
                        if len(over_threshold[0]) > 0:
                            accumulated_pixels_weight.fill(0)
                            accumulated_pixels_weight[over_threshold] = 1.0
                        else:
                            accumulated_pixels_weight.fill(1.0)
                    else:
                        accumulated_pixels_weight *= 1.0 / max_temp

                    subap *= accumulated_pixels_weight
                    n_weight_applied += 1

            if self._winMatWindowed is not None:
                if i == 0 and self._verbose:
                    print("self._winMatWindowed applied")
                subap *= self._winMatWindowed[:, :, i]

            if self._dotemplate:
                corr_template[:, :, i] = subap

            flux_per_subaperture[i] = self.xp.sum(subap)
            max_flux_per_subaperture[i] = self.xp.max(subap)

            thr = 0
            if self._thr_value > 0:
                thr = self._thr_value
            if self._thr_ratio_value > 0:
                thr = self._thr_ratio_value * self.xp.max(subap)

            if self._thr_pedestal:
                thr_idx = self.xp.where(subap < thr)
            else:
                subap -= thr
                thr_idx = self.xp.where(subap < 0)

            if len(thr_idx[0]) > 0:
                subap[thr_idx] = 0

            if self._store_thr_mask_cube:
                thr_mask.fill(0)
                if len(thr_idx[0]) > 0:
                    thr_mask[thr_idx] = 1
                thr_mask_cube[:, :, i] = thr_mask

            if self._correlation:
                if self._corrWindowSidePix > 0:
                    subap = self.xp.convolve(
                        subap[np_sub // 2 - self._corrWindowSidePix // 2: np_sub // 2 + self._corrWindowSidePix // 2],
                        corr_template[np_sub // 2 - self._corrWindowSidePix // 2: np_sub // 2 + self._corrWindowSidePix // 2, i],
                        mode='same'
                    )
                else:
                    subap = self.xp.convolve(subap, corr_template[:, :, i], mode='same')
                thr_idx = self.xp.where(subap < 0)
                if len(thr_idx[0]) > 0:
                    subap[thr_idx] = 0

            # CoG in two steps logic (simplified here)
            if self._two_steps_cog:
                pass  # Further logic for two-step centroid calculation can go here.

            subap_total = self.xp.sum(subap)
            factor = 1.0 / subap_total if subap_total > 0 else 0

            sx[i] = self.xp.sum(subap * self._xweights) * factor
            sy[i] = self.xp.sum(subap * self._yweights) * factor

        if self._weightFromAccumulated:
            print(f"Weights mask has been applied to {n_weight_applied} sub-apertures")

        if self._dotemplate:
            self._corr_template = corr_template

        if self._mult_factor != 0:
            sx *= self._mult_factor
            sy *= self._mult_factor
            print("WARNING: multiplication factor in the slope computer!")

        if accumulated:
            self._accumulatedSlopes.xslopes = sx
            self._accumulatedSlopes.yslopes = sy
            self._accumulatedSlopes.generation_time = t
        else:
            if self._store_thr_mask_cube:
                self._thr_mask_cube.value = thr_mask_cube
                self._thr_mask_cube.generation_time = t

            self._slopes.xslopes = sx
            self._slopes.yslopes = sy
            self._slopes.generation_time = t

            self._fluxPerSubapertureVector.value = flux_per_subaperture
            self._fluxPerSubapertureVector.generation_time = t
            self._maxFluxPerSubapertureVector.value = max_flux_per_subaperture
            self._total_counts.value = self.xp.sum(self._fluxPerSubapertureVector.value)
            self._total_counts.generation_time = t
            self._subap_counts.value = self.xp.mean(self._fluxPerSubapertureVector.value)
            self._subap_counts.generation_time = t

        if self._verbose:
            print(f"Slopes min, max and rms : {self.xp.min(sx)}, {self.xp.max(sx)}, {self.xp.sqrt(self.xp.mean(sx ** 2))}")

    
    def calc_slopes_nofor(self, t, accumulated=False):
        """
        Calculate slopes without a for-loop over subapertures.
        
        Parameters:
        t (float): The time for which to calculate the slopes.
        accumulated (bool): If True, use accumulated pixels for slope calculation.
        """
        if self._verbose and self._subapdata is None:
            print('subapdata is not valid.')
            return

        n_subaps = self._subapdata.n_subaps
        np_sub = self._subapdata.np_sub
        pixels = self._accumulatedPixels.pixels if accumulated else self._pixels.pixels

        if self._store_thr_mask_cube:
            thr_mask_cube = self.xp.zeros((np_sub, np_sub, n_subaps), dtype=int)

        flux_per_subaperture_vector = self.xp.zeros(n_subaps, dtype=float)
        max_flux_per_subaperture_vector = self.xp.zeros(n_subaps, dtype=float)

        if self._thr_value > 0 and self._thr_ratio_value > 0:
            raise ValueError("Only one between _thr_value and _thr_ratio_value can be set.")

        # Reform pixels based on the subaperture index
        pixels = pixels[self._subap_idx].T

        if self._weightFromAccumulated:
            n_weight_applied = 0
            if self._accumulatedPixelsPtr is not None and t >= self._accumulationDt:
                accumulated_pixels_weight = self._accumulatedPixelsPtr[self._subap_idx].T
                accumulated_pixels_weight -= self.xp.min(accumulated_pixels_weight, axis=1, keepdims=True)
                max_temp = self.xp.max(accumulated_pixels_weight, axis=1)
                idx0 = self.xp.where(max_temp <= 0)[0]
                if len(idx0) > 0:
                    accumulated_pixels_weight[:, idx0] = 1.0

                if self._weightFromAccWithWindow:
                    window_threshold = 0.05
                    one_over_max_temp = 1.0 / max_temp[:, self.xp.newaxis]
                    accumulated_pixels_weight *= one_over_max_temp
                    over_threshold = self.xp.where(
                        (accumulated_pixels_weight >= window_threshold) | 
                        (accumulated_pixels_weight[:, ::-1] >= window_threshold)
                    )
                    if len(over_threshold[0]) > 0:
                        accumulated_pixels_weight.fill(0)
                        accumulated_pixels_weight[over_threshold] = 1.0
                    else:
                        accumulated_pixels_weight.fill(1.0)
                    n_weight_applied += self.xp.sum(self.xp.any(accumulated_pixels_weight > 0, axis=1))

                pixels *= accumulated_pixels_weight

        # Calculate flux and max flux per subaperture
        flux_per_subaperture_vector = self.xp.sum(pixels, axis=1)
        max_flux_per_subaperture_vector = self.xp.max(pixels, axis=1)

        if self._winMatWindowed is not None:
            if self._verbose:
                print("self._winMatWindowed applied")
            pixels *= self._winMatWindowed.reshape(np_sub * np_sub, n_subaps)

        # Thresholding logic
        if self._thr_ratio_value > 0:
            thr = self._thr_ratio_value * max_flux_per_subaperture_vector
        elif self._thr_pedestal or self._thr_value > 0:
            thr = self._thr_value
        else:
            thr = 0

        if self.xp.size(thr) == 1:
            thr = self.xp.tile(thr, (np_sub * np_sub, n_subaps))
        else:
            thr = thr[:, self.xp.newaxis] * self.xp.ones((1, np_sub * np_sub))

        if self._thr_pedestal:
            thr_idx = self.xp.where(pixels < thr)
        else:
            pixels -= thr
            thr_idx = self.xp.where(pixels < 0)

        if len(thr_idx[0]) > 0:
            pixels[thr_idx] = 0

        if self._store_thr_mask_cube:
            thr_mask_cube = thr.reshape(np_sub, np_sub, n_subaps)

        # Compute denominator for slopes
        subap_tot = self.xp.sum(pixels * self._mask_weighted.reshape(np_sub * np_sub, 1) * self.xp.ones((1, n_subaps)), axis=1)
        mean_subap_tot = self.xp.mean(subap_tot)
        idx_le_0 = self.xp.where(subap_tot <= mean_subap_tot * 1e-3)[0]
        if len(idx_le_0) > 0:
            subap_tot[idx_le_0] = mean_subap_tot
        factor = 1.0 / subap_tot
        if len(idx_le_0) > 0:
            factor[idx_le_0] = 0.0

        factor = factor[:, self.xp.newaxis] * self.xp.ones((1, np_sub * np_sub))

        # Compute slopes
        sx = self.xp.sum(pixels * self._xweights.reshape(np_sub * np_sub, 1) * self.xp.ones((1, n_subaps)) * factor, axis=1)
        sy = self.xp.sum(pixels * self._yweights.reshape(np_sub * np_sub, 1) * self.xp.ones((1, n_subaps)) * factor, axis=1)

        if self._weightFromAccumulated:
            print(f"Weights mask has been applied to {n_weight_applied} sub-apertures")

        if self._mult_factor != 0:
            sx *= self._mult_factor
            sy *= self._mult_factor
            print("WARNING: multiplication factor in the slope computer!")

        if accumulated:
            self._accumlatedSlopes.xslopes = sx
            self._accumlatedSlopes.yslopes = sy
            self._accumlatedSlopes.generation_time = t
        else:
            if self._store_thr_mask_cube:
                self._thr_mask_cube.value = thr_mask_cube
                self._thr_mask_cube.generation_time = t

            self._slopes.xslopes = sx
            self._slopes.yslopes = sy
            self._slopes.generation_time = t

            self._fluxPerSubapertureVector.value = flux_per_subaperture_vector
            self._fluxPerSubapertureVector.generation_time = t
            self._maxFluxPerSubapertureVector.value = max_flux_per_subaperture_vector
            self._total_counts.value = self.xp.sum(self._fluxPerSubapertureVector.value)
            self._total_counts.generation_time = t
            self._subap_counts.value = self.xp.mean(self._fluxPerSubapertureVector.value)
            self._subap_counts.generation_time = t

        if self._verbose:
            print(f"Slopes min, max and rms : {self.xp.min(sx)}, {self.xp.max(sx)}, {self.xp.sqrt(self.xp.mean(sx ** 2))}")

    def psf_gaussian(self, np_sub, fwhm):
        x = self.xp.linspace(-1, 1, np_sub)
        y = self.xp.linspace(-1, 1, np_sub)
        x, y = self.xp.meshgrid(x, y)
        gaussian = self.xp.exp(-4 * self.xp.log(2) * (x ** 2 + y ** 2) / fwhm[0] ** 2)
        return gaussian
