import numpy as np
import cupy as cp # TODO fuse
from pyssata.lib.toccd import toccd
from pyssata.lib.make_mask import make_mask
from pyssata.connections import InputValue
from pyssata.data_objects.ef import ElectricField
from pyssata.data_objects.intensity import Intensity
from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.data_objects.lenslet import Lenslet
from pyssata.data_objects.subap_data import SubapData


# TODO
if 0:
        convolGaussSpotSize = self.extract(params, 'convolGaussSpotSize', default=None)        
        
        if convolGaussSpotSize is not None:
            kernelobj = KernelGauss()
            kernelobj.spotsize = convolGaussSpotSize
            kernelobj.lenslet = sh.lenslet
            kernelobj.cm = self._cm
            sh.kernelobj = kernelobj
        

@cp.fuse(kernel_name='abs2')
def abs2(u_fp):
     psf = cp.real(u_fp * cp.conj(u_fp))
     return psf

class SH(BaseProcessingObj):
    def __init__(self,
                 wavelengthInNm: float,
                 subap_wanted_fov: float,
                 sensor_pxscale: float,
                 subap_on_diameter: int,
                 subap_npx: int,
                 FoVres30mas: bool = False,
                 squaremask: bool = True,
                 gauss_kern = None, 
                 fov_ovs_coeff: float = 0,
                 xShiftPhInPixel: float = 0,
                 yShiftPhInPixel: float = 0,
                 aXShiftPhInPixel: float = 0,
                 aYShiftPhInPixel: float = 0,
                 rotAnglePhInDeg: float = 0,
                 aRotAnglePhInDeg: float = 0,
                 do_not_double_fov_ovs: bool = False,
                 set_fov_res_to_turbpxsc: bool = False,
                 target_device_idx: int = None, 
                 precision: int = None,
        ):

        super().__init__(target_device_idx=target_device_idx, precision=precision)        

        self._kernelobj = None   # TODO
        rad2arcsec = 180 / self.xp.pi * 3600
        self._wavelengthInNm = wavelengthInNm
        self._lenslet = Lenslet(subap_on_diameter)
        self._subap_wanted_fov = subap_wanted_fov / rad2arcsec
        self._sensor_pxscale = sensor_pxscale / rad2arcsec
        self._subap_npx = subap_npx
        if gauss_kern is not None:
            self._gkern = True
            self._gkern_size = gauss_kern / rad2arcsec
        else:
            self._gkern = False
            self._gkern_size = 0

        self._kernel_fov_scale = 1.0
        self._fov_ovs_coeff = fov_ovs_coeff
        self._squaremask = squaremask
        self._fov_resolution_arcsec = 0.03 if FoVres30mas else 0
        self._kernel_application = ''
        self._kernel_precalc_fft = False
        self._debugOutput = False
        self._noprints = False
        self._subap_idx = None
        self._idx_valid = False
        self._scale_ovs = 1.0
        self._floatShifts = False
        self._rotAnglePhInDeg = rotAnglePhInDeg
        self._aRotAnglePhInDeg = aRotAnglePhInDeg  # TODO if intended to change, set _trigger_geometry_calculated=False to force recalculation
        self._xShiftPhInPixel = xShiftPhInPixel
        self._yShiftPhInPixel = yShiftPhInPixel
        self._aXShiftPhInPixel = aXShiftPhInPixel  # Same TODO as above
        self._aYShiftPhInPixel = aYShiftPhInPixel
        self._fov_ovs = 1
        self._set_fov_res_to_turbpxsc = set_fov_res_to_turbpxsc
        self._do_not_double_fov_ovs = do_not_double_fov_ovs
        self._np_sub = 0
        self._fft_size = 0
        self._input_ef_set = False
        self._trigger_geometry_calculated = False
        self._extrapol_mat = None
        
        self._ccd_side = self._subap_npx * self._lenslet.n_lenses
        self._out_i = Intensity(self._ccd_side, self._ccd_side, precision=self.precision, target_device_idx=self._target_device_idx)

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_i'] = self._out_i

    @property
    def wavelengthInNm(self):
        return self._wavelengthInNm

    @wavelengthInNm.setter
    def wavelengthInNm(self, value):
        self._wavelengthInNm = value

    @property
    def sensor_pxscale(self):
        return self._sensor_pxscale

    @sensor_pxscale.setter
    def sensor_pxscale(self, value):
        self._sensor_pxscale = value

    @property
    def fov_resolution_arcsec(self):
        return self._fov_resolution_arcsec

    @fov_resolution_arcsec.setter
    def fov_resolution_arcsec(self, value):
        self._fov_resolution_arcsec = value

    @property
    def kernel_application(self):
        return self._kernel_application

    @kernel_application.setter
    def kernel_application(self, str_val):
        if str_val not in ['FFT', 'FOV', 'SUBAP']:
            raise ValueError("Kernel application string must be one of 'FFT', 'FOV', or 'SUBAP'")
        self._kernel_application = str_val

    @property
    def lenslet(self):
        return self._lenslet
    
    def set_in_ef(self, in_ef, noprints=False):
        rad2arcsec = 180 / np.pi * 3600
        arcsec2rad = 1.0 / rad2arcsec

        lens = self._lenslet.get(0, 0)
        n_lenses = self._lenslet.n_lenses
        ef_size = in_ef.size[0]

        self._np_sub = max([1, round((ef_size * lens[2]) / 2.0)])
        if self._np_sub * n_lenses > ef_size:
            self._np_sub -= 1

        np_sub = (ef_size * lens[2]) / 2.0

        sensor_pxscale_arcsec = self._sensor_pxscale * rad2arcsec
        dSubApInM = np_sub * in_ef.pixel_pitch
        turbulence_pxscale = self._wavelengthInNm * 1e-9 / dSubApInM * rad2arcsec
        subap_wanted_fov_arcsec = self._subap_wanted_fov * rad2arcsec
        subap_real_fov_arcsec = self._sensor_pxscale * self._subap_npx * rad2arcsec

        if self._fov_resolution_arcsec == 0:
            if not self._noprints:
                print('FoV internal resolution parameter not set.')
            if self._set_fov_res_to_turbpxsc:
                if turbulence_pxscale >= sensor_pxscale_arcsec:
                    raise ValueError('set_fov_res_to_turbpxsc property should be set to one only if turb. pix. sc. is < sensor pix. sc.')
                self._fov_resolution_arcsec = turbulence_pxscale
                if not self._noprints:
                    print('WARNING: set_fov_res_to_turbpxsc property is set.')
                    print('FoV internal resolution parameter will be set to turb. pix. sc.')
            elif turbulence_pxscale < sensor_pxscale_arcsec and sensor_pxscale_arcsec / 2.0 > 0.5:
                self._fov_resolution_arcsec = turbulence_pxscale * 0.5
            else:
                i = 0
                resTry = turbulence_pxscale / (i + 2)
                while resTry >= sensor_pxscale_arcsec:
                    i += 1
                    resTry = turbulence_pxscale / (i + 2)
                iMin = i

                nTry = 10
                resTry = np.zeros(nTry)
                scaleTry = np.zeros(nTry)
                fftScaleTry = np.zeros(nTry)
                subapRealTry = np.zeros(nTry)
                mcmxTry = np.zeros(nTry)

                for i in range(nTry):
                    resTry[i] = turbulence_pxscale / (iMin + i + 2)
                    scaleTry[i] = round(turbulence_pxscale / resTry[i])
                    fftScaleTry[i] = self._wavelengthInNm / 1e9 * self._lenslet.dimx / (ef_size * in_ef.pixel_pitch * scaleTry[i]) * rad2arcsec
                    subapRealTry[i] = round(subap_wanted_fov_arcsec / fftScaleTry[i] / 2.0) * 2
                    mcmxTry[i] = np.lcm(int(self._subap_npx), int(subapRealTry[i]))

                # Search for resolution factor with FoV error < 1%
                FoV = subapRealTry * fftScaleTry
                FoVerror = np.abs(FoV - subap_wanted_fov_arcsec) / subap_wanted_fov_arcsec
                idxGood = np.where(FoVerror < 0.02)[0]

                # If no resolution factor gives low error, consider all scale values
                if len(idxGood) == 0:
                    idxGood = np.arange(nTry)

                # Search for index with minimum ratio between M.C.M. and resolution factor
                ratioMcm = mcmxTry[idxGood] / scaleTry[idxGood]
                idxMin = np.argmin(ratioMcm)
                if idxGood[idxMin] != 0 and mcmxTry[idxGood[0]] / mcmxTry[idxGood[idxMin]] > scaleTry[idxGood[idxMin]] / scaleTry[idxGood[0]]:
                    self._fov_resolution_arcsec = resTry[idxGood[idxMin]]
                else:
                    self._fov_resolution_arcsec = resTry[idxGood[0]]

        if not self._noprints:
            print(f'FoV internal resolution parameter set as [arcsec]: {self._fov_resolution_arcsec}')

        # Compute FFT FoV resolution element in arcsec
        self._scale_ovs = round(turbulence_pxscale / self._fov_resolution_arcsec)

        dTelPaddedInM = ef_size * in_ef.pixel_pitch * self._scale_ovs
        dSubApPaddedInM = dTelPaddedInM / self._lenslet.dimx
        fft_pxscale_arcsec = self._wavelengthInNm * 1e-9 / dSubApPaddedInM * rad2arcsec

        # Compute real FoV
        subap_real_fov_pix = round(subap_real_fov_arcsec / fft_pxscale_arcsec / 2.0) * 2.0
        subap_real_fov_arcsec = subap_real_fov_pix * fft_pxscale_arcsec
        mcmx = np.lcm(int(self._subap_npx), int(subap_real_fov_pix))

        turbulence_fov_pix = int(self._scale_ovs * np_sub)

        # Avoid increasing the FoV if it's already more than twice the requested one
        if turbulence_fov_pix > 2 * subap_real_fov_pix:
            self._fov_ovs = 1
            if self._fov_ovs_coeff != 0.0:
                self._fov_ovs = self._fov_ovs_coeff
        else:
            ratio = float(subap_real_fov_pix) / float(turbulence_fov_pix)
            np_factor = 1 if abs(np_sub - int(np_sub)) < 1e-3 else int(np_sub)
            if self._do_not_double_fov_ovs and self._fov_ovs_coeff == 0.0:
                self._fov_ovs_coeff = 1.0
                self._fov_ovs = np.ceil(np_factor * ratio / 2.0) * 2.0 / float(np_factor)
            else:
                if self._fov_ovs_coeff == 0.0:
                    self._fov_ovs_coeff = 2.0
                if ratio < 2:
                    self._fov_ovs = np.ceil(np_factor * self._fov_ovs_coeff) / float(np_factor)
                else:
                    self._fov_ovs = np.ceil(np_factor * ratio * self._fov_ovs_coeff) / float(np_factor)

        self._sensor_pxscale = subap_real_fov_arcsec / self._subap_npx / rad2arcsec
        self._congrid_np_sub = int(ef_size * self._fov_ovs * lens[2] * 0.5)
        self._fft_size = self._congrid_np_sub * self._scale_ovs

        if self._verbose:
            print('\n-->     FoV resolution [asec], {}'.format(self._fov_resolution_arcsec))
            print('-->     turb. pix. sc.,        {}'.format(turbulence_pxscale))
            print('-->     sc. over sampl.,       {}'.format(self._scale_ovs))
            print('-->     FoV over sampl.,       {}'.format(self._fov_ovs))
            print('-->     FFT pix. sc. [asec],   {}'.format(fft_pxscale_arcsec))
            print('-->     no. elements FoV,      {}'.format(subap_real_fov_pix))
            print('-->     FFT size (turb. FoV),  {}'.format(self._fft_size))
            print('-->     L.C.M. for toccd,      {}'.format(mcmx))

        # Kernel object initialization
        if self._kernelobj is not None:
            self._kernelobj.ef = in_ef

            if self.kernel_at_fft():
                self._kernelobj.pxscale = pixel_pitch_fft * rad2arcsec
                self._kernelobj.dimension = self._fft_size
                self._kernelobj.oversampling = 1
                self._kernelobj.positiveShiftTT = True
                self._kernelobj.returnFft = self._kernel_precalc_fft

            if self.kernel_at_fov():
                self._kernelobj.pxscale = pixel_pitch_fft * rad2arcsec
                self._kernelobj.dimension = (self._fft_size - self._cutpixels) * self._kernel_fov_scale
                self._kernelobj.oversampling = 1

            if self.kernel_at_subap():
                self._kernelobj.pxscale = self._sensor_pxscale * rad2arcsec
                self._kernelobj.dimension = self._subap_npx * self._kernel_fov_scale
                self._kernelobj.oversampling = 3

        # Check for valid phase size
        if abs((ef_size * round(self._fov_ovs) * lens[2]) / 2.0 - round((ef_size * round(self._fov_ovs) * lens[2]) / 2.0)) > 1e-4:
            raise ValueError('ERROR: interpolated input phase size is not divisible by subapertures.')
        elif not self._noprints:
            print(f'GOOD: interpolated input phase size is divisible by {self._lenslet.n_lenses} subapertures.')

    def detect_subaps(self, image, lenslet, energy_th):
        np = image.shape[0]
        mask_subap = self.xp.zeros_like(image)

        idxs = {}
        map = {}
        count = 0
        spot_intensity = self.xp.zeros((self._lenslet.dimx, self._lenslet.dimy))
        x = self.xp.zeros((self._lenslet.dimx, self._lenslet.dimy))
        y = self.xp.zeros((self._lenslet.dimx, self._lenslet.dimy))

        for i in range(self._lenslet.dimx):
            for j in range(self._lenslet.dimy):
                lens = self._lenslet.get(i, j)
                x[i, j] = np / 2.0 * (1 + lens[0])
                y[i, j] = np / 2.0 * (1 + lens[1])
                np_sub = round(np / 2.0 * lens[2])

                mask_subap *= 0
                mask_subap[self.xp.round(x[i, j] - np_sub / 2):self.xp.round(x[i, j] + np_sub / 2),
                           self.xp.round(y[i, j] - np_sub / 2):self.xp.round(y[i, j] + np_sub / 2)] = 1

                spot_intensity[i, j] = self.xp.sum(image * mask_subap)

        for i in range(self._lenslet.dimx):
            for j in range(self._lenslet.dimy):
                if spot_intensity[i, j] > energy_th * self.xp.max(spot_intensity):
                    mask_subap *= 0
                    mask_subap[self.xp.round(x[i, j] - np_sub / 2):self.xp.round(x[i, j] + np_sub / 2),
                               self.xp.round(y[i, j] - np_sub / 2):self.xp.round(y[i, j] + np_sub / 2)] = 1
                    idxs[count] = self.xp.where(mask_subap == 1)
                    map[count] = j * self._lenslet.dimx + i
                    count += 1

        if count == 0:
            raise ValueError("Error: no subapertures selected")

        v = self.xp.zeros((len(idxs), np_sub*np_sub), dtype=int)
        m = self.xp.zeros(len(idxs), dtype=int)
        for k, idx in idxs.items():
            v[k] = self.xp.ravel_multi_index(idx, image.shape)
            m[k] = map[k]
        
        subap_data = SubapData(idxs=v, map=m, nx=self._lenslet.dimx, ny=self._lenslet.dimy, energy_th=energy_th,
                           target_device_idx=self._target_device_idx, precision=self._precision)
      
        return subap_data
    
    def calc_trigger_geometry(self):
        
        in_ef = self.inputs['in_ef'].get(self._target_device_idx)
        s = in_ef.size

        # Calculate subap chunks (all of equal x/y size)
        # subap_chunks = [ (x slice, y slice), (x slice, y slice) ... ]
        # All chunks must have the same size and shape
        self._subap_chunks = []

        # Test: chunks of 2 full rows each
#        for i in range(0, self._lenslet.dimx, 2):
            #self._subap_chunks.append((slice(i, i+2), slice(0, self._lenslet.dimy)))
            
        # Whole SH in one go
        self._subap_chunks.append((slice(0, self._lenslet.dimx), slice(0, self._lenslet.dimy)))

        nx = self._subap_chunks[0][0].stop - self._subap_chunks[0][0].start
        ny = self._subap_chunks[0][1].stop - self._subap_chunks[0][1].start
        n = nx * ny

        fov_oversample = self._fov_ovs

        subap_wanted_fov = self._subap_wanted_fov
        sensor_pxscale = self._sensor_pxscale
        subap_npx = self._subap_npx

        self._rotAnglePhInDeg = self._rotAnglePhInDeg + self._aRotAnglePhInDeg
        self._xyShiftPhInPixel = np.array([self._xShiftPhInPixel + self._aXShiftPhInPixel, self._yShiftPhInPixel + self._aYShiftPhInPixel]) * fov_oversample

        if not self._floatShifts:
            self._xyShiftPhInPixel = np.round(self._xyShiftPhInPixel).astype(int)

        if fov_oversample != 1 or self._rotAnglePhInDeg != 0 or np.sum(np.abs(self._xyShiftPhInPixel)) != 0:
            M0 = s[0] * fov_oversample
            M1 = s[1] * fov_oversample
            wf1 = ElectricField(M0, M1, in_ef.pixel_pitch / fov_oversample)
        else:
            wf1 = in_ef            
        
        # Reuse geometry calculated in set_in_ef
        np_sub = self._congrid_np_sub
        fft_size = self._fft_size

        # Subaperture extracted from full pupil
        wf3 = self.xp.zeros((n, fft_size, fft_size), dtype=self.complex_dtype)
   
        # Focal plane result from FFT
        fp4_pixel_pitch = self._wavelengthInNm / 1e9 / (wf1.pixel_pitch * fft_size)
        fov_complete = fft_size * fp4_pixel_pitch

        # Calculate subaperture indexes
        self._subap_idx = self.xp.zeros((self._lenslet.dimx, self._lenslet.dimy, np_sub * np_sub), dtype=int)
        for i in range(self._lenslet.dimx):
            for j in range(self._lenslet.dimy):
                lens = self._lenslet.get(i, j)  # [x, y, size]
                x = wf1.size[0] / 2.0 * (1 + lens[0])
                y = wf1.size[1] / 2.0 * (1 + lens[1])

                f = self.xp.zeros(wf1.size, dtype=int)
                f[int(np.round(x - np_sub / 2.)):int(np.round(x + np_sub / 2.)),
                  int(np.round(y - np_sub / 2.)):int(np.round(y + np_sub / 2.))] = 1
                idx = self.xp.ravel_multi_index(f.nonzero(), f.shape)
                self._subap_idx[i, j, :] = idx

        sensor_subap_fov = sensor_pxscale * subap_npx
        fov_cut = fov_complete - sensor_subap_fov
        
        print(f'{fft_size=} {fp4_pixel_pitch=} {fov_cut=} {fov_complete=} {sensor_subap_fov=} {sensor_pxscale=} {subap_npx=}')
        self._cutpixels = int(np.round(fov_cut / fp4_pixel_pitch) / 2 * 2)
        self._cutsize = fft_size - self._cutpixels
        self._psfimage = self.xp.zeros((self._cutsize * self._lenslet.dimx, self._cutsize * self._lenslet.dimy))
        
        # 1/2 Px tilt
        self._tltf = self.get_tlt_f(np_sub, fft_size - np_sub)

        self._fp_mask = make_mask(fft_size, diaratio=subap_wanted_fov / fov_complete, square=self._squaremask, xp=self.xp)

        # Remember a few things
        self._wf1 = wf1
        self._wf3 = wf3

    def trigger(self, t):
        
        in_ef = self.inputs['in_ef'].get(self._target_device_idx)
        if in_ef.generation_time != t:
            return

        if not self._input_ef_set:
            self.set_in_ef(in_ef)
            self._input_ef_set = True

        if not self._trigger_geometry_calculated:
            self.calc_trigger_geometry()
            self._trigger_geometry_calculated = True

        if self._kernelobj is not None:
            self._kernelobj.trigger(t)

        s = in_ef.size

        fov_oversample = self._fov_ovs

        subap_wanted_fov = self._subap_wanted_fov
        subap_npx = self._subap_npx

        # Interpolation of input array if needed
        if fov_oversample != 1 or self._rotAnglePhInDeg != 0 or np.sum(np.abs([self._xyShiftPhInPixel])) != 0:
            if self._extrapol_mat is None:
                self._extrapol_mat = extrapolate_edge_pixel_mat_define(in_ef.a, do_ext_2_pix=True)

            phaseInNmNew = extrapolate_edge_pixel(in_ef.phaseInNm, self._extrapol_mat)
            tempA = congrid(in_ef.a, s[0] * fov_oversample, s[1] * fov_oversample, interp=True, minus_one=True)
            tempW = congrid(phaseInNmNew, s[0] * fov_oversample, s[1] * fov_oversample, interp=True, minus_one=True)

            if self._rotAnglePhInDeg != 0 or np.sum(np.abs([self._xyShiftPhInPixel])) != 0:
                self._wf1.A = (rot_and_shift_image(tempA, self._rotAnglePhInDeg, self._xyShiftPhInPixel, 1.0, use_interpolate=True) >= 0.5).astype(np.uint8)
                self._wf1.phaseInNm = rot_and_shift_image(tempW, self._rotAnglePhInDeg, self._xyShiftPhInPixel, 1.0, use_interpolate=True)
            else:
                self._wf1.A = tempA
                self._wf1.phaseInNm = tempW
        else:
            # wf1 already set to in_ef
            pass

        fft_size = self._fft_size
        psfTotalAtFft = 0
 
        if self._debugOutput:
            tempefcpu = np.zeros((self._lenslet.dimx * self._fft_size, self._lenslet.dimy * self._fft_size), dtype=complex)
            tempfftcpu = np.zeros_like(tempefcpu)
            tempconvcpu = np.zeros_like(tempefcpu)
            temppsfcpu = np.zeros_like(tempefcpu, dtype=float)

        congrid_np_sub = self._congrid_np_sub

        ef_whole = self._wf1.ef_at_lambda(self._wavelengthInNm)

        for chunk in self._subap_chunks:
            xslice, yslice = chunk
            nx = xslice.stop - xslice.start
            ny = yslice.stop - yslice.start
            x1 = xslice.start * congrid_np_sub
            x2 = xslice.stop * congrid_np_sub
            y1 = yslice.start * congrid_np_sub
            y2 = yslice.stop * congrid_np_sub
            n = nx * ny

            ef = ef_whole[x1:x2, y1:y2]

            # Transform from 2d subap tiling into N x np x np
            # For an explanation of how this works, ask ChatGPT
            ef = ef.reshape(nx, congrid_np_sub, ny, congrid_np_sub)
            ef = ef.transpose((2, 0, 1, 3))
            ef = ef.reshape(n, congrid_np_sub, congrid_np_sub)

            # Insert into padded array
            self._wf3[:, :congrid_np_sub, :congrid_np_sub] = ef * self._tltf[self.xp.newaxis, :, :]

            if self._debugOutput:
                tempefcpu[i * fft_size:(i + 1) * fft_size, j * fft_size:(j + 1) * fft_size] = self._wf3

            # PSF generation
            fp4 = self.xp.fft.fft2(self._wf3, axes=(1, 2))
            psf_shifted = abs2(fp4)
            psfTotalAtFft += self.xp.sum(psf_shifted)

            # Full resolution kernel
            if self._kernelobj is not None and self.kernel_at_fft():
                if not self._kernelobj.returnFft:
                    subap_kern = self.xp.fft.fftshift(self._kernelobj.out_kernels.value[j * self._lenslet.dimx + i])
                    subap_kern /= self.xp.sum(subap_kern)
                    subap_kern_fft = self.xp.fft.fft2(subap_kern)
                else:
                    subap_kern_fft = self._kernelobj.out_kernels.value[j * self._lenslet.dimx + i]

                psf_fft = self.xp.fft.fft2(psf_shifted)
                psf = self.xp.fft.ifft2(psf_fft * subap_kern_fft).real
                psf = self.xp.fft.fftshift(psf)

                if self._debugOutput:
                    temppsfcpu[i * fft_size:(i + 1) * fft_size, j * fft_size:(j + 1) * fft_size] = psf_shifted
                    tempfftcpu[i * fft_size:(i + 1) * fft_size, j * fft_size:(j + 1) * fft_size] = psf_fft
                    tempconvcpu[i * fft_size:(i + 1) * fft_size, j * fft_size:(j + 1) * fft_size] = psf

                psf *= self._fp_mask
            else:
                psf = self.xp.fft.fftshift(psf_shifted, axes=(1, 2))
                psf *= self._fp_mask[self.xp.newaxis, :, :]

            cutsize = self._cutsize
            cutpixels = self._cutpixels
            psf_cut = psf[:, cutpixels // 2: -cutpixels // 2, cutpixels // 2: -cutpixels // 2]

            # FoV kernel
            if self._kernelobj is not None and self.kernel_at_fov():
                subap_kern = self._kernelobj.out_kernels.value[j * self._lenslet.dimx + i]
                subap_kern /= self.xp.sum(subap_kern)
                sKernel = subap_kern.shape[0]
                if sKernel == cutsize:
                    psf_cut = self.xp.fft.fftshift(self.xp.convolve(psf_cut, subap_kern, mode='same'))
                else:
                    psf_cut = self.xp.fft.fftshift(self.xp.convolve(psf_cut, self.xp.fft.fftshift(subap_kern, axes=(-2, -1)), mode='same'))

            # Back-transform from N x np x np to 2d subap tiling
            # For an explanation of how this works, ask ChatGPT
            psf_cut = psf_cut.reshape(ny, nx, cutsize, cutsize)
            psf_cut = psf_cut.transpose(1, 2, 0, 3)
            psf_cut = psf_cut.reshape(nx * cutsize, ny * cutsize)
            
            # Insert subap strip into overall PSF image
            self._psfimage[xslice.start * cutsize: xslice.stop * cutsize, yslice.start * cutsize: yslice.stop * cutsize] = psf_cut

        self._psfimage /= psfTotalAtFft + 1e-6 # Avoid dividing by zero

        # Post-processing kernel (Gaussian convolution)
        if self._gkern:
            psf_size = self._psfimage.shape[0] // self._lenslet.dimx
            kern_pixsize = self._gkern_size / subap_wanted_fov * psf_size
            subap_gkern = gaussian_function(kern_pixsize / (2.0 * np.sqrt(2 * np.log(2))), width=psf_size, maximum=1.0)
            subap_gkern /= np.sum(subap_gkern)

            for i in range(self._lenslet.dimx):
                for j in range(self._lenslet.dimy):
                    x1 = i * psf_size
                    x2 = x1 + psf_size
                    y1 = j * psf_size
                    y2 = y1 + psf_size
                    subap = self._psfimage[x1:x2, y1:y2]
                    subap_tmp = self.xp.convolve(subap, subap_gkern, mode='same')
                    self._psfimage[x1:x2, y1:y2] = self.xp.abs(self.xp.fft.fftshift(self.xp.fft.fft2(subap_tmp)))

        ccd = toccd(self._psfimage, (self._ccd_side, self._ccd_side), xp=self.xp)

        # Apply kernel at subaperture level if necessary
        if self._kernelobj is not None and self.kernel_at_subap():
            for i in range(self._lenslet.dimx):
                for j in range(self._lenslet.dimy):
                    x1 = i * subap_npx
                    x2 = x1 + subap_npx
                    y1 = j * subap_npx
                    y2 = y1 + subap_npx
                    subap = ccd[x1:x2, y1:y2]
                    subap_kern = self._kernelobj.out_kernels.value[j * self._lenslet.dimx + i]
                    subap_kern /= self.xp.sum(subap_kern)
                    ccd[x1:x2, y1:y2] = self.xp.fft.fftshift(self.xp.convolve(subap, subap_kern, mode='same'))

        phot = in_ef.S0 * in_ef.masked_area()
        ccd *= phot

        self._out_i.i = ccd
        self._out_i.generation_time = t

    def get_tlt_f(self, p, c):
        iu = complex(0, 1)
        xx, yy = self.xp.meshgrid(self.xp.arange(-p // 2, p // 2), self.xp.arange(-p // 2, p // 2))
        tlt_g = xx + yy
        tlt_f = self.xp.exp(-2 * self.xp.pi * iu * tlt_g / (2 * (p + c)), dtype=self.complex_dtype)
        return tlt_f
