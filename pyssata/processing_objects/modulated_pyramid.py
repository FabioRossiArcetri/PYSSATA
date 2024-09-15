import numpy as np

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.base_value import BaseValue
from pyssata.lib.make_xy import make_xy
from pyssata.data_objects.intensity import Intensity
from pyssata.lib.make_mask import make_mask
from pyssata.lib.toccd import toccd

class ModulatedPyramid(BaseProcessingObj):
    def __init__(self, wavelength_in_nm, fov_res, fp_masking, fft_res, tilt_scale, fft_sampling, 
                 fft_padding, fft_totsize, toccd_side, final_ccd_side, fp_obs=None, pyr_tlt_coeff=None, 
                 pyr_edge_def_ld=0.0, pyr_tip_def_ld=0.0, pyr_tip_maya_ld=0.0):
        
        super().__init__()

        # Compute focal plane central obstruction dimension ratio                 
        fp_obsratio = fp_obs / (fft_totsize / fft_res) if fp_obs is not None else 0

        self._wavelength_in_nm = wavelength_in_nm
        self._fov_res = fov_res
        self._fp_masking = fp_masking
        self._fp_obsratio = fp_obsratio
        self._fft_res = fft_res
        self._tilt_scale = tilt_scale
        self._fft_sampling = fft_sampling
        self._fft_padding = fft_padding
        self._fft_totsize = fft_totsize
        self._toccd_side = toccd_side
        self._final_ccd_side = final_ccd_side
        self._pyr_tlt_coeff = pyr_tlt_coeff
        self._pyr_edge_def_ld = pyr_edge_def_ld
        self._pyr_tip_def_ld = pyr_tip_def_ld
        self._pyr_tip_maya_ld = pyr_tip_maya_ld
        self._rotAnglePhInDeg = 0
        self._mod_amp = 0
        self._mod_steps = 0
        self._pup_shifts = None

        if not all([fft_res, fov_res, tilt_scale, fft_sampling, fft_totsize, toccd_side, final_ccd_side]):
            return

        self._out_i = Intensity(final_ccd_side, final_ccd_side)

        self._psf_tot = BaseValue(np.zeros((fft_totsize, fft_totsize)))
        self._psf_bfm = BaseValue(np.zeros((fft_totsize, fft_totsize)))
        self._out_transmission = BaseValue(0)

        self._pyr_tlt = self.get_pyr_tlt(fft_sampling, fft_padding)
        self._tlt_f = self.get_tlt_f(fft_sampling, fft_padding)
        self._tilt_x = self.get_modulation_tilt(fft_sampling, X=True)
        self._tilt_y = self.get_modulation_tilt(fft_sampling, Y=True)
        self._fp_mask = self.get_fp_mask(fft_totsize, fp_masking, obsratio=fp_obsratio)
        self._extended_source_in_on = False
        iu = 1j  # complex unit
        self._myexp = np.exp(-2 * np.pi * iu * self._pyr_tlt)

        # Pre-computation of ttexp will be done when mod_steps will be set or re-set
        self._ttexp = {}
        # Trigger cache
        self.mod_amp = 3
        self.mod_steps = 32

    @property
    def mod_amp(self):
        return self._mod_amp

    @mod_amp.setter
    def mod_amp(self, value):
        if value != self._mod_amp:
            self._mod_amp = value
            self.cache_ttexp()

    @property
    def mod_steps(self):
        return self._mod_steps

    @mod_steps.setter
    def mod_steps(self, value):
        if value != self._mod_steps:
            self._mod_steps = value
            self.cache_ttexp()

    @property
    def in_ef(self):
        return self._in_ef

    @in_ef.setter
    def in_ef(self, value):
        self._in_ef = value

    @property
    def fp_masking(self):
        return self._fp_masking

    @fp_masking.setter
    def fp_masking(self, value):
        self._fp_masking = value

    @property
    def pup_shifts(self):
        return self._pup_shifts

    @pup_shifts.setter
    def pup_shifts(self, value):
        self._pup_shifts = value

    @property
    def ext_source_psf(self):
        return self._extSourcePsf

    @ext_source_psf.setter
    def ext_source_psf(self, value):
        self._extSourcePsf = value

    @property
    def rot_angle_ph_in_deg(self):
        return self._rotAnglePhInDeg

    @rot_angle_ph_in_deg.setter
    def rot_angle_ph_in_deg(self, value):
        self._rotAnglePhInDeg = value

    @property
    def out_i(self):
        return self._out_i

    @staticmethod
    def calc_geometry(
        DpupPix,                # number of pixels of input phase array
        pixel_pitch,            # pixel sampling [m] of DpupPix
        lambda_,                # working lambda of the sensor [nm]
        FoV,                    # requested FoV in arcsec
        pup_diam,               # pupil diameter in subapertures
        ccd_side,               # requested output ccd side, in pixels
        fov_errinf=0.1,         # accepted error in reducing FoV, default = 0.1 (-10%)
        fov_errsup=0.5,         # accepted error in enlarging FoV, default = 0.5 (+50%)
        pup_dist=None,          # pupil distance in subapertures, optional
        pup_margin=2,           # zone of respect around pupils for margins, optional, default=2px
        fft_res=3.0,            # requested minimum PSF sampling, 1.0 = 1 pixel / PSF, default=3.0
        min_pup_dist=None,
        NOTEST=False            # skip the time estimation done with a test pyramid
    ):
        # Calculate pup_distance if not given, using the pup_margin
        if pup_dist is None:
            pup_dist = pup_diam + pup_margin * 2

        if min_pup_dist is None:
            min_pup_dist = pup_diam + pup_margin * 2

        if pup_dist < min_pup_dist:
            print(f"Error: pup_dist (px) = {pup_dist} is not enough to hold the pupil geometry. Minimum allowed distance is {min_pup_dist}")
            return 0

        min_ccd_side = pup_dist + pup_diam + pup_margin * 2
        if ccd_side < min_ccd_side:
            print(f"Error: ccd_side (px) = {ccd_side} is not enough to hold the pupil geometry. Minimum allowed side is {min_ccd_side}")
            return 0

        RAD2ARCSEC = 206265
        D = DpupPix * pixel_pitch
        Fov_internal = lambda_ * 1e-9 / D * (D / pixel_pitch) * RAD2ARCSEC

        minfov = FoV * (1 - fov_errinf)
        maxfov = FoV * (1 + fov_errsup)
        fov_res = 1.0

        if Fov_internal < minfov:
            fov_res = int(minfov / Fov_internal)
            if Fov_internal * fov_res < minfov:
                fov_res += 1

        if Fov_internal > maxfov:
            print("Error: Calculated FoV is higher than maximum accepted FoV.")
            print("Please revise error margin, or the input phase dimension and/or pitch")
            return 0

        if fov_res > 1:
            Fov_internal *= fov_res
            print(f"Interpolated FoV (arcsec): {Fov_internal:.2f}")
            print(f"Warning: reaching the requested FoV requires {fov_res}x interpolation of input phase array.")
            print("Consider revising the input phase dimension and/or pitch to improve performance.")

        fp_masking = FoV / Fov_internal

        if Fov_internal != FoV:
            print(f"FoV reduction from {Fov_internal:.2f} to {FoV:.2f} will be performed with a focal plane mask")

        DpupPixFov = DpupPix * fov_res
        pitch_internal = pixel_pitch / fov_res

        fft_res_min = (pup_dist + pup_diam) / pup_diam * 1.1
        if fft_res < fft_res_min:
            fft_res = fft_res_min

        internal_ccd_side = round(fft_res * pup_diam / 2) * 2
        fft_res = internal_ccd_side / float(pup_diam)

        totsize = round(DpupPixFov * fft_res / 2) * 2
        fft_res = totsize / float(DpupPixFov)

        padding = round((DpupPixFov * fft_res - DpupPixFov) / 2) * 2

        factors = np.array([])
        exponents = np.array([])

        if not NOTEST:
            # Placeholder for the test pyramid calculations
            pass

        results = {
            'fov_res': fov_res,
            'fp_masking': fp_masking,
            'fft_res': fft_res,
            'tilt_scale': fft_res / ((pup_dist / float(pup_diam)) / 2.0),
            'fft_sampling': DpupPixFov,
            'fft_padding': padding,
            'fft_totsize': totsize,
            'wavelengthInNm': lambda_,
            'toccd_side': internal_ccd_side,
            'final_ccd_side': ccd_side
        }

        return results

    
    def set_extended_source(self, source):
        self._extSource = source
        self._extended_source_in_on = True

        self._ext_xtilt = self.zern(2, make_xy(self._fft_sampling, 1.0))
        self._ext_ytilt = self.zern(3, make_xy(self._fft_sampling, 1.0))
        self._ext_focus = self.zern(4, make_xy(self._fft_sampling, 1.0))

        if source.npoints == 0:
            raise ValueError('ERROR: number of points of extended source is 0!')
        else:
            self._mod_steps = source.npoints

        self._ttexp.clear()

        print(f'modulated_pyramid --> Setting up extended source with {self._mod_steps} points')

        if self._mod_steps <= 0:
            return

        iu = 1j  # complex unit

        for tt in range(self._mod_steps):
            angle = 2 * np.pi * (tt / self._mod_steps)
            pup_tt = source.coeff_tiltx[tt] * self._ext_xtilt + source.coeff_tilty[tt] * self._ext_ytilt
            pup_focus = -1 * source.coeff_focus[tt] * self._ext_focus
            self._ttexp[tt] = np.exp(-iu * (pup_tt + pup_focus))

        i = source.coeff_flux
        idx = np.where(np.abs(i) < np.max(np.abs(i)) * 1e-5)[0]
        if len(idx[0]) > 0:
            i[idx] = 0
        self._flux_factor_vector = i

    def get_pyr_tlt(self, p, c):
        A = int((p + c) // 2)
        pyr_tlt = np.zeros((2 * A, 2 * A))
        #tlt_basis = np.tile(np.arange(A), (A, 1))
        y, x = np.mgrid[0:A,0:A]

        if self._pyr_tlt_coeff is not None:
            k = self._pyr_tlt_coeff

            tlt_basis -= np.mean(tlt_basis)

            pyr_tlt[0:A, 0:A] = k[0, 0] * tlt_basis + k[1, 0] * tlt_basis.T
            pyr_tlt[A:2*A, 0:A] = k[0, 1] * tlt_basis + k[1, 1] * tlt_basis.T
            pyr_tlt[A:2*A, A:2*A] = k[0, 2] * tlt_basis + k[1, 2] * tlt_basis.T
            pyr_tlt[0:A, A:2*A] = k[0, 3] * tlt_basis + k[1, 3] * tlt_basis.T
            pyr_tlt[0:A, 0:A] -= np.min(pyr_tlt[0:A, 0:A])
            pyr_tlt[A:2*A, 0:A] -= np.min(pyr_tlt[A:2*A, 0:A])
            pyr_tlt[A:2*A, A:2*A] -= np.min(pyr_tlt[A:2*A, A:2*A])
            pyr_tlt[0:A, A:2*A] -= np.min(pyr_tlt[0:A, A:2*A])

        else:
            #pyr_tlt[0:A, 0:A] = tlt_basis + tlt_basis.T
            #pyr_tlt[A:2*A, 0:A] = A - 1 - tlt_basis + tlt_basis.T
            #pyr_tlt[A:2*A, A:2*A] = 2 * A - 2 - tlt_basis - tlt_basis.T
            #pyr_tlt[0:A, A:2*A] = A - 1 + tlt_basis - tlt_basis.T
            pyr_tlt[:A, :A] = x + y
            pyr_tlt[:A, A:] = x[:,::-1] + y
            pyr_tlt[A:, :A] = x + y[::-1]
            pyr_tlt[A:, A:] = x[:,::-1] + y[::-1]

        xx, yy = make_xy(A * 2, A)

        # distance from edge
        dx = np.sqrt(xx ** 2)
        dy = np.sqrt(yy ** 2)
        idx_edge = np.where((dx <= self._pyr_edge_def_ld * self._fft_res / 2) | 
                            (dy <= self._pyr_edge_def_ld * self._fft_res / 2))[0]
        if len(idx_edge) > 0:
            pyr_tlt[idx_edge] = np.max(pyr_tlt) * np.random.rand(len(idx_edge[0]))
            print(f'get_pyr_tlt: {len(idx_edge[0])} pixels set to 0 to consider pyramid imperfect edges')

        # distance from tip
        d = np.sqrt(xx ** 2 + yy ** 2)
        idx_tip = np.where(d <= self._pyr_tip_def_ld * self._fft_res / 2)[0]
        if len(idx_tip) > 0:
            pyr_tlt[idx_tip] = np.max(pyr_tlt) * np.random.rand(len(idx_tip[0]))
            print(f'get_pyr_tlt: {len(idx_tip[0])} pixels set to 0 to consider pyramid imperfect tip')

        # distance from tip
        idx_tip_m = np.where(d <= self._pyr_tip_maya_ld * self._fft_res / 2)[0]
        if len(idx_tip_m) > 0:
            pyr_tlt[idx_tip_m] = np.min(pyr_tlt[idx_tip_m])
            print(f'get_pyr_tlt: {len(idx_tip_m[0])} pixels set to 0 to consider pyramid imperfect tip')

        return pyr_tlt / self._tilt_scale

    def get_tlt_f(self, p, c):
        iu = 1j  # complex unit
        p = int(p)
        xx, yy = make_xy(2 * p, p, quarter=True, zero_sampled=True)
        tlt_g = xx + yy

        tlt_f = np.exp(-2 * np.pi * iu * tlt_g / (2 * (p + c)))
        return tlt_f

    def get_fp_mask(self, totsize, mask_ratio, obsratio=0):
        return make_mask(totsize, diaratio=mask_ratio, obsratio=obsratio)

    def get_modulation_tilt(self, p, X=False, Y=False):
        p = int(p)
        xx, yy = make_xy(p, p // 2)
        mm = self.minmax(xx)
        tilt_x = xx * np.pi / ((mm[1] - mm[0]) / 2)
        tilt_y = yy * np.pi / ((mm[1] - mm[0]) / 2)

        if X:
            return tilt_x
        if Y:
            return tilt_y

    def cache_ttexp(self):
        if not self._extended_source_in_on:
            self._ttexp.clear()
            if self._mod_steps <= 0:
                return

            iu = 1j  # complex unit

            for tt in range(self._mod_steps):
                angle = 2 * np.pi * (tt / self._mod_steps)
                pup_tt = self._mod_amp * np.sin(angle) * self._tilt_x + \
                         self._mod_amp * np.cos(angle) * self._tilt_y

                self._ttexp[tt] = np.exp(-iu * pup_tt)

            self._flux_factor_vector = np.ones(self._mod_steps)

    def trigger(self, t):
        if self._in_ef.generation_time != t:
            return

        if self._extended_source_in_on and self._extSourcePsf is not None:
            if self._extSourcePsf.generation_time == t:
                if np.sum(np.abs(self._extSourcePsf.value)) > 0:
                    self._extSource.updatePsf(self._extSourcePsf.value)
                    self._flux_factor_vector = self._extSource.coeff_flux

        s = self._in_ef.size

        if self._rotAnglePhInDeg != 0:
            A = (self.ROT_AND_SHIFT_IMAGE(self._in_ef.A, self._rotAnglePhInDeg, [0, 0], 1, use_interpolate=True) >= 0.5).astype(np.uint8)
            phi_at_lambda = self.ROT_AND_SHIFT_IMAGE(self._in_ef.phi_at_lambda(self._wavelength_in_nm), self._rotAnglePhInDeg, [0, 0], 1, use_interpolate=True)
            ef = np.complex64(np.rebin(A, (s[0] * self._fov_res, s[1] * self._fov_res)) + 
                              np.rebin(phi_at_lambda, (s[0] * self._fov_res, s[1] * self._fov_res)) * 1j)
        else:
            if self._fov_res != 1:
                ef = np.complex64(np.rebin(self._in_ef.A, (s[0] * self._fov_res, s[1] * self._fov_res)) + 
                                  np.rebin(self._in_ef.phi_at_lambda(self._wavelength_in_nm), (s[0] * self._fov_res, s[1] * self._fov_res)) * 1j)
            else:
                ef = self._in_ef.ef_at_lambda(self._wavelength_in_nm)

        u_tlt_const = ef * self._tlt_f

        pup_pyr_tot = np.zeros((self._fft_totsize, self._fft_totsize))
        psf_bfm = np.zeros((self._fft_totsize, self._fft_totsize))
        psf_tot = np.zeros((self._fft_totsize, self._fft_totsize))

        u_tlt = np.zeros((self._fft_totsize, self._fft_totsize), dtype=np.complex64)

        for tt in range(self._mod_steps):
            if self._flux_factor_vector[tt] <= np.median(self._flux_factor_vector) * 1e-3:
                continue

            tmp = u_tlt_const * self._ttexp[tt]
            ss = tmp.shape
            u_tlt[0:ss[0], 0:ss[1]] = tmp

            u_fp = np.fft.fftshift(np.fft.fft2(u_tlt))

            psf = np.abs(u_fp) ** 2

            psf_bfm += psf * self._flux_factor_vector[tt]

            u_fp *= self._fp_mask
            psf *= self._fp_mask

            psf_tot += psf * self._flux_factor_vector[tt]

            u_fp_pyr = u_fp * self._myexp

            pup_pyr_tot += np.abs(np.fft.ifft2(u_fp_pyr)) ** 2 * self._flux_factor_vector[tt]

        pup_pyr_tot = np.roll(pup_pyr_tot, (self._fft_padding//2, self._fft_padding//2), (0,1))

        factor = 1.0 / np.sum(self._flux_factor_vector)
        pup_pyr_tot *= factor
        psf_tot *= factor
        psf_bfm *= factor

        sum_psf = np.sum(psf_tot)
        sum_bfm = np.sum(psf_bfm)
        sum_pup = np.sum(pup_pyr_tot)
        transmission = sum_psf / sum_bfm
        phot = self._in_ef.S0 * self._in_ef.masked_area()
        pup_pyr_tot *= (phot / sum_pup) * transmission

        if phot == 0:
            print('WARNING: total intensity at PYR entrance is zero')

        if self._pup_shifts is not None:
            self._pup_shifts.trigger(t)
            image = np.pad(pup_pyr_tot, 1, mode='constant')
            imscale = float(self._fft_totsize) / float(self._toccd_side)

            pup_shiftx = self._pup_shifts.output.value[0] * imscale
            pup_shifty = self._pup_shifts.output.value[1] * imscale

            image = self.interpolate(image, np.arange(self._fft_totsize + 2) - pup_shiftx, 
                                     np.arange(self._fft_totsize + 2) - pup_shifty, grid=True, missing=0)
            pup_pyr_tot = image[1:-1, 1:-1]

        ccd_internal = toccd(pup_pyr_tot, (self._toccd_side, self._toccd_side))

        if self._final_ccd_side > self._toccd_side:
            delta = (self._final_ccd_side - self._toccd_side) // 2
            ccd = np.zeros((self._final_ccd_side, self._final_ccd_side))
            ccd[delta:delta + ccd_internal.shape[0], delta:delta + ccd_internal.shape[1]] = ccd_internal
        elif self._final_ccd_side < self._toccd_side:
            delta = (self._toccd_side - self._final_ccd_side) // 2
            ccd = ccd_internal[delta:delta + self._final_ccd_side, delta:delta + self._final_ccd_side]
        else:
            ccd = ccd_internal

        self._out_i.i = ccd
        self._out_i.generation_time = t
        self._psf_tot.value = psf_tot
        self._psf_tot.generation_time = t
        self._psf_bfm.value = psf_bfm
        self._psf_bfm.generation_time = t
        self._out_transmission.value = transmission
        self._out_transmission.generation_time = t

    def run_check(self, time_step):
        if self._extended_source_in_on:
            return 1
        elif self._mod_steps < round(2 * np.pi * self._mod_amp):
            raise Exception(f'Number of modulation steps is too small ({self._mod_steps}), it must be at least 2*pi times the modulation amplitude ({round(2 * np.pi * self._mod_amp)})!')
        return 1

    def hdr(self, hdr):
        hdr['MODAMP'] = self._mod_amp
        hdr['MODSTEPS'] = self._mod_steps

    @staticmethod
    def minmax(array):
        return np.min(array), np.max(array)

    @staticmethod
    def zern(mode, xx, yy):
        raise NotImplementedError


    @staticmethod
    def interpolate(image, x, y, grid=False, missing=0):
        raise NotImplementedError

    @staticmethod
    def ROT_AND_SHIFT_IMAGE(image, angle, shift, scale, use_interpolate=False):
        raise NotImplementedError
