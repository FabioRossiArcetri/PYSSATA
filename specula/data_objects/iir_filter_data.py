import numpy as np

from specula.base_data_obj import BaseDataObj

from astropy.io import fits


class IIRFilterData(BaseDataObj):
    def __init__(self, ordnum, ordden, num, den, target_device_idx=None, precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.ordnum = self.xp.array(ordnum, dtype=int)
        self.ordden = self.xp.array(ordden, dtype=int) 
        self.zeros = None
        self.poles = None
        self.gain = None
        self.set_num(num)
        self.set_den(den)

    @property
    def nfilter(self):
        return len(self.num)
    
    def zeros(self):
        if self.zeros is None:
            snum1 = self.num.shape[1]
            zeros = self.xp.zeros((self.nfilter, snum1 - 1), dtype=self.dtype)
            for i in range(self.nfilter):
                if self.ordnum[i] > 1:
                    roots = self.xp.roots(self._num[i, snum1 - int(self.ordnum[i]):])
                    if np.sum(np.abs(roots)) > 0:
                        zeros[i, :int(self.ordnum[i]) - 1] = roots
            self.zeros = zeros
        return self.zeros

    def poles(self):
        if self.poles is None:
            sden1 = self.den.shape[1]
            poles = self.xp.zeros((self.nfilter, sden1 - 1), dtype=self.dtype)
            for i in range(self.nfilter):
                if self.ordden[i] > 1:
                    poles[i, :int(self.ordden[i]) - 1] = self.xp.roots(self._den[i, sden1 - int(self.ordden[i]):])
            self.poles = poles
        return self.poles
        
    def set_num(self, num):
        snum1 = num.shape[1]
        mynum = num.copy()
        for i in range(len(mynum)):
            if self.ordnum[i] < snum1:
                if np.sum(self.xp.abs(mynum[i, int(self.ordnum[i]):])) == 0:
                    mynum[i, :] = self.xp.roll(mynum[i, :], snum1 - int(self.ordnum[i]))

        gain = self.xp.zeros(len(mynum), dtype=self.dtype)
        for i in range(len(gain)):
            gain[i] = mynum[i, - 1]
        self.gain = gain
        self.zeros = None 
        self.num = self.xp.array(mynum, dtype=self.dtype)

    def set_den(self, den):
        sden1 = den.shape[1]
        myden = den.copy()
        for i in range(len(myden)):
            if self.ordden[i] < sden1:
                if np.sum(self.xp.abs(myden[i, int(self.ordden[i]):])) == 0:
                    myden[i, :] = self.xp.roll(myden[i, :], sden1 - int(self.ordden[i]))

        self.den = self.xp.array(myden, dtype=self.dtype)
        self.poles = None

    def set_zeros(self, zeros):
        self.zeros = self.xp.array(zeros, dtype=self.dtype)
        num = self.xp.zeros((self.nfilter, self.zeros.shape[1] + 1), dtype=self.dtype)
        snum1 = num.shape[1]
        for i in range(self.nfilter):
            if self.ordnum[i] > 1:
                num[i, snum1 - int(self.ordnum[i]):] = self.xp.poly(self.zeros[i, :int(self.ordnum[i]) - 1])
        self.num = num

    def set_poles(self, poles):
        self.poles = self.xp.array(poles, dtype=self.dtype)
        den = self.xp.zeros((self.nfilter, self.poles.shape[1] + 1), dtype=self.dtype)
        sden1 = den.shape[1]
        for i in range(self.nfilter):
            if self.ordden[i] > 1:
                den[i, sden1 - int(self.ordden[i]):] = self.xp.poly(self._poles[i, :int(self.ordden[i]) - 1])
        self._den = den

    def set_gain(self, gain, verbose=False):
        if verbose:
            print('original gain:', self._gain)
        if self.xp.size(gain) < self.nfilter:
            nfilter = np.size(gain)
        else:
            nfilter = self.nfilter
        if self.gain is None:
            for i in range(nfilter):
                if self.xp.isfinite(gain[i]):
                    if self.ordnum[i] > 1:
                        self.num[i, :] *= gain[i]
                    else:
                        self.num[i, - 1] = gain[i]
                else:
                    gain[i] = self._num[i, - 1]
        else:
            for i in range(nfilter):
                if self.xp.isfinite(gain[i]):
                    if self.ordnum[i] > 1:
                        self.num[i, :] *= (gain[i] / self.gain[i])
                    else:
                        self.num[i, - 1] = gain[i] / self.gain[i]
                else:
                    gain[i] = self._gain[i]
        self.gain = self.xp.array(gain, dtype=self.dtype)
        if verbose:
            print('new gain:', self._gain)

    def complexRTF(self, mode, fs, delay, freq=None, verbose=False):
        if delay > 1:
            dm = self.xp.array([0.0, 1.0], dtype=self.dtype)
            nm = self.xp.array([1.0, 0.0], dtype=self.dtype)
            wTf = self.discrete_delay_tf(delay - 1)
        else:
            dm = self.xp.array([1.0], dtype=self.dtype)
            nm = self.xp.array([1.0], dtype=self.dtype)
            wTf = self.discrete_delay_tf(delay)
        nw = wTf[:, 0]
        dw = wTf[:, 1]
        complex_yt_tf = self.plot_iirfilter_tf(self.num[mode, :], self.den[mode, :], fs, dm=dm, nw=nw, dw=dw, freq=freq, noplot=True, verbose=verbose)
        return complex_yt_tf

    def RTF(self, mode, fs, freq=None, tf=None, dm=None, nw=None, dw=None, verbose=False, title=None, overplot=False, **extra):
        plotTitle = title if title else '!17Rejection Transfer Function'
        tf = self.plot_iirfilter_tf(self._num[mode, :], self._den[mode, :], fs, dm=dm, nw=nw, dw=dw, freq=freq, noplot=True, verbose=verbose)
        import matplotlib.pyplot as plt
        if overplot:
            color = extra.get('color', 255)
            plt.plot(freq, tf, color=color, **extra)
        else:
            plt.plot(freq, tf, label=plotTitle)
            plt.xlabel('!17frequency !4[!17Hz!4]!17')
            plt.ylabel('!17magnitude')
            plt.title(plotTitle)
            plt.show()

    def NTF(self, mode, fs, freq=None, tf=None, dm=None, nw=None, dw=None, verbose=False, title=None, overplot=False, **extra):
        plotTitle = title if title else '!17Noise Transfer Function'
        tf = self.plot_iirfilter_tf(self.num[mode, :], self.den[mode, :], fs, dm=dm, nw=nw, dw=dw, freq=freq, noplot=True, verbose=verbose)
        import matplotlib.pyplot as plt
        if overplot:
            color = extra.get('color', 255)
            plt.plot(freq, tf, color=color, **extra)
        else:
            plt.plot(freq, tf, label=plotTitle)
            plt.xlabel('!17frequency !4[!17Hz!4]!17')
            plt.ylabel('!17magnitude')
            plt.title(plotTitle)
            plt.show()

    def is_stable(self, mode, nm=None, dm=None, nw=None, dw=None, gain=None, no_margin=False, verbose=False):
        nm = nm if nm is not None else self.xp.array([1, 0], dtype=self.dtype)
        nw = nw if nw is not None else self.xp.array([1, 0], dtype=self.dtype)
        dm = dm if dm is not None else self.xp.array([0, 1], dtype=self.dtype)
        dw = dw if dw is not None else self.xp.array([0, 1], dtype=self.dtype)

        temp1 = self.xp.polymul(dm, dw)
        while temp1[-1] == 0:
            temp1 = temp1[:-1]
        DDD = self.xp.polymul(temp1, self.den[mode, :])
        while DDD[-1] == 0:
            DDD = DDD[:-1]

        temp2 = self.xp.polymul(nm, nw)
        while temp2[-1] == 0:
            temp2 = temp2[:-1]
        NNN = self.xp.polymul(temp2, self.num[mode, :])
        if self.xp.sum(self.xp.abs(NNN)) != 0:
            while NNN[-1] == 0:
                NNN = NNN[:-1]

        if gain is not None:
            NNN *= gain / self.gain[mode]

        stable, ph_margin, g_margin, mroot, m_one_dist = self.nyquist(NNN, DDD, no_margin=no_margin)

        if verbose:
            print('max root (closed loop) =', mroot)
            print('phase margin =', ph_margin)
            print('gain margin =', g_margin)
            print('min. distance from (-1;0) =', m_one_dist)
        return stable

    def save(self, filename):
        hdr = fits.Header()
        hdr['VERSION'] = 1

        hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=self.ordnum, name='ORDNUM'))
        hdul.append(fits.ImageHDU(data=self.ordden, name='ORDDEN'))
        hdul.append(fits.ImageHDU(data=self.num, name='NUM'))
        hdul.append(fits.ImageHDU(data=self.den, name='DEN'))
        hdul.writeto(filename, overwrite=True)

    def restore(self, filename, target_device_idx=None):
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            version = hdr['VERSION']
            if version != 1:
                raise ValueError(f"Error: unknown version {version} in file {filename}")
            ordnum = hdul['ORDNUM'].data
            ordden = hdul['ORDDEN'].data
            num = hdul['NUM'].data
            den = hdul['DEN'].data
            return IIRFilterData(ordnum, ordden, num, den, target_device_idx=target_device_idx)

    def get_fits_header(self):
        # TODO
        raise NotImplementedError()

    @staticmethod
    def from_header(hdr):
        # TODO
        raise NotImplementedError()

    def discrete_delay_tf(self, delay):
        # If not-integer delay TF:
        # DelayTF = z^(−l) * ( m * (1−z^(−1)) + z^(−1) )
        # where delay = (l+1)*T − mT, T integration time, l integer, 0<m<1

        if delay - np.fix(delay) != 0:
            d_m = np.ceil(delay)
            den = np.zeros(int(d_m)+1)
            den[int(d_m)] = 1
            num = den*0 
            num[0] = delay - np.fix(delay) 
            num[1] = 1. - num[0]
        else:
            d_m = delay
            den = np.zeros(int(d_m)+1)
            den[int(d_m)] = 1
            num = den*0
            num[0] = 1.

        return num, den

    def plot_iirfilter_tf(self, num, den, fs, dm, nw, dw, freq, noplot, verbose):
        # Placeholder for the actual plotting of IIR filter transfer function
        pass

    def nyquist(self, NNN, DDD, no_margin):
        # Placeholder for the actual Nyquist stability criterion implementation
        pass

    @staticmethod
    def from_gain_and_ff(gain, ff=None, target_device_idx=None):
        '''Build an IIRFilterData object from a gain value/vector
        and an optional forgetting factor value/vector'''

        gain = np.array(gain)
        n = len(gain)

        if ff is None:
            ff = np.ones(n)
        elif len(ff) != n:
            ff = np.full(n, ff)
        else:
            ff = np.array(ff)

        # Filter initialization
        num = np.zeros((n, 2))
        ord_num = np.zeros(n)
        den = np.zeros((n, 2))
        ord_den = np.zeros(n)
        
        for i in range(n):
            num[i, 0] = 0
            num[i, 1] = gain[i]
            ord_num[i] = 2
            den[i, 0] = -ff[i]
            den[i, 1] = 1
            ord_den[i] = 2
        
        return IIRFilterData(ord_num, ord_den, num, den, target_device_idx=target_device_idx)


