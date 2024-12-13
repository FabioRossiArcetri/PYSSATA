import numpy as np

from specula import float_dtype
from specula.base_data_obj import BaseDataObj

from astropy.io import fits

class IIRFilter(BaseDataObj):
    def __init__(self, target_device_idx=None, precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.nfilter = 0
        self._ordnum = None
        self._ordden = None
        self._num = None
        self._den = None
        self._zeros = None
        self._poles = None
        self._gain = None

    @property
    def nfilter(self):
        return self._nfilter
    
    @nfilter.setter
    def nfilter(self, value):
        self._nfilter = value

    @property
    def ordnum(self):
        return self._ordnum
    
    @ordnum.setter
    def ordnum(self, ordnum):
        self._ordnum = self.xp.array(ordnum, dtype=int)

    @property
    def ordden(self):
        return self._ordden
    
    @ordden.setter
    def ordden(self, ordden):
        self._ordden = self.xp.array(ordden, dtype=int)    

    @property
    def num(self):
        return self._num
    
    @num.setter
    def num(self, num):
        self.set_num(num)    

    @property
    def den(self):
        return self._den
    
    @den.setter
    def den(self, den):
        self.set_den(den)    

    @property
    def zeros(self):
        return self._zeros
    
    @zeros.setter
    def zeros(self, zeros):
        self.set_zeros(zeros)    

    @property
    def poles(self):
        return self._poles
    
    @poles.setter
    def poles(self, poles):
        self.set_poles(poles) 

    @property
    def gain(self):
        return self._gain
    
    @gain.setter
    def gain(self, gain):
        self.set_gain(gain) 

    def set_num(self, num):
        snum1 = num.shape[1]
        for i in range(self._nfilter):
            if self._ordnum[i] < snum1:
                if np.sum(self.xp.abs(num[i, int(self._ordnum[i]):])) == 0:
                    num[i, :] = self.xp.roll(num[i, :], snum1 - int(self._ordnum[i]))

        self._num = self.xp.array(num, dtype=float_dtype)
        zeros = self.xp.zeros((self._nfilter, snum1 - 1), dtype=float_dtype)
        gain = self.xp.zeros(self._nfilter, dtype=float_dtype)
        for i in range(self._nfilter):
            if self._ordnum[i] > 1:
                roots = self.xp.roots(self._num[i, snum1 - int(self._ordnum[i]):])
                if np.sum(np.abs(roots)) > 0:
                    zeros[i, :int(self._ordnum[i]) - 1] = roots
            gain[i] = self._num[i, - 1]
        self._zeros = zeros
        self._gain = gain

    def set_den(self, den):
        sden1 = den.shape[1]
        for i in range(self._nfilter):
            if self._ordden[i] < sden1:
                if np.sum(self.xp.abs(den[i, int(self._ordden[i]):])) == 0:
                    den[i, :] = self.xp.roll(den[i, :], sden1 - int(self._ordden[i]))

        self._den = self.xp.array(den, dtype=float_dtype)
        poles = self.xp.zeros((self._nfilter, sden1 - 1), dtype=float_dtype)
        for i in range(self._nfilter):
            if self._ordden[i] > 1:
                poles[i, :int(self._ordden[i]) - 1] = self.xp.roots(self._den[i, sden1 - int(self._ordden[i]):])
        self._poles = poles

    def set_zeros(self, zeros):
        self._zeros = self.xp.array(zeros, dtype=float_dtype)
        num = self.xp.zeros((self._nfilter, self._zeros.shape[1] + 1), dtype=float_dtype)
        snum1 = num.shape[1]
        for i in range(self._nfilter):
            if self._ordnum[i] > 1:
                num[i, snum1 - int(self._ordnum[i]):] = self.xp.poly(self._zeros[i, :int(self._ordnum[i]) - 1])
        self._num = num

    def set_poles(self, poles):
        self._poles = self.xp.array(poles, dtype=float_dtype)
        den = self.xp.zeros((self._nfilter, self._poles.shape[1] + 1), dtype=float_dtype)
        sden1 = den.shape[1]
        for i in range(self._nfilter):
            if self._ordden[i] > 1:
                den[i, sden1 - int(self._ordden[i]):] = self.xp.poly(self._poles[i, :int(self._ordden[i]) - 1])
        self._den = den

    def set_gain(self, gain, verbose=False):
        if verbose:
            print('original gain:', self._gain)
        if self.xp.size(gain) < self._nfilter:
            nfilter = np.size(gain)
        else:
            nfilter = self._nfilter
        if self._gain is None:
            for i in range(nfilter):
                if self.xp.isfinite(gain[i]):
                    if self._ordnum[i] > 1:
                        self._num[i, :] *= gain[i]
                    else:
                        self._num[i, - 1] = gain[i]
                else:
                    gain[i] = self._num[i, - 1]
        else:
            for i in range(nfilter):
                if self.xp.isfinite(gain[i]):
                    if self._ordnum[i] > 1:
                        self._num[i, :] *= (gain[i] / self._gain[i])
                    else:
                        self._num[i, - 1] = gain[i] / self._gain[i]
                else:
                    gain[i] = self._gain[i]
        self._gain = self.xp.array(gain, dtype=float_dtype)
        if verbose:
            print('new gain:', self._gain)

    def complexRTF(self, mode, fs, delay, freq=None, verbose=False):
        if delay > 1:
            dm = self.xp.array([0.0, 1.0], dtype=float_dtype)
            nm = self.xp.array([1.0, 0.0], dtype=float_dtype)
            wTf = self.discrete_delay_tf(delay - 1)
        else:
            dm = self.xp.array([1.0], dtype=float_dtype)
            nm = self.xp.array([1.0], dtype=float_dtype)
            wTf = self.discrete_delay_tf(delay)
        nw = wTf[:, 0]
        dw = wTf[:, 1]
        complex_yt_tf = self.plot_iirfilter_tf(self._num[mode, :], self._den[mode, :], fs, dm=dm, nw=nw, dw=dw, freq=freq, noplot=True, verbose=verbose)
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

    def is_stable(self, mode, nm=None, dm=None, nw=None, dw=None, gain=None, no_margin=False, verbose=False):
        nm = nm if nm is not None else self.xp.array([1, 0], dtype=float_dtype)
        nw = nw if nw is not None else self.xp.array([1, 0], dtype=float_dtype)
        dm = dm if dm is not None else self.xp.array([0, 1], dtype=float_dtype)
        dw = dw if dw is not None else self.xp.array([0, 1], dtype=float_dtype)

        temp1 = self.xp.polymul(dm, dw)
        while temp1[-1] == 0:
            temp1 = temp1[:-1]
        DDD = self.xp.polymul(temp1, self._den[mode, :])
        while DDD[-1] == 0:
            DDD = DDD[:-1]

        temp2 = self.xp.polymul(nm, nw)
        while temp2[-1] == 0:
            temp2 = temp2[:-1]
        NNN = self.xp.polymul(temp2, self._num[mode, :])
        if self.xp.sum(self.xp.abs(NNN)) != 0:
            while NNN[-1] == 0:
                NNN = NNN[:-1]

        if gain is not None:
            NNN *= gain / self._gain[mode]

        stable, ph_margin, g_margin, mroot, m_one_dist = self.nyquist(NNN, DDD, no_margin=no_margin)

        if verbose:
            print('max root (closed loop) =', mroot)
            print('phase margin =', ph_margin)
            print('gain margin =', g_margin)
            print('min. distance from (-1;0) =', m_one_dist)
        return stable

    def save(self, filename, hdr=None):
        hdr = hdr if hdr is not None else fits.Header()
        hdr['VERSION'] = 1

        hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=self._ordnum, name='ORDNUM'))
        hdul.append(fits.ImageHDU(data=self._ordden, name='ORDDEN'))
        hdul.append(fits.ImageHDU(data=self._num, name='NUM'))
        hdul.append(fits.ImageHDU(data=self._den, name='DEN'))
        hdul.writeto(filename, overwrite=True)

    def read(self, filename, hdr=None, exten=0):
        hdul = fits.open(filename)
        hdr = hdul[0].header
        self._ordnum = hdul['ORDNUM'].data
        self._ordden = hdul['ORDDEN'].data
        self._nfilter = len(self._ordnum)
        self.set_num(hdul['NUM'].data)
        self.set_den(hdul['DEN'].data)

    def restore(self, filename):
        obj = IIRFilter()
        obj.read(filename)
        return obj

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
