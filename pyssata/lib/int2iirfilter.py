import numpy as np


def int2iirfilter(gain, ff=None):
    n = len(gain)
    
    if ff is None:
        ff = np.ones(n, dtype=np.float64)
    elif len(ff) != n:
        ff = np.full(n, ff, dtype=np.float64)
    
    # Filter initialization
    num = np.zeros((n, 2), dtype=np.float64)
    ord_num = np.zeros(n, dtype=np.float64)
    den = np.zeros((n, 2), dtype=np.float64)
    ord_den = np.zeros(n, dtype=np.float64)
    ost = np.zeros((n, 2), dtype=np.float64)
    ist = np.zeros((n, 2), dtype=np.float64)
    
    for i in range(n):
        num[i, 0:2] = [0, gain[i]]
        ord_num[i] = 2
        den[i, 0:2] = [-ff[i], 1]
        ord_den[i] = 2
    
    iirfilter = IIRFilter()
    
    iirfilter.nfilter = n
    iirfilter.ordnum = ord_num
    iirfilter.ordden = ord_den
    iirfilter.num = num
    iirfilter.den = den
    iirfilter.ost = ost
    iirfilter.ist = ist

    return iirfilter

