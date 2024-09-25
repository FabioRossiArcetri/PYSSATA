 #+
 # NAME:
 #   int2iirfilter 
 # PURPOSE:
 #   this function returns an iirfilter object
 #   made by integrators                              
 # CATEGORY:
 #   AO simulation.
 # CALLING SEQUENCE:
 # function int2iirfilter, gain, ff=ff 
 # INPUTS:
 #   gain        gain vector
 # KEYWORD
 #   ff          forgetting factor vector
 # OUTPUTS:
 #   iirfilter   iirfilter object
 # COMMON BLOCKS:
 #   None.
 # SIDE EFFECTS:
 #   None.
 # RESTRICTIONS:
 #   None
 # MODIFICATION HISTORY:
 #   Created 24-SEP-2014 by Guido Agapito agapito@arcetri.astro.it
 #-

from pyssata.data_objects.iirfilter import IIRFilter

import numpy as np
from pyssata import cpuArray
from pyssata import float_dtype

def int2iirfilter(gain, ff=None, xp=None):
    n = len(gain)
    
    if ff is None:
        ff = xp.ones(n, dtype=float_dtype)
    elif len(ff) != n:
        ff = xp.full(n, ff, dtype=float_dtype)
    
    # Filter initialization
    num = xp.zeros((n, 2), dtype=float_dtype)
    ord_num = xp.zeros(n, dtype=float_dtype)
    den = xp.zeros((n, 2), dtype=float_dtype)
    ord_den = xp.zeros(n, dtype=float_dtype)
    ost = xp.zeros((n, 2), dtype=float_dtype)
    ist = xp.zeros((n, 2), dtype=float_dtype)
    
    for i in range(n):
        num[i, 0:2] = xp.array([0, cpuArray(gain)[i]], dtype=float_dtype)
        ord_num[i] = 2
        den[i, 0:2] = xp.array([-cpuArray(ff)[i], 1], dtype=float_dtype)
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

