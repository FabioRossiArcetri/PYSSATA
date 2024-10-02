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

def int2iirfilter(gain, ff=None, target_device_idx=None, precision=None):

    # This routine is called before a parent object can be initialized,
    # so get the xp and dtpye information from this IIRFilter instead.
    
    iirfilter = IIRFilter(target_device_idx=target_device_idx, precision=precision)
    xp = iirfilter.xp
    float_dtype = iirfilter.dtype

    gain = xp.array(gain)
    n = len(gain)

    if ff is None:
        ff = xp.ones(n, dtype=float_dtype)
    elif len(ff) != n:
        ff = xp.full(n, ff, dtype=float_dtype)
    else:
        ff = xp.array(ff)

    # Filter initialization
    num = xp.zeros((n, 2), dtype=float_dtype)
    ord_num = xp.zeros(n, dtype=float_dtype)
    den = xp.zeros((n, 2), dtype=float_dtype)
    ord_den = xp.zeros(n, dtype=float_dtype)
    ost = xp.zeros((n, 2), dtype=float_dtype)
    ist = xp.zeros((n, 2), dtype=float_dtype)
    
    for i in range(n):
        num[i, 0] = 0
        num[i, 1] = gain[i]
        ord_num[i] = 2
        den[i, 0] = -ff[i]
        den[i, 1] = 1
        ord_den[i] = 2
    
    # TODO property sets in IIRfilter do not work
    iirfilter.nfilter = n
    iirfilter._ordnum = ord_num
    iirfilter._ordden = ord_den
    iirfilter._num = num
    iirfilter._den = den
    iirfilter._ost = ost
    iirfilter._ist = ist

    return iirfilter

