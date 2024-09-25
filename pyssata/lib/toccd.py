# -*- coding: utf-8 -*-
#########################################################
# PySimul project.
#
# who       when        what
# --------  ----------  ---------------------------------
# apuglisi  2019-09-28  Created
#
#########################################################


from pyssata.lib.rebin import rebin2d

def gcd(a,b):
    '''
    Returns the greatest common divisor of a and b.
    '''
    while b:
        a, b = b, a % b

    return a

def lcm(a,b):
    '''
    Returns the least common multiple of a and b.
    '''
    return (a*b) // gcd(a,b)


def toccd(a, newshape, set_total=None, xp=None):
    '''
    Clone of oaalib's toccd() function, using least common multiple
    to rebin an array similar to openvc's INTER_AREA interpolation.
    '''
    if a.shape == newshape:
        return a

    if len(a.shape) != 2:
        raise ValueError('Input array has shape %s, cannot continue' % str(a.shape))

    if len(newshape) != 2:
        raise ValueError('Output shape is %s, cannot continue' % str(newshape))

    if set_total is None:
        set_total = a.sum()

    mcmx = lcm(a.shape[0], newshape[0])
    mcmy = lcm(a.shape[1], newshape[1])

    temp = rebin2d(a, (mcmx, a.shape[1]), sample=True, xp=xp)
    temp = rebin2d(temp, (newshape[0], a.shape[1]), xp=xp)
    temp = rebin2d(temp, (newshape[0], mcmy), sample=True, xp=xp)
    rebinned = rebin2d(temp, newshape, xp=xp)

    return rebinned / rebinned.sum() * set_total