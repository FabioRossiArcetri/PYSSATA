# -*- coding: utf-8 -*-

#########################################################
# PySimul project.
#
# who       when        what
# --------  ----------  ---------------------------------
# apuglisi  2019-09-20  Created
#
#########################################################

import numpy as np

def rebin2d(a, shape, sample=False):
    '''
    Replacement of IDL's rebin() function for 2d arrays.
    '''
    if a.shape == shape:
        return a

    if sample:
        slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,shape) ]
        idx = np.mgrid[slices].astype(np.int32)
        return a[tuple(idx)]

    else:
        M, N = a.shape
        m, n = shape

        if m<=M and n<=M:
            if (M//m != M/m) or (N//n != N/n):
                raise ValueError('Resampling by non-integer factors is not supported')
            return a.reshape((m,M//m,n,N//n)).mean(3).mean(1)
        elif m>=M and n>=M:
            raise ValueError('Upsampling with sample=False is not supported')
        else:
            raise ValueError('Upsampling and downsampling in different axes it not supported')
