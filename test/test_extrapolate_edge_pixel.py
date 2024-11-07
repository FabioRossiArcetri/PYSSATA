

import specula
specula.init(1)

import os
import unittest
import numpy as np
from astropy.io import fits

from specula.lib.extrapolate_edge_pixel import extrapolate_edge_pixel

class TestExtrapolateEdgePixel(unittest.TestCase):

    def test_extrapolage_edge_pixel(self):
        
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        phase = fits.getdata(os.path.join(datadir, 'phase.fits'))
        ref = fits.getdata(os.path.join(datadir, 'extrapolated1.fits'))
        mat1 = fits.getdata(os.path.join(datadir, 'extrapol_array_doExt2PixFalse.fits'), ext=0).astype(int)
        mat2 = fits.getdata(os.path.join(datadir, 'extrapol_array_doExt2PixFalse.fits'), ext=1).astype(int)
        
        test = extrapolate_edge_pixel(phase, mat1, mat2, xp=np)
        np.testing.assert_array_almost_equal(test, ref)

    def test_extrapolage_edge_pixel_doExt2Pix(self):
        
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        phase = fits.getdata(os.path.join(datadir, 'phase.fits'))
        ref = fits.getdata(os.path.join(datadir, 'extrapolated2.fits'))
        mat1 = fits.getdata(os.path.join(datadir, 'extrapol_array_doExt2PixTrue.fits'), ext=0).astype(int)
        mat2 = fits.getdata(os.path.join(datadir, 'extrapol_array_doExt2PixTrue.fits'), ext=1).astype(int)
        
        test = extrapolate_edge_pixel(phase, mat1, mat2, xp=np)
        np.testing.assert_array_almost_equal(test, ref)

