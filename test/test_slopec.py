

import specula
specula.init(1)

import unittest
import numpy as np

from specula import xp
from specula import cpuArray

from specula.data_objects.pixels import Pixels
from specula.data_objects.pupdata import PupData
from specula.processing_objects.pyr_slopec import PyrSlopec


class TestSlopec(unittest.TestCase):

    def test_slopec(self):
        pixels = Pixels(4, 2)
        pixels.pixels = xp.arange(25,  dtype=xp.uint16).reshape((5,5))
        pixels.generation_time = 1
        pupdata = PupData()
        pupdata.ind_pup = xp.array([[1,3,6,8], [15,16,21,24]], dtype=int)
        slopec = PyrSlopec(pupdata)
        slopec.in_pixels = pixels
        slopec.trigger(1)
        slopes = slopec.out_slopes
        
        s1 = cpuArray(slopes.slopes)
        np.testing.assert_array_almost_equal(s1, np.array([-0.21276595, -0.29787233,  0. , -0.04255319]))
