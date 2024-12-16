

import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.iirfilter import IIRFilter
from specula.processing_objects.iircontrol import IIRControl

from test.specula_testlib import cpu_and_gpu

class TestIIRControl(unittest.TestCase):
   
    # Building an IIRControl object will use or not use numba
    # depending on the Python version, so we just check that it goes through.
    @cpu_and_gpu
    def test_iircontrol_instantiation(self, target_device_idx, xp):
        iir_filter = IIRFilter()
        iir_filter.num = np.ones((2,2))
        iir_control = IIRControl(iir_filter)

