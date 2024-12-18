

import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.iir_filter_data import IIRFilterData
from specula.processing_objects.iir_filter import IIRFilter

from test.specula_testlib import cpu_and_gpu

class TestIIRFilter(unittest.TestCase):
   
    # Building an IIRFilter object will use or not use numba
    # depending on the Python version, so we just check that it goes through.
    @cpu_and_gpu
    def test_iir_filter_instantiation(self, target_device_idx, xp):
        iir_filter = IIRFilterData(ordnum=(1,1), ordden=(1,1), num=xp.ones((2,2)), den=xp.ones((2,2)),
                                   target_device_idx=target_device_idx)
        iir_control = IIRFilter(iir_filter)

