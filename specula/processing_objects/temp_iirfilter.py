
from specula.processing_objects.iircontrol import IIRControl
from specula.lib.int2iirfilter import int2iirfilter
import numpy as np

from specula import float_dtype
    
class TempIIRFilter(IIRControl):
    def __init__(self, num_coeffs, den_coeffs, ff=None, delay=0, offset=None, og_shaper=None,                 
                target_device_idx=None, 
                precision=None
                ):       
         
        # TODO initialize an iirfilter from num_coeffs and den_coeffs
        iirfilter = int2iirfilter(gain=num_coeffs, ff=ff, target_device_idx=target_device_idx, precision=precision)

        # Initialize IIRControl object
        super().__init__(iirfilter, delay=delay, target_device_idx=target_device_idx, precision=precision)
        
        if offset is not None:
            self._offset = offset
        if og_shaper is not None:
            self._og_shaper = og_shaper

    @property
    def ff(self):
        return self._iirfilter.poles
