
from pyssata.processing_objects.iircontrol import IIRControl
from pyssata.lib.int2iirfilter import int2iirfilter
import numpy as np

from pyssata import float_dtype
    
class IntControl(IIRControl):
    def __init__(self, int_gain, ff=None, delay=0, offset=None, og_shaper=None,                 
                target_device_idx=None, 
                precision=None
                ):        
        iirfilter = int2iirfilter(np.array(int_gain, dtype=float_dtype), ff=ff, xp=np)

        # Initialize IIRControl object
        super().__init__(iirfilter, delay=delay,target_device_idx=target_device_idx, precision=precision)
        
        if offset is not None:
            self._offset = offset
        if og_shaper is not None:
            self._og_shaper = og_shaper

    @property
    def ff(self):
        return self._iirfilter.poles

    def run_check(self, time_step, errmsg=""):
        return True

