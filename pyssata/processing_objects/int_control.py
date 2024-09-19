
from pyssata.processing_objects.iircontrol import IIRControl
from pyssata.lib.int2iirfilter import int2iirfilter
import numpy as np

from pyssata import xp
from pyssata import standard_dtype
    
class IntControl(IIRControl):
    def __init__(self, int_gain, ff=None, delay=0, offset=None, og_shaper=None):
        
        iirfilter = int2iirfilter(xp.array(int_gain, dtype=standard_dtype), ff=ff)

        # Initialize IIRControl object
        super().__init__(iirfilter, delay=delay)
        
        if offset is not None:
            self._offset = offset
        if og_shaper is not None:
            self._og_shaper = og_shaper

    @property
    def ff(self):
        return self._iirfilter.poles

    @staticmethod
    def revision_track():
        return "$Rev$"

    def cleanup(self):
        super().cleanup()

    def run_check(self, time_step, errmsg=""):
        return True

