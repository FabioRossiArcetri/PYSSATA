
from pyssata.processing_objects.iircontrol import IIRControl


class IntControl(IIRControl):
    def __init__(self, gain, ff=None, delay=0):
        iirfilter = int2iirfilter(gain, ff=ff)
        
        # Initialize IIRControl object
        if not super().__init__(iirfilter, delay=delay):
            raise ValueError("Initialization of IIRControl failed")
        
        # Initialize BaseProcessingObj
        if not BaseProcessingObj.Init(self, 'intcontrol', 'Integrator based Time Control'):
            raise ValueError("Initialization of BaseProcessingObj failed")

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

# Define the necessary functions used in IntControl
def int2iirfilter(gain, ff=None):
    # Implementation of the int2iirfilter function goes here
    pass
