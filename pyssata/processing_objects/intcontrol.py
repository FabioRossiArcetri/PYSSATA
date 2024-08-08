
from pyssata.processing_objects.iircontrol import IIRControl
from pyssata.lib.int2iirfilter import int2iirfilter


class IntControl(IIRControl):
    def __init__(self, gain, ff=None, delay=0):
        iirfilter = int2iirfilter(gain, ff=ff)

        # Initialize IIRControl object
        super().__init__(iirfilter, delay=delay)

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

