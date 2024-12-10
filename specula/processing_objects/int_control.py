
from specula.processing_objects.iircontrol import IIRControl
from specula.data_objects.iirfilter import IIRFilter

    
class IntControl(IIRControl):
    def __init__(self, int_gain, ff=None, delay=0, offset=None, og_shaper=None,                 
                target_device_idx=None, 
                precision=None
                ):
        
        iirfilter = IIRFilter.from_gain_and_ff(int_gain, ff=ff,
                                               target_device_idx=target_device_idx)

        # Initialize IIRControl object
        super().__init__(iirfilter, delay=delay, offset=offset, og_shaper=og_shaper,
                         target_device_idx=target_device_idx, precision=precision)
