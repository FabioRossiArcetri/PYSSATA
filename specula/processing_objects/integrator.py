
from specula.processing_objects.iir_filter import IIRFilter
from specula.data_objects.iir_filter_data import IIRFilterData

    
class Integrator(IIRFilter):
    def __init__(self, int_gain, ff=None, delay=0, offset=None, og_shaper=None,                 
                target_device_idx=None, 
                precision=None
                ):
        
        iir_filter_data = IIRFilterData.from_gain_and_ff(int_gain, ff=ff,
                                               target_device_idx=target_device_idx)

        # Initialize IIRFilter object
        super().__init__(iir_filter_data, delay=delay, offset=offset, og_shaper=og_shaper,
                         target_device_idx=target_device_idx, precision=precision)
