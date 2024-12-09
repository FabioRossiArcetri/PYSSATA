
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputList


class LinearCombination(BaseProcessingObj):
    def __init__(self,
                 target_device_idx: int = None, 
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)       

        self.inputs['in_vectors_list'] = InputList(type=BaseValue)
        
        self._out_vector = BaseValue()
        self.outputs['out_vector'] = self._out_vector