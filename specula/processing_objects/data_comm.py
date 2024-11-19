from astropy.io import fits

import os
import numpy as np

from specula import xp
from collections import OrderedDict
import pickle
import yaml
import time

from specula import cpuArray
from specula import process_rank
from specula import process_comm
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.base_data_obj import BaseDataObj
from specula.connections import InputValue, InputList
from specula.data_objects.ef import ElectricField
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.data_objects.layer import Layer


class DataComm(BaseProcessingObj):
    '''Data communication object'''

    def __init__(self,
                comm_port_in: str,
                comm_port_out: str,
                outputs_origin_proc: list,
                items_to_receive: list
                ):
        super().__init__()
        self.items_to_send = {}
        self.storage = {}
        self.decimation_t = 0
        prpp = comm_port_in.split(':')
        self.peer_rank, self.peer_port = int(prpp[0]), int(prpp[1])
        self.out_port = int(comm_port_out)
        self.outputs_origin_proc = outputs_origin_proc

        self.items_to_receive = items_to_receive
        self.outputs['out_layer'] = Layer(160, 160, 1, 0)
        self.outputs['out_on_axis_source_ef'] = ElectricField(160, 160, 1)

    def add(self, data_obj, name=None):
        if name is None:
            name = data_obj.__class__.__name__
        if name in self.items_to_send:
            raise ValueError(f'Communication already has an object with name {name}')
        self.items_to_send[name] = data_obj        

    def trigger_code(self):
        send_reqs = []
        recv_reqs = []
        tag_idx = -1
        print('Process ', process_rank, ' sending requests')
        for k, item in self.items_to_send.items():
            tag_idx += 1
            if item is not None and item.generation_time == self.current_time:
                # here send item to the peer
                sreq = process_comm.isend(item, dest=self.peer_rank, tag=tag_idx)
                sreq.wait()
                # send_reqs.append(sreq)
                
        print('Process ', process_rank, ' receiving requests')
        # here receive from the peer        
        tag_idx = -1
        for k in self.items_to_receive:
            tag_idx += 1
            rreq = process_comm.irecv(source=self.outputs_origin_proc[tag_idx], tag=tag_idx)
            self.outputs[k] = rreq.wait()
            # recv_reqs.append(rreq)
            
#        print('Process ', process_rank, ' waiting to send')
#        for sreq in send_reqs:
#            sreq.wait()
#        print('Process ', process_rank, ' waiting to receive')
#        tag_idx = -1
#        for k in self.items_to_receive:
#            tag_idx += 1            
#            self.outputs[k] = recv_reqs[tag_idx].wait()             

        print('Process ', process_rank, ' step done')

    def run_check(self, time_step, errmsg=''):
        return True

    def finalize(self):
        pass
