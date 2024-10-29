from astropy.io import fits

from pyssata.base_time_obj import BaseTimeObj
from pyssata import default_target_device, cp
from pyssata.connections import InputValue, InputList
from contextlib import nullcontext

class BaseProcessingObj(BaseTimeObj):
    def __init__(self, target_device_idx=None, precision=None):
        """
        Initialize the base processing object.

        Parameters:
        precision (int, optional): if None will use the global_precision, otherwise pass 0 for double, 1 for single
        target_device_idx (int, optional): if None will use the default_target_device_idx, otherwise pass -1 for cpu, i for GPU of index i
        """
        BaseTimeObj.__init__(self, target_device_idx=target_device_idx, precision=precision)

        if self.target_device_idx>=0:
            from cupyx.scipy.ndimage import rotate
            from cupyx.scipy.interpolate import RegularGridInterpolator
            from cupyx.scipy.fft import get_fft_plan
        else:
            from scipy.ndimage import rotate
            from scipy.interpolate import RegularGridInterpolator
            get_fft_plan = None


        self.rotate = rotate        
        self.RegularGridInterpolator = RegularGridInterpolator
        self._get_fft_plan = get_fft_plan

        self.current_time = 0
        self.current_time_seconds = 0

        self._verbose = 0
        self._loop_dt = int(0)
        self._loop_niters = 0
        
        # Will be populated by derived class
        self.inputs = {}
        self.local_inputs = {}
        self.last_seen = {}
        self.outputs = {}
        self.stream  = None
        self.ready = False

    def get_fft_plan(self, a, shape=None, axes=None, value_type='C2C'):
        if self._get_fft_plan:
            return self._get_fft_plan(a, shape, axes, value_type)
        else:
            return nullcontext()

    def checkInputTimes(self):        
        if len(self.inputs)==0:
            return True
        for input_name, input_obj in self.inputs.items():
            if type(input_obj) is InputValue:
                if input_name not in self.last_seen and input_obj.get_time() is not None and input_obj.get_time() >= 0:  # First time
                    return True
                if input_name in self.last_seen and input_obj.get_time() > self.last_seen[input_name]:
                    return True
            elif type(input_obj) is InputList:
                if input_name not in self.last_seen:
                    for tt in input_obj.get_time():
                        if tt >= 0:
                            return True
#                if input_name not in self.last_seen and input_obj.get_time() >= 0:  # First time
#                    return True
                for tt, last in zip(input_obj.get_time(), self.last_seen[input_name]):
                    if tt > last:
                        return True
        return False

    def prepare_trigger(self, t):
        self.current_time_seconds = self.t_to_seconds(self.current_time)
        for input_name, input_obj in self.inputs.items():
            if type(input_obj) is InputValue:
                self.local_inputs[input_name] =  input_obj.get(self.target_device_idx)
                if self.local_inputs[input_name] is not None:
                    self.last_seen[input_name] = self.local_inputs[input_name].generation_time
            elif type(input_obj) is InputList:
                self.local_inputs[input_name] = []
                self.last_seen[input_name] = []
                for tt in input_obj.get(self.target_device_idx):
                    self.local_inputs[input_name].append(tt)
                    if self.local_inputs[input_name] is not None:
                        self.last_seen[input_name].append(tt.generation_time)

    def trigger_code(self):
        pass

    def post_trigger(self):
        if self.target_device_idx>=0 and self.cuda_graph:
            self.stream.synchronize()

#        if self.checkInputTimes():
#         if self.target_device_idx>=0 and self.cuda_graph:
#             self.stream.synchronize()
#             self._target_device.synchronize()
#             self.xp.cuda.runtime.deviceSynchronize()
## at the end of the derevide method should call this?
#            default_target_device.use()
#            self.xp.cuda.runtime.deviceSynchronize()                
#            cp.cuda.Stream.null.synchronize()

    def build_stream(self):
        if self.target_device_idx>=0:
            self._target_device.use()
            self.stream = cp.cuda.Stream(non_blocking=False)
            self.capture_stream()
            default_target_device.use()

    def capture_stream(self):
        with self.stream:
            self.stream.begin_capture()
            self.trigger_code()
            self.cuda_graph = self.stream.end_capture()

    def check_ready(self, t):
        self.current_time = t
        if self.checkInputTimes():
            if self.target_device_idx>=0:
                self._target_device.use()
            self.prepare_trigger(t)
            self.ready = True
        else:
            if self.verbose:
                print(f'No inputs have been refreshed, skipping trigger')
        return self.ready
    
    def trigger(self):        
        if self.ready:
            if self.target_device_idx>=0 and self.cuda_graph:
                self.cuda_graph.launch(stream=self.stream)
            else:
                self.trigger_code()
            self.ready = False
                    
    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    @property
    def loop_dt(self):
        return self._loop_dt

    @loop_dt.setter
    def loop_dt(self, value):
        self._loop_dt = value

    @property
    def loop_niters(self):
        return self._loop_niters

    @loop_niters.setter
    def loop_niters(self, value):
        self._loop_niters = value


    def run_check(self, time_step, errmsg=None):
        """
        Must be implemented by derived classes.

        Parameters:
        time_step (int): The time step for the simulation
        errmsg (str, optional): Error message
        """
        print(f"Problem with {self}: please implement run_check() in your derived class!")
        return 1

    def finalize(self):
        '''
        Override this method to perform any actions after
        the simulation is completed
        '''
        pass

    def save(self, filename):
        hdr = fits.Header()
        hdr['VERBOSE'] = self._verbose
        hdr['LOOP_DT'] = self._loop_dt
        hdr['LOOP_NITERS'] = self._loop_niters        
        with fits.open(filename, mode='update') as hdul:
            hdr = hdul[0].header
            hdr['VERBOSE'] = self._verbose
            hdr['LOOP_DT'] = self._loop_dt
            hdr['LOOP_NITERS'] = self._loop_niters
            hdul.flush()

    def read(self, filename):        
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            self._verbose = hdr.get('VERBOSE', 0)
            self._loop_dt = hdr.get('LOOP_DT', int(0))
            self._loop_niters = hdr.get('LOOP_NITERS', 0)

