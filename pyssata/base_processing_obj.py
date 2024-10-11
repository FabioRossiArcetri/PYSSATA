from astropy.io import fits

from pyssata.base_time_obj import BaseTimeObj
from pyssata.base_parameter_obj import BaseParameterObj
from pyssata import default_target_device, cp
from pyssata.connections import InputValue, InputList

class BaseProcessingObj(BaseTimeObj, BaseParameterObj):
    def __init__(self, target_device_idx=None, precision=None):
        """
        Initialize the base processing object.

        Parameters:
        precision (int, optional): if None will use the global_precision, otherwise pass 0 for double, 1 for single
        target_device_idx (int, optional): if None will use the default_target_device_idx, otherwise pass -1 for cpu, i for GPU of index i
        """
        BaseTimeObj.__init__(self, target_device_idx=target_device_idx, precision=precision)

        if self._target_device_idx>=0:
            from cupyx.scipy.ndimage import rotate
            from cupyx.scipy.interpolate import RegularGridInterpolator
        else:
            from scipy.ndimage import rotate
            from scipy.interpolate import RegularGridInterpolator


        self.rotate = rotate        
        self.RegularGridInterpolator = RegularGridInterpolator

        BaseParameterObj.__init__(self)

        self.current_time = 0
        self.current_time_seconds = 0

        self._verbose = 0
        self._loop_dt = int(0)
        self._loop_niters = 0
        
        # Will be populated by derived class
        self.inputs = {}
        self.local_inputs = {}
        self.outputs = {}
        self.stream  = None

    def checkInputTimes(self):        
        if len(self.inputs)==0:
            return True
        for input_obj in self.inputs.values():            
            if type(input_obj) is InputValue:
                if input_obj.get_time() == self.current_time:
                    return True
            elif type(input_obj) is InputList:
                for tt in input_obj.get_time():
                    if tt == self.current_time:
                        return True
        return False

    def prepare_trigger(self, t):                
        self.current_time_seconds = self.t_to_seconds(self.current_time)
        for input_name, input_obj in self.inputs.items():
            if type(input_obj) is InputValue:
                self.local_inputs[input_name] =  input_obj.get(self._target_device_idx)
            elif type(input_obj) is InputList:
                self.local_inputs[input_name] = []
                for tt in input_obj.get(self._target_device_idx):
                    self.local_inputs[input_name].append(tt)
        
    def trigger_code(self):
        pass

    def build_stream(self):
        if self._target_device_idx>=0:
            #self.prepare_trigger(0)
            self._target_device.use()
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.capture_stream()
            default_target_device.use()

    def capture_stream(self):
        with self.stream:
            self.stream.begin_capture()
            self.trigger_code()
            self.cuda_graph = self.stream.end_capture()

    def trigger(self, t):
        self.current_time = t
        if self.checkInputTimes():
            self.prepare_trigger(t)
            if self._target_device_idx>=0 and self.cuda_graph:
                self._target_device.use()
                self.cuda_graph.launch(stream=self.stream)
                self.stream.synchronize()
                default_target_device.use()
            else:
                self.trigger_code()
        else:
            if self.verbose:
                print(f'No inputs have been refreshed, skipping trigger')
                    
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

    def save(self, filename):
        hdr = fits.Header()
        hdr['VERBOSE'] = self._verbose
        hdr['LOOP_DT'] = self._loop_dt
        hdr['LOOP_NITERS'] = self._loop_niters
        super().save(filename)
        with fits.open(filename, mode='update') as hdul:
            hdr = hdul[0].header
            hdr['VERBOSE'] = self._verbose
            hdr['LOOP_DT'] = self._loop_dt
            hdr['LOOP_NITERS'] = self._loop_niters
            hdul.flush()

    def read(self, filename):
        super().read(filename)
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            self._verbose = hdr.get('VERBOSE', 0)
            self._loop_dt = hdr.get('LOOP_DT', int(0))
            self._loop_niters = hdr.get('LOOP_NITERS', 0)

