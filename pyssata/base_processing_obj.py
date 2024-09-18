from astropy.io import fits

from pyssata.base_time_obj import BaseTimeObj
from pyssata.base_parameter_obj import BaseParameterObj

class BaseProcessingObj(BaseTimeObj, BaseParameterObj):
    def __init__(self, precision=0):
        """
        Initialize the base processing object.

        Parameters:
        precision (int, optional): double 1 or single 0, defaults to single precision
        """
        BaseTimeObj.__init__(self, precision)
        BaseParameterObj.__init__(self)
        self._verbose = 0
        self._loop_dt = int(0)
        self._loop_niters = 0
        
        # Will be populated by derived class
        self.inputs = {}
        self.outputs = {}

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

    def trigger(self, t):
        """
        Must be implemented by derived classes.
        """
        raise NotImplementedError("Derived class must implement this method")

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

