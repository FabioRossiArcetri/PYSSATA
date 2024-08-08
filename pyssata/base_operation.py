from astropy.io import fits

from pyssata.base_processing_obj import BaseProcessingObj
from pyssata.base_value import BaseValue

class BaseOperation(BaseProcessingObj):
    def __init__(self, constant_mult=None, constant_div=None, constant_sum=None, constant_sub=None, mult=False, div=False, sum=False, sub=False):
        """
        Initialize the base operation object.

        Parameters:
        constant_mult (float, optional): Constant for multiplication
        constant_div (float, optional): Constant for division
        constant_sum (float, optional): Constant for addition
        constant_sub (float, optional): Constant for subtraction
        mult (bool, optional): Flag for multiplication operation
        div (bool, optional): Flag for division operation
        sum (bool, optional): Flag for addition operation
        sub (bool, optional): Flag for subtraction operation
        """
        super().__init__('base_operation', 'Simple operations with base value(s)')

        self._constant_mult = constant_mult
        self._constant_div = 1.0 / constant_div if constant_div is not None else None
        self._constant_sum = constant_sum
        self._constant_sub = -constant_sub if constant_sub is not None else None

        self._mult = mult
        self._div = div
        self._sum = sum
        self._sub = sub

        self._in_value1 = None
        self._in_value2 = None
        self._out_value = BaseValue()

    @property
    def in_value1(self):
        return self._in_value1

    @in_value1.setter
    def in_value1(self, value):
        self._in_value1 = value

    @property
    def in_value2(self):
        return self._in_value2

    @in_value2.setter
    def in_value2(self, value):
        self._in_value2 = value

    @property
    def out_value(self):
        return self._out_value

    def trigger(self, t):
        if self._in_value1 and self._in_value1.generation_time == t:
            if self._constant_mult is not None:
                self._out_value.value = self._in_value1.value * self._constant_mult
            if self._constant_sum is not None:
                self._out_value.value = self._in_value1.value + self._constant_sum
            self._out_value.generation_time = t

        if self._in_value1 and self._in_value2 and (self._in_value1.generation_time == t or self._in_value2.generation_time == t):
            if self._mult:
                self._out_value.value = self._in_value1.value * self._in_value2.value
            if self._div:
                self._out_value.value = self._in_value1.value / self._in_value2.value
            if self._sum:
                self._out_value.value = self._in_value1.value + self._in_value2.value
            if self._sub:
                self._out_value.value = self._in_value1.value - self._in_value2.value
            self._out_value.generation_time = t

    def run_check(self, time_step, errmsg=None):
        """
        Check the validity of the operation.

        Parameters:
        time_step (int): The time step for the simulation
        errmsg (str, optional): Error message

        Returns:
        bool: True if the check is successful, False otherwise
        """
        return self._out_value is not None and self._in_value1 is not None

    def save(self, filename):
        hdr = fits.Header()
        hdr['CONST_MULT'] = self._constant_mult
        hdr['CONST_DIV'] = self._constant_div
        hdr['CONST_SUM'] = self._constant_sum
        hdr['CONST_SUB'] = self._constant_sub
        hdr['MULT'] = self._mult
        hdr['DIV'] = self._div
        hdr['SUM'] = self._sum
        hdr['SUB'] = self._sub
        super().save(filename)
        with fits.open(filename, mode='update') as hdul:
            hdr = hdul[0].header
            hdr['CONST_MULT'] = self._constant_mult
            hdr['CONST_DIV'] = self._constant_div
            hdr['CONST_SUM'] = self._constant_sum
            hdr['CONST_SUB'] = self._constant_sub
            hdr['MULT'] = self._mult
            hdr['DIV'] = self._div
            hdr['SUM'] = self._sum
            hdr['SUB'] = self._sub
            hdul.flush()

    def read(self, filename):
        super().read(filename)
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            self._constant_mult = hdr.get('CONST_MULT', None)
            self._constant_div = hdr.get('CONST_DIV', None)
            self._constant_sum = hdr.get('CONST_SUM', None)
            self._constant_sub = hdr.get('CONST_SUB', None)
            self._mult = hdr.get('MULT', False)
            self._div = hdr.get('DIV', False)
            self._sum = hdr.get('SUM', False)
            self._sub = hdr.get('SUB', False)

    def cleanup(self):
        pass
