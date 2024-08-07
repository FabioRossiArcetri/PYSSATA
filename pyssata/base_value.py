class BaseValue(BaseTimeObj):
    def __init__(self, objname="base_value", objdescr="Base data object for simple data", value=None):
        """
        Initialize the base value object.

        Parameters:
        objname (str, optional): object name
        objdescr (str, optional): object description
        value (any, optional): data to store. If not set, the value is initialized to None.
        """
        super().__init__(objname, objdescr)
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

    @property
    def ptr_value(self):
        return self._value

    def cleanup(self):
        self._value = None
        super().cleanup()

    def save(self, filename):
        hdr = fits.Header()
        if self._value is not None:
            hdr['VALUE'] = str(self._value)  # Store as string for simplicity
        super().save(filename)
        with fits.open(filename, mode='update') as hdul:
            hdr = hdul[0].header
            if self._value is not None:
                hdr['VALUE'] = str(self._value)
            hdul.flush()

    def read(self, filename):
        super().read(filename)
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            value_str = hdr.get('VALUE', None)
            if value_str is not None:
                self._value = eval(value_str)  # Convert back from string to original type
