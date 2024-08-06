class BaseParameterObj:
    def __init__(self):
        """
        Initialize the base parameter object.
        """
        pass

    def apply_properties(self, properties, ignore_extra_keywords=False):
        """
        Apply properties to the object.

        Parameters:
        properties (dict): Dictionary of properties to apply
        ignore_extra_keywords (bool, optional): Flag to ignore extra keywords
        """
        if not hasattr(self, 'setproperty'):
            raise AttributeError(f"{self} Error: this object does not have a setproperty method")

        if not isinstance(properties, dict):
            raise TypeError(f"{self} Error: properties must be either a dictionary")

        # Assuming setproperty method exists and can handle the properties
        self.setproperty(**properties)

    def get_properties_list(self):
        """
        Get a list of properties of the object.

        Returns:
        dict: Dictionary of properties
        """
        return vars(self)

    def setproperty(self, **kwargs):
        """
        Placeholder for setproperty method, to be overridden by derived classes.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

# Define a dummy parameter object attribute as in the IDL struct
class DummyParameterObj(BaseParameterObj):
    def __init__(self):
        super().__init__()
        self._dummy_parameter_obj = 0
