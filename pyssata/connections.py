

class InputValue():
    def __init__(self, type):
        """
        Wrapper for simple input values
        """
        self.wrapped_type = type
        self.wrapped_value = None

    def get(self):
        return self.wrapped_value
    
    def set(self, value):
        if not isinstance(value, self.wrapped_type):
            raise ValueError(f'Value must be of type {self.wrapped_type}')
        self.wrapped_value = value
    
    def type(self):
        return self.wrapped_type


class InputList():
    def __init__(self, type):
        """
        Wrapper for input lists
        """
        self.wrapped_type = type
        self.wrapped_list = None

    def get(self):
        return self.wrapped_list
    
    def set(self, new_list):
        for value in new_list:
            if not isinstance(value, self.wrapped_type):
                raise ValueError(f'List element must be of type {self.wrapped_type}')
        self.wrapped_list = new_list

    def type(self):
        return self.wrapped_type
