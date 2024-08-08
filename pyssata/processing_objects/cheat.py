
from pyssata.base_processing_obj import BaseProcessingObj


class Cheat(BaseProcessingObj):
    def __init__(self, expr, timeobj=None, objname="cheat", objdescr="Base data object for simple data", **kwargs):
        super().__init__(objname, objdescr)

        self._expr = expr
        self._timeobj = timeobj if timeobj else None
        self._refs = {}

        if kwargs:
            for name, obj in kwargs.items():
                self._refs[name] = obj
                if not self._timeobj:
                    self._timeobj = obj
                self._expr = self._expr.replace(f"{name}.", f"self._refs['{name}'].")

    def trigger(self, t):
        try:
            exec(self._expr)
        except Exception as e:
            raise ValueError(f"Error in expression: {self._expr}") from e

    @property
    def value(self):
        try:
            local_vars = {}
            exec(self._expr, globals(), local_vars)
            return local_vars.get('value')
        except Exception as e:
            raise ValueError(f"Error in expression: {self._expr}") from e

    @property
    def ptr_value(self):
        return self.value

    @property
    def generation_time(self):
        return self._timeobj.generation_time if self._timeobj else None

    @generation_time.setter
    def generation_time(self, value):
        if self._timeobj:
            self._timeobj.generation_time = value

    def set_property(self, **kwargs):
        super().set_property(**kwargs)
        if 'generation_time' in kwargs:
            self.generation_time = kwargs['generation_time']

    def get_property(self, **kwargs):
        super().get_property(**kwargs)
        props = {}
        if 'value' in kwargs:
            props['value'] = self.value
        if 'ptr_value' in kwargs:
            props['ptr_value'] = self.ptr_value
        if 'generation_time' in kwargs:
            props['generation_time'] = self.generation_time
        return props

    def revision_track(self):
        return '$Rev$'

    def run_check(self, time_step):
        return True

    def cleanup(self):
        super().cleanup()

    def save(self, filename, hdr):
        hdr['VERSION'] = 1
        super().save(filename, hdr)

    def read(self, filename, hdr, exten=0):
        super().read(filename, hdr, exten)

    def restore(self, filename):
        data = fits.getdata(filename, header=True)
        hdr = data[1]
        obj = Cheat('')
        obj.read(filename, hdr)
        return obj
