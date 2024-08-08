
from pyssata.base_processing_obj import BaseProcessingObj

class ProcessingContainer(BaseProcessingObj):
    '''Container of processing objects'''

    def __init__(self):
        super().__init__()
        self._objs = {}
        self._outputs = {}
        self._inputs = {}

    def add(self, obj, name=None, output=None, input=None):
        if name is None:
            name = f'obj{len(self._objs)}'
        self._objs[name.upper()] = obj
        if output:
            self.add_output(name, output)
        if input:
            self.add_input(name, input)

    def add_output(self, name, keyword):
        self._outputs[keyword.upper()] = name.upper()

    def add_input(self, name, keyword):
        self._inputs[keyword.upper()] = name.upper()

    def trigger(self, t):
        for item in self._objs.values():
            item.trigger(t)

    def get(self, name):
        return self._objs[name.upper()]

    def set_property(self, loop_dt=None, **kwargs):
        ex = []
        for key in kwargs:
            name = key.upper()
            if name in self._inputs:
                objname = self._inputs[name]
                getattr(self._objs[objname], 'set_property')(**{key: kwargs[key]})
            else:
                ex.append(key)

        if loop_dt is not None:
            self._loop_dt = loop_dt
            for item in self._objs.values():
                item.loop_dt = loop_dt

        if ex:
            super().set_property(**{key: kwargs[key] for key in ex})

    def get_property(self, **kwargs):
        ex = []
        for key in kwargs:
            name = key.upper()
            if name in self._outputs:
                objname = self._outputs[name]
                getattr(self._objs[objname], 'get_property')(**{key: kwargs[key]})
            else:
                ex.append(key)

        if ex:
            super().get_property(**{key: kwargs[key] for key in ex})

    def run_check(self, time_step, errmsg=''):
        ok = True
        for item in self._objs.values():
            if not item.run_check(time_step, errmsg):
                print(f'run_check failed on {item}')
                ok = False
        return ok

    def cleanup(self):
        for obj in self._objs.values():
            obj.cleanup()
        super().cleanup()
