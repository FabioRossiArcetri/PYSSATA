
import re
import typing
import inspect
import importlib
from copy import deepcopy
from pyssata.base_processing_obj import BaseProcessingObj

from pyssata.loop_control import LoopControl
from pyssata.lib.flatten import flatten
from pyssata.calib_manager import CalibManager
from pyssata.processing_objects.datastore import Datastore

import yaml
import io


class Simul():
    '''
    Simulation organizer
    '''
    def __init__(self, *param_files):
        if len(param_files) < 1:
            raise ValueError('At least one Yaml parameter file must be present')
        self.param_files = param_files
        self.objs = {}

    def _camelcase_to_snakecase(self, s):
        tokens = re.findall('[A-Z]+[0-9a-z]*', s)
        return '_'.join([x.lower() for x in tokens])

    def _import_class(self, classname):
        modulename = self._camelcase_to_snakecase(classname)
        try:
            try:
                mod = importlib.import_module(f'pyssata.processing_objects.{modulename}')
            except ModuleNotFoundError:
                try:
                    mod = importlib.import_module(f'pyssata.data_objects.{modulename}')
                except ModuleNotFoundError:
                    mod = importlib.import_module(f'pyssata.display.{modulename}')
        except ModuleNotFoundError:
            raise ImportError(f'Class {classname} must be defined in a file called {modulename}.py but it cannot be found')

        try:
            return getattr(mod, classname)
        except AttributeError:
            raise AttributeError(f'Class {classname} not found in file {modulename}.py')

    def _get_type_hints(self, type):
        hints ={}
        for x in type.__mro__:
            hints.update(typing.get_type_hints(getattr(x, '__init__')))
        return hints
    
    def output_owner(self, output_name):
        if '.' in output_name:
            obj_name, attr_name = output_name.split('.')
            return obj_name
        else:
            return output_name

    def resolve_output(self, output_name):
        if '.' in output_name:
            obj_name, attr_name = output_name.split('.')
            if not obj_name in self.objs:
                raise ValueError(f'Object {obj_name} does not exist')
            if not attr_name in self.objs[obj_name].outputs:
                raise ValueError(f'Object {obj_name} does not define an output with name {attr_name}')
            output_ref = self.objs[obj_name].outputs[attr_name]
        else:
            output_ref = self.objs[output_name]
        return output_ref

    def connect_datastore(self, store, params):
        if 'store' in params['main']:
            for name, output_ref in params['main']['store'].items():
                output = self.resolve_output(output_ref)
                store.add(output, name=name)

    def build_objects(self, params):
        main = params['main']
        cm = CalibManager(main['root_dir'])
        skip_pars = 'class inputs'.split()

        for key, pars in params.items():
            if key == 'main':
                continue
            try:
                classname = pars['class']
            except KeyError:
                raise KeyError(f'Object {key} does not define the "class" parameter')

            klass = self._import_class(classname)
            args = inspect.getfullargspec(getattr(klass, '__init__')).args
            hints = self._get_type_hints(klass)

            target_device_idx = pars.get('target_device_idx', None)
                
            pars2 = {}
            for name, value in pars.items():
                if name in skip_pars:
                    continue

                if name.endswith('_list_ref'):
                    data = [self.resolve_output(x) for x in value]
                    pars2[name[:-4]] = data

                elif name.endswith('_dict_ref'):
                    data = {x : self.resolve_output(x) for x in value}
                    pars2[name[:-4]] = data

                elif name.endswith('_ref'):
                    data = self.resolve_output(value)
                    pars2[name[:-4]] = data

                elif name.endswith('_data'):
                    data = cm.read_data(value)
                    pars2[name[:-5]] = data

                elif name.endswith('_object'):
                    parname = name[:-7]
                    if parname in hints:
                        partype = hints[parname]
                        filename = cm.filename(parname, value)  # TODO use partype instead of parname?
                        parobj = partype.restore(filename, target_device_idx=target_device_idx)
                        pars2[parname] = parobj
                    else:
                        raise ValueError(f'No type hint for parameter {parname} of class {classname}')

                else:
                    pars2[name] = value

            # Add global and class-specific params if needed
            my_params = {k: main[k] for k in args if k in main}
            if 'data_dir' in args:  # TODO special case
                my_params['data_dir'] = cm.root_subdir(classname)
            my_params.update(pars2)
            self.objs[key] = klass(**my_params)

    def connect_objects(self, params):
        for dest_object, pars in params.items():
            if 'inputs' not in pars:
                continue
            for input_name, output_name in pars['inputs'].items():
                if not input_name in self.objs[dest_object].inputs:
                    raise ValueError(f'Object {dest_object} does does not have an input called {input_name}')
                if not isinstance(output_name, (str, list)):
                    raise ValueError(f'Object {dest_object}: invalid input definition type {type(output_name)}')
                
                wanted_type = self.objs[dest_object].inputs[input_name].type()
                
                if isinstance(output_name, str):
                    output_ref = self.resolve_output(output_name)
                    if not isinstance(output_ref, wanted_type):
                        raise ValueError(f'Input {input_name}: output {output_ref} is not of type {wanted_type}')

                elif isinstance(output_name, list):
                    outputs = [self.resolve_output(x) for x in output_name]
                    output_ref = flatten(outputs)
                    for output in output_ref:
                        if not isinstance(output, wanted_type):
                            raise ValueError(f'Input {input_name}: output {output} is not of type {wanted_type}')

                self.objs[dest_object].inputs[input_name].set(output_ref)

    def remove_inputs(self, params, obj_to_remove):
        '''
        Modify params removing all references to the specificed object name
        '''
        for objname, obj in params.items():
            for key in ['inputs', 'store']:
                if key not in obj:
                    continue
                obj_inputs_copy = deepcopy(obj[key])
                for input_name, output_name in obj[key].items():
                    if isinstance(output_name, str):
                        owner = self.output_owner(output_name)
                        if owner == obj_to_remove:
                            del obj_inputs_copy[input_name]
                            print(f'Deleted {input_name} from {obj[key]}')
                    elif isinstance(output_name, list):
                        newlist = [x for x in output_name if self.output_owner(x) != obj_to_remove]
                        diff = set(output_name).difference(set(newlist))
                        obj_inputs_copy[input_name] = newlist
                        if len(diff) > 0:
                            print(f'Deleted {diff} from {obj[key]}')
                obj[key] = obj_inputs_copy
        return params

    def combine_params(self, params, additional_params):
        '''
        Add/update/remove params with additional_params
        '''
        for name, values in additional_params.items():
            if name == 'remove':
                for objname in values:
                    if objname not in params:
                        raise ValueError(f'Parameter file has no object named {objname}')
                    del params[objname]
                    print(f'Removed {objname}')

                    # Remove corresponding inputs
                    params = self.remove_inputs(params, objname)

            elif name.endswith('_override'):
                objname = name[:-9]
                if objname not in params:
                    raise ValueError(f'Parameter file has no object named {objname}')
                for k, v in values.items():
                    if k not in params[objname]:
                        raise ValueError(f'Object {objname} has not parameter {k} to override')
                params[objname].update(values)
            else:
                if name in params:
                    raise ValueError(f'Parameter file already has an object named {name}')
                params[name] = values
        
    def run(self):
        params = {}
        # Read YAML file(s)
        print('Reading parameters from', self.param_files[0])
        with open(self.param_files[0], 'r') as stream:
            params = yaml.safe_load(stream)
                
        for filename in self.param_files[1:]:
            print('Reading additional parameters from', self.param_files[0])
            with open(filename, 'r') as stream:
                additional_params = yaml.safe_load(stream)
                self.combine_params(params, additional_params)

        # Initialize housekeeping objects
        loop = LoopControl(run_time=params['main']['total_time'], dt=params['main']['time_step'])
        store = Datastore(params['main']['store_dir'])

        # Actual creation code
        self.build_objects(params)

        # TODO temporary hack, locals() does not work
        for name, obj in self.objs.items():
            globals()[name] = obj
                        
        self.connect_objects(params)
        self.connect_datastore(store, params)

        # Build loop
        for name, obj in self.objs.items():
            if isinstance(obj, BaseProcessingObj):
                loop.add(obj)
        loop.add(store)

        # Run simulation loop
        loop.run(run_time=params['main']['total_time'], dt=params['main']['time_step'], speed_report=True)

        if store.has_key('sr'):
            print(f"Mean Strehl Ratio (@{params['psf']['wavelengthInNm']}nm) : {store.mean('sr', init=min([50, 0.1 * params['main']['total_time'] / params['main']['time_step']])) * 100.}")

        # Saving method with a single sav file
        store.save('save_file.pickle')

        # Alternative saving method:
        # tn = store.save_tracknum(dir=dir, params=params, nodlm=True, noolformat=True, compress=True, saveFloat=saveFloat)
