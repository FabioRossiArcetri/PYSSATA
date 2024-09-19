
import re
import typing
import inspect
import importlib
from pyssata.base_processing_obj import BaseProcessingObj

from pyssata.factory import Factory
from pyssata.lib.flatten import flatten
from pyssata.calib_manager import CalibManager
from pyssata.processing_objects.datastore import Datastore


from pyssata.display.slopec_display import SlopecDisplay
from pyssata.display.plot_display import PlotDisplay
from pyssata.display.phase_display import PhaseDisplay
from pyssata.display.psf_display import PSFDisplay

class Simul():
    '''
    Simulation organizer
    '''
    def __init__(self, param_file):
        self.param_file = param_file
        self.objs = {}

    def _camelcase_to_snakecase(self, s):
        tokens = re.findall('[A-Z]+[0-9a-z]*', s)
        return '_'.join([x.lower() for x in tokens])

    def _import_class(self, classname):
        modulename = self._camelcase_to_snakecase(classname)
        try:
            mod = importlib.import_module(f'pyssata.processing_objects.{modulename}')
        except ModuleNotFoundError:
            mod = importlib.import_module(f'pyssata.data_objects.{modulename}')
        return getattr(mod, classname)

    def _get_type_hints(self, type):
        hints ={}
        for x in type.__mro__:
            hints.update(typing.get_type_hints(getattr(x, '__init__')))
        return hints
    
    def _handle_lists(self, parname, f, lst):
        if parname.endswith('_list'):
            return parname[:-5], [f(y) for y in lst]
    
    def _resolve(self, x):
        return self.resolve_output(x)

    def resolve_output(self, output_name):
        if '.' in output_name:
            obj_name, attr_name = output_name.split('.')
            # TODO will be replaced by input/output get/set methods
            output_ref = getattr(self.objs[obj_name], attr_name)
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

        # Initialize housekeeping objects
        factory = Factory(params['main'])

        for key, pars in params.items():
            if key in 'pupilstop slopec psf on_axis_source prop atmo seeing wind_speed wind_direction pyramid detector control dm rec'.split():
                print(key, pars)
                try:
                    classname = pars['class']
                except KeyError:
                    raise KeyError(f'Object {key} does not define the "class" parameter')

                klass = self._import_class(classname)
                args = inspect.getfullargspec(getattr(klass, '__init__')).args
                hints = self._get_type_hints(klass)

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
                            parobj = partype.restore(filename)
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
            print(pars)
            if 'inputs' not in pars:
                continue

            for input_name, output_name in pars['inputs'].items():
                if not input_name in self.objs[dest_object].inputs:
                    raise ValueError(f'Object {dest_object} does does not have an input called {input_name}')
                if not isinstance(output_name, (str, list)):
                    raise ValueError(f'Object {dest_object}: invalid input definition type {type(output_name)}')

                wanted_type = self.objs[dest_object].inputs[input_name].type

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

                # TODO will be replaced by input/output get/set methods
                setattr(self.objs[dest_object], input_name, output_ref)

    def run(self):
        params = {}
        exec(open(self.param_file).read(), params)
        del params['__builtins__']
        if 'np' in params:
            del params['np']
        if 'xp' in params:
            del params['xp']

        # Initialize housekeeping objects
        factory = Factory(params['main'])
        loop = factory.get_loop_control()
        store = Datastore(params['main']['store_dir'])

        # Actual creation code
        self.build_objects(params)

        # TODO temporary hack, locals() does not work
        for name, obj in self.objs.items():
            globals()[name] = obj
                        
        # Initialize display objects
        sc_disp = SlopecDisplay(self.objs['slopec'], disp_factor=4)
        sr_disp = PlotDisplay(self.objs['psf'].out_sr, window=11, title='SR')
        ph_disp = PhaseDisplay(self.objs['prop'].pupil_dict['on_axis_source'], window=12, disp_factor=2)
        dm_disp = PhaseDisplay(self.objs['dm'].out_layer, window=13, title='DM')
        psf_disp = PSFDisplay(self.objs['psf'].out_psf, window=14,  title='PSF')

        self.connect_objects(params)
        self.connect_datastore(store, params)

        # Build loop
        for name, obj in self.objs.items():
            if isinstance(obj, BaseProcessingObj):
                if name not in ['control']:
                    loop.add(obj)
        loop.add(store)
    
       # loop.add(sc_disp)
       # loop.add(sr_disp)
       # loop.add(ph_disp)
       # loop.add(dm_disp)
       # loop.add(psf_disp)

        # Run simulation loop
        loop.run(run_time=params['main']['total_time'], dt=params['main']['time_step'], speed_report=True)

        print(f"Mean Strehl Ratio (@{params['psf']['wavelengthInNm']}nm) : {store.mean('sr', init=min([50, 0.1 * params['main']['total_time'] / params['main']['time_step']]) * 100.)}")

        # Saving method with a single sav file
        store.save('save_file.pickle')

        # Alternative saving method:
        # tn = store.save_tracknum(dir=dir, params=params, nodlm=True, noolformat=True, compress=True, saveFloat=saveFloat)
