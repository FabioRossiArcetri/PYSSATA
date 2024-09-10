
import re
import typing
import inspect
import importlib

from pyssata.factory import Factory
from pyssata.calib_manager import CalibManager

from pyssata.display.slopec_display import SlopecDisplay
from pyssata.display.plot_display import PlotDisplay
from pyssata.display.phase_display import PhaseDisplay
from pyssata.display.psf_display import PSFDisplay

class Simul():
    
    def __init__(self, param_file):
        self.param_file = param_file

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

    def run(self):
        params = {}
        exec(open(self.param_file).read(), params)

        main = params['main']
        cm = CalibManager(main['root_dir'])
        for key, pars in params.items():
            if key in 'pupilstop slopec psf wfs_source prop atmo seeing wind_speed wind_direction control'.split():
                print(key, pars)
                try:
                    classname = pars['class']
                except KeyError:
                    raise KeyError(f'Object {key} does not define the "class" parameter')
                del pars['class']

                klass = self._import_class(classname)
                args = inspect.getfullargspec(getattr(klass, '__init__')).args
                hints = self._get_type_hints(klass)

                pars2 = pars.copy()  # Cannot modify original dict during iteration
                for name, value in pars.items():
                    if name.endswith('_data'):
                        parname = name[:-7]
                        data = cm.read_data(value)
                        pars2[name[:-5]] = data
                        del pars2[name]
                    if name.endswith('_object'):
                        parname = name[:-7]
                        if parname in hints:
                            partype = hints[parname]
                            filename = cm.filename(parname, value)
                            parobj = partype.restore(filename)
                            pars2[parname] = parobj
                            del pars2[name]
                        else:
                            raise ValueError(f'No type hint for parameter {parname} of class {classname}')
                        
                # TODO special cases
                if classname == 'AtmoEvolution':
                    pars2['directory'] = cm.root_subdir('phasescreen')
                    print(classname, pars2['source_list'])
                    sources = [globals()[x] for x in pars2['source_list']]  
                    pars2['source_list'] = sources
                if classname == 'AtmoPropagation':
                    sources = [globals()[x] for x in pars2['source_list']]  
                    pars2['source_list'] = sources
                        
                # Add global params if needed
                my_params = {k: main[k] for k in args if k in main}
                my_params.update(pars2)
                globals()[key] = klass(**my_params)  # TODO temporary hack, locals() does not work

        # Initialize housekeeping objects
        factory = Factory(params['main'])
        loop = factory.get_loop_control()
        store = factory.get_datastore()

        # Initialize processing objects
        pyr = factory.get_modulated_pyramid(params['pyramid'])
        ccd = factory.get_ccd(params['detector'])
        rec = factory.get_modalrec(params['modalrec'])
        dm = factory.get_dm(params['dm'])

        # Initialize display objects
        sc_disp = SlopecDisplay(slopec, disp_factor=4)
        sr_disp = PlotDisplay(psf.out_sr, window=11, title='SR')
        ph_disp = PhaseDisplay(prop.pupil_list[0], window=12, disp_factor=2)
        dm_disp = PhaseDisplay(dm.out_layer, window=13, title='DM')
        psf_disp = PSFDisplay(psf.out_psf, window=14,  title='PSF')

        # Add atmospheric and DM layers to propagation object
        atmo_layers = atmo.layer_list
        for layer in atmo_layers:
            prop.add_layer_to_layer_list(layer)
        prop.add_layer_to_layer_list(pupilstop)
        prop.add_layer_to_layer_list(dm.out_layer)

        # Connect processing objects
        pyr.in_ef = prop.pupil_list[0]
        ccd.in_i = pyr.out_i
        slopec.in_pixels = ccd.out_pixels
        rec.in_slopes = slopec.out_slopes
        control.in_delta_comm = rec.out_modes
        #dm.in_command = control.out_comm
        dm.in_command = rec.out_modes
        psf.in_ef = pyr.in_ef
        atmo.seeing = seeing.output
        atmo.wind_speed = wind_speed.output
        atmo.wind_direction = wind_direction.output
        
        # Set store data
        store.add(psf.out_sr, name='sr')
        store.add(pyr.in_ef, name='res_ef')

        # Build loop
        loop.add(seeing)
        loop.add(wind_speed)
        loop.add(wind_direction)
        loop.add(atmo)
        loop.add(prop)
        loop.add(pyr)
        loop.add(ccd)
        loop.add(slopec)
        loop.add(rec)
        loop.add(control)
        loop.add(dm)
        loop.add(psf)
        loop.add(store)
        loop.add(sc_disp)
        loop.add(sr_disp)
        loop.add(ph_disp)
        loop.add(dm_disp)
        loop.add(psf_disp)

        # Run simulation loop
        loop.run(run_time=params['main']['total_time'], dt=params['main']['time_step'])

        # Add integrated PSF to store
        store.add(psf.out_int_psf)

        print(f"Mean Strehl Ratio (@{params['psf']['wavelengthInNm']}nm) : {store.mean('sr', init=min([50, 0.1 * params['main']['total_time'] / params['main']['time_step']]) * 100.)}")

        # Saving method with a single sav file
        store.save('save_file.pickle')

        # Alternative saving method:
        # tn = store.save_tracknum(dir=dir, params=params, nodlm=True, noolformat=True, compress=True, saveFloat=saveFloat)
