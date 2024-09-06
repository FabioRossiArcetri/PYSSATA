
import re
import typing
import inspect
import importlib

from pyssata.factory import Factory

from pyssata.data_objects.source import Source
from pyssata.processing_objects.atmo_propagation import AtmoPropagation

from pyssata.calib_manager import CalibManager


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
        for key, v in params.items():
            if key in 'pupilstop slopec psf'.split():
                print(key, v)
                classname = v['class']
                del v['class']

                klass = self._import_class(classname)
                args = inspect.getfullargspec(getattr(klass, '__init__')).args
                hints = self._get_type_hints(klass)

                v2 = v.copy()  # Cannot modify original dict during iteration
                for name, value in v.items():
                    if name.endswith('_data'):
                        parname = name[:-7]
                        data = cm.read_data(value)
                        v2[name[:-5]] = data
                        del v2[name]
                    if name.endswith('_object'):
                        parname = name[:-7]
                        if parname in hints:
                            partype = hints[parname]
                            filename = cm.filename(parname, value)
                            parobj = getattr(partype, 'restore').__call__(filename)
                            v2[parname] = parobj
                            del v2[name]
                        else:
                            raise ValueError(f'No type hint for parameter {parname} of class {classname}')
                        
                # Add global params if needed
                my_params = {k: main[k] for k in args if k in main}
                my_params.update(v2)
                globals()[key] = klass(**my_params)  # TODO temporary hack, locals() does not work

        # Initialize housekeeping objects
        factory = Factory(params['main'])
        loop = factory.get_loop_control()
        store = factory.get_datastore()


        # Initialize processing objects
        source = [Source(**p) for p in params['wfs_source']]
        prop = AtmoPropagation(source,
                            pixel_pupil = params['main']['pixel_pupil'],
                            pixel_pitch = params['main']['pixel_pitch'],
                            )
        pyr = factory.get_modulated_pyramid(params['pyramid'])
        ccd = factory.get_ccd(params['detector'])
        rec = factory.get_modalrec(params['modalrec'])
        intc = factory.get_control(params['control'])
        dm = factory.get_dm(params['dm'])
        atmo = factory.get_atmo_container(source, params['atmo'],
                                        params['seeing'], params['wind_speed'], params['wind_direction'])

        # Initialize display objects
        sc_disp = factory.get_slopec_display(slopec)
        sr_disp = factory.get_plot_display(psf.out_sr)
        ph_disp = factory.get_phase_display(prop.pupil_list[0])
        dm_disp = factory.get_phase_display(dm.out_layer)
        psf_disp = factory.get_psf_display(psf.out_psf)

        sc_disp.window = 10
        sr_disp.window = 11
        ph_disp.window = 12
        dm_disp.window = 13
        psf_disp.window = 14
        sr_disp.title = 'SR'
        dm_disp.title = 'DM'
        psf_disp.title = 'PSF'
        sc_disp.disp_factor = 4
        ph_disp.disp_factor = 2

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
        intc.in_delta_comm = rec.out_modes
        #dm.in_command = intc.out_comm
        dm.in_command = rec.out_modes
        psf.in_ef = pyr.in_ef

        # Set store data
        store.add(psf.out_sr, name='sr')
        store.add(pyr.in_ef, name='res_ef')

        # Build loop
        loop.add(atmo)
        loop.add(prop)
        loop.add(pyr)
        loop.add(ccd)
        loop.add(slopec)
        loop.add(rec)
        loop.add(intc)
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
