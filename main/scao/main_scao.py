
from copy import deepcopy
from pyssata.factory import Factory

from pyssata.data_objects.source import Source
from pyssata.data_objects.pupilstop import PupilStop
from pyssata.processing_objects.psf import PSF
from pyssata.processing_objects.pyr_slopec import PyrSlopec
from pyssata.processing_objects.atmo_propagation import AtmoPropagation

from pyssata.calib_manager import CalibManager

# Read parameters file
dir = '/home/alfio/git/PYSSATA/main/scao/'

import os
os.chdir(dir)

params = {}
exec(open(dir + 'params_scao.py').read(), params)

# Hack to keep the original params dict after factory deletes entries
del params['__builtins__']
del params['np']
params_orig = deepcopy(params)

# Initialize housekeeping objects
factory = Factory(params['main'])
loop = factory.get_loop_control()
store = factory.get_datastore()
cm = CalibManager(params['main']['root_dir'])

# Initialize processing objects
source = [Source(**p) for p in params['wfs_source']]
prop = AtmoPropagation(source,
                       pixel_pupil = params['main']['pixel_pupil'],
                       pixel_pitch = params['main']['pixel_pitch']
                       )
pupilstop = PupilStop(pixel_pupil = params['main']['pixel_pupil'],
                      pixel_pitch = params['main']['pixel_pitch'],
                      **params['pupilstop'])
pyr = factory.get_modulated_pyramid(params['pyramid'])
ccd = factory.get_ccd(params['detector'])
sc = PyrSlopec(**params['slopec'], cm=cm)
rec = factory.get_modalrec(params['modalrec'])
intc = factory.get_control(params['control'])
dm = factory.get_dm(params['dm'])
psf = PSF(**params['camera'])
atmo = factory.get_atmo_container(source, params['atmo'],
                                  params['seeing'], params['wind_speed'], params['wind_direction'])

# Initialize display objects
sc_disp = factory.get_slopec_display(sc)
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
sc.in_pixels = ccd.out_pixels
rec.in_slopes = sc.out_slopes
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
loop.add(sc)
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

print(f"Mean Strehl Ratio (@{params_orig['camera']['wavelengthInNm']}nm) : {store.mean('sr', init=min([50, 0.1 * params['main']['total_time'] / params['main']['time_step']]) * 100.)}")

# Saving method with a single sav file
store.save('save_file.pickle')

# Alternative saving method:
# tn = store.save_tracknum(dir=dir, params=params, nodlm=True, noolformat=True, compress=True, saveFloat=saveFloat)
