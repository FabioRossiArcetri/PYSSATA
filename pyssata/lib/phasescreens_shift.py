import numpy as np

from pyssata import xp
from scipy.ndimage import rotate

def phasescreens_shift(phasescreens, pixel_layer, wind_speed, wind_direction, delta_time, pixel_pitch, scale_coeff, layer_list, position=None, cycle_screens=False):
    # Compute the delta position in pixels
    delta_position = wind_speed * delta_time / pixel_pitch  # [pixel]
    
    # Update the position
    if position is not None:
        new_position = position + delta_position
    else:
        new_position = delta_position  # [pixel]
    
    # Get quotient and remainder
    new_position_quo = xp.floor(new_position).astype(int)
    new_position_rem = new_position - new_position_quo
    
    for ii, p in enumerate(phasescreens):
        # Check if we need to cycle the screens
        # print(ii, new_position[ii], pixel_layer[ii], p.shape[1]) # Verbose?
        if cycle_screens:
            if new_position[ii] + pixel_layer[ii] > p.shape[1]:
                new_position[ii] = 0.
        
        if new_position[ii] + pixel_layer[ii] > p.shape[1]:
            print(f'phasescreens size: {xp.around(p.shape[0], 2)}')
            print(f'requested position: {xp.around(new_position[ii], 2)}')
            raise ValueError(f'phasescreens_shift cannot go out of the {ii}-th phasescreen!')
        
        pos = new_position_quo[ii]
        # print(pos, pixel_layer) # Verbose?
        ps_Shift1 = p[0: int(pixel_layer[ii]), pos: int(pos + pixel_layer[ii])]
        ps_Shift2 = p[0: int(pixel_layer[ii]), pos + 1: int(pos + pixel_layer[ii]) + 1]
        ps_ShiftInterp = (1 - new_position_rem[ii]) * ps_Shift1 + new_position_rem[ii] * ps_Shift2
        
        layer = ps_ShiftInterp[:, :]

        # Meta-pupil rotation
        if wind_direction[ii] != 0:
            if wind_direction[ii] == 90:
                layer = xp.rot90(layer, 3)
            elif wind_direction[ii] == 180:
                layer = xp.rot90(layer, 2)
            elif wind_direction[ii] == 270 or wind_direction[ii] == -90:
                layer = xp.rot90(layer, 1)
            elif wind_direction[ii] == -180:
                layer = xp.rot90(layer, 2)
            elif wind_direction[ii] == -270:
                layer = xp.rot90(layer, 3)
            else:
                layer = rotate(layer, wind_direction[ii], reshape=False, order=1)
        
        layer_list[ii].phaseInNm = layer * scale_coeff

    # print(f'Phasescreen_shift: {new_position=}') # Verbose?
    # Update position output
    return new_position
