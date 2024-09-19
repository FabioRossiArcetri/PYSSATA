import numpy as np
from pyssata import gpuEnabled
from pyssata import xp

def online_filter(num, den, input_data, ost=None, ist=None):
    # Initialize state vectors if not provided
    if ost is None:
        ost = xp.zeros(len(den))
    if ist is None:
        ist = xp.zeros(len(num))
    
    sden = xp.shape(den)
    snum = xp.shape(num)
    
    if len(sden) == 1 and len(snum) == 1:
        no = sden[0]
        ni = max(snum[0], no)

        if ni != snum[0]:
            if len(ist) != ni:
                ist = xp.concatenate((ist, xp.zeros(ni - snum[0])))
            num_temp = xp.concatenate((num, xp.zeros(ni - snum[0])))
        else:
            num_temp = num

        # Delay the vectors
        ost = xp.concatenate((ost[1:], [0]))
        ist = xp.concatenate((ist[1:], [0]))

        # New input
        ist[ni - 1] = input_data

        # New output
        ost[no - 1] = (xp.sum(num_temp * ist) - xp.sum(den[:no - 1] * ost[:no - 1])) / den[no - 1]
        output = ost[no - 1]

    else:
        no = sden[1]
        ni = max(snum[1], no)

        if ni != snum[1]:
            if len(ist) != ni:
                ist = xp.concatenate((ist, xp.zeros((snum[0], ni - snum[1]))), axis=1)
            num_temp = xp.concatenate((num, xp.zeros((snum[0], ni - snum[1]))), axis=1)
        else:
            num_temp = num

        # Delay the vectors
        print(ost[:,1:].shape, xp.zeros((sden[0], 1)).shape)
        ost = xp.concatenate((ost[:, 1:], xp.zeros((sden[0], 1))), axis=1)
        ist = xp.concatenate((ist[:, 1:], xp.zeros((sden[0], 1))), axis=1)

        # New input
        ist[:len(input_data), ni - 1] = input_data

        # New output
        if no > 2:
            ost[:, no - 1] = (xp.sum(num_temp * ist, axis=1) - xp.sum(den[:, :no - 1] * ost[:, :no - 1], axis=1)) / den[:, no - 1]
        else:
            ost[:, no - 1] = (xp.sum(num_temp * ist, axis=1) - xp.squeeze(den[:, :no - 1] * ost[:, :no - 1])) / den[:, no - 1]

        output = ost[:len(input_data), no - 1]

    return output
