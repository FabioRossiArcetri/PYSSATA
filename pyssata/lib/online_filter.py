import numpy as np

from pyssata import xp

def online_filter(num, den, input_data, ost=None, ist=None):
    # Initialize state vectors if not provided
    if ost is None:
        ost = xp.zeros(len(den), dtype=self.dtype)
    if ist is None:
        ist = xp.zeros(len(num), dtype=self.dtype)
    
    sden = xp.shape(den)
    snum = xp.shape(num)
    
    if len(sden) == 1 and len(snum) == 1:
        no = sden[0]
        ni = snum[0]

        # Delay the vectors
        ost = xp.concatenate((ost[1:], [0]))
        ist = xp.concatenate((ist[1:], [0]))

        # New input
        ist[ni - 1] = input_data

        # New output
        factor = 1/den[no - 1]
        ost[no - 1] = factor * xp.sum(num * ist)
        ost[no - 1] -= factor * xp.sum(den[:no - 1] * ost[:no - 1])
        output = ost[no - 1]

    else:
        no = sden[1]
        ni = snum[1]

        # Delay the vectors
        ost = xp.concatenate((ost[:, 1:], xp.zeros((sden[0], 1), dtype=self.dtype)), axis=1)
        ist = xp.concatenate((ist[:, 1:], xp.zeros((sden[0], 1), dtype=self.dtype)), axis=1)

        # New input
        ist[:len(input_data), ni - 1] = input_data

        # New output
        factor = 1/den[:, no - 1]
        ost[:, no - 1] = factor * xp.sum(num * ist, axis=1)
        ost[:, no - 1] -= factor * xp.sum(den[:, :no - 1] * ost[:, :no - 1], axis=1)

        output = ost[:len(input_data), no - 1]

    return output
