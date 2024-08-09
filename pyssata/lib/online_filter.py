import numpy as np

def online_filter(num, den, input_data, ost=None, ist=None):
    # Initialize state vectors if not provided
    if ost is None:
        ost = np.zeros(len(den))
    if ist is None:
        ist = np.zeros(len(num))
    
    sden = np.shape(den)
    snum = np.shape(num)
    
    if len(sden) == 1 and len(snum) == 1:
        no = sden[0]
        ni = max(snum[0], no)

        if ni != snum[0]:
            if len(ist) != ni:
                ist = np.concatenate((ist, np.zeros(ni - snum[0])))
            num_temp = np.concatenate((num, np.zeros(ni - snum[0])))
        else:
            num_temp = num

        # Delay the vectors
        ost = np.concatenate((ost[1:], [0]))
        ist = np.concatenate((ist[1:], [0]))

        # New input
        ist[ni - 1] = input_data

        # New output
        ost[no - 1] = (np.sum(num_temp * ist) - np.sum(den[:no - 1] * ost[:no - 1])) / den[no - 1]
        output = ost[no - 1]

    else:
        no = sden[1]
        ni = max(snum[1], no)

        if ni != snum[1]:
            if len(ist) != ni:
                ist = np.concatenate((ist, np.zeros((snum[0], ni - snum[1]))), axis=1)
            num_temp = np.concatenate((num, np.zeros((snum[0], ni - snum[1]))), axis=1)
        else:
            num_temp = num

        # Delay the vectors
        print(ost[:,1:].shape, np.zeros((sden[0], 1)).shape)
        ost = np.concatenate((ost[:, 1:], np.zeros((sden[0], 1))), axis=1)
        ist = np.concatenate((ist[:, 1:], np.zeros((sden[0], 1))), axis=1)

        # New input
        ist[:len(input_data), ni - 1] = input_data

        # New output
        if no > 2:
            ost[:, no - 1] = (np.sum(num_temp * ist, axis=1) - np.sum(den[:, :no - 1] * ost[:, :no - 1], axis=1)) / den[:, no - 1]
        else:
            ost[:, no - 1] = (np.sum(num_temp * ist, axis=1) - np.squeeze(den[:, :no - 1] * ost[:, :no - 1])) / den[:, no - 1]

        output = ost[:len(input_data), no - 1]

    return output
