import numpy as np

from pyssata.lib.online_filter import online_filter


def compute_comm(filter_obj, input_data, ist=None, ost=None):
    nfilter = filter_obj.num.shape[0]
    ninput = len(input_data)

    if nfilter < ninput:
        raise ValueError(f"Error: IIR filter needs no more than {nfilter} coefficients ({ninput} given)")

    ordnum = np.sort(filter_obj.ordnum)
    ordden = np.sort(filter_obj.ordden)
    idx_onu = np.unique(ordnum, return_index=True)[1]
    idx_odu = np.unique(ordden, return_index=True)[1]
    onu = ordden[idx_onu]
    odu = ordden[idx_odu]

    output = np.zeros_like(input_data)

    if len(onu) == 1 and len(odu) == 1:
        idx_finite = np.where(np.isfinite(input_data))
        temp_ist = ist[idx_finite]
        temp_ost = ost[idx_finite]
        output[idx_finite] = online_filter(
            filter_obj.num[idx_finite, :int(onu[0])],
            filter_obj.den[idx_finite, :int(odu[0])],
            input_data[idx_finite],
            ost=temp_ost,
            ist=temp_ist
        )
        ost[idx_finite] = temp_ost
        ist[idx_finite] = temp_ist
    else:
        for j in range(len(idx_onu)):
            for k in range(len(idx_odu)):
                idx = np.where((filter_obj.ordnum == onu[j]) & (filter_obj.ordden == odu[k]))[0]
                if len(idx) > 0:
                    ord_num = onu[j]
                    ord_den = odu[k]
                    idx_finite = np.where(np.isfinite(input_data[idx]))[0]
                    if len(idx_finite) > 0:
                        idx = idx[idx_finite]
                        temp_ist = ist[idx, :ord_num]
                        temp_ost = ost[idx, :ord_den]
                        output[idx] = online_filter(
                            filter_obj.num[idx, :ord_num].reshape(-1, ord_num),
                            filter_obj.den[idx, :ord_den].reshape(-1, ord_den),
                            input_data[idx],
                            ost=temp_ost,
                            ist=temp_ist
                        )
                        ost[idx, :ord_den] = temp_ost
                        ist[idx, :ord_num] = temp_ist

    return output
