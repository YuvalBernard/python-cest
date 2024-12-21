# %%

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks, savgol_filter


def Z_to_MTR_Assym(offsets, data):
    def locate_peaks(data, prominence: float):
        idx, _ = find_peaks(data, prominence)
        if len(idx) == 0:
            prominence /= 1.5
            return locate_peaks(data, prominence)
        elif len(idx) == 1:
            return prominence
        else:
            prominence *= 2
            return locate_peaks(data, prominence)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    if offsets[0] >= offsets[1] and offsets[-2] >= offsets[-1]:  # make sure offsets are monotonically increasing
        offsets = np.flip(offsets)
        data = np.flip(data, axis=-1)

    ds_loc = find_nearest(offsets, 0)
    if data.ndim > 1:
        spectrum = (np.flip(data, axis=-1) - data)[-1]
    else:
        spectrum = np.flip(data, axis=-1) - data
    filt = savgol_filter(spectrum, 15, 2)
    prominence = locate_peaks(filt, 0.5)
    peak_loc, _ = find_peaks(filt, prominence)
    if offsets[peak_loc] > 0:
        N = len(offsets[ds_loc:])
        x_lab = np.linspace(offsets[ds_loc], offsets[-1], N)
        x_ref = np.sort(-x_lab)
    else:
        N = len(offsets[:ds_loc])
        x_lab = np.linspace(offsets[0], offsets[ds_loc], N)
        x_ref = np.sort(-x_lab)
    if data.ndim > 1:
        MTR = np.zeros((np.size(data, 0), N))
        for i, val in enumerate(MTR):
            z_lab = PchipInterpolator(offsets, data[i])(x_lab)
            z_ref = np.flip(PchipInterpolator(offsets, data[i])(x_ref))
            MTR[i] = z_ref - z_lab
    else:
        z_lab = PchipInterpolator(offsets, data)(x_lab)
        z_ref = np.flip(PchipInterpolator(offsets, data)(x_ref))
        MTR = z_ref - z_lab

    QUOTIENT_TO_SKIP = 0.1
    idx1 = int(N * QUOTIENT_TO_SKIP)
    idx2 = int(N * (1 - QUOTIENT_TO_SKIP))
    if offsets[peak_loc] > 0:
        MTR = MTR[:, idx1:]
        x_lab = x_lab[idx1:]
    else:
        MTR = MTR[:, :idx2]
        x_lab = x_lab[:idx2]
    return x_lab, MTR
