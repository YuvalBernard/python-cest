# %%
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.linalg import expm
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks, savgol_filter


@partial(jnp.vectorize, excluded=[0, 1, 3, 4, 5], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4, 5], signature="()->()")  # offsets
def batch_gen_spectrum_numerical(model_parameters, offset, power, B0, gamma, tp):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    dwa *= B0 * gamma
    dwb *= B0 * gamma
    offset *= B0 * gamma
    power *= gamma
    ka = k * f
    M0 = jnp.array([0, 0, 1, 0, 0, f, 1])
    A = jnp.array(
        [
            [-(R2a + ka), dwa - offset, 0, k, 0, 0, 0],
            [offset - dwa, -(R2a + ka), power, 0, k, 0, 0],
            [0, -power, -(R1a + ka), 0, 0, k, R1a],
            [ka, 0, 0, -(R2b + k), dwb - offset, 0, 0],
            [0, ka, 0, offset - dwb, -(R2b + k), power, 0],
            [0, 0, ka, 0, -power, -(R1b + k), R1b * f],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    Z = jnp.matmul(expm(A * tp, max_squarings=24), M0)[2]
    return Z


offsets = np.linspace(6, -6, 500)
powers = np.array([1.5, 3.0, 5.0, 7.5])
# powers = 1.5
B0 = 9.4
gamma = 267.522
tp = 10.0
args = (offsets, powers, B0, gamma, tp)
#                    R1b   R2b  dwa  R1b  R2b   k      f     dwb
fit_pars = np.array([0.33, 0.5, 0.0, 1.0, 30.0, 200.0, 5e-4, 3.5])
Z = batch_gen_spectrum_numerical(fit_pars, offsets, powers, B0, gamma, tp)
rng = np.random.default_rng()
sigma = 0.015
data = rng.normal(Z, sigma)

MTR_Assym = np.flip(data, axis=-1) - data

plt.plot(offsets, data.T)
plt.gca().set_prop_cycle(None)
plt.plot(offsets, MTR_Assym.T)


# %%


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
spectrum = (np.flip(data, axis=-1) - data)[-1]
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
if MTR_Assym.ndim > 1:
    MTR = np.zeros((np.size(MTR_Assym, 0), N))
    for i, val in enumerate(MTR):
        z_lab = PchipInterpolator(offsets, data[i])(x_lab)
        z_ref = np.flip(PchipInterpolator(offsets, data[i])(x_ref))
        MTR[i] = z_ref - z_lab
else:
    z_lab = PchipInterpolator(offsets, data)(x_lab)
    z_ref = np.flip(PchipInterpolator(offsets, data)(x_ref))
    MTR = z_ref - z_lab
plt.plot(x_lab, MTR.T)

# %%


def Z_to_MTR_Assym(offsets, data):
    def locate_peaks(data, prominence: float):
        idx, _ = find_peaks(data, prominence=prominence)
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
    prominence = locate_peaks(filt, 0.05)
    peak_loc, _ = find_peaks(filt, prominence=prominence)
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

    QUOTIENT_TO_SKIP = 0.
    idx1 = int(N * QUOTIENT_TO_SKIP)
    idx2 = int(N * (1 - QUOTIENT_TO_SKIP))
    if offsets[peak_loc] > 0:
        MTR = MTR[:, idx1:]
        x_lab = x_lab[idx1:]
    else:
        MTR = MTR[:, :idx2]
        x_lab = x_lab[:idx2]
    return x_lab, MTR


import pandas as pd

df = pd.read_excel("/home/yuval/Documents/Weizmann/python-cest-bkp/fresh-start/results/H-CEST-0.02-single-sim-symbolic/data.xlsx")
offsets = df["ppm"].astype(float).to_numpy()
data = df.drop("ppm", axis=1).astype(float).T.to_numpy()
x_mtr, y_mtr = Z_to_MTR_Assym(offsets, data)

plt.plot(x_mtr, y_mtr.T)
plt.gca().set_prop_cycle(None)
plt.plot(offsets, data.T)
