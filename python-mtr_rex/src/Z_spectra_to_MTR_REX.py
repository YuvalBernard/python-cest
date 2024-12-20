# %%
from functools import partial

import jax.numpy as jnp
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


def Z_to_MTR_REX(x_zspec, zspec):
    def single_Z_to_MTR_REX(x_zspec, zspec):
        if x_zspec[0] >= x_zspec[1] and x_zspec[-2] >= x_zspec[-1]:  # array is decreasing
            x_zspec = np.flip(x_zspec)
            zspec = np.flip(zspec)

        def locate_peaks(data, prominence: float):
            idx, _ = find_peaks(-data, prominence=prominence)
            if len(idx) == 2:
                return prominence
            elif len(idx) > 2:
                prominence *= 2
                return locate_peaks(data, prominence)
            else:
                prominence /= 1.5
                return locate_peaks(data, prominence)

        filtered_zspec = savgol_filter(zspec, 15, 2)
        prominence = locate_peaks(filtered_zspec, prominence=0.1)
        idx, _ = find_peaks(-filtered_zspec, prominence=prominence)

        ds_idx = idx[0] if zspec[idx[0]] < zspec[idx[1]] else idx[1]
        cest_idx = idx[0] if zspec[idx[0]] > zspec[idx[1]] else idx[1]

        QUOTIENT_TO_SKIP = 0.15
        if ds_idx < cest_idx:  # peak is at positive frequency
            N = len(x_zspec[ds_idx:])
            x_lab = np.linspace(x_zspec[ds_idx], x_zspec[-1], N)
            x_ref = np.sort(-x_lab)
            z_lab = PchipInterpolator(x_zspec, zspec)(x_lab)
            z_ref = np.flip(PchipInterpolator(x_zspec, zspec)(x_ref))
            MTR_REX = (1 / z_lab - 1 / z_ref)[int(len(x_lab) * QUOTIENT_TO_SKIP) :]
            x_lab = x_lab[int(len(x_lab) * QUOTIENT_TO_SKIP) :]
        else:  # peak is at negative frequency
            N = len(x_zspec[:ds_idx])
            x_lab = np.linspace(x_zspec[0], x_zspec[ds_idx], N)
            x_ref = np.sort(-x_lab)
            z_lab = np.flip(PchipInterpolator(x_zspec, zspec)(x_lab))
            z_ref = PchipInterpolator(x_zspec, zspec)(x_ref)

            MTR_REX = (1 / z_lab - 1 / z_ref)[int(len(x_lab) * QUOTIENT_TO_SKIP) :]
            x_lab = x_lab[int(len(x_lab) * QUOTIENT_TO_SKIP) :]

        return x_lab, MTR_REX

    if not zspec.size:
        raise RuntimeError("Data matrix is not valid!")
        return

    if zspec.ndim > 1:
        MTR_REX = []
        for i, _ in enumerate(zspec):
            x_lab, mtr_rex = single_Z_to_MTR_REX(x_zspec, zspec[i])
            MTR_REX.append(mtr_rex)
        return x_lab, np.array(MTR_REX)
    else:
        return single_Z_to_MTR_REX(x_zspec, zspec)
