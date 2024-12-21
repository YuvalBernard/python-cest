# %%
import os
from functools import partial

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.linalg import expm
from Z_spectra_to_MTR_Assym import Z_to_MTR_Assym


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
    Z = jnp.matmul(expm(A * tp, max_squarings=18), M0)[2]
    return Z


offsets = np.arange(6, -6, -0.15, dtype=float)
powers = np.array([1.5, 3.0, 5.0, 7.5])
B0 = 9.4
gamma = 267.522
tp = 10.0
args = (offsets, powers, B0, gamma, tp)
#                    R1b   R2b  dwa  R1b  R2b   k      f     dwb
fit_pars = np.array([0.33, 0.5, 0.0, 1.0, 30.0, 200.0, 5e-4, 3.5])
Z = batch_gen_spectrum_numerical(fit_pars, offsets, powers, B0, gamma, tp)
rng = np.random.default_rng()
sigma = 0.01
data = rng.normal(Z, sigma)

x_mtr, y_mtr = Z_to_MTR_Assym(offsets, data)

df = pd.DataFrame(np.c_[x_mtr, y_mtr.T], columns=["ppm"] + [f"{power:.3g} μT" for power in powers])
with pd.ExcelWriter(os.path.join(os.getcwd(), "python-mtr_assym/example_data.xlsx")) as writer:
    df.to_excel(writer, sheet_name="data", index=False, float_format="%.3g")
