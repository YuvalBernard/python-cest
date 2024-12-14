# %%
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import config
from jax.scipy.linalg import expm

config.update("jax_enable_x64", True)


@partial(jnp.vectorize, excluded=[0, 1, 3, 4, 5], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4, 5], signature="()->()")  # offsets
def gen_AREX(model_parameters, offset, power, B0, gamma, tp):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    dwa *= B0 * gamma
    dwb *= B0 * gamma
    offset *= B0 * gamma
    power *= gamma
    ka = k * f
    M0 = jnp.array([0, 0, 1, 0, 0, f, 1])
    A_lab = jnp.array(
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
    A_ref = jnp.array(
        [
            [-(R2a + ka), dwa + offset, 0, k, 0, 0, 0],
            [-offset - dwa, -(R2a + ka), power, 0, k, 0, 0],
            [0, -power, -(R1a + ka), 0, 0, k, R1a],
            [ka, 0, 0, -(R2b + k), dwb + offset, 0, 0],
            [0, ka, 0, -offset - dwb, -(R2b + k), power, 0],
            [0, 0, ka, 0, -power, -(R1b + k), R1b * f],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    Z_lab = jnp.matmul(expm(A_lab * tp, max_squarings=18), M0)[2]
    Z_ref = jnp.matmul(expm(A_ref * tp, max_squarings=18), M0)[2]
    return R1a * (1 / Z_lab - 1 / Z_ref)


offsets = jnp.arange(6, 0, -0.05, dtype=float)
powers = jnp.array(
    [
        0.5,
        1.5,
        3.0,
        5.0,
    ]
)
B0 = 9.4
gamma = 267.522
tp = 10.0
args = (offsets, powers, B0, gamma, tp)
#                    R1a   R2a  dwa  R1b  R2b   k      f     dwb
fit_pars = jnp.array([0.33, 0.5, 0.0, 1.0, 30.0, 200.0, 5e-4, 3.5])
AREX = gen_AREX(fit_pars, offsets, powers, B0, gamma, tp).T

plt.plot(offsets, AREX)

# df = pd.read_excel("/home/yuval/Documents/Weizmann/python-cest/sandbox/F_CEST_283K.xlsx", sheet_name=4, usecols="G:J")
# Z_lab = df.iloc[1:128, 0].to_numpy()
# Z_ref = df.iloc[129:, 0][::-1].to_numpy()
# offsets = df.iloc[1:128, 2].to_numpy()
# plt.plot(offsets[:50], 0.65*(1/Z_ref - 1/Z_lab)[:50])
