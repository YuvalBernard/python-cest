# %%
from functools import partial
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


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
    Z = jnp.matmul(expm(A * tp), M0)[2]
    return Z


def gen_MTR_assym(model_parameters, offset, power, B0, gamma, tp):
    Z_lab = batch_gen_spectrum_numerical(model_parameters, offset, power, B0, gamma, tp)
    Z_ref = batch_gen_spectrum_numerical(model_parameters, -offset, power, B0, gamma, tp)
    return Z_ref - Z_lab

# @partial(jnp.vectorize, excluded=[0], signature="()->(k)")
def calc_MTR_assym(offsets, Z):
    pks_loc, _ = find_peaks(-Z)
    if len(pks_loc) != 2:
        print("ERROR")
    if Z[pks_loc[0]] < Z[pks_loc[0]]:
        ds_loc = pks_loc[0]
        cest_loc = pks_loc[1]
    else:
        ds_loc = pks_loc[1]
        cest_loc = pks_loc[0]

    idx = jnp.argsort(offsets)[::-1]
    Z = Z[idx]

    if cest_loc < ds_loc:
        x_asym = offsets[:ds_loc]
        y_asym = Z[:ds_loc]
    else:
        x_asym = offsets[ds_loc:]
        y_asym = Z[ds_loc:]
    return x_asym, y_asym

@partial(jnp.vectorize, excluded=[0, 1, 3, 4, 5], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4, 5], signature="()->()")  # offsets
def Rex_anal(model_parameters, offset, power, B0, gamma, tp):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    dwa *= B0 * gamma
    dwb *= B0 * gamma
    offset *= B0 * gamma
    power *= gamma
    ka = k * f

    A = jnp.array(
        [
            [-(R2a + ka), dwa - offset, 0, k, 0, 0],
            [offset - dwa, -(R2a + ka), power, 0, k, 0],
            [0, -power, -(R1a + ka), 0, 0, k],
            [ka, 0, 0, -(R2b + k), dwb - offset, 0],
            [0, ka, 0, offset - dwb, -(R2b + k), power],
            [0, 0, ka, 0, -power, -(R1b + k)]
        ]
    )
    b = jnp.array([0, 0, R1a, 0, 0, R1b*f])
    Z_ss = jnp.linalg.solve(A, -b)[2]
    theta = jnp.atan(power/(offset - dwa))
    R_eff = R1a*jnp.cos(theta)**2 + R2a*jnp.sin(theta)**2
    R_1p = (R1a*jnp.cos(theta)**2)/Z_ss
    R_ex = R_1p - R_eff
    return R_ex

@partial(jnp.vectorize, excluded=[0, 1, 3, 4, 5], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4, 5], signature="()->()")  # offsets
def Rex_num(model_parameters, offset, power, B0, gamma, tp):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    dwa *= B0 * gamma
    dwb *= B0 * gamma
    offset *= B0 * gamma
    power *= gamma
    ka = k * f

    A = jnp.array(
        [
            [-(R2a + ka), dwa - offset, 0, k, 0, 0],
            [offset - dwa, -(R2a + ka), power, 0, k, 0],
            [0, -power, -(R1a + ka), 0, 0, k],
            [ka, 0, 0, -(R2b + k), dwb - offset, 0],
            [0, ka, 0, offset - dwb, -(R2b + k), power],
            [0, 0, ka, 0, -power, -(R1b + k)]
        ]
    )
    b = jnp.array([0, 0, R1a, 0, 0, R1b*f])
    theta = jnp.atan(power/(offset - dwa))
    R_eff = R1a*jnp.cos(theta)**2 + R2a*jnp.sin(theta)**2
    lambdas = jnp.linalg.eigvals(A)
    R1p = jnp.min(jnp.abs(lambdas))
    Rex = R1p - R_eff
    return Rex


offsets = jnp.linspace(6, -6, 100)
powers = jnp.array([1, 2.5, 5, 7])
B0 = 9.4
gamma = 267.5153
tp = 15
args = (offsets, powers, B0, gamma, tp)
#                    R1a   R2a  dwa  R1b  R2b   k      f     dwb
fit_pars = jnp.array([0.33, 0.5, 0.0, 1.0, 30, 200, 5e-4, 3.5])
Z = batch_gen_spectrum_numerical(fit_pars, offsets, powers, B0, gamma, tp)
rng = np.random.default_rng()
sigma = 0.01
data = rng.normal(Z, sigma)

rex_num = Rex_num(fit_pars, offsets, powers, B0, gamma, tp)
rex_anal = Rex_anal(fit_pars, offsets, powers, B0, gamma, tp)

ax = plt.gca()
ax.plot(offsets, rex_num.T)
ax.set_prop_cycle(None)
ax.plot(offsets, rex_anal.T, "--")
