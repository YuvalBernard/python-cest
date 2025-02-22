# %%
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import config, jit
from jax.scipy.linalg import expm

config.update("jax_enable_x64", True)

from gen_eigenvalue_laguerre import gen_eigenvalue_laguerre
from gen_eigenvalue_trace import gen_eigenvalue_trace


@jit
@partial(jnp.vectorize, excluded=[0, 1, 3, 4], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4], signature="()->()")  # offsets
def gen_R1p_numerical(model_parameters, offset, power, B0, gamma):
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
            [0, 0, ka, 0, -power, -(R1b + k)],
        ]
    )
    R1p = jnp.min(jnp.abs(jnp.linalg.eigvals(A)))
    return R1p


@jit
@partial(jnp.vectorize, excluded=[0, 1, 3, 4], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4], signature="()->()")  # offsets
def gen_R1p_zaiss(model_parameters, offset, power, B0, gamma):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    dwa *= B0 * gamma
    dwb *= B0 * gamma
    offset *= B0 * gamma
    power *= gamma
    theta = jnp.arctan(power / (offset - dwa))
    R_eff = R1a * jnp.cos(theta) ** 2 + R2a * jnp.sin(theta) ** 2
    REXMAX = ((k * f * power**2) / ((offset - dwa) ** 2 + power**2)) * (
        (dwb) ** 2 + ((offset - dwa) ** 2 + power**2) * R2b / k + R2b * (k + R2b)
    )
    GAMMA = 2 * jnp.sqrt((k + R2b) / k * power**2 + (k + R2b) ** 2)
    R_ex = REXMAX / ((GAMMA / 2) ** 2 + (offset - dwb) ** 2)
    R1p = R_eff + R_ex
    return R1p

@jit
@partial(jnp.vectorize, excluded=[0, 1, 3, 4], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4], signature="()->()")  # offsets
def gen_R1p_laguerre(model_parameters, offset, power, B0, gamma):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    return -gen_eigenvalue_laguerre(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp)


@jit
@partial(jnp.vectorize, excluded=[0, 1, 3, 4], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4], signature="()->()")  # offsets
def gen_R1p_trace(model_parameters, offset, power, B0, gamma):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    return -gen_eigenvalue_trace(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp)


@partial(jnp.vectorize, excluded=[0, 1, 3, 4, 5], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4, 5], signature="()->()")  # offsets
def batch_gen_spectrum_analytical(model_parameters, offset, power, B0, gamma, tp):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    dwa *= B0 * gamma
    dwb *= B0 * gamma
    offset *= B0 * gamma
    power *= gamma
    theta = jnp.arctan(power / (offset - dwa))
    R_eff = R1a * jnp.cos(theta) ** 2 + R2a * jnp.sin(theta) ** 2
    REXMAX = ((k * f * power**2) / ((offset - dwa) ** 2 + power**2)) * (
        (dwb) ** 2 + ((offset - dwa) ** 2 + power**2) * R2b / k + R2b * (k + R2b)
    )
    GAMMA = 2 * jnp.sqrt((k + R2b) / k * power**2 + (k + R2b) ** 2)
    R_ex = REXMAX / ((GAMMA / 2) ** 2 + (offset - dwb) ** 2)
    R1p = R_eff + R_ex
    Z_ss = jnp.cos(theta) ** 2 * R1a / R1p
    Z = (jnp.cos(theta) ** 2 - Z_ss) * jnp.exp(-R1p * tp) + Z_ss
    return Z


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


@partial(jnp.vectorize, excluded=[0, 1, 3, 4, 5], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4, 5], signature="()->()")  # offsets
def batch_gen_spectrum_symbolic_laguerre(model_parameters, offset, power, B0, gamma, tp):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    lambda_1 = gen_eigenvalue_laguerre(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp)
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
            [0, 0, ka, 0, -power, -(R1b + k)],
        ]
    )
    b = jnp.array([0, 0, R1a, 0, 0, R1b * f])
    Z_ss = jnp.linalg.solve(A, -b)[2]
    cos2 = (offset - dwa) ** 2 / (power**2 + (offset - dwa) ** 2)
    Z = (cos2 - Z_ss) * jnp.exp(lambda_1 * tp) + Z_ss
    return Z


@partial(jnp.vectorize, excluded=[0, 1, 3, 4, 5], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4, 5], signature="()->()")  # offsets
def batch_gen_spectrum_symbolic_trace(model_parameters, offset, power, B0, gamma, tp):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    lambda_1 = gen_eigenvalue_trace(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp)
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
            [0, 0, ka, 0, -power, -(R1b + k)],
        ]
    )
    b = jnp.array([0, 0, R1a, 0, 0, R1b * f])
    Z_ss = jnp.linalg.solve(A, -b)[2]
    cos2 = (offset - dwa) ** 2 / (power**2 + (offset - dwa) ** 2)
    Z = (cos2 - Z_ss) * jnp.exp(lambda_1 * tp) + Z_ss
    return Z


offsets = jnp.linspace(6, -6, 200)
powers = jnp.array(2)
B0 = 9.4
gamma = 267.522
tp = 10
args = (offsets, powers, B0, gamma, tp)
#                    R1a   R2a  dwa  R1b  R2b   k      f     dwb
fit_pars = jnp.array([0.33, 0.5, 0.0, 50.0, 30, 20, 2e-2, 3.5])

R1p_numerical = gen_R1p_numerical(fit_pars, offsets, powers, B0, gamma)
R1p_zaiss = gen_R1p_zaiss(fit_pars, offsets, powers, B0, gamma)
R1p_Laguerre = gen_R1p_laguerre(fit_pars, offsets, powers, B0, gamma)
R1p_Trace = gen_R1p_trace(fit_pars, offsets, powers, B0, gamma)

Z_numerical = batch_gen_spectrum_numerical(fit_pars, offsets, powers, B0, gamma, tp)
Z_zaiss = batch_gen_spectrum_analytical(fit_pars, offsets, powers, B0, gamma, tp)
Z_symbolic_laguerre = batch_gen_spectrum_symbolic_laguerre(fit_pars, offsets, powers, B0, gamma, tp)
Z_symbolic_trace = batch_gen_spectrum_symbolic_trace(fit_pars, offsets, powers, B0, gamma, tp)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(offsets, R1p_numerical, linestyle="solid", label="Num.")
ax1.plot(offsets, R1p_zaiss, linestyle="dashed", label="zaiss")
ax1.plot(offsets, R1p_Laguerre, linestyle="dotted", color="r", label="laguerre")
ax1.plot(offsets, R1p_Trace, linestyle="dashdot", label="trace")
ax1.legend()

ax2.plot(offsets, Z_numerical, linestyle="solid", label="Num.")
ax2.plot(offsets, Z_zaiss, linestyle="dashed", label="zaiss")
ax2.plot(offsets, Z_symbolic_laguerre, linestyle="dotted", color="r", label="laguerre")
ax2.plot(offsets, Z_symbolic_trace, linestyle="dashdot", label="trace")
ax2.legend()
# %%
jnp.linalg.norm(Z_numerical - Z_symbolic_trace)
