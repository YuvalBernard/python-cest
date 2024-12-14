# %%
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from jax import config, jit

config.update("jax_enable_x64", True)


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


def gen_symbolic_expressions():
    R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp = sp.symbols(
        "R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp", real=True
    )
    ka = f * k
    # Define coefficient matrix of BM equations
    A = sp.Matrix(
        [
            [-(R2a + ka), (dwa - offset) * B0 * gamma, 0, k, 0, 0],
            [(offset - dwa) * B0 * gamma, -(R2a + ka), power * gamma, 0, k, 0],
            [0, -power * gamma, -(R1a + ka), 0, 0, k],
            [ka, 0, 0, -(R2b + k), (dwb - offset) * B0 * gamma, 0],
            [0, ka, 0, (offset - dwb) * B0 * gamma, -(R2b + k), power * gamma],
            [0, 0, ka, 0, -power * gamma, -(R1b + k)],
        ]
    )
    A_eff = A.subs(k, 0)
    lambda_eff = A_eff.det("lu") / A_eff.adjugate("lu").trace()
    A_rescaled = A - sp.eye(6) * lambda_eff
    lambda_1_rescale = A_rescaled.det("lu") / A_rescaled.adjugate("lu").trace() + lambda_eff
    coeffs = list(reversed(A.charpoly().coeffs()))
    lambda_1_laguerre = -6 * coeffs[0] / (coeffs[1] + sp.sqrt(25 * coeffs[1] ** 2 - 60 * coeffs[0] * coeffs[2]))
    lambda_1_newton = -coeffs[0] / coeffs[1]
    lambda_1_halley = -(2 * coeffs[0] * coeffs[1]) / (2 * coeffs[1] ** 2 - coeffs[0] * coeffs[2])

    R1p_rescale = sp.lambdify(
        [R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma],
        -lambda_1_rescale,
        modules="jax",
        cse=True,
        docstring_limit=0,
    )
    R1p_laguerre = sp.lambdify(
        [R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma],
        -lambda_1_laguerre,
        modules="jax",
        cse=True,
        docstring_limit=0,
    )
    R1p_netwon = sp.lambdify(
        [R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma],
        -lambda_1_newton,
        modules="jax",
        cse=True,
        docstring_limit=0,
    )
    R1p_halley = sp.lambdify(
        [R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma],
        -lambda_1_halley,
        modules="jax",
        cse=True,
        docstring_limit=0,
    )
    return R1p_rescale, R1p_laguerre, R1p_netwon, R1p_halley


R1p_rescale, R1p_laguerre, R1p_newton, R1p_halley = gen_symbolic_expressions()


@jit
@partial(jnp.vectorize, excluded=[0, 1, 3, 4], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4], signature="()->()")  # offsets
def gen_R1p_rescale(model_parameters, offset, power, B0, gamma):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    return R1p_rescale(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma)


@jit
@partial(jnp.vectorize, excluded=[0, 1, 3, 4], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4], signature="()->()")  # offsets
def gen_R1p_laguerre(model_parameters, offset, power, B0, gamma):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    return R1p_laguerre(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma)


@jit
@partial(jnp.vectorize, excluded=[0, 1, 3, 4], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4], signature="()->()")  # offsets
def gen_R1p_newton(model_parameters, offset, power, B0, gamma):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    return R1p_newton(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma)


@jit
@partial(jnp.vectorize, excluded=[0, 1, 3, 4], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4], signature="()->()")  # offsets
def gen_R1p_halley(model_parameters, offset, power, B0, gamma):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    return R1p_halley(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma)


offsets = jnp.linspace(20, -20, 500)
powers = jnp.array(1.5)
B0 = 11.7
gamma = 251.815
tp = 5
args = (offsets, powers, B0, gamma, tp)
#                    R1a   R2a  dwa  R1b  R2b   k      f     dwb
fit_pars = jnp.array([0.65, 10, 0.0, 70.0, 200, 250, 1.5e-1, -17.5])

R1p_numerical = gen_R1p_numerical(fit_pars, offsets, powers, B0, gamma)
R1p_zaiss = gen_R1p_zaiss(fit_pars, offsets, powers, B0, gamma)
R1p_Rescale = gen_R1p_rescale(fit_pars, offsets, powers, B0, gamma)
R1p_Newton = gen_R1p_newton(fit_pars, offsets, powers, B0, gamma)
R1p_Laguerre = gen_R1p_laguerre(fit_pars, offsets, powers, B0, gamma)
R1p_Halley = gen_R1p_halley(fit_pars, offsets, powers, B0, gamma)

jnp.linalg.norm(R1p_numerical - R1p_Laguerre)

# %%
ax = plt.gca()
ax.plot(offsets, jnp.exp(-R1p_numerical * tp), linestyle="solid", label="Num.")
ax.plot(offsets, jnp.exp(-R1p_Halley * tp), linestyle="dashed", label="halley")
ax.plot(offsets, jnp.exp(-R1p_Rescale * tp), linestyle="dotted", color="r", label="rescale")
ax.plot(offsets, jnp.exp(-R1p_Newton * tp), linestyle="dashdot", label="netwon")
ax.plot(offsets, jnp.exp(-R1p_Laguerre * tp), linestyle="dashdot", label="laguerre")
ax.legend()
