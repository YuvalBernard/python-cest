"""
Module comprising different methods to simulate Z-spectra (multi-B1).
Includes:
    - Analytical solution by Moritz Zaiss
    - Numerical solution
    - Symbolic solution
"""

from functools import partial

import jax.numpy as jnp
from jax.scipy.linalg import expm

from solve_bloch_mcconnell import gen_spectrum_symbolic


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
def batch_gen_spectrum_symbolic(model_parameters, offset, power, B0, gamma, tp):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    return gen_spectrum_symbolic(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp)
