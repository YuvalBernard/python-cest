"""
Module comprising different methods to simulate Z-spectra (multi-B1).
Includes:
    - Analytical solution by Moritz Zaiss
    - Numerical solution
    - Symbolic solution
"""

from functools import partial

import jax.numpy as jnp
from jax import config
from jax.scipy.linalg import expm

config.update("jax_platform_name", "cpu")


import gen_MTR_Assym
import gen_MTR_Assym_direct_ss


@partial(jnp.vectorize, excluded=[0, 1, 3, 4, 5], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4, 5], signature="()->()")  # offsets
def batch_gen_MTR_Assym_analytical(model_parameters, offset, power, B0, gamma, tp):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    dwa *= B0 * gamma
    dwb *= B0 * gamma
    offset *= B0 * gamma
    power *= gamma
    theta_lab = jnp.arctan(power / (offset - dwa))
    theta_ref = jnp.arctan(power / (-offset - dwa))

    R_eff_lab = R1a * jnp.cos(theta_lab) ** 2 + R2a * jnp.sin(theta_lab) ** 2
    R_eff_ref = R1a * jnp.cos(theta_ref) ** 2 + R2a * jnp.sin(theta_ref) ** 2

    Rex_MAX_lab = ((k * f * power**2) / ((offset - dwa) ** 2 + power**2)) * (
        (dwb) ** 2 + ((offset - dwa) ** 2 + power**2) * R2b / k + R2b * (k + R2b)
    )
    Rex_MAX_ref = ((k * f * power**2) / ((-offset - dwa) ** 2 + power**2)) * (
        (dwb) ** 2 + ((-offset - dwa) ** 2 + power**2) * R2b / k + R2b * (k + R2b)
    )

    GAMMA = 2 * jnp.sqrt((k + R2b) / k * power**2 + (k + R2b) ** 2)

    R_ex_lab = Rex_MAX_lab / ((GAMMA / 2) ** 2 + (offset - dwb) ** 2)
    R_ex_ref = Rex_MAX_ref / ((GAMMA / 2) ** 2 + (-offset - dwb) ** 2)

    R1p_lab = R_eff_lab + R_ex_lab
    R1p_ref = R_eff_ref + R_ex_ref

    Z_ss_lab = jnp.cos(theta_lab) ** 2 * R1a / R1p_lab
    Z_ss_ref = jnp.cos(theta_ref) ** 2 * R1a / R1p_ref

    Z_lab = (jnp.cos(theta_lab) ** 2 - Z_ss_lab) * jnp.exp(-R1p_lab * tp) + Z_ss_lab
    Z_ref = (jnp.cos(theta_ref) ** 2 - Z_ss_ref) * jnp.exp(-R1p_ref * tp) + Z_ss_ref
    return Z_ref - Z_lab


@partial(jnp.vectorize, excluded=[0, 1, 3, 4, 5], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4, 5], signature="()->()")  # offsets
def batch_gen_MTR_Assym_ss_analytical(model_parameters, offset, power, B0, gamma, tp):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    dwa *= B0 * gamma
    dwb *= B0 * gamma
    offset *= B0 * gamma
    power *= gamma
    theta_lab = jnp.arctan(power / (offset - dwa))
    theta_ref = jnp.arctan(power / (-offset - dwa))

    R_eff_lab = R1a * jnp.cos(theta_lab) ** 2 + R2a * jnp.sin(theta_lab) ** 2
    R_eff_ref = R1a * jnp.cos(theta_ref) ** 2 + R2a * jnp.sin(theta_ref) ** 2

    Rex_MAX_lab = ((k * f * power**2) / ((offset - dwa) ** 2 + power**2)) * (
        (dwb) ** 2 + ((offset - dwa) ** 2 + power**2) * R2b / k + R2b * (k + R2b)
    )
    Rex_MAX_ref = ((k * f * power**2) / ((-offset - dwa) ** 2 + power**2)) * (
        (dwb) ** 2 + ((-offset - dwa) ** 2 + power**2) * R2b / k + R2b * (k + R2b)
    )

    GAMMA = 2 * jnp.sqrt((k + R2b) / k * power**2 + (k + R2b) ** 2)

    R_ex_lab = Rex_MAX_lab / ((GAMMA / 2) ** 2 + (offset - dwb) ** 2)
    R_ex_ref = Rex_MAX_ref / ((GAMMA / 2) ** 2 + (-offset - dwb) ** 2)

    R1p_lab = R_eff_lab + R_ex_lab
    R1p_ref = R_eff_ref + R_ex_ref

    Z_ss_lab = jnp.cos(theta_lab) ** 2 * R1a / R1p_lab
    Z_ss_ref = jnp.cos(theta_ref) ** 2 * R1a / R1p_ref

    return Z_ss_ref - Z_ss_lab


@partial(jnp.vectorize, excluded=[0, 1, 3, 4, 5], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4, 5], signature="()->()")  # offsets
def batch_gen_MTR_Assym_numerical(model_parameters, offset, power, B0, gamma, tp):
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
    return Z_ref - Z_lab


@partial(jnp.vectorize, excluded=[0, 1, 3, 4, 5], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 2, 3, 4, 5], signature="()->()")  # offsets
def batch_gen_MTR_Assym_ss_numerical(model_parameters, offset, power, B0, gamma, tp):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    dwa *= B0 * gamma
    dwb *= B0 * gamma
    offset *= B0 * gamma
    power *= gamma
    ka = k * f
    A_lab = jnp.array(
        [
            [-(R2a + ka), dwa - offset, 0, k, 0, 0],
            [offset - dwa, -(R2a + ka), power, 0, k, 0],
            [0, -power, -(R1a + ka), 0, 0, k],
            [ka, 0, 0, -(R2b + k), dwb - offset, 0],
            [0, ka, 0, offset - dwb, -(R2b + k), power],
            [0, 0, ka, 0, -power, -(R1b + k)],
        ]
    )
    A_ref = jnp.array(
        [
            [-(R2a + ka), dwa + offset, 0, k, 0, 0],
            [-offset - dwa, -(R2a + ka), power, 0, k, 0],
            [0, -power, -(R1a + ka), 0, 0, k],
            [ka, 0, 0, -(R2b + k), dwb + offset, 0],
            [0, ka, 0, -offset - dwb, -(R2b + k), power],
            [0, 0, ka, 0, -power, -(R1b + k)],
        ]
    )
    b = jnp.array([0, 0, R1a, 0, 0, R1b * f])
    Z_ss_lab = jnp.linalg.solve(A_lab, -b)[2]
    Z_ss_ref = jnp.linalg.solve(A_ref, -b)[2]
    return Z_ss_ref - Z_ss_lab


@partial(jnp.vectorize, excluded=[0, 1, 3, 4, 5], signature="()->(k)")  # powers
def batch_gen_MTR_Assym_symbolic(model_parameters, offset, power, B0, gamma, tp):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    return gen_MTR_Assym.gen_spectrum_symbolic(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp)


@partial(jnp.vectorize, excluded=[0, 1, 3, 4, 5], signature="()->(k)")  # powers
def batch_gen_MTR_Assym_ss_direct_symbolic(model_parameters, offset, power, B0, gamma, tp):
    R1a, R2a, dwa, R1b, R2b, k, f, dwb = model_parameters
    return gen_MTR_Assym_direct_ss.gen_spectrum_symbolic(
        R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp
    )
