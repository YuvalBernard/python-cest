# %% Load modules
import inspect

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from pyutil import filereplace
from sympy.physics.quantum import TensorProduct

plt.style.use("bmh")
jax.config.update("jax_enable_x64", True)


# %% Solve Bloch-McConnell equations for 3 pools
def solve_bloch_mcconnell_3_pools():
    (
        R1a,
        R2a,
        R1b,
        R2b,
        R1c,
        R2c,
        kb,
        kc,
        fb,
        fc,
        offset,
        dwa,
        dwb,
        dwc,
        power,
        tp,
        B0,
        gamma
    ) = sp.symbols("R1a, R2a, R1b, R2b, R1c, R2c,  kb, kc,  fb, fc,  offset, dwa, dwb, dwc,  power, tp, B0, gamma", real=True)
    kab = fb * kb
    kac = fc * kc
    L_a = sp.Matrix(
        [
            [-R2a, (dwa - offset) * B0 * gamma, 0],
            [(offset - dwa) * B0 * gamma, -R2a, power * gamma],
            [0, -power * gamma, -R1a],
        ]
    )
    L_b = sp.Matrix(
        [
            [-R2b, (dwb - offset) * B0 * gamma, 0],
            [(offset - dwb) * B0 * gamma, -R2b, power * gamma],
            [0, -power * gamma, -R1b],
        ]
    )
    L_c = sp.Matrix(
        [
            [-R2c, (dwc - offset) * B0 * gamma, 0],
            [(offset - dwc) * B0 * gamma, -R2c, power * gamma],
            [0, -power * gamma, -R1c],
        ]
    )
    K = sp.Matrix(
        [
            [-(kab + kac), kb, kc],
            [kab, -kb, 0],
            [kac, 0, -kc],
        ]
    )
    # Define coefficient matrix of BM equations
    A = sp.diag(L_a, L_b, L_c) + TensorProduct(K, sp.eye(3))
    b = sp.Matrix([0, 0, R1a, 0, 0, R1b * fb, 0, 0, R1c * fc])
    lambda_eff = R1a + (R2a - R1a) * (power**2) / (power**2 + (B0 * (offset - dwa)) ** 2)
    A_rescaled = A - sp.eye(A.shape[0]) * lambda_eff
    det_A_rescaled = A_rescaled.det("lu")
    N = A.shape[0]
    lambda_1 = (
        det_A_rescaled / (sp.Matrix([A_rescaled.cofactor(i, i) for i in range(N)]).dot(sp.ones(N, 1))) + lambda_eff
    )
    v1 = sp.Matrix([power * gamma, 0, offset * B0 * gamma]).normalized()
    # Calculate steady-state solution
    Z_ss = A.LUsolve(-b)[2]
    # Simulate spectrum according to Zaiss 2013, 10.1002/nbm.2887
    Z = (v1[2] ** 2 - Z_ss) * sp.exp(lambda_1 * tp) + Z_ss
    gen_eigenvalue = sp.lambdify(
        [
            R1a,
            R2a,
            R1b,
            R2b,
            R1c,
            R2c,
            kb,
            kc,
            fb,
            fc,
            offset,
            dwa,
            dwb,
            dwc,
            power,
            B0,
            gamma
        ],
        lambda_1,
        modules="jax",
        cse=True,
        docstring_limit=0,
    )
    gen_spectrum = sp.lambdify(
        [
            R1a,
            R2a,
            R1b,
            R2b,
            R1c,
            R2c,
            kb,
            kc,
            fb,
            fc,
            offset,
            dwa,
            dwb,
            dwc,
            power,
            tp,
            B0,
            gamma
        ],
        Z,
        modules="jax",
        cse=True,
        docstring_limit=0,
    )
    with open("solve_bloch_mcconnell_3_pools.py", "w") as text_file:
        text_file.write("import jax.numpy as jnp\n")
        text_file.write(inspect.getsource(gen_spectrum))

    filereplace("solve_bloch_mcconnell_3_pools.py", "_lambdifygenerated", "gen_spectrum_symbolic")
    filereplace("solve_bloch_mcconnell_3_pools.py", "exp", "jnp.exp")
    filereplace("solve_bloch_mcconnell_3_pools.py", "abs", "jnp.absolute")
    filereplace("solve_bloch_mcconnell_3_pools.py", "sqrt", "jnp.sqrt")
    return gen_spectrum, gen_eigenvalue


gen_spectrum_symbolic, gen_eigenvalue_symbolic = solve_bloch_mcconnell_3_pools()

# # %% Check eigenvalue and Z-spectrum validity
# import sys

# from functools import partial

# from solve_bloch_mcconnell_3_pools import gen_spectrum_symbolic


# @jax.jit
# @partial(jnp.vectorize, excluded=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15], signature="()->(k)")  # powers
# def batch_gen_spectrum_symbolic(R1a, R2a, R1b, R2b, R1c, R2c, kb, kc, fb, fc, offset, dwa, dwb, dwc, power, tp):
#     return gen_spectrum_symbolic(R1a, R2a, R1b, R2b, R1c, R2c, kb, kc, fb, fc, offset, dwa, dwb, dwc, power, tp)


# @jax.jit
# @partial(jnp.vectorize, excluded=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15], signature="()->(k)")  # powers
# @partial(jnp.vectorize, excluded=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15], signature="()->()")  # offsets
# def batch_gen_spectrum_numerical(R1a, R2a, R1b, R2b, R1c, R2c, kb, kc, fb, fc, offset, dwa, dwb, dwc, power, tp):
#     dwa *= B0 * gamma
#     dwb *= B0 * gamma
#     dwc *= B0 * gamma
#     offset *= B0 * gamma
#     power *= gamma

#     kab = fb * kb
#     kac = fc * kc
#     L_a = jnp.array([[-R2a, dwa - offset, 0], [offset - dwa, -R2a, power], [0, -power, -R1a]])
#     L_b = jnp.array([[-R2b, dwb - offset, 0], [offset - dwb, -R2b, power], [0, -power, -R1b]])
#     L_c = jnp.array([[-R2c, dwc - offset, 0], [offset - dwc, -R2c, power], [0, -power, -R1c]])
#     K = jnp.array(
#         [
#             [-(kab + kac), kb, kc],
#             [kab, -kb, 0],
#             [kac, 0, -kc],
#         ]
#     )
#     # Define coefficient matrix of BM equations
#     A = jax.scipy.linalg.block_diag(L_a, L_b, L_c) + jnp.linalg.tensordot(K, jnp.eye(3))
#     b = jnp.array([0, 0, R1a, 0, 0, R1b * fb, 0, 0, R1c * fc])

#     M0 = jnp.array([0, 0, 1, 0, 0, fb, 0, 0, fc])
#     M_ss = jnp.linalg.solve(A, -b)
#     Z = ((M0 - M_ss) @ jax.scipy.linalg.expm(A * tp, max_squarings=18) + M_ss)[2]
#     return Z


# R1a = 1 / 3.0
# R2a = 1 / 2.0
# dwa = 0.0
# R1b = 1.0
# R2b = 30.0
# R1c = 2.0
# R2c = 15.0
# kb = 200.0
# kc = 30.0
# fb = 5e-4
# fc = 1e-3
# dwb = 3.5
# dwc = 2.0
# offsets = jnp.arange(-6, 6, 0.01, dtype=float)
# powers = jnp.array([1.0, 3.0], dtype=float)
# B0 = 4.7
# gamma = 267.522
# tp = 15.0

# Z_symbolic = batch_gen_spectrum_symbolic(
#     R1a, R2a, R1b, R2b, R1c, R2c, kb, kc, fb, fc, offsets, dwa, dwb, dwc, powers, tp
# )
# Z_numerical = batch_gen_spectrum_numerical(
#     R1a, R2a, R1b, R2b, R1c, R2c, kb, kc, fb, fc, offsets, dwa, dwb, dwc, powers, tp
# )

# fig, axs = plt.subplots(ncols=2)
# fig.set_figwidth(14)
# axs[0].plot(offsets, Z_symbolic.T)
# axs[1].plot(offsets, Z_numerical.T)
