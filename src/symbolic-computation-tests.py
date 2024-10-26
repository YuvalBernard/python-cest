# %% Load modules
import inspect
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from pyutil import filereplace

plt.style.use("bmh")
jax.config.update("jax_enable_x64", True)


# %% Save solution to Bloch-McConnell equations to file.
def solve_bloch_mcconnell():
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
    # Define non-homogeneous part of BM-equations
    b = sp.Matrix([0, 0, R1a, 0, 0, R1b * f])
    lambda_eff = R1a + (R2a - R1a) * ((power * gamma) ** 2) / (
        (power * gamma) ** 2 + (B0 * gamma * (offset - dwa)) ** 2
    )
    # A_perturbed = A.subs(k, 0)
    # coeffs = list(reversed(A_perturbed.charpoly().coeffs()))
    # lambda_eff = -6 * coeffs[0] / (coeffs[1] + sp.sqrt(25 * coeffs[1] ** 2 - 60 * coeffs[0] * coeffs[2]))
    A_rescaled = A - sp.eye(6) * lambda_eff
    lambda_1 = A_rescaled.det("berkowitz") / A_rescaled.adjugate("berkowitz").trace() + lambda_eff
    v1 = sp.Matrix([power * gamma, 0, offset * B0 * gamma]).normalized()
    # Calculate steady-state solution
    Z_ss = A.LUsolve(-b)[2]
    # Simulate spectrum according to Zaiss 2013, 10.1002/nbm.2887
    Z = (v1[2] ** 2 - Z_ss) * sp.exp(lambda_1 * tp) + Z_ss
    gen_eigenvalue = sp.lambdify(
        [R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma],
        lambda_1,
        modules="jax",
        cse=True,
        docstring_limit=0,
    )
    gen_spectrum = sp.lambdify(
        [R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp],
        Z,
        modules="jax",
        cse=True,
        docstring_limit=0,
    )
    with open("lib/solve_bloch_mcconnell.py", "w") as text_file:
        text_file.write("import jax.numpy as jnp\n")
        text_file.write(inspect.getsource(gen_spectrum))

    filereplace("lib/solve_bloch_mcconnell.py", "_lambdifygenerated", "gen_spectrum_symbolic")
    filereplace("lib/solve_bloch_mcconnell.py", "exp", "jnp.exp")
    filereplace("lib/solve_bloch_mcconnell.py", "abs", "jnp.absolute")
    filereplace("lib/solve_bloch_mcconnell.py", "sqrt", "jnp.sqrt")
    return gen_spectrum, gen_eigenvalue


gen_spectrum_symbolic, gen_eigenvalue_symbolic = solve_bloch_mcconnell()


# %% Check eigenvalue and Z-spectrum validity
@jax.jit
@partial(jnp.vectorize, excluded=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12], signature="()->(k)")  # powers
def batch_gen_spectrum_symbolic(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offsets, power, B0, gamma, tp):
    return gen_spectrum_symbolic(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offsets, power, B0, gamma, tp)


@jax.jit
@partial(jnp.vectorize, excluded=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11], signature="()->(k)")  # powers
def batch_gen_eigenvalue_symbolic(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offsets, power, B0, gamma):
    return gen_eigenvalue_symbolic(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offsets, power, B0, gamma)


@jax.jit
@partial(jnp.vectorize, excluded=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12], signature="()->()")  # offsets
def batch_gen_spectrum_numerical(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp):
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
    Z = jnp.matmul(jax.scipy.linalg.expm(A * tp, max_squarings=18), M0)[2]
    return Z


@jax.jit
@partial(jnp.vectorize, excluded=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11], signature="()->()")  # offsets
def batch_gen_eigenvalue_numerical(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma):
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
    eigs = jnp.linalg.eigvals(A)
    return -jnp.min(jnp.absolute(eigs))


@partial(jnp.vectorize, excluded=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12], signature="()->(k)")  # powers
@partial(jnp.vectorize, excluded=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12], signature="()->()")  # offsets
def batch_gen_spectrum_analytical(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp):
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


R1a = 0.64
R2a = 11.47
dwa = 0.0
R1b = 50.0
R2b = 260
k = 260.0
f = 0.01
dwb = -17.8
offsets = jnp.arange(-25, 15, 0.1, dtype=float)
powers = jnp.array([1.0, 3.0], dtype=float)
B0 = 11.7
gamma = 251.815
tp = 4.72

# R1a = 0.33
# R2a = 0.5
# dwa = 0.0
# R1b = 1.0
# R2b = 30.0
# k = 200.0
# f = 5e-4
# dwb = 3.5
# offsets = jnp.arange(-6, 6, 0.01, dtype=float)
# powers = jnp.array([1.0, 3.0], dtype=float)
# B0 = 4.7
# gamma = 267.522
# tp = 15.0

Z_symbolic = batch_gen_spectrum_symbolic(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offsets, powers, B0, gamma, tp)
Z_numerical = batch_gen_spectrum_numerical(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offsets, powers, B0, gamma, tp)
Z_analytical = batch_gen_spectrum_analytical(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offsets, powers, B0, gamma, tp)
# lambda_1_symbolic = batch_gen_eigenvalue_symbolic(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offsets, powers, B0, gamma)
# lambda_1_numerical = batch_gen_eigenvalue_numerical(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offsets, powers, B0, gamma)

fig, axs = plt.subplots(ncols=3)
fig.set_figwidth(14)
axs[0].plot(offsets, Z_symbolic.T)
axs[1].plot(offsets, Z_numerical.T)
axs[2].plot(offsets, Z_analytical.T)
axs[0].set_title("Symbolic")
axs[1].set_title("Numerical")
axs[2].set_title("Analytical")
fig.show()
