# %% Load modules
import inspect

import jax
import sympy as sp
from pyutil import filereplace

jax.config.update("jax_enable_x64", True)


# Save solution to Bloch-McConnell equations to file.
def gen_symbolic_expressions():
    R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp = sp.symbols(
        "R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp", real=True
    )
    ka = f * k
    # Define coefficient matrix of BM equations
    A_lab = sp.Matrix(
        [
            [-(R2a + ka), (dwa - offset) * B0 * gamma, 0, k, 0, 0],
            [(offset - dwa) * B0 * gamma, -(R2a + ka), power * gamma, 0, k, 0],
            [0, -power * gamma, -(R1a + ka), 0, 0, k],
            [ka, 0, 0, -(R2b + k), (dwb - offset) * B0 * gamma, 0],
            [0, ka, 0, (offset - dwb) * B0 * gamma, -(R2b + k), power * gamma],
            [0, 0, ka, 0, -power * gamma, -(R1b + k)],
        ]
    )
    A_ref = sp.Matrix(
        [
            [-(R2a + ka), (dwa + offset) * B0 * gamma, 0, k, 0, 0],
            [(-offset - dwa) * B0 * gamma, -(R2a + ka), power * gamma, 0, k, 0],
            [0, -power * gamma, -(R1a + ka), 0, 0, k],
            [ka, 0, 0, -(R2b + k), (dwb + offset) * B0 * gamma, 0],
            [0, ka, 0, (-offset - dwb) * B0 * gamma, -(R2b + k), power * gamma],
            [0, 0, ka, 0, -power * gamma, -(R1b + k)],
        ]
    )

    # Define non-homogeneous part of BM-equations
    b = sp.Matrix([0, 0, R1a, 0, 0, R1b * f])
    theta_lab = sp.atan(power / (B0 * (offset - dwa)))
    theta_ref = sp.atan(power / (B0 * (-offset - dwa)))
    # lambda_eff = (R1a * sp.cos(theta) + R2a * sp.sin(theta)).simplify()

    A_ex_lab = A_lab.subs(k, 0)
    A_ex_ref = A_ref.subs(k, 0)

    lambda_eff_lab = A_ex_lab.det("lu") / A_ex_lab.adjugate("lu").trace()
    lambda_eff_ref = A_ex_ref.det("lu") / A_ex_ref.adjugate("lu").trace()

    A_rescaled_lab = A_lab - sp.eye(6) * lambda_eff_lab
    A_rescaled_ref = A_ref - sp.eye(6) * lambda_eff_ref

    lambda_ex_lab = A_rescaled_lab.det("lu") / A_rescaled_lab.adjugate("lu").trace()
    lambda_ex_ref = A_rescaled_ref.det("lu") / A_rescaled_ref.adjugate("lu").trace()

    lambda_1_lab = lambda_ex_lab + lambda_eff_lab
    lambda_1_ref = lambda_ex_ref + lambda_eff_ref

    Z_ss_lab = A_lab.LUsolve(-b)[2]
    Z_ss_ref = A_ref.LUsolve(-b)[2]

    # Simulate spectrum according to Zaiss 2013, 10.1002/nbm.2887
    Z_lab = (sp.cos(theta_lab) ** 2 - Z_ss_lab) * sp.exp(lambda_1_lab * tp) + Z_ss_lab
    Z_ref = (sp.cos(theta_ref) ** 2 - Z_ss_ref) * sp.exp(lambda_1_ref * tp) + Z_ss_ref

    MTR_Assym = Z_ref - Z_lab
    MTR_Assym_direct_ss = Z_ss_ref - Z_ss_lab

    gen_MTR_Assym = sp.lambdify(
        [R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp],
        MTR_Assym,
        modules="jax",
        cse=True,
        docstring_limit=0,
    )
    gen_MTR_Assym_direct_ss = sp.lambdify(
        [R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp],
        MTR_Assym_direct_ss,
        modules="jax",
        cse=True,
        docstring_limit=0,
    )

    with open("src/gen_MTR_Assym.py", "w") as text_file:
        text_file.write("import jax.numpy as jnp\n")
        text_file.write(inspect.getsource(gen_MTR_Assym))

    filereplace("src/gen_MTR_Assym.py", "_lambdifygenerated", "gen_spectrum_symbolic")
    filereplace("src/gen_MTR_Assym.py", "exp", "jnp.exp")
    filereplace("src/gen_MTR_Assym.py", "abs", "jnp.absolute")
    filereplace("src/gen_MTR_Assym.py", "sqrt", "jnp.sqrt")

    with open("src/gen_MTR_Assym_direct_ss.py", "w") as text_file:
        text_file.write("import jax.numpy as jnp\n")
        text_file.write(inspect.getsource(gen_MTR_Assym_direct_ss))

    filereplace("src/gen_MTR_Assym_direct_ss.py", "_lambdifygenerated", "gen_spectrum_symbolic")
    filereplace("src/gen_MTR_Assym_direct_ss.py", "exp", "jnp.exp")
    filereplace("src/gen_MTR_Assym_direct_ss.py", "abs", "jnp.absolute")
    filereplace("src/gen_MTR_Assym_direct_ss.py", "sqrt", "jnp.sqrt")

    return gen_MTR_Assym, gen_MTR_Assym_direct_ss


gen_MTR_Assym, gen_MTR_Assym_direct_ss = gen_symbolic_expressions()
