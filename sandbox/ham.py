# %%
import sympy as sp
R1a, R2a, dwa_, R1b, R2b, k, f, dwb_, offset_, power_, B0, gamma, tp = sp.symbols(
    "R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp", real=True
)
offset = offset_ * B0 * gamma
dwa = dwa_ * B0 * gamma
dwb = dwb_ * B0 * gamma
power = power_ * gamma
ka = f * k
# Define coefficient matrix of BM equations
A = sp.Matrix(
    [
        [-(R2a + ka), dwa - offset, 0, k, 0, 0],
        [offset - dwa, -(R2a + ka), power, 0, k, 0],
        [0, -power, -(R1a + ka), 0, 0, k],
        [ka, 0, 0, -(R2b + k), dwb - offset, 0],
        [0, ka, 0, offset - dwb, -(R2b + k), power],
        [0, 0, ka, 0, -power, -(R1b + k)],
    ]
)
A_prime = sp.Matrix(
    [
        [-ka, dwa - offset, 0, k, 0, 0],
        [offset - dwa, -ka, power, 0, k, 0],
        [0, -power, -ka, 0, 0, k],
        [ka, 0, 0, -(R2b + k), dwb - offset, 0],
        [0, ka, 0, offset - dwb, -(R2b + k), power],
        [0, 0, ka, 0, -power, -(R1b + k)],
    ]
)
sin2 = power**2 / (power**2 + (offset-dwa)**2)
cos2 = (offset-dwa)**2 / (power**2 + (offset-dwa)**2)
b_eff = sp.Matrix([power, 0, offset - dwa]).normalized()
b = sp.Matrix([0, 0, R1a, 0, 0, R1b * f])

coeffs = list(reversed(A.charpoly().coeffs()))
# lambda_1 = -coeffs[0] / (coeffs[1]  - coeffs[2]*coeffs[0]/coeffs[1])

lambda_eff = -(R1a * cos2 + R2a * sin2)
coeffs_prime = list(reversed(A_prime.charpoly().coeffs()))
lambda_ex_laguerre = -coeffs_prime[0] / (coeffs_prime[1]  - coeffs_prime[2]*coeffs_prime[0]/coeffs_prime[1])
lambda_ex_trace = -coeffs_prime[0]/coeffs_prime[1]

lambda_ex_trace
