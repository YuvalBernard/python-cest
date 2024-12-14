# %%
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.linalg import expm


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

# %% simulate data
offsets = np.arange(-6, 6, 0.15, dtype=float)
powers = 1.5
B0 = 9.4
gamma = 267.522
tp = 10.0
args = (offsets, powers, B0, gamma, tp)
#                    R1b   R2b  dwa  R1b  R2b   k      f     dwb
fit_pars = np.array([0.33, 0.5, 0.0, 1.0, 30.0, 200.0, 4e-4, 3.5])
Z = batch_gen_spectrum_numerical(fit_pars, offsets, powers, B0, gamma, tp)
rng = np.random.default_rng()
sigma = 0.01
data = rng.normal(Z, sigma)

peaks, _ = find_peaks(-data, prominence=0.1)
plt.plot(offsets, data)
plt.plot(offsets[peaks], data[peaks], 'r*')





# %%
#

def Z_to_MTR_REX(x_zspec, zspec):
    # Locate peaks

def arex_ps(x_zspec, zspec):
    """
    Calculates AREX of z-spectra.

    Parameters:
    x_zspec (numpy array): x values of the z-spectrum.
    Z (numpy array): 1D, 2D, 3D or 4D z-spectrum or zspec stack.
    R1A (numpy array, optional): R1A-value, 2D-matrix or 3D-matrix. Defaults to 1.

    Returns:
    x_arex (numpy array): new x vector.
    AREX (numpy array): AREX vector or 4D stack.
    """

    # Create new x vector
    int1 = x_zspec[1:]
    int2 = x_zspec[:-1]
    step = np.abs(np.min(int1 - int2))
    offset = np.max([np.abs(np.min(x_zspec)), np.abs(np.max(x_zspec))])
    x_zspec_int = np.arange(-offset, offset, step)
    int3 = np.arange(len(x_zspec_int) // 2, 0, -1)
    int4 = np.arange(len(x_zspec_int) // 2 + 1, len(x_zspec_int) + 1)
    x_arex = np.arange(0, offset + step, step)

    # Initialize AREX array
    AREX = np.zeros(len(int4) + 1)
    # Interpolate zspec data
    y_zspec_int = PchipInterpolator(x_zspec, zspec)(x_zspec_int)
    # Calculate AREX-vector
    AREX[0] = 0
    AREX[1:] = 1 / y_zspec_int[int4] - 1 / y_zspec_int[int3]

    return x_arex, AREX




# %%
x_zspec = np.arange(-6, 6, 0.1, dtype=float)
z_spec = batch_gen_spectrum_numerical(fit_pars, x_zspec, powers, B0, gamma, tp)
if (x_zspec[0] >= x_zspec[1] and x_zspec[-2] >= x_zspec[-1]): # array is decreasing
    x_zspec = np.flip(x_zspec)
    z_spec = np.flip(z_spec)

int1 = x_zspec[1:]
int2 = x_zspec[:-1]
step = np.abs(np.min(int1 - int2))
offset = np.max([np.abs(np.min(x_zspec)), np.abs(np.max(x_zspec))])
x_zspec_int = np.arange(-offset, offset, step)
midpoint = len(x_zspec_int) // 2
int3 = np.arange(midpoint-1, -1, -1)
int4 = np.arange(midpoint, len(x_zspec_int))
x_arex = np.arange(0, offset, step)
# Interpolate zspec data
y_zspec_int = PchipInterpolator(x_zspec, z_spec)(x_zspec_int)
AREX = 1 / y_zspec_int[int4] - 1 / y_zspec_int[int3]
plt.plot(x_arex, AREX)
