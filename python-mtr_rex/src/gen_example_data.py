import os

import numpy as np
import openpyxl
import pandas as pd

from simulate_spectra import batch_gen_spectrum_numerical as gen_spectrum

offsets = np.arange(6, -6, -0.15, dtype=float)
powers = np.array([1.5, 3.0, 5.0, 7.5])
B0 = 9.4
gamma = 267.522
tp = 10.0
args = (offsets, powers, B0, gamma, tp)
#                    R1b   R2b  dwa  R1b  R2b   k      f     dwb
fit_pars = np.array([0.33, 0.5, 0.0, 1.0, 30.0, 200.0, 5e-4, 3.5])
Z = gen_spectrum(fit_pars, offsets, powers, B0, gamma, tp)
rng = np.random.default_rng()
sigma = 0.02
data = rng.normal(Z, sigma)

df = pd.DataFrame(np.c_[offsets, data.T], columns=["ppm"] + [f"{power:.3g} μT" for power in powers])
with pd.ExcelWriter(os.path.join(os.getcwd(), "example_data.xlsx")) as writer:
    df.to_excel(writer, sheet_name="data", index=False, float_format="%.3g")
