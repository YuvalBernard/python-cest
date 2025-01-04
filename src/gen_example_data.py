import os

import numpy as np
import pandas as pd

from simulate_spectra import batch_gen_spectrum_numerical as gen_spectrum

offsets = np.arange(-175, 175, 5, dtype=float)
powers = np.array([5.8, 11.6, 23.2, 57.9])
B0 = 14.1
gamma = -27.116
tp = 1.0
args = (offsets, powers, B0, gamma, tp)
#                    R1b   R2b  dwa  R1b  R2b   k      f     dwb
fit_pars = np.array([1.0, 10.0, 0.0, 1.0, 10_000, 10.0, 0.05, 0])
Z = gen_spectrum(fit_pars, offsets, powers, B0, gamma, tp)
rng = np.random.default_rng()
sigma = 0.02
data = rng.normal(Z, sigma)

df = pd.DataFrame(np.c_[offsets, data.T], columns=["ppm"] + [f"{power:.3g} μT" for power in powers])
with pd.ExcelWriter(os.path.join(os.getcwd(), "paper_nytrogen.xlsx")) as writer:
    df.to_excel(writer, sheet_name="data", index=False, float_format="%.3g")
