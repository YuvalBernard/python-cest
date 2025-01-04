import arviz as az
import os
from cmdstanpy import CmdStanModel
import pandas as pd
import numpy as np


b0 = 9.4
gamma = 267.522
tp = 10.0

dwa_min = -1
dwa_max = 1
R1a_min = 0.2
R1a_max = 0.4
R2a_min = 0.5
R2a_max = 0.8
dwb_min = 3
dwb_max = 4
R1b_min = 0.1
R1b_max = 10
R2b_min = 1
R2b_max = 100
kb_min = 1
kb_max = 100
fb_min = 1e-4
fb_max = 1e-2

df = pd.read_excel("paper_hydrogen.xlsx")
b1_list = list(df.columns[1:].str.extract(r"([-+]?\d*\.?\d+)", expand=False))

if df.columns[1:].str.contains("T").any():
    powers = np.array([float(b1) for b1 in b1_list])
else:
    powers = np.array([float(b1)/gamma for b1 in b1_list])

offsets = df.to_numpy(dtype=float).T[0]
y = df.to_numpy(dtype=float).T[1:]


stan_file = os.path.join("bloch_mcconnell.stan")
model = CmdStanModel(stan_file=stan_file, cpp_options = {"CXXFLAGS": "-O3"})
data = {
    "N": len(offsets),
    "M": len(powers),
    "b0": b0,
    "gamma": gamma,
    "tp": tp,
    "w_rfs": offsets,
    "powers": powers,
    "y": y.T,
    "dwa_min": dwa_min,
    "R1a_min": R1a_min,
    "R2a_min": R2a_min,
    "dwb_min": dwb_min,
    "R1b_min": R1b_min,
    "R2b_min": R2b_min,
    "kb_min": kb_min,
    "fb_min": fb_min,
    "dwa_max": dwa_max,
    "R1a_max": R1a_max,
    "R2a_max": R2a_max,
    "dwb_max": dwb_max,
    "R1b_max": R1b_max,
    "R2b_max": R2b_max,
    "kb_max": kb_max,
    "fb_max": fb_max
}
fit = model.pathfinder(data=data)
inits = fit.create_inits()
fit2 = model.sample(data=data, inits=inits, iter_warmup=100)
print(fit2.summary())
print(fit2.diagnose())