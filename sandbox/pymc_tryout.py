import matplotlib.pyplot as plt

from functools import partial
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from simulate_spectra import batch_gen_spectrum_numerical
import numpyro, blackjax
# numpyro.set_host_device_count(8)
import arviz as az

@partial(pt.vectorize, signature="(),(),(),(),(),(),(),(),(),(),(),(),(a)->(a)")
def gen_spectrum_symbolic(R1a, R2a, dwa, R1b, R2b, k, f, dwb, B0, gamma, tp, offset, power):
    x0 = f*k
    x1 = R2a + x0
    x2 = -x1
    x3 = R2b + k
    x4 = gamma**2
    x5 = power**2
    x6 = x4*x5
    x7 = R1b*f
    x8 = x0*x3
    x9 = -R1b - k
    x10 = R1a*x9
    x11 = -x3
    x12 = f**2
    x13 = k**2
    x14 = x12*x13
    x15 = R1b*x14
    x16 = x11*x15
    x17 = x11*x3
    x18 = B0*gamma
    x19 = -B0*gamma*offset
    x20 = dwb*x18 + x19
    x21 = -x20
    x22 = x20*x21
    x23 = x11*x9 + x6
    x24 = x11*x23 + x21**2*x9
    x25 = -R1a*x24 - k*(x17*x7 + x22*x7)
    x26 = -dwa*x18 - x19
    x27 = x0*x21
    x28 = x10*x27
    x29 = f**3
    x30 = k**3
    x31 = k*x6
    x32 = x0*x23
    x33 = x21*x6
    x34 = f*x13
    x35 = x12*x30
    x36 = R1a + x0
    x37 = x36*x9
    x38 = x27*x37
    x39 = -x36
    x40 = -k*(-x0*x17 - x0*x22) + x24*x39
    x41 = gamma*power
    x42 = x11*x35
    x43 = (-k*(-k*(-R1a*x14*x9 + R1b*x12*x31 + R1b*x29*x30) + x1*(R1a*x32 - x16) + x26*(x15*x20 + x28 - x33*x7)) + x2*(-k*(x10*x8 + x16 - x3*x6*x7) + x2*x25) + x26*(-k*(-x15*x21 + x28) + x25*x26))/(-k*(-k*(-k**4*x29 + x12*x13*x39*x9 - x14*x6) + x1*(x32*x36 + x42) + x26*(-x20*x35 + x27*x6 + x38) + x41*(x32*x41 + x35*x41)) + x2*(-k*(x37*x8 - x42 + x6*x8) + x2*x40 - x41*(-x24*x41 + x3*x34*x41)) + x26*(-k*(x21*x35 + x38) + x26*x40 - x33*x34))
    x44 = x26**2
    x45 = (x44 + x6)**(-1.0)
    x46 = R2b*x13
    x47 = x46*x5
    x48 = R1b*x5
    x49 = R2b*k
    x50 = x48*x49
    x51 = B0**2
    x52 = R2b**2
    x53 = power**4*x4
    x54 = k*x52
    x55 = dwa**2
    x56 = R1b*x51
    x57 = x52*x56
    x58 = offset**2
    x59 = x13*x56
    x60 = B0**4*x4
    x61 = offset**4*x60
    x62 = dwb**2
    x63 = x56*x6
    x64 = x55*x60
    x65 = R1b*x64
    x66 = R1b*x60
    x67 = x58*x62
    x68 = x51*x55
    x69 = R2b*x6
    x70 = x51*x58
    x71 = x31*x51
    x72 = dwa*offset
    x73 = 2*x72
    x74 = 2*offset**3
    x75 = dwa*x66
    x76 = dwb*x74
    x77 = x49*x56
    x78 = 2*x77
    x79 = 4*x72
    x80 = x62*x73
    x81 = 2*dwb*offset
    x82 = 4*dwb*x58
    x83 = R1b*x61 + R2b*x53 + x31*x68 + x48*x52 + x5*x54 - x51*x69*x73 + x55*x57 + x55*x59 + x55*x78 + x57*x58 - x57*x73 + x58*x59 + x58*x63 + x58*x65 + x58*x78 - x59*x73 + x62*x63 + x62*x65 + x62*x71 - x63*x81 - x65*x81 + x66*x67 - x66*x76 - x66*x80 + x68*x69 + x69*x70 - x74*x75 + x75*x82 - x77*x79
    x84 = x30*x51
    x85 = x51*x54
    x86 = 2*x46
    x87 = k*x64
    x88 = k*x60
    x89 = dwa*x88
    return x43 + (-x43 + x44*x45)*pt.exp(tp*(-R1a*x44*x45 - R2a*x45*x6 + f*k*(2*dwa*dwb*k*x4*x5*x51 - x47 - x50 -
                                                                              x83)/(k*x53 + k*x61 + x13*x48 + x30*x5 + 2*x31*x70 - x46*x51*x79 + 2*x47 + 2*x50 + x55*x84 + x55*x85 + x58*x84 + x58*x85 + x58*x87 + x62*x87 + x67*x88 + x68*x86 + x70*x86 - x71*x73 - x71*x81 - x73*x84 - x73*x85 - x74*x89 - x76*x88 - x80*x88 - x81*x87 + x82*x89 + x83)))

R1a_par = pt.scalar("R1a")
R2a_par = pt.scalar("R2a")
dwa_par = pt.scalar("dwa")
R1b_par = pt.scalar("R1b")
R2b_par = pt.scalar("R2b")
kb_par = pt.scalar("kb")
fb_par = pt.scalar("fb")
dwb_par = pt.scalar("dwb")
offset_par = pt.vector("offset")
power_par = pt.vector("power")
B0_par = pt.scalar("B0")
gamma_par = pt.scalar("gamma")
tp_par = pt.scalar("tp")

Z = gen_spectrum_symbolic(R1a_par, R2a_par, dwa_par, R1b_par, R2b_par, kb_par, fb_par, dwb_par, B0_par, gamma_par, tp_par, offset_par, power_par)
model_fn = pytensor.function(inputs=[R1a_par, R2a_par, dwa_par, R1b_par, R2b_par, kb_par, fb_par, dwb_par, B0_par,
                                gamma_par, tp_par, offset_par, power_par], outputs=Z)

offsets = np.linspace(-6, 6, 75)
powers = np.array([0.5, 2, 5])
B0 = 7.0
gamma = 267.522
tp = 10.0
args = (offsets, powers, B0, gamma, tp)

R1a = 0.33  # Hz
R2a = 0.67  # Hz
dwa = 0  # ppm
R1b = 1.0  # Hz
R2b = 66.67  # Hz
kb = 30  # Hz
fb = 0.007  # dimensionless.
dwb = 3.5  # ppm

Z = model_fn(R1a, R2a, dwa, R1b, R2b, kb, fb, dwb, B0, gamma, tp, offsets, powers)

rng = np.random.default_rng()
sigma = 0.02
data = rng.normal(Z, sigma)

with pm.Model() as model:
    # constants
    B0 = pm.Data("B0", B0)
    gamma = pm.Data("gamma", gamma)
    tp = pm.Data("tp", tp)
    offsets = pm.Data("offsets", offsets)
    powers = pm.Data("powers", powers)
    # priors
    R1a = pm.TruncatedNormal("R1a", mu=0.55, sigma=0.15, lower=0.1, upper=1)
    R2a = pm.TruncatedNormal("R2a", mu=0.55, sigma=0.15, lower=0.1, upper=1)
    dwa = pm.TruncatedNormal("dwa", mu=0, sigma=0.33, lower=-1, upper=1)
    R1b = pm.TruncatedNormal("R1b", mu=5.05, sigma=1.65, lower=0.1, upper=10)
    R2b = pm.TruncatedNormal("R2b", mu=50.5, sigma=16.5, lower=1, upper=100)
    kb = pm.TruncatedNormal("kb", mu=500.5, sigma=166.5, lower=1, upper=1000)
    fb = pm.TruncatedNormal("fb", mu=0.005, sigma=0.0016, lower=1e-6, upper=1e-2)
    dwb = pm.TruncatedNormal("dwb", mu=3.5, sigma=0.16, lower=3, upper=4)
    sigma = pm.Gamma("sigma", alpha=2, beta=100)
    # model prediction
    pred = gen_spectrum_symbolic(R1a, R2a, dwa, R1b, R2b, kb, fb, dwb, B0, gamma, tp, offsets, powers)
    # model likelihood
    likelihood = pm.Normal("y", mu=pred, sigma=sigma, observed=data)
    # sample
    idata = pm.sample(cores=4, progressbar=True)

print(az.summary(idata, round_to=5))