import jax.numpy as jnp
def gen_eigenvalue_trace(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp):
    x0 = gamma**2
    x1 = power**2
    x2 = x0*x1
    x3 = (-B0*dwa*gamma + B0*gamma*offset)**2
    x4 = (x2 + x3)**(-1.0)
    x5 = gamma**4
    x6 = power**4*x5
    x7 = R2b*x6
    x8 = k**3
    x9 = x2*x8
    x10 = R1b*x5
    x11 = B0**4
    x12 = offset**4*x11
    x13 = x10*x12
    x14 = k*x5
    x15 = R2b**2
    x16 = R1b*x2
    x17 = x15*x16
    x18 = k**2
    x19 = x16*x18
    x20 = k*x15
    x21 = 2*R2b
    x22 = k*x21
    x23 = x18*x2
    x24 = dwa**2
    x25 = B0**2
    x26 = x0*x25
    x27 = x26*x8
    x28 = x24*x27
    x29 = offset**2
    x30 = x27*x29
    x31 = 2*dwa
    x32 = offset*x31
    x33 = offset**3
    x34 = x11*x33
    x35 = x10*x34
    x36 = 2*dwb
    x37 = x14*x34
    x38 = R1b*x26
    x39 = dwa*offset
    x40 = 4*R2b*x39
    x41 = x15*x38
    x42 = x18*x38
    x43 = x24*x42
    x44 = dwb**2
    x45 = x10*x44
    x46 = x1*x25
    x47 = x45*x46
    x48 = x29*x42
    x49 = x10*x29
    x50 = x46*x49
    x51 = x24*x46
    x52 = R2b*x5
    x53 = x29*x46
    x54 = x20*x26
    x55 = x14*x51
    x56 = x14*x44
    x57 = x11*x24
    x58 = x45*x57
    x59 = x49*x57
    x60 = x11*x29
    x61 = x45*x60
    x62 = x14*x57
    x63 = x22*x38
    x64 = offset*x36
    x65 = x10*x64
    x66 = x18*x26
    x67 = x14*x46
    x68 = 4*dwa*dwb
    x69 = x11*x49*x68
    x70 = x11*x32
    x71 = x21*x66
    x72 = x14*x53
    x73 = (-k*x38*x40 + k*x6 + x12*x14 + x13 + x14*x60*x68 + x16*x22 + x17 + x19 + x2*x20 + x21*x23 - x21*x39*x46*x5 + x24*x41 + x24*x54 + x24*x63 + x24*x71 - x27*x32 + x28 + x29*x41 + x29*x54 + x29*x62 + x29*x63 + x29*x71 + x30 - x31*x35 - x31*x37 - x32*x41 - x32*x42 - x32*x54 - x32*x67 - x35*x36 - x36*x37 - x40*x66 + x43 - x45*x70 + x46*x56 - x46*x65 + x47 + x48 + x50 + x51*x52 + x52*x53 + x55 + x56*x57 + x56*x60 - x56*x70 - x57*x65 + x58 + x59 + x61 - x62*x64 - x64*x67 + x69 + x7 + 2*x72 + x9)**(-1.0)
    x74 = k*x73
    x75 = R2b*x73
    x76 = R1b*x73
    x77 = x20*x38*x73
    x78 = x18*x5*x73
    x79 = x21*x73
    return -R1a*x3*x4 - R2a*x2*x4 + f*(4*R1b*R2b*dwa*offset*x0*x18*x25*x73 + 2*R1b*dwa*k*offset*x0*x15*x25*x73 + 2*R1b*dwa*k*offset*x11*x44*x5*x73 + 2*R1b*dwa*k*x11*x33*x5*x73 + 2*R1b*dwa*offset*x0*x25*x73*x8 + 2*R1b*dwb*k*offset*x1*x25*x5*x73 + 2*R1b*dwb*k*offset*x11*x24*x5*x73 + 2*R1b*dwb*k*x11*x33*x5*x73 + 2*R2b*dwa*k*offset*x1*x25*x5*x73 + 2*dwa*dwb*x1*x18*x25*x5*x73 - x13*x74 - x15*x23*x73 - x17*x74 - x19*x75 - x24*x77 - x28*x76 - x29*x77 - x30*x76 - x43*x79 - x44*x46*x78 - x47*x74 - x48*x79 - x50*x74 - x51*x78 - x55*x75 - x58*x74 - x59*x74 - x61*x74 - x69*x74 - x7*x74 - x72*x75 - x75*x9)
