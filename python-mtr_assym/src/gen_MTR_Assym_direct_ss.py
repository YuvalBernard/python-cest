import jax.numpy as jnp
def gen_spectrum_symbolic(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp):
    x0 = gamma**2
    x1 = power**2
    x2 = f*k
    x3 = R2a + x2
    x4 = -x3
    x5 = x4**(-1.0)
    x6 = -offset
    x7 = dwa + x6
    x8 = -x7
    x9 = B0**2
    x10 = x0*x9
    x11 = x10*x7
    x12 = -x11*x5*x8 - x3
    x13 = x12**(-1.0)
    x14 = R1a + x2
    x15 = x0*x1*x13 - x14
    x16 = x15**(-1.0)
    x17 = k**2
    x18 = f*x17
    x19 = x16*x18
    x20 = x13*x18
    x21 = x4**(-2.0)
    x22 = x21*x8
    x23 = gamma**4
    x24 = x12**(-2.0)
    x25 = x18*x5
    x26 = R2b + k
    x27 = x25 + x26
    x28 = (f*x1*x16*x17*x21*x23*x24*x7*x8*x9 - x11*x20*x22 - x27)**(-1.0)
    x29 = dwb + x6
    x30 = B0*gamma
    x31 = x13*x30
    x32 = x25*x8
    x33 = gamma**3
    x34 = x1*x24
    x35 = x33*x34
    x36 = B0*x16
    x37 = x32*x36
    x38 = x28*(-x29*x30 + x31*x32 - x35*x37)
    x39 = x25*x7
    x40 = x36*x39
    x41 = power*x0
    x42 = x13*x41
    x43 = x40*x42
    x44 = gamma*power
    x45 = x13*x44
    x46 = x19*x45 + x44
    x47 = x38*x43 + x46
    x48 = x29*x30 + x31*x39 - x35*x40
    x49 = (f*x0*x1*x16*x17*x24 - x20 - x26 - x38*x48)**(-1.0)
    x50 = x49*(-x28*x37*x42*x48 - x46)
    x51 = f**2
    x52 = k**4
    x53 = x15**(-2.0)
    x54 = R1b + k
    x55 = R1b*f
    x56 = x23*x9
    x57 = R1a*k**3*x51
    x58 = R1a*x2
    x59 = x16*x58
    x60 = B0*x13*x41*x5*x59*x7
    x61 = x38*x60 + x45*x59
    x62 = (R1a*f*k*x16 - x22*x28*x34*x53*x56*x57*x7 + x50*x61 - x55)/(x1*x21*x23*x24*x28*x51*x52*x53*x7*x8*x9 - x19 - x47*x50 - x54)
    x63 = x49*(-x47*x62 - x61)
    x64 = dwa + offset
    x65 = x5*x64
    x66 = -x64
    x67 = x10*x66
    x68 = -x3 - x65*x67
    x69 = x68**(-1.0)
    x70 = x0*x1*x69 - x14
    x71 = x70**(-1.0)
    x72 = x18*x71
    x73 = x18*x69
    x74 = x21*x64
    x75 = x68**(-2.0)
    x76 = (f*x1*x17*x21*x23*x64*x66*x71*x75*x9 - x27 - x67*x73*x74)**(-1.0)
    x77 = dwb + offset
    x78 = x30*x73
    x79 = x5*x66
    x80 = x1*x33*x72*x75
    x81 = B0*x79
    x82 = x76*(-x30*x77 + x78*x79 - x80*x81)
    x83 = B0*x65
    x84 = x41*x69*x83
    x85 = x72*x84
    x86 = x44*x69
    x87 = x44 + x72*x86
    x88 = x82*x85 + x87
    x89 = x30*x77 + x65*x78 - x80*x83
    x90 = (f*x0*x1*x17*x71*x75 - x26 - x73 - x82*x89)**(-1.0)
    x91 = x90*(-x41*x69*x72*x76*x81*x89 - x87)
    x92 = x70**(-2.0)
    x93 = x58*x71
    x94 = x84*x93
    x95 = x82*x94 + x86*x93
    x96 = (R1a*f*k*x71 - x1*x56*x57*x66*x74*x75*x76*x92 - x55 + x91*x95)/(x1*x21*x23*x51*x52*x64*x66*x75*x76*x9*x92 - x54 - x72 - x88*x91)
    x97 = x90*(-x88*x96 - x95)
    return -x16*(B0*k*power*x0*x13*x28*x5*x8*(x43*x62 - x48*x63 + x60) - R1a - k*x45*x63 - k*x62) + x71*(B0*k*power*x0*x5*x66*x69*x76*(x85*x96 - x89*x97 + x94) - R1a - k*x86*x97 - k*x96)
