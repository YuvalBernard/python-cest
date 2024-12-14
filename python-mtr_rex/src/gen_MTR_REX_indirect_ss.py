import jax.numpy as jnp
def gen_spectrum_symbolic(R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp):
    x0 = dwa + offset
    x1 = -x0
    x2 = B0**2
    x3 = power**2
    x4 = x3/x2
    x5 = gamma**2
    x6 = -R2b
    x7 = R2b**(-1.0)
    x8 = x2*x5
    x9 = x7*x8
    x10 = dwb + offset
    x11 = -x10
    x12 = x10*x11
    x13 = x12*x9 + x6
    x14 = -R1b + x3*x5/x13
    x15 = R2a*R2b
    x16 = R1a*x15
    x17 = x13*x14
    x18 = -R2a
    x19 = R2a**(-1.0)
    x20 = x19*x8
    x21 = x0*x1
    x22 = x18 + x20*x21
    x23 = x15*x22
    x24 = x23*(-R1a + x3*x5/x22)
    x25 = x3*x5
    x26 = x15*(-R1a - x19*x25)
    x27 = -R1b - x25*x7
    x28 = x13*x24
    x29 = x14*x28/(-R1b*x24 - x16*x17 + x17*x23 + x17*x26 + x24*x27 + x28)
    x30 = f*k
    x31 = R2a + x30
    x32 = x29 + x31
    x33 = -x32
    x34 = x33**(-1.0)
    x35 = k**2
    x36 = f*x35
    x37 = x34*x36
    x38 = k + x29
    x39 = R2b + x38
    x40 = x37 + x39
    x41 = -x40
    x42 = x12*x8
    x43 = -x39 - x42/x41
    x44 = R1b + x38
    x45 = x3*x5/x43 - x44
    x46 = R1a + x30
    x47 = x29 + x46
    x48 = f**2
    x49 = gamma**4
    x50 = k**4
    x51 = x33**(-2.0)
    x52 = x21*x8
    x53 = -x32 - x34*x52
    x54 = x53**(-2.0)
    x55 = x53**(-1.0)
    x56 = x3*x5*x55 - x47
    x57 = x56**(-1.0)
    x58 = x36*x57
    x59 = -x44 - x58
    x60 = x59**(-1.0)
    x61 = x36*x55
    x62 = x40 + x51*x52*x61
    x63 = -f*x0*x1*x2*x3*x35*x49*x51*x54*x57 + x62
    x64 = -x0*x1*x2*x3*x48*x49*x50*x51*x54*x60/x56**2 + x63
    x65 = x33*x53
    x66 = x56*x65
    x67 = x59*x66
    x68 = gamma*power
    x69 = x55*x58*x68 + x68
    x70 = -x60*x69
    x71 = x39 + x61
    x72 = -f*x3*x35*x5*x54*x57 + x71
    x73 = -x69*x70 - x72
    x74 = x67*x73
    x75 = x3*x34*x5 - x47
    x76 = x75**(-1.0)
    x77 = x36*x76
    x78 = -x44 - x77
    x79 = x34*x68*x77 + x68
    x80 = f*x3*x35*x5*x51*x76 - x40 + x79**2/x78
    x81 = -x71
    x82 = x81**(-1.0)
    x83 = B0*gamma
    x84 = x0*x37
    x85 = x55*x83
    x86 = x10*x83 + x84*x85
    x87 = x1*x37
    x88 = x11*x83 + x85*x87
    x89 = -x62 - x82*x86*x88
    x90 = -x72
    x91 = B0*x57
    x92 = x84*x91
    x93 = gamma**3
    x94 = x3*x54*x93
    x95 = x86 - x92*x94
    x96 = x87*x91
    x97 = x94*x96
    x98 = power*x5
    x99 = x55*x98
    x100 = -offset
    x101 = dwa + x100
    x102 = -x101
    x103 = dwb + x100
    x104 = -x103
    x105 = x103*x104
    x106 = x105*x9 + x6
    x107 = -R1b + x3*x5/x106
    x108 = x106*x107
    x109 = x101*x102
    x110 = x109*x20 + x18
    x111 = x110*x15
    x112 = x111*(-R1a + x3*x5/x110)
    x113 = x106*x112
    x114 = x107*x113/(-R1b*x112 + x108*x111 - x108*x16 + x108*x26 + x112*x27 + x113)
    x115 = x114 + x31
    x116 = -x115
    x117 = x116**(-1.0)
    x118 = x117*x36
    x119 = k + x114
    x120 = R2b + x119
    x121 = x118 + x120
    x122 = -x121
    x123 = x105*x8
    x124 = -x120 - x123/x122
    x125 = R1b + x119
    x126 = -x125 + x3*x5/x124
    x127 = x114 + x46
    x128 = x116**(-2.0)
    x129 = x109*x8
    x130 = -x115 - x117*x129
    x131 = x130**(-2.0)
    x132 = x130**(-1.0)
    x133 = -x127 + x132*x3*x5
    x134 = x133**(-1.0)
    x135 = x134*x36
    x136 = -x125 - x135
    x137 = x136**(-1.0)
    x138 = x132*x36
    x139 = x121 + x128*x129*x138
    x140 = -f*x101*x102*x128*x131*x134*x2*x3*x35*x49 + x139
    x141 = -x101*x102*x128*x131*x137*x2*x3*x48*x49*x50/x133**2 + x140
    x142 = x116*x130
    x143 = x133*x142
    x144 = x136*x143
    x145 = x132*x135*x68 + x68
    x146 = -x137*x145
    x147 = x120 + x138
    x148 = -f*x131*x134*x3*x35*x5 + x147
    x149 = -x145*x146 - x148
    x150 = x144*x149
    x151 = x117*x3*x5 - x127
    x152 = x151**(-1.0)
    x153 = x152*x36
    x154 = -x125 - x153
    x155 = x117*x153*x68 + x68
    x156 = f*x128*x152*x3*x35*x5 - x121 + x155**2/x154
    x157 = -x147
    x158 = x157**(-1.0)
    x159 = x132*x83
    x160 = x102*x118
    x161 = x104*x83 + x159*x160
    x162 = x101*x118
    x163 = x103*x83 + x159*x162
    x164 = -x139 - x158*x161*x163
    x165 = -x148
    x166 = x131*x3*x93
    x167 = B0*x134
    x168 = x160*x167
    x169 = x166*x168
    x170 = x162*x167
    x171 = x163 - x166*x170
    x172 = x132*x98
    return (1 + x4/x1**2)*(x29 + x74*(-x64 - (x70*x92*x99 + x95)*(B0*f*gamma*x1*x34*x35*x55 + B0*gamma*x11 - x60*x69*x96*x99 - x97)/x73)/(x33*x41*x43*x45*(-x36/x45 - x47) + x33*x75*x78*x80*(-x39 - x42/x80) - x64*x67 + x65*x81*x89*(x3*x5*x82 + x3*x5*x86*x88/(x81**2*x89) - x44) + x66*x90*(-x63 - x95*(x88 - x97)/x90) + x74)) + (1 + x4/x102**2)*(-x114 - x150*(-x141 - (x146*x170*x172 + x171)*(B0*f*gamma*x102*x117*x132*x35 + B0*gamma*x104 - x137*x145*x168*x172 - x169)/x149)/(x116*x122*x124*x126*(-x127 - x36/x126) + x116*x151*x154*x156*(-x120 - x123/x156) - x141*x144 + x142*x157*x164*(-x125 + x158*x3*x5 + x161*x163*x3*x5/(x157**2*x164)) + x143*x165*(-x140 - x171*(x161 - x169)/x165) + x150))