functions {
  real solve_bloch_mcconnell(real R1a, real R2a, real dwa, real R1b, real R2b, real k, real f, real dwb, real w_rf, real power, real B0, real gamma, real tp){
      real x0 = f * k;
      real x1 = R2a + x0;
      real x2 = -x1;
      real x3 = R2b + k;
      real x4 = gamma^2;
      real x5 = power^2;
      real x6 = x4 * x5;
      real x7 = R1b * f;
      real x8 = x0 * x3;
      real x9 = -R1b - k;
      real x10 = R1a * x9;
      real x11 = -x3;
      real x12 = f^2;
      real x13 = k^2;
      real x14 = x12 * x13;
      real x15 = R1b * x14;
      real x16 = x11 * x15;
      real x17 = x11 * x3;
      real x18 = B0 * gamma;
      real x19 = -B0 * gamma * w_rf;
      real x20 = dwb * x18 + x19;
      real x21 = -x20;
      real x22 = x20 * x21;
      real x23 = x11 * x9 + x6;
      real x24 = x11 * x23 + x21^2 * x9;
      real x25 = -R1a * x24 - k * (x17 * x7 + x22 * x7);
      real x26 = -dwa * x18 - x19;
      real x27 = x0 * x21;
      real x28 = x10 * x27;
      real x29 = f^3;
      real x30 = k^3;
      real x31 = k * x6;
      real x32 = x0 * x23;
      real x33 = x21 * x6;
      real x34 = f * x13;
      real x35 = x12 * x30;
      real x36 = R1a + x0;
      real x37 = x36 * x9;
      real x38 = x27 * x37;
      real x39 = -x36;
      real x40 = -k * (-x0 * x17 - x0 * x22) + x24 * x39;
      real x41 = gamma * power;
      real x42 = x11 * x35;
      real x43 = (-k * (-k * (-R1a * x14 * x9 + R1b * x12 * x31 + R1b * x29 * x30) + x1 * (R1a * x32 - x16) + x26 * (x15 * x20 + x28 - x33 * x7)) + x2 * (-k * (x10 * x8 + x16 - x3 * x6 * x7) + x2 * x25) + x26 * (-k * (-x15 * x21 + x28) + x25 * x26)) / (-k * (-k * (-k^4 * x29 + x12 * x13 * x39 * x9 - x14 * x6) + x1 * (x32 * x36 + x42) + x26 * (-x20 * x35 + x27 * x6 + x38) + x41 * (x32 * x41 + x35 * x41)) + x2 * (-k * (x37 * x8 - x42 + x6 * x8) + x2 * x40 - x41 * (-x24 * x41 + x3 * x34 * x41)) + x26 * (-k * (x21 * x35 + x38) + x26 * x40 - x33 * x34));
      real x44 = x26^2;
      real x45 = (x44 + x6)^(-1.0);
      real x46 = R2b * x13;
      real x47 = x46 * x5;
      real x48 = R1b * x5;
      real x49 = R2b * k;
      real x50 = x48 * x49;
      real x51 = B0^2;
      real x52 = R2b^2;
      real x53 = power^4 * x4;
      real x54 = k * x52;
      real x55 = dwa^2;
      real x56 = R1b * x51;
      real x57 = x52 * x56;
      real x58 = w_rf^2;
      real x59 = x13 * x56;
      real x60 = B0^4 * x4;
      real x61 = w_rf^4 * x60;
      real x62 = dwb^2;
      real x63 = x56 * x6;
      real x64 = x55 * x60;
      real x65 = R1b * x64;
      real x66 = R1b * x60;
      real x67 = x58 * x62;
      real x68 = x51 * x55;
      real x69 = R2b * x6;
      real x70 = x51 * x58;
      real x71 = x31 * x51;
      real x72 = dwa * w_rf;
      real x73 = 2 * x72;
      real x74 = 2 * w_rf^3;
      real x75 = dwa * x66;
      real x76 = dwb * x74;
      real x77 = x49 * x56;
      real x78 = 2 * x77;
      real x79 = 4 * x72;
      real x80 = x62 * x73;
      real x81 = 2 * dwb * w_rf;
      real x82 = 4 * dwb * x58;
      real x83 = R1b * x61 + R2b * x53 + x31 * x68 + x48 * x52 + x5 * x54 - x51 * x69 * x73 + x55 * x57 + x55 * x59 + x55 * x78 + x57 * x58 - x57 * x73 + x58 * x59 + x58 * x63 + x58 * x65 + x58 * x78 - x59 * x73 + x62 * x63 + x62 * x65 + x62 * x71 - x63 * x81 - x65 * x81 + x66 * x67 - x66 * x76 - x66 * x80 + x68 * x69 + x69 * x70 - x74 * x75 + x75 * x82 - x77 * x79;
      real x84 = x30 * x51;
      real x85 = x51 * x54;
      real x86 = 2 * x46;
      real x87 = k * x64;
      real x88 = k * x60;
      real x89 = dwa * x88;
      real result = x43 + (-x43 + x44 * x45) * exp(tp * (-R1a * x44 * x45 - R2a * x45 * x6 + f * k * (2 * dwa * dwb * k * x4 * x5 * x51 - x47 - x50 - x83) / (k * x53 + k * x61 + x13 * x48 + x30 * x5 + 2 * x31 * x70 - x46 * x51 * x79 + 2 * x47 + 2 * x50 + x55 * x84 + x55 * x85 + x58 * x84 + x58 * x85 + x58 * x87 + x62 * x87 + x67 * x88 + x68 * x86 + x70 * x86 - x71 * x73 - x71 * x81 - x73 * x84 - x73 * x85 - x74 * x89 - x76 * x88 - x80 * x88 - x81 * x87 + x82 * x89 + x83)));
      return result;
    }
}

data {
  int N;
  int M;
  real<lower=0> b0;
  real<lower=0> gamma;
  real<lower=0> tp;
  array[N] real w_rfs;
  array[M] real powers;
  array[N] vector[M] y;

  real dwa_min;
  real<lower=0> R1a_min;
  real<lower=0> R2a_min;
  real dwb_min;
  real<lower=0> R1b_min;
  real<lower=0> R2b_min;
  real<lower=0> kb_min;
  real<lower=0> fb_min;
  real dwa_max;
  real<lower=0> R1a_max;
  real<lower=0> R2a_max;
  real dwb_max;
  real<lower=0> R1b_max;
  real<lower=0> R2b_max;
  real<lower=0> kb_max;
  real<lower=0> fb_max;
}

transformed data {
  real dwa_mean = (dwa_min + dwa_max)/2;
  real dwa_std = (dwa_max - dwa_min)/6;
  real R1a_mean = (R1a_min + R1a_max)/2;
  real R1a_std = (R1a_max - R1a_min)/6;
  real R2a_mean = (R2a_min + R2a_max)/2;
  real R2a_std = (R2a_max - R2a_min)/6;
  real dwb_mean = (dwb_min + dwb_max)/2;
  real dwb_std = (dwb_max - dwb_min)/6;
  real R1b_mean = (R1b_min + R1b_max)/2;
  real R1b_std = (R1b_max - R1b_min)/6;
  real R2b_mean = (R2b_min + R2b_max)/2;
  real R2b_std = (R2b_max - R2b_min)/6;
  real kb_mean = (kb_min + kb_max)/2;
  real kb_std = (kb_max - kb_min)/6;
  real fb_mean = (fb_min + fb_max)/2;
  real fb_std = (fb_max - fb_min)/6;
}

parameters {
  real<lower=dwa_min,upper=dwa_max> dwa;
  real<lower=R1a_min,upper=R1a_max> R1a;
  real<lower=R2a_min,upper=R2a_max> R2a;
  real<lower=dwb_min,upper=dwb_max> dwb;
  real<lower=R1b_min,upper=R1b_max> R1b;
  real<lower=R2b_min,upper=R2b_max> R2b;
  real<lower=kb_min,upper=kb_max> kb;
  real<lower=fb_min,upper=fb_max> fb;
  real<lower=0> sigma;
}
model {
  array[N] vector[M] y_hat;

  dwa ~ normal(dwa_mean, dwa_std);
  R1a ~ normal(R1a_mean, R1a_std);
  R2a ~ normal(R2a_mean, R2a_std);
  dwb ~ normal(dwb_mean, dwb_std);
  R1b ~ normal(R1b_mean, R1b_std);
  R2b ~ normal(R2b_mean, R2b_std);
  kb ~ normal(kb_mean, kb_std);
  fb ~ normal(fb_mean, fb_std);
  sigma ~ gamma(2, 100);

  for (m in 1:M) {
    for (n in 1:N) {
      y_hat[n, m] = solve_bloch_mcconnell(R1a, R2a, dwa, R1b, R2b, kb, fb, dwb, w_rfs[n], powers[m], b0, gamma, tp);
    }
  }

  for (i in 1:N) {
    y[i] ~ normal(y_hat[i], sigma);
  }
}