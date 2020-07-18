data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] x;
  vector[N] y;
}
parameters {
  real alpha;
  vector[K] beta;
  real s;
}
model {
  s ~ uniform(0.5, 8);
  alpha ~ normal(0, 5);
  beta ~ normal(0, 5);
  y ~ normal(alpha + x * beta, s * s + 2);
}
