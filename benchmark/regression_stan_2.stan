data {
  int N;
  vector[N] x1;
  vector[N] x2;
  vector[N] x3;
  vector[N] y;
}
parameters {
  real w1;
  real w2;
  real w3;
  real b;
}
model {
  b ~ normal(0,5);
  w1 ~ normal(0,5);
  w2 ~ normal(0,5);
  w3 ~ normal(0,5);
  y ~ normal(b + w1 * x1 + w2 * x2 + w3 * x3, 1);
}
