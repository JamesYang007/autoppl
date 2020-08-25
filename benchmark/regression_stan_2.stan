data {
  int N;
  int K;
  matrix[N, K] X;
  vector[N] y;
}
parameters {
  vector[K] w;
  real b;
}
model {
  b ~ normal(0,5);
  w ~ normal(0,5);
  y ~ normal(b + X*w, 1);
}
