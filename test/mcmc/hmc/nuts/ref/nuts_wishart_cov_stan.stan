data {
    int N;
    vector[N] y;
    cov_matrix[N] V;
}

transformed data {
    vector[N] mu;
    for (i in 1:N) {
        mu[i] = 0;
    }
}

parameters {
    cov_matrix[N] Sigma;
}

model {
    Sigma ~ wishart(2, V);
    y ~ multi_normal(mu, Sigma);
}
