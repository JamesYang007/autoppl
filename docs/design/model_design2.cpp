Y ~ W.x + epsilon
Y ~ N(W.x, sigma^2)

Parameter<double> X {4.0};  // observed
Parameter<double> Y {5.0}; // observed
Parameter<double> W; // hidden
​
Model m1 = Model(  // Model class defines a distribution over existing Parameters.
	W |= Uniform(-10, 10), // linear regression
	Y |= Normal(W * X, 3), // overload multiplication to build a graph from W * X
​);

Model m2 = Model(
	W |= Normal(0, 1), // ridge regression instead
	Y |= Normal(W * X, 3),
);
​
m1.sample(1000);

(3*x).pdf(10) => x.pdf(10 / 3)

X.observe(3); // observe more data

// P(Y, W | X) = P(Y | W, X) P(W | X) which is doable for multiple samples, just need to 
// assert len(Y) == len(X) and then multiply out over all pairs of (X, Y) values.

// P(Y | X) => this is a fine distribution, but I can't talk about P(Y, X) or P(X | Y) until I put a prior on Y.
// I don't have a joint distribution yet.

// Some issues:
// how do we do (x ** 2).pdf(5)? This is pretty damn hard for non-bijective functions, need to integrate?
// 