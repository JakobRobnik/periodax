This is an implementation of Lomb-Scargle periodogram with several non-conventional benefits:

- It can compute the false positive probability in a very roboust way (data can be non-Gaussian with almost arbitrary systematics).
- Periodogram score takes into account correlated noise.
- Typically, periodogram score is the likelihood ratio, here it can also be the Bayes factor, marginalizing over the parameters of the noise correlation kernel and the frequency of the signal.
- Everything is implemented in JAX with all the usual benefits: easy parallelization, GPU support, speed, etc.

Application:
reanalysis of https://academic.oup.com/mnras/article/463/2/2145/2589684 .
