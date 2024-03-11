This is an implementation of Lomb-Scargle periodogram with several non-conventional benefits:

- noise is allowed to be correlated
- it does not just compute the likelihood ratio but also the Bayes factor, marginalizing over the parameters of the noise correlation kernel and the frequency of the signal
- it is implemented in JAX with all the usual benefits: easy parallelization, GPU support, speed, etc.
- it can compute the false positive probability in a very roboust way (not assuming Gaussian noise)

Application usage:
reanalysis of https://academic.oup.com/mnras/article/463/2/2145/2589684 .
