This is an implementation of Lomb-Scargle periodogram with several non-conventional benefits:

- **Roboust false positive probability** computations (data can be non-Gaussian with almost arbitrary systematics). This is achieved by constructing special null signal templates (NST) that act as effective null simulations when used on the real data.
- Periodogram score takes into account **correlated noise**.
- Typically, periodogram score is the likelihood ratio, here it can also be the **Bayes factor**, marginalizing over the parameters of the noise correlation kernel and the frequency of the signal.
- Everything is implemented in [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) with all the usual benefits: easy parallelization, GPU support, speed, etc.

Example Application:
Reanalysis of Supermassive Black Hole Binaries search from [here](https://academic.oup.com/mnras/article/463/2/2145/2589684) is on the 'quasars' branch.

To get started, checkout the [tutorial](tutorial.ipynb). See also the [associated paper](https://arxiv.org/abs/2407.17565).

If you encounter any issues, feel free to contact me at jakob_robnik@berkeley.edu .
