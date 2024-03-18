import jax.random as rand
import jax.numpy as jnp


### alternative prior

def loguniform_freq(freq_arr):
    """Uniform prior in log frequency:
        p(log f) = 1 / log(fmax / fmin)
       Args:
            freq_arr: array of trial frequencies that is only used to determine the minimal and maximal frequency
       Returns:
            nlogp function, taking f and returning -log(p(log (f))) = log log(fmax/fmin)
    """
    
    const = jnp.log(jnp.log(jnp.max(freq_arr) / jnp.min(freq_arr)))
    
    return lambda f: const


### null priors

def log_normal():
    log_mu = jnp.log(jnp.array([0.1, 120.]))
    sigma = jnp.array([0.2, 0.9])
    nlogp = lambda y: jnp.sum(0.5 * jnp.square((y - log_mu)/sigma) + 0.5 * jnp.log(2 * jnp.pi * jnp.square(sigma)))    
    generate = lambda key: jnp.exp(rand.normal(key, shape= (2,)) * sigma + log_mu)
    
    return nlogp, generate
    

