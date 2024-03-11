import jax
import jax.numpy as jnp

from LombScargle import periodogram
from simulations.util import *



# setup
key = jax.random.PRNGKey(42)
mode_spread = 10
num_sim = 1000


def do_sim(time, data_generator):
    def func(key, freq):
        data = data_generator(key)
        score, _ = jax.vmap(periodogram.func(time, data))(freq)
        return score

    return jax.vmap(func, (0, None))



def get_mask():
    mask = jnp.concatenate((jnp.ones(50, dtype = bool), jnp.zeros(29, dtype = bool), 
                            jnp.ones(72, dtype = bool), jnp.zeros(100, dtype = bool), 
                            jnp.ones(61, dtype = bool), jnp.zeros(45, dtype = bool), 
                            jnp.ones(146, dtype = bool), jnp.zeros(23, dtype = bool),
                            jnp.ones(53, dtype = bool)))
    bits = jax.random.bernoulli(jax.random.PRNGKey(0), p = 0.9, shape = mask.shape)
    return mask * bits
    

def masked():
    mask = get_mask()
    dt = 1.
    time = jnp.arange(len(mask))[mask] * dt
    time -= (jnp.max(time) + jnp.min(time)) * 0.5
    PSD = jnp.ones(len(mask))

    T = len(mask) * dt
    freq = jnp.arange(1, len(mask) // 2) / T 
    
    do_sims = do_sim(time, lambda k: regular_spaced(k, PSD, mask))

    return time, freq, do_sims

def randtime():
    time = jax.random.uniform(key, shape= (1000,))
    freq = jnp.arange(len(time))

    do_sims = do_sim(time, lambda k: jax.random.normal(k, shape= time.shape))

    return time, freq, do_sims

        
time, freq, do_sims = masked()


freq_drift = periodogram.drifting_freq(time, freq, mode_spread)


def compute(null, key):
    
    keys = jax.random.split(key, num_sim)
    score = do_sims(keys, freq_drift if null else freq)
    
    return score
    

key1, key2 = jax.random.PRNGKey(42)
cdf(compute(False, key1), compute(False, key2))
    