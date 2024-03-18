import os
import jax

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(128)
num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

import jax.numpy as jnp
from jax.tree_util import tree_map


import pandas as pd
from simulations.util import irregular_spaced, prepare_data
from quasars.prior import loguniform_freq, log_normal
from LombScargle import psd
from hypothesis_testing.bayes_factor import logB


# Here the data is signal + correlated Gaussian noise with the unknown kernel (but we have a prior)
# We compare Bayes factor against the likelihood ratio


# setup
key = jax.random.PRNGKey(42)
repeat = 2

time, _, mag_err, freq = prepare_data(2)

prior_freq = loguniform_freq(freq)
prior_null, generator_null = log_normal()


def noise(key):
    key1, key2 = jax.random.split(key)
    hyp = jnp.exp(generator_null(key1))
    cov = psd.covariance(time, psd.drw_kernel(*hyp), mag_err)
    return irregular_spaced(key2, cov)

def signal(key):
    key1, key2 = jax.random.split(key)
    phase = jax.random.uniform(key1) * 2 * jnp.pi
    freq_injected = jax.random.choice(key2, freq)
    return jnp.sin(2 * jnp.pi * freq_injected * time + phase), 1./freq_injected


def sim(key, amplitude):
    key1, key2 = jax.random.split(key)
    model, period_injected = signal(key2)
    data = noise(key1) + amplitude * model
    results= logB(time, data, mag_err, freq, prior_freq, prior_null)
    results['period_injected'] = period_injected
    return results


def roc(key):

    keys= jax.random.split(key, repeat * num_cores).reshape(num_cores, repeat, 2)
    
    for amp in [0.1, ]:
        
        simm = lambda k: sim(k, amp)
        results = jax.pmap(jax.vmap(simm))(keys)
        results = tree_map(lambda x: x.reshape((num_cores * repeat, )), results)
        df = pd.DataFrame.from_dict(results)
        df.to_csv('simulations/results/amp_' + str(amp) + '.csv', index= False)

        
    
import time
t0 = time.time()

roc(key)

print((t0 - time.time())/60.)