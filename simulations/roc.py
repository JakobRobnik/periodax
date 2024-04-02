import os, sys
import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu') # we will use cpu here (because we can reserve so many), this line is only to avoid jax warning
jax.config.update("jax_enable_x64", True)

import pandas as pd
from simulations.util import irregular_spaced
from quasars.main import prepare_data, scratch
from quasars import prior
from LombScargle import psd, periodogram
from hypothesis_testing.bayes_factor import logB
from quasars import parallel

# Here the data is signal + correlated Gaussian noise with the unknown kernel (but we have a prior)
# We compare Bayes factor against the likelihood ratio

amplitude = float(sys.argv[1])


# setup
cores, jobs = 128, 5000
keys = jax.random.split(jax.random.PRNGKey(42), jobs)
time, _, mag_err, freq, prior_params = prepare_data(2)

nlogpr_logfreq, generator_logfreq = prior.uniform_with_smooth_edge(*prior_params)
nlogpr_null, generator_null = prior.normal()


def noise(key):
    key1, key2 = jax.random.split(key)
    hyp = jnp.exp(generator_null(key1))
    cov = psd.covariance(time, psd.drw_kernel(*hyp), mag_err)
    return irregular_spaced(key2, cov)


def signal(key):
    key1, key2 = jax.random.split(key)
    phase = jax.random.uniform(key1) * 2 * jnp.pi
    freq_injected = jnp.exp(generator_logfreq(key2))
    return jnp.sin(2 * jnp.pi * freq_injected * time + phase), 1./freq_injected


def sim(simulation_id):
    key = keys[simulation_id]
    key1, key2, key3 = jax.random.split(key, 3)
    model, period_injected = signal(key2)
    data = noise(key1) + amplitude * model
    
    templates = {'basic': periodogram.basic, 
                 'randomized': periodogram.randomized_period(key3, 1000, 0.1)}

    for temp in templates.keys():
        results= logB(time, data, mag_err, freq, nlogpr_logfreq, nlogpr_null, temp_func= templates[temp])
        results['period_injected'] = period_injected
        df = pd.DataFrame(results, index= [0])
        df.to_csv(scratch + 'candidates/' + temp + '/' + str(simulation_id) + '.csv', index= False)



prep = parallel.error_handling(sim)
parallel.for_loop_with_job_manager(prep, 128, jobs)
