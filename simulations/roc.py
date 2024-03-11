
import jax
import jax.numpy as jnp
import pandas as pd

from LombScargle import periodogram
from simulations.util import *
from simulations.quasars import *


num_sim = 50000

def sim_tp(key, sqrt_cov, amplitude):
    key1, key2, key3 = jax.random.split(key, 3)
    phase = jax.random.uniform(key1) * 2 * jnp.pi
    freq_injected = jax.random.choice(key3, freq)
    signal= jnp.sin(2 * jnp.pi * freq_injected * time + phase) * amplitude
    data = irregular_spaced(key2, cov) + signal
    score, _ = periodogram.func(time, data, floating_mean= True, sqrt_cov= sqrt_cov)(freq_injected)
    return score


def sim_fp(key, sqrt_cov):
    data = irregular_spaced(key, cov)
    score, _ = jax.vmap(periodogram.func(time, data, floating_mean= True, sqrt_cov= sqrt_cov))(freq)
    return jnp.max(score)


def roc(key, sqrt_cov, name):

    keys= jax.random.split(key, num_sim)
    fp = jax.vmap(sim_fp, (0, None))(keys, sqrt_cov)
    jnp.save('simulations/results/fp_'+name+'LS', fp)
    
    amps = jnp.logspace(-1, 0., 30, endpoint= True)
    tp = jax.vmap(lambda amp: jax.vmap(sim_tp, (0, None, None))(keys, sqrt_cov, amp))(amps)
    df = pd.DataFrame(tp.T, columns= amps)
    df.to_csv('simulations/results/tp_'+name+'LS.csv', index= False)
    

roc(key, mag_err, 'white')
roc(key, jnp.linalg.cholesky(cov), 'corr')

