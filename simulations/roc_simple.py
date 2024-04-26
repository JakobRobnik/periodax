import jax
import jax.numpy as jnp
import pandas as pd

from LombScargle import periodogram, psd
from simulations.util import *
from quasars.prep import prepare_data

# Here the data is signal + correlated Gaussian noise with the known kernel
# We compare correlated Lomb-Scargle against white Lomb-Scargle

# setup
key = jax.random.PRNGKey(42)
mode_spread = 10
num_sim = 10000

time, _, mag_err, freq = prepare_data(2)
cov = psd.covariance(time, psd.drw_kernel(sigma= 0.1, tau= 200.), mag_err)
mag_err_corr = jnp.linalg.cholesky(cov)



def sim_tp(key, sqrt_cov, amplitude):
    key1, key2, key3 = jax.random.split(key, 3)
    phase = jax.random.uniform(key1) * 2 * jnp.pi
    freq_injected = jax.random.choice(key3, freq)
    signal= jnp.sin(2 * jnp.pi * freq_injected * time + phase) * amplitude
    data = gauss_noise(key2, cov) + signal
    
    return periodogram.lomb_scargle(time, data, sqrt_cov= sqrt_cov)(freq_injected)[0]


def sim_fp(key, sqrt_cov):
    data = gauss_noise(key, cov)
    return jnp.max(jax.vmap(periodogram.func(time, data, sqrt_cov= sqrt_cov))(freq)[0])



def roc(key, err, name):

    # false positive rate
    keys= jax.random.split(key, num_sim)
    scores = jax.vmap(sim_fp, (0, None))(keys, err)
    jnp.save('simulations/results/fp_simple_'+name, scores)
    
    # true positive rate    
    amps = jnp.logspace(-1, 0., 30, endpoint= True)
    tp = jax.vmap(lambda amp: jax.vmap(sim_tp, (0, None, None))(keys, err, amp))(amps)
    df = pd.DataFrame(tp.T, columns= amps)
    df.to_csv('simulations/results/tp_simple_'+name+'.csv', index= False)



roc(key, mag_err, 'white')
roc(key, mag_err_corr, 'corr')

