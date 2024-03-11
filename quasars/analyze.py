import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

from LombScargle import periodogram


scratch = '/pscratch/sd/j/jrobnik/quasars/'
ids = jnp.load(scratch + 'ids.npy')

# drw_kernel = lambda sigma, tau: lambda t1, t2: jnp.square(sigma) * jnp.exp(-jnp.abs(t2-t1) / tau)
# cov = periodogram.covariance(time, drw_kernel(1., 10.)) + jnp.diag(jnp.square(mag_err))

def process(i):
    df = pd.read_csv(scratch + str(ids[i]) + '.csv')
    time = jnp.array(df['time'])
    mag = jnp.array(df['mag'])
    mag_err = jnp.array(df['mag_err'])
    T = jnp.max(time) - jnp.min(time)
    fmin, fmax = 1./T, 1./60.
    freq = jnp.logspace(jnp.log10(fmin), jnp.log10(fmax), 1000)
    score, best_params = jax.vmap(periodogram.func(time, mag, floating_mean= True, sqrt_cov= mag_err))(freq)
    
    ibest = jnp.argmax(score)
    score, best_params = score[ibest], best_params[ibest]
    
    return score
    

from time import time

t0 = time()

for i in range(100):
    process(i)

print(time()-t0)

    
        
    
    