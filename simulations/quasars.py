import jax
import jax.numpy as jnp
import pandas as pd

from LombScargle import periodogram, psd


# setup
key = jax.random.PRNGKey(42)
mode_spread = 10
num_sim = 10000

scratch = '/pscratch/sd/j/jrobnik/quasars/'
id = jnp.load(scratch + 'ids.npy')[2]
drw_kernel = lambda sigma, tau: lambda t1, t2: jnp.square(sigma) * jnp.exp(-jnp.abs(t2-t1) / tau)
df = pd.read_csv(scratch + str(id) + '.csv')
time = jnp.array(df['time'])
print(len(time))
#mag = jnp.array(df['mag'])
mag_err = jnp.array(df['mag_err'])
T = jnp.max(time) - jnp.min(time)
print(T)
fmin, fmax = 1./T, 1./60.
#freq_injected = 1./200.
freq = jnp.logspace(jnp.log10(fmin), jnp.log10(fmax), 1000)
cov = psd.covariance(time, drw_kernel(sigma= 0.1, tau= 200.), mag_err) 
