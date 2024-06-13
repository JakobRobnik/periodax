import jax
import jax.numpy as jnp
import os

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=128' # specififc to nersc
num_cores = jax.local_device_count()
print(num_cores, jax.lib.xla_bridge.get_backend().platform)

from LombScargle import periodogram
import numpy as np


def _main(key, cycles, spread):
    """period = 1, phase = 0, without loss of generality"""
    
    # data = unmodified template
    time = jnp.linspace(0, cycles, (int)(cycles * 20))
    data = jnp.sin(2 * jnp.pi * time)
    c= jnp.log10(3.)
    freq = jnp.logspace(-c, c, 1000)
    
    # scan with the modified template
    temp = periodogram.null_signal_template(key, (int)(cycles * 2), spread)
    periodogram_modified = periodogram.lomb_scargle(time, data, floating_mean= False, temp_func= temp)
    score_modified = jnp.max(jax.vmap(periodogram_modified)(freq)[0])

    score_og = periodogram.lomb_scargle(time, data, floating_mean= False)(1.)[0]
    
    return score_modified / score_og



key = jax.random.key(42)
pmap_num, vmap_num = 128, 4
keys = jax.random.split(key, pmap_num * vmap_num).reshape(pmap_num, vmap_num)
mainn = lambda c, s: jax.pmap(jax.vmap(lambda k: _main(k, c, s)))(keys).reshape(pmap_num*vmap_num)

cycles = np.logspace(np.log10(3.), np.log10(100), 10)
spread = jnp.logspace(jnp.log10(1.), jnp.log10(30.), 10)
r = np.array([jax.vmap(mainn, (None, 0))(c, spread) for c in cycles]) # cycles cannot be jax-ed so they are run as a for loop
print(r.shape)
np.save('notebooks/r.npy', r)