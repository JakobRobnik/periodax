import jax.numpy as jnp
import jax
import numpy as np 
from LombScargle import periodogram, psd
from simulations.util import gauss_noise
import matplotlib.pyplot as plt


key, key_data = jax.random.split(jax.random.key(0), 2)
keys = jax.random.split(key, 6).reshape(2, 3)


# time sampling
t1 = jax.random.uniform(keys[0, 0], (100,), minval=0, maxval=10)
t2 = jax.random.uniform(keys[0, 1], (30,), minval=6, maxval=10)
t3 = jax.random.uniform(keys[0, 2], (20,), minval=2, maxval=7)
times= (t1, t2, t3)
time = jnp.concatenate(times)
band_mask = periodogram.band_mask(times)

# errors
errors1 = jnp.square(jax.random.normal(keys[1, 0], t1.shape)) * 0.2
errors2 = jnp.square(jax.random.normal(keys[1, 0], t2.shape)) * 0.5
errors3 = jnp.square(jax.random.normal(keys[1, 0], t3.shape)) * 0.1
errors = (errors1, errors2, errors3)
errors= jnp.concatenate(errors)


# simualte the data (here the entire data in all bands is simulated at once, when you use real lightcurves, construct them like errors and time was constructed here)
# DRW params
sigmas = jnp.array([1., 3., 0.5])
tau = 3
# construct the covariance matrix (also between the bands)
cov = psd.multiband_covariance(time, psd.drw_kernel(1., tau), errors, drw_amp= sigmas, sizes= [len(t1), len(t2), len(t3)])
data = gauss_noise(key_data, cov) # generate the noise with this covariance matrix

# inject signal
period_true = 2.
signal =  jnp.sin(2 * jnp.pi * time / period_true) * jnp.sum(band_mask * jnp.array([3., 2., 0.5])[:, None], axis = 0)
data += signal

# make different bands have different magnitudes
shift = jnp.sum(band_mask * jnp.array([10., 20., 5.])[:, None], axis = 0)
data += shift


def plot_lc():

    # get the fit (taking the ground truth period)
    ls = periodogram.lomb_scargle(time, data, sqrt_cov= jnp.linalg.cholesky(cov), temp_func= periodogram.basic, band_mask= band_mask)
    _, amps = ls()
    T = jnp.tile(jnp.linspace(0, 10, 100), 3)
    o, z = jnp.ones(100), jnp.zeros(100)
    _mask = jnp.block([[o, z, z], [z, o, z], [z, z, o]])
    model = np.array(periodogram.fit_bands(T, 0.5, amps, _mask, temp_func= periodogram.basic))


    t, d, e, m = np.array(time), np.array(data), np.array(errors), np.array(band_mask)
    band_names = ['r', 'g', 'i']
    band_colors = ['tab:red', 'tab:green', 'tab:blue']

    plt.figure(figsize= (10, 10))
    
    for i, mask in enumerate(m):

        plt.subplot(len(band_mask) + 1, 1, i + 1)
        plt.errorbar(t[mask], d[mask], yerr= e[mask], fmt='o', color = band_colors[i])
        plt.plot(np.array(T)[:100], model[100 * i : 100*(i+1)], color = 'black')

        plt.title(f'Band {band_names[i]}', fontweight = 'bold')
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.xlim(0, 10.)


    # periodogram
    freq = jnp.linspace(0.2, 5, 1000)
    ls = periodogram.lomb_scargle(time, data, sqrt_cov= jnp.linalg.cholesky(cov), temp_func= periodogram.basic, band_mask= band_mask)
    scores, _ = jax.vmap(ls)(freq)

    plt.title('Periodogram score', fontweight = 'bold')

    plt.subplot(len(band_mask) + 1, 1, len(band_mask) + 1)
    plt.plot(1./freq, scores, '.', color = 'teal')
    plt.axvline(period_true, ls='--', alpha = 0.5, color= 'black')
    plt.xlabel('period')
    plt.ylabel('periodogram score')

    plt.tight_layout()
    plt.savefig("img/simulated_data_injected.png")
    #plt.savefig("img/simulated_data.png")
    
    plt.close()



plot_lc()
