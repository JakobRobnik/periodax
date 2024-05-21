import jax
import jax.numpy as jnp
import os
cpus = 128 #specific to nersc
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=' + str(cpus)

import matplotlib.pyplot as plt
import itertools
from typing import NamedTuple, Any
from LombScargle import periodogram, psd
from simulations.util import *
plt.style.use('img/style.mplstyle')





# Goals: testing LEE3 (single effective simulation out of every dataset) on different:
#   - time samplings: equally-spaced, equally-spaced with masks, random, quasar times
#   - data: white noise, correlated noise, cauchy
#
# We will here always take the periodogram score as a test statistic.
#


class Noise(NamedTuple):
    name: str
    generator: Any # functions taking jax random key and time array and generating the data at these times

    
def white_noise(key, time):
    return jax.random.normal(key, shape = time.shape)

def cauchy_noise(key, time):
    return jax.random.cauchy(key, shape = time.shape)

def correlated_noise(key, time):
    tau = (jnp.max(time) - jnp.min(time)) * 0.05
    cov = psd.covariance(time, psd.drw_kernel(sigma= 1., tau= tau), jnp.ones(time.shape) * 0.2)
    return gauss_noise(key, cov)


noise = {'white': Noise('white Gaussian', white_noise), 
         'cauchy': Noise('Cauchy', cauchy_noise), 
         'correlated': Noise('correlated Gaussian', correlated_noise)}



class TimeSampling(NamedTuple):
    name: str
    time: jax.typing.ArrayLike
    freq: jax.typing.ArrayLike


def equally_spaced_times():
    size = 1000
    time = jnp.arange(size)
    freq = jnp.arange(1, size//2) / size
    return TimeSampling('Equally spaced', time, freq)


def equally_spaced_gaps_times():
    """time series of size 382 (in the range [0, 578])"""
    
    # large gaps
    mask = jnp.concatenate((jnp.ones(50, dtype = bool), jnp.zeros(29, dtype = bool), 
                            jnp.ones(72, dtype = bool), jnp.zeros(100, dtype = bool), 
                            jnp.ones(61, dtype = bool), jnp.zeros(45, dtype = bool), 
                            jnp.ones(146, dtype = bool), jnp.zeros(23, dtype = bool),
                            jnp.ones(53, dtype = bool)))
    
    # individual missing data
    bits = jax.random.bernoulli(jax.random.PRNGKey(0), p = 0.9, shape = mask.shape)
    mask = mask * bits   
    
    time = jnp.arange(len(mask))[mask]
    freq = jnp.arange(1, len(mask)//2) / len(mask)
    return TimeSampling('Gapped, equally spaced', time, freq)


def quasar_times():
    dir_data = '/pscratch/sd/j/jrobnik/quasars/'
    id = jnp.load(dir_data + 'ids.npy')[2]
    df = pd.read_csv(dir_data + str(id) + '.csv')
    time = jnp.array(df['time'])
    T = jnp.max(time) - jnp.min(time)
    fmin, fmax = 2./T, 1./60.
    freq = jnp.logspace(jnp.log10(fmin), jnp.log10(fmax), 1000)

    return TimeSampling('Quasar with', time, freq)


def random_times():
    key = jax.random.PRNGKey(101)
    time = jax.random.uniform(key, shape= (1000,))
    freq = jnp.arange(1, len(time)//2)
    return TimeSampling('Randomly spaced', time, freq)


time_sampling = {'equal': equally_spaced_times(), 
                 'equalgaps': equally_spaced_gaps_times(), 
                 'random': random_times(), 
                 'quasar': quasar_times()}



def main(time_name, noise_name):
    
    time, freq = time_sampling[time_name].time, time_sampling[time_name].freq
    get_data = noise[noise_name].generator
    
    def sim(key, temp_func):
        key_data, key_template = jax.random.split(key)
        data = get_data(key_data, time)
        temp = temp_func(key_template)
        score, _ = jax.vmap(periodogram.lomb_scargle(time, data, floating_mean= True, temp_func= temp))(freq)
        return jnp.max(score)
    
    num_sim = 2**12
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, num_sim)
    
    # templates 
    basic = lambda rng_key: periodogram.basic
    rand_temp = lambda rng_key: periodogram.randomized_period(rng_key, 2000, 1.)
    
    sims= lambda keys, temp_func: jax.pmap(jax.vmap(lambda k: sim(k, temp_func)))(keys.reshape(cpus, num_sim//cpus, 2)).reshape(num_sim) # for cpu
    #sims = jax.vmap(sim, (0, None)) # for gpu
                   
    # true simulations
    score = sims(keys, basic)

    # LEE3 simulations
    score_lee3 = sims(keys, rand_temp)
    
    jnp.save('data/synthetic/' + time_name + noise_name + '.npy', [score, score_lee3])

    
def mainn():
    # iterate over all combinations of time sampling and noise
    for name in itertools.product(time_sampling.keys(), noise.keys()):
       print(*name)
       main(*name)
       

def plot():
    
    num_x, num_y = len(time_sampling), len(noise)
    plt.figure(figsize= (num_x * 5, num_y * 3))
    counter = 0
    for (noise_name, time_name) in itertools.product(noise.keys(), time_sampling.keys()):
        plt.subplot(num_y, num_x, counter + 1)
        
        data = jnp.load('data/synthetic/' + time_name + noise_name + '.npy')
        plt.title(time_sampling[time_name].name + ' ' + noise[noise_name].name + ' noise')
        
        # truth 
        p, x, xmin, xmax = cdf_with_err(data[0])
        plt.plot(x, p, color = 'black', lw = 2)
        
        # LEE3
        p, x, xmin, xmax = cdf_with_err(data[1])
        color = plt.cm.inferno(0.05 + (counter + 0.5) / (num_x * num_y) * 0.9)
        plt.plot(x, p, color = color, lw = 2)
        plt.fill_betweenx(p, xmin, xmax, color = color, alpha= 0.3)
        
        plt.yscale('log')
        ylim_only(x, p, 5e-5, 1.1)
        
        if counter // num_x == num_y -1:
            plt.xlabel('q(x)')
        
        if counter % num_x == 0:
            plt.ylabel('P(q > q(x))')
        
        counter += 1
    
    plt.tight_layout()
    plt.savefig('img/synthetic/dirichlet_1.png')
    plt.close()
    
    
    
def show_times():
    import numpy as np
    i = 0
    plt.figure(figsize = (15, 2))
    for times in time_sampling.values():
        time = times.time
        plt.plot(np.array((time - jnp.min(time)) / (jnp.max(time) - jnp.min(time))), np.ones(len(time))*i, '.')
        i-=1
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('img/time_sampling.png')
    plt.close()
    


if __name__ == '__main__':
    # show_times()
    mainn()
    plot()