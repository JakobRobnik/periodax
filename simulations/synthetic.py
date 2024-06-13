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

inject= False # weather to inject signal in the noise simulations
name_inject = 'injected' if inject else 'null'


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
                 'random': random_times()} 
                 #'quasar': quasar_times()}



def get_signal(key, time, freq):
    key1, key2 = jax.random.split(key)
    phase = jax.random.uniform(key1) * 2 * jnp.pi
    freq_injected = jax.random.choice(key2, freq)
    return jnp.sin(2 * jnp.pi * freq_injected * time + phase)



def main(time_name, noise_name, injected_chi2):
    
    time, freq = time_sampling[time_name].time, time_sampling[time_name].freq
    get_noise = noise[noise_name].generator
    
    
    def sim(key, temp_func):
        key_noise, key_signal, key_template = jax.random.split(key, 3)
        n = get_noise(key_noise, time)
        s = get_signal(key_signal, time, freq)
        s *= jnp.sqrt(injected_chi2 / jnp.sum(jnp.square(s)))
        data = inject * s + n
        temp = temp_func(key_template)
        score, _ = jax.vmap(periodogram.lomb_scargle(time, data, floating_mean= True, temp_func= temp))(freq)
        return jnp.max(score)
    
    num_sim = 2**13
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, num_sim)
    
    # templates 
    true_temp = lambda rng_key: periodogram.basic
    null_temp = lambda rng_key: periodogram.null_signal_template(rng_key, 2000)
    
    sims= lambda keys, temp_func: jax.pmap(jax.vmap(lambda k: sim(k, temp_func)))(keys.reshape(cpus, num_sim//cpus, 2)).reshape(num_sim) # for cpu
    #sims = jax.vmap(sim, (0, None)) # for gpu
                   
    # true simulations
    score_true = sims(keys, true_temp)

    # NST simulations
    score_nst = sims(keys, null_temp)
    
    jnp.save('data/synthetic/' + name_inject + '/'+ time_name + noise_name + '.npy', [score_true, score_nst])

    
def mainn():
    
    amp = [15, 7e7, 700,
           10, 3e7, 300,
           10, 7e7, 600,
           4, 5e6, 60,
        ]
    count = 0
    
    # iterate over all combinations of time sampling and noise
    for name in itertools.product(time_sampling.keys(), noise.keys()):
       print(*name, amp[count])
       main(*name, amp[count])
       count += 1
       

def plot():
    
    num_x, num_y = len(time_sampling), len(noise)

    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['font.size'] = 15
    
    plt.figure(figsize= (num_x * 7, num_y * 4))
    counter = 0
    cut = 5
    for (noise_name, time_name) in itertools.product(noise.keys(), time_sampling.keys()):
        plt.subplot(num_y, num_x, counter + 1)
        
        plt.title(time_sampling[time_name].name + ' ' + noise[noise_name].name + ' noise', fontsize= 18, y = 1.05)
        
        # truth 
        data = jnp.load('data/synthetic/null/' + time_name + noise_name + '.npy')
        p, x, xmin, xmax = cdf_with_err(data[0])
        plt.plot(x[cut:], p[cut:], color = 'black', lw = 2)
        plt.fill_betweenx(p[cut:], xmin[cut:], xmax[cut:], color = 'black', alpha= 0.3)
        
        
        data = jnp.load('data/synthetic/' + name_inject + '/' + time_name + noise_name + '.npy')
        # NST
        p, x, xmin, xmax = cdf_with_err(data[1])
        color = plt.cm.inferno(0.05 + (counter + 0.5) / (num_x * num_y) * 0.9)
        if inject:
            color= 'teal'
        plt.plot(x[cut:], p[cut:], color = color, lw = 2)
        plt.fill_betweenx(p[cut:], xmin[cut:], xmax[cut:], color = color, alpha= 0.3)
        
        if inject: # real template on injected data
            p, x, xmin, xmax = cdf_with_err(data[0])
            color = 'chocolate'
            plt.plot(x[cut:], p[cut:], color = color, lw = 2)
            plt.fill_betweenx(p[cut:], xmin[cut:], xmax[cut:], color = color, alpha= 0.3)
                
        
        plt.yscale('log')
        ylim_only(x[cut:], p[cut:], 5e-4, 1.1)
        
        if counter // num_x == num_y -1:
            plt.xlabel(r'$q_{LS}(X)$')
        
        if counter % num_x == 0:
            plt.ylabel(r'$P(q_{LS} > q_{LS}(x))$')
        
        counter += 1
    
    plt.tight_layout()
    plt.savefig('img/synthetic/'+name_inject+'.png')
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
    #mainn()
    plot()