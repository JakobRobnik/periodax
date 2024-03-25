import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import itertools
from LombScargle import periodogram, psd
from simulations.util import *
plt.style.use('img/style.mplstyle')

# Goals: testing LEE3 (single effective simulation out of every dataset) on different:
#   - time samplings: equally-spaced, equally-spaced with masks, random, quasar times
#   - data: white noise, correlated noise, cauchy
#
# We will here always take the periodogram score as a test statistic.
#


### noise ###
# functions taking jax random key and time array and generating the data at these times
def white_noise(key, time):
    return jax.random.normal(key, shape = time.shape)

def cauchy_noise(key, time):
    return jax.random.cauchy(key, shape = time.shape)

def correlated_noise(key, time):
    tau = (jnp.max(time) - jnp.min(time)) * 0.01
    cov = psd.covariance(time, psd.drw_kernel(sigma= 1., tau= tau), jnp.ones(time.shape) * 0.2)
    return irregular_spaced(key, cov)


noise = {'white': ('white Gaussian', white_noise), 'cauchy': ('Cauchy', cauchy_noise), 'correlated': ('Correlated Gaussian', correlated_noise)}



### time sampling ### 
# functions without arguments and returning time and frequencies to be searched over

def equally_spaced_times():
    size = 1000
    time = jnp.arange(size) + 0.
    freq = jnp.arange(1, size//2) / size
    return time, freq


def equally_spaced_gaps_times():
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
    return time, freq


def quasar_times():
    dir_data = '/pscratch/sd/j/jrobnik/quasars/'
    id = jnp.load(dir_data + 'ids.npy')[2]
    df = pd.read_csv(dir_data + str(id) + '.csv')
    time = jnp.array(df['time'])
    T = jnp.max(time) - jnp.min(time)
    fmin, fmax = 2./T, 1./60.
    freq = jnp.logspace(jnp.log10(fmin), jnp.log10(fmax), 1000)

    return time, freq


def random_times():
    key = jax.random.PRNGKey(101)
    time = jax.random.uniform(key, shape= (1000,))
    freq = jnp.arange(1, len(time)//2)
    return time, freq

time_sampling = {'equal': ('Equally spaced', equally_spaced_times), 'equalgaps': ('Gapped equally spaced', equally_spaced_gaps_times), 
                 'random': ('Randomly spaced', random_times), 'quasar': ('Quasar with ', quasar_times)}



def lee3_freq(time, freq):
    mode_spread = 1
    freq_drift = periodogram.drifting_freq(time, freq, mode_spread)
    return freq_drift


def main(time_name, noise_name):
    
    get_time = time_sampling[time_name][1]
    get_data = noise[noise_name][1]
    
    time, freq = get_time()
    
    def sim(key, freq):
        data = get_data(key, time)
        score, _ = jax.vmap(periodogram.lomb_scargle(time, data))(freq)
        return jnp.max(score)
    
    sims= jax.vmap(sim, (0, None))
    
    num_sim = 10000
    keys = jax.random.split(jax.random.PRNGKey(42), num_sim)
    
    score = sims(keys, freq)
    score_lee3 = sims(keys, lee3_freq(time, freq))
    
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
        plt.title(time_sampling[time_name][0] + ' ' + noise[noise_name][0] + ' noise')
        
        # truth 
        p, x, xmin, xmax = cdf_with_err(data[0])
        plt.plot(x, p, color = 'tab:red', lw = 2)
        
        # LEE3
        p, x, xmin, xmax = cdf_with_err(data[1])
        color = plt.cm.ocean((counter + 0.5) / (num_x * num_y))
        plt.plot(x, p, color = color, lw = 2)
        #plt.fill_betweenx(p, xmin, xmax, color = color, alpha= 0.3)
        
        plt.yscale('log')
        plt.xlabel('q(x)')
        plt.ylabel('P(q > q(x))')
        counter += 1
    
    plt.tight_layout()
    plt.savefig('img/synthetic.png')
    plt.close()
    


if __name__ == '__main__':
    
    mainn()
    plot()