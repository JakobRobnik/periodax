import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd



dir_data = '/pscratch/sd/j/jrobnik/quasars/'
ids = jnp.load(dir_data + 'ids.npy')


def prepare_data(myid):
    
    df = pd.read_csv(dir_data + str(ids[myid]) + '.csv')
    time = jnp.array(df['time'])
    mag = jnp.array(df['mag'])
    mag_err = jnp.array(df['mag_err'])
    T = jnp.max(time) - jnp.min(time)
    fmin, fmax = 1./T, 1./60.

    freq = jnp.logspace(jnp.log10(fmin), jnp.log10(fmax), 1000)
    
    return time, mag, mag_err, freq



def regular_spaced(key, PSD, mask):
    
    key1, key2 = jax.random.split(key)
    
    # data in the Fourier domain
    phases = jax.random.uniform(key1, shape = mask.shape) * 2 * jnp.pi
    amplitudes = jax.random.normal(key2, shape = mask.shape) * jnp.sqrt(PSD * len(mask))
    modes = amplitudes * (jnp.cos(phases) + 1j * jnp.sin(phases))
    
    # time-domain
    data = jnp.fft.irfft(jnp.insert(modes, 0, 0.0), n= len(mask))
    
    return data[mask]


def irregular_spaced(key, cov):
    
    return jnp.linalg.cholesky(cov) @ jax.random.normal(key, shape= (cov.shape[0], ))



def maximize_over_bin(q, bin):
    return jnp.max(q.reshape(jnp.shape(q)[0], jnp.shape(q)[-1] // bin, bin), axis = -1)


def cdf_with_err(x_given, sgn = -1):
    key = jax.random.PRNGKey(0)
    
    repeat = 100

    #sigma          1         2      3
    #percentiles = 34.1     47.7    49.8
    percentile = 34.1
    ilow = (int) (repeat * (50 - percentile) / 100)
    ihigh = (int)(repeat * (50 + percentile) / 100)

    x = sgn*jnp.sort(sgn*x_given)
    num_realizations = len(x)
    pMC = (jnp.arange(num_realizations) + 1.0) / (num_realizations + 1.0)

    index = jax.random.randint(key, minval = 0, maxval = num_realizations, shape = (repeat, num_realizations))
    #X = x[index]
    X = sgn*jnp.sort(jnp.sort(sgn*x[index], axis = 1), axis = 0)

    return pMC, x, X[ilow, :], X[ihigh, :]



def freq_dependence(freq, score, score_null, save_dir):
    num_sim = score.shape[0]
    
    bin = 8
    freq_bined = freq[bin//2::bin]

    def func(score, fmt):
        score = maximize_over_bin(score, bin)
        score = jnp.sort(score, axis = 0)
        cmap = plt.cm.magma
        plt.plot(freq_bined, score[num_sim//2, :], fmt, color = cmap(0.1), label= 'p-value = 50%')
        plt.plot(freq_bined, score[(num_sim * 9) //10, :], fmt, color = cmap(0.5), label= 'p-value = 10%')
        plt.plot(freq_bined, score[(num_sim * 99) //100, :], fmt, color = cmap(0.9), label= 'p-value = 1%')

    
    func(score, '-')
    plt.legend()
    func(score_null, '--')

    plt.savefig(save_dir)
    plt.close()



def cdf(score, score_null, save_dir):
        
    
    def func(score, color, label):
        score1 = jnp.max(score, axis= 1)
        p, x, xmin, xmax = cdf_with_err(score1)
        plt.plot(x, p, color = color, label = label)
        plt.fill_betweenx(p, xmin, xmax, color = color, alpha= 0.3)
        
        
    func(score, 'black', 'truth')
    func(score_null, 'teal', 'null')
    
    plt.yscale('log')
    plt.xlabel('q(x)')
    plt.ylabel('P(q > q(x))')
    plt.legend()
    plt.savefig(save_dir)
    plt.close()
    
    
def ROC(score_fp, score_true):
    
    F = jnp.sort(score_fp)
    T = jnp.sort(score_true)
    numT, numF = T.shape[0], F.shape[0]
    
    fpr = jnp.arange(numF, 0, -1) / numF
    tpr = (numT - jnp.searchsorted(T, F)) / numT

    return fpr, tpr

