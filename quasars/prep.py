import jax
import jax.numpy as jnp
import pandas as pd


scratch_base = '/pscratch/sd/j/jrobnik/'
scratch = scratch_base + 'quasars_scratch/'
dir_data = scratch_base + 'quasars/'


ids = jnp.load(dir_data + 'ids.npy')
#ids = jnp.array(pd.read_csv('quasars/known.csv')['id'], dtype= int)


def average_within_night(time, data, err):
    """average the data within the same night"""
    
    midnight = 8./24. # specific to the time zone
    night = jnp.round(time - midnight).astype(int) # which night does each observation belong to
    t0 = jnp.min(night) 
    night -= t0 # let the first night have index 0
    num_nights = jnp.max(night).astype(int)
    
    # combining Gaussian measurements
    invvar = 1./jnp.square(err)
    invVar = jnp.zeros(num_nights).at[night].add(invvar) # sum within night: 1 / sigma^2
    Time = jnp.zeros(num_nights).at[night].add(time * invvar) # sum within night: t / sigma^2
    Data = jnp.zeros(num_nights).at[night].add(data * invvar) # sum withing night data / sigma^2
    
    has_data = invVar != 0. # eliminate nights where there are no measurements
    return Time[has_data] / invVar[has_data], Data[has_data] / invVar[has_data], jnp.sqrt(1./invVar[has_data]) # normalize the sums and return results
    

def prepare_data(id, night_average= True):
    
    # load the data
    df = pd.read_csv(dir_data + str(id) + '.csv')
    time = jnp.array(df['time'])
    mag = jnp.array(df['mag'])
    mag_err = jnp.array(df['mag_err'])
    
    if night_average:
        time, mag, mag_err = average_within_night(time, mag, mag_err)
    
    
    # determine the frequency grid
    T = jnp.max(time) - jnp.min(time)
    fmin, fmax = 2./T, 1./60.
    factor = 2./1.5 # prior starts to die off at 2/T and is zero at 1.5/T
    prior_params = (jnp.log(fmin), jnp.log(fmax), jnp.log(factor))
    #freq = jnp.logspace(jnp.log10(fmin/factor), jnp.log10(fmax*factor), 1000)
    freq = jnp.logspace(jnp.log10(1./T), jnp.log10(fmax), 1000)

    
    return time, mag, mag_err, freq, prior_params
