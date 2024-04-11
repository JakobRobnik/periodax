import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern



scratch_base = '/pscratch/sd/j/jrobnik/'
scratch = scratch_base + 'quasars_scratch/'
dir_data = scratch_base + 'quasars/'


ids = jnp.load(dir_data + 'ids.npy')
#ids = jnp.array(pd.read_csv('quasars/known.csv')['id'], dtype= int)


def within_night_averaging(time, data, err):
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


def fit_GP(x, y, yerr):
    kernel = Matern(length_scale= 100., nu= 0.5) # = DRW kernel with tau = length_scale
    sigma_kernel = 0.1
    gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha= np.square(yerr / sigma_kernel), optimizer= None)
    gaussian_process.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    return gaussian_process


def _outlier_removal_gp(time, data, data_err, mask):
    gp = fit_GP(time[mask], data[mask], data_err[mask])
    model, err_model = gp.predict(time.reshape(-1, 1), return_std=True)
    mask_new = np.abs((data-model)/err_model) < 3.
    return mask_new

    
def outlier_removal(time, data, data_err):
    
    avg = np.average(data, weights=1./np.square(data_err)) #for subtracting the average
    mask = np.ones(time.shape, dtype = bool)
    niter = 5
    
    for i in range(niter): # iteratively fit GP to the data without outliers and identify outliers
        mask = _outlier_removal_gp(time, data-avg, data_err, mask)
            
    return time[mask], data[mask], data_err[mask]
    


def prepare_data(id, remove_outliers= True, average_within_night= True):
    
    # load the data
    df = pd.read_csv(dir_data + str(id) + '.csv')
    time = np.array(df['time'])
    mag = np.array(df['mag'])
    mag_err = np.array(df['mag_err'])
    
    # sort the data
    perm = np.argsort(time)
    time, mag, mag_err = time[perm], mag[perm], mag_err[perm]
    
    if remove_outliers:
        time, mag, mag_err = outlier_removal(time, mag, mag_err)
    
    time, mag, mag_err = jnp.array(time), jnp.array(mag), jnp.array(mag_err)
    if average_within_night:
        time, mag, mag_err = within_night_averaging(time, mag, mag_err)
    
    # determine the frequency grid
    T = jnp.max(time) - jnp.min(time)
    fmin, fmax = 2./T, 1./60.
    factor = 2./1.5 # prior starts to die off at 2/T and is zero at 1.5/T
    prior_params = (jnp.log(fmin), jnp.log(fmax), jnp.log(factor))
    freq = jnp.logspace(jnp.log10(fmin/factor), jnp.log10(fmax*factor), 1000)
    #freq = jnp.logspace(jnp.log10(1./T), jnp.log10(fmax), 1000)

    return time, mag, mag_err, freq, prior_params
