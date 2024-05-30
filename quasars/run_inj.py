import multiprocessing as mp
import numpy as np
import os, sys
import jax
import jax.numpy as jnp

# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREAD"] = "1"

jax.config.update('jax_platform_name', 'cpu') # we will use cpu here (because we can reserve so many), this line is only to avoid jax warning
jax.config.update("jax_enable_x64", True) # since we are on cpu we do not mind using 64 bits

import pandas as pd
from quasars.prep import load_data
from quasars import prior, scratch_structure
from LombScargle import psd, periodogram
from hypothesis_testing.bayes_factor import logB
from simulations.util import gauss_noise

plot= 0


def signal(time, period, A_const, A_sin, A_cos):
    return periodogram.fit(time, 1./period, np.array([A_const, A_sin, A_cos]))



def save(id, results, base,
         log_prior_odds, time, period_injected= None):
    
    ### add to the results ###
    results['data_points'] = len(time)
    results['T'] = np.max(time) - np.min(time)
    results['log_prior_odds'] = log_prior_odds
    
    results['id'] = id
    
    if period_injected != None:
        results['period_injected'] = period_injected
    
    ### save the results ###
    df = pd.DataFrame(results, index= [0])
    df.to_csv(base + str(id) + '.csv', index= False)
    
    return True


def injected(params):
    id, redshift, key, temp, period, A_const, A_sin, A_cos = params
    mode, temp, amp = 'inj', 0, 0
    base = scratch_structure.scratch + scratch_structure.base_name(mode, temp, amp) + '/'

    time, data, mag_err, freq = load_data(id)
    PriorAlterantive, PriorNull, log_prior_odds = prior.prepare(freq, redshift)
    data += signal(time, period, A_const, A_sin, A_cos)
    
    results= logB(time, data, mag_err, freq, PriorAlterantive.nlogp, PriorNull.nlogp, 
                    plot_name= str(id) + '.png' if plot else None) 

    return save(id, results, base, log_prior_odds, time)


def injected_lee3(params):
    id, redshift, key, temp, period, A_const, A_sin, A_cos = params
    mode, temp, amp = 'inj', temp, 0
    base = scratch_structure.scratch + scratch_structure.base_name(mode, temp, amp) + '/'

    time, data, mag_err, freq = load_data(id)
    PriorAlterantive, PriorNull, log_prior_odds = prior.prepare(freq, redshift)
    data += signal(time, period, A_const, A_sin, A_cos)
    
    results= logB(time, data, mag_err, freq, PriorAlterantive.nlogp, PriorNull.nlogp, 
                    temp_func= periodogram.randomized_period(key, 2000, concentration= 3.), 
                    plot_name= str(id) + '.png' if plot else None) 

    return save(id, results, base, log_prior_odds, time)

if __name__ == "__main__":

    info = pd.read_csv('data/twins.csv')
    id = np.array(info['best_match'], dtype= int)
    redshift = np.array(info['redshift'])
    period = np.array(info['period'])
    A_const = np.array(info['A_const'])
    A_sin = np.array(info['A_sin'])
    A_cos = np.array(info['A_cos'])
    
    temp = int(sys.argv[1])
    num_cores = 30
    mp.set_start_method('spawn')
    from time import time as tt
    t1 = tt()
    
    
    keys = jax.random.split(jax.random.key(42), len(id))
    
    params_transposed = [id, redshift, keys, [temp, ] * len(id), period, A_const, A_sin, A_cos]
    params = [[row[i] for row in params_transposed] for i in range(len(id))]
    
    if temp == 0:
        with mp.Pool(processes=num_cores) as pool:
            results = pool.imap_unordered(injected, params)

            for result in results: #useless, but otherwise multiprocessing doesn't think it is neccessary to actually run the previous line
                None        
    else:
        with mp.Pool(processes=num_cores) as pool:
            results = pool.imap_unordered(injected_lee3, params)

            for result in results: #useless, but otherwise multiprocessing doesn't think it is neccessary to actually run the previous line
                None        
    print((tt() - t1)/60.)