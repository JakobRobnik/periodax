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




def noise(key, time, mag_err, generator_null):
    key1, key2 = jax.random.split(key)
    hyp = jnp.exp(generator_null(key1))
    cov = psd.covariance(time, psd.drw_kernel(*hyp), mag_err)
    return gauss_noise(key2, cov)


def signal(key, time, generator_logfreq):
    key1, key2 = jax.random.split(key)
    phase = jax.random.uniform(key1) * 2 * jnp.pi
    freq_injected = jnp.exp(generator_logfreq(key2))
    return jnp.sin(2 * jnp.pi * freq_injected * time + phase), 1./freq_injected


def save(id, results, base,
         log_prior_odds, data_points, period_injected= None):
    
    ### add to the results ###
    results['data_points'] = data_points
    results['log_prior_odds'] = log_prior_odds
    
    results['id'] = id
    
    if period_injected != None:
        results['period_injected'] = period_injected
    
    ### save the results ###
    df = pd.DataFrame(results, index= [0])
    print(df)
    df.to_csv(base + str(id) + '.csv', index= False)
    
    return True

    

def real(base, ids, amplitude, temp_func, keys):
    
    def mainn(job_id):
        id, key = ids[job_id], keys[job_id]
        time, data, mag_err, freq, redshift = load_data(id)
        PriorAlterantive, PriorNull, log_prior_odds = prior.prepare(freq, redshift)

        results= logB(time, data, mag_err, freq, PriorAlterantive.nlogp, PriorNull.nlogp, temp_func= temp_func(key))
        return save(id, results, base, log_prior_odds, len(data))

    return mainn


def sim(base, ids, amplitude, temp_func, keys):

    def mainn(job_id):
        id, key = ids[job_id], keys[job_id]
        time, _, mag_err, freq, redshift = load_data(id, remove_outliers= False, average_within_night= True)
        PriorAlterantive, PriorNull, log_prior_odds = prior.prepare(freq, redshift)
        
        key1, key2, key3 = jax.random.split(key, 3)
        model, period_injected = signal(key2, time, PriorAlterantive.rvs)
        data = noise(key1, time, mag_err, PriorNull.rvs) + amplitude * model
        
        results= logB(time, data, mag_err, freq, PriorAlterantive.nlogp, PriorNull.nlogp, temp_func= temp_func(key3))
        
        save(id, results, base, log_prior_odds, len(data), period_injected= period_injected)
    
    return mainn
    
# def injected(job_id):
#     time, _, mag_err, freq, redshift = load_data(ids[job_id], remove_outliers= False, average_within_night= True)
#     PriorAlterantive, PriorNull, log_prior_odds = prior.prepare(freq[0], redshift)

    
#     key1, key2, key3 = jax.random.split(key, 3)
#     model, period_injected = signal(key2, time, PriorAlterantive.rvs)
#     data = noise(key1, time, mag_err, PriorNull.rvs) + amplitude * model
#     results= logB(time, data, mag_err, freq, PriorAlterantive.nlogp, PriorNull.nlogp, temp_func= temp_func(key3))

#     save(job_id, results, log_prior_odds, len(data), id= ids[job_id], period_injected= period_injected)
    


def get_main(amplitudes, delta, ids):
        
    # settings of the script

    mode = sys.argv[3]
    temp = int(sys.argv[4])
    amp = int(sys.argv[5])
    
    base = scratch_structure.scratch + scratch_structure.base_name(mode, temp, amp)
    amplitude = amplitudes[amp]
        
    if temp == 0:
        print('yes')
        temp_func = lambda key: periodogram.basic

    else:
        temp_func = lambda key: periodogram.randomized_period(key, 2000, delta)


    # setup
    keys = jax.random.split(jax.random.PRNGKey(42), 10 * len(ids)).reshape(10, len(ids), 2)[temp]

    if mode == 'real':
        return real(base, ids, amplitude, temp_func, keys)

    elif mode == 'sim':
        return sim(base, ids, amplitude, temp_func, keys)
    else:
        raise ValueError("mode= " + mode + " is not a valid option. Should be 'real' or 'sim'.")    


def run_main(mainn, ids):
    
    start, finish = int(sys.argv[1]), int(sys.argv[2])
    finish = min(finish, len(ids))
    
    num_cores = 128
    mp.set_start_method('spawn')
    from time import time as tt
    t1 = tt()

    with mp.Pool(processes=num_cores) as pool:
        results = pool.imap_unordered(mainn, np.arange(start, finish))

        for result in results: #useless, but otherwise multiprocessing doesn't think it is neccessary to actually run the previous line
            None
        
    print((tt() - t1)/60.)


if __name__ == "__main__":
    # parameters to the script:
    #  start, finish: integers, quasar_ids[start:finish] will be processed. If finish is larger than the number of quasars, it will be set to the number of quasars
    #  mode: 'real' or 'sim' standing for analysis of the real data or simulations
    #  temp: integer. If 0, the real (sinusoidal) template will be used. If non-negative, period will be randomized and different integers correspond to different realizations
    #  amp: amplitudes[amp] will be the amplitude of the injected signal. Should be 0, if no signal is to be injected.

    ids = np.load(scratch_structure.dir_data + 'ids.npy')

    mainn = get_main(amplitudes = [0.0, 0.1, 0.2, 0.3, 0.4], delta= 2., ids= ids)
    mainn(10984)
    #mainn(0)
    #run_main(mainn, ids)