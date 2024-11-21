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
amplitudes = [0.0, 0.08, 0.2, 0.3, 0.4]


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



def bernoulli_with_odds(key, log_odds, shape= None):
    """odds = p / (1-p)
        p= 1 / (1 + 1/odds) = 1 / (1 + exp(-log_odds)) = sigmoid(log_odds)
    """
    return jax.random.bernoulli(key, p= jax.nn.sigmoid(log_odds), shape= shape)
    
    

def real(params):
    id, redshift = params
    mode, temp, amp = 'real', 0, 0
    base = scratch_structure.scratch + scratch_structure.base_name(mode, temp, amp) + '/'

    time, data, mag_err, freq = load_data(id)
    PriorAlterantive, PriorNull, log_prior_odds = prior.prepare(freq, redshift)
    results= logB(time, data, mag_err, freq, PriorAlterantive.nlogp, PriorNull.nlogp, 
                    plot_name= str(id) + '.png' if plot else None) 

    return save(id, results, base, log_prior_odds, time)


def real_nst(params):
    id, redshift, key, temp, amp = params

    mode = 'real'
    base = scratch_structure.scratch + scratch_structure.base_name(mode, temp, amp) + '/'

    time, data, mag_err, freq = load_data(id)
    PriorAlterantive, PriorNull, log_prior_odds = prior.prepare(freq, redshift)
    
    results= logB(time, data, mag_err, freq, PriorAlterantive.nlogp, PriorNull.nlogp, 
                    temp_func= periodogram.null_signal_template(key, 2000), 
                    plot_name= str(id) + '.png' if plot else None) 

    return save(id, results, base, log_prior_odds, time)


def sim(params):
    id, redshift, key, key_num, amp = params
    amplitude = amplitudes[amp]
    mode = 'sim'
    base = scratch_structure.scratch + scratch_structure.base_name(mode, key_num, amp) + '/'
    time, _, mag_err, freq = load_data(id)
    PriorAlterantive, PriorNull, log_prior_odds = prior.prepare(freq, redshift)
    
    # simulate the data
    key_noise, key_signal, key_inject = jax.random.split(key, 3)
    inject = bernoulli_with_odds(key_inject, log_prior_odds) # True if signal is injected
    model, period_injected = signal(key_signal, time, PriorAlterantive.rvs)
    data = noise(key_noise, time, mag_err, PriorNull.rvs) + inject * amplitude * model
    
    
    results= logB(time, data, mag_err, freq, PriorAlterantive.nlogp, PriorNull.nlogp, 
                    plot_name= None)
    
    save(id, results, base, log_prior_odds, time, period_injected= period_injected if inject else np.NAN)


def sim_nst(params):
    id, redshift, key, key_num, amp = params
    amplitude = amplitudes[amp]
    mode = 'sim'
    base = scratch_structure.scratch + scratch_structure.base_name(mode, key_num, amp) + '/'
    time, _, mag_err, freq = load_data(id)
    PriorAlterantive, PriorNull, log_prior_odds = prior.prepare(freq, redshift)
    
    # simulate the data
    key_noise, key_signal, key_inject, key_nst = jax.random.split(key, 4)
    inject = bernoulli_with_odds(key_inject, log_prior_odds) # True if signal is injected
    model, period_injected = signal(key_signal, time, PriorAlterantive.rvs)
    data = noise(key_noise, time, mag_err, PriorNull.rvs) + inject * amplitude * model
    
    
    results= logB(time, data, mag_err, freq, PriorAlterantive.nlogp, PriorNull.nlogp, 
                    temp_func= periodogram.null_signal_template(key_nst, 2000), 
                    plot_name= None)
    
    save(id, results, base, log_prior_odds, time, period_injected= period_injected if inject else np.NAN)


if __name__ == "__main__":
    # parameters to the script:
    #  start, finish: integers, quasar_ids[start:finish] will be processed. Used only in run_main. If finish is larger than the number of quasars, it will be set to the number of quasars
    #  mode: 'real' or 'sim' standing for analysis of the real data or simulations
    #  temp: integer. If 0, the real (sinusoidal) template will be used. If non-negative, period will be randomized and different integers correspond to different realizations
    #  amp: amplitudes[amp] will be the amplitude of the injected signal. Should be 0, if no signal is to be injected.

    quasar_info = pd.read_csv(scratch_structure.dir_data + 'quasar_info.txt', delim_whitespace= True)
    ids, redshifts = np.array(quasar_info['id']), np.array(quasar_info['redshift'])   
    
    mode = sys.argv[3]
    key_num = int(sys.argv[4])
    amp = int(sys.argv[5])
            
    if key_num >= 1000: # key numbers above (and including) 1000 represent null template
        nst, subtract = True, 1000
    else:
        nst, subtract = False, 0
    
    start, finish = int(sys.argv[1]), int(sys.argv[2])
    finish = min(finish, len(ids))
    
    num_cores = 128
    mp.set_start_method('spawn')
    from time import time as tt
    t1 = tt()
    
    
    id = ids[start:finish]
    keys = jax.random.split(jax.random.key(42), 100 * len(ids)).reshape(100, len(ids))[key_num-subtract][start:finish]  #for drw simulations
    #keys = jax.random.split(jax.random.key(42), 10 * len(ids)).reshape(10, len(ids))[key_num-subtract][start:finish]  # if you change this, change also in analyze.ipynb
    
    if mode == 'sim': # simulations
        params_transposed = [id, redshifts, keys, [key_num, ] * len(id), [amp, ] * len(id)]
        params = [[row[i] for row in params_transposed] for i in range(len(id))]
        
        if not nst: # real template
            with mp.Pool(processes=num_cores) as pool:
                results = pool.imap_unordered(sim, params)

                for result in results: #useless, but otherwise multiprocessing doesn't think it is neccessary to actually run the previous line
                    None 
        else: # nst template
            with mp.Pool(processes=num_cores) as pool:
                results = pool.imap_unordered(sim_nst, params)

                for result in results: #useless, but otherwise multiprocessing doesn't think it is neccessary to actually run the previous line
                    None
    else: # real data 
        if not nst: # real template
            
            params_transposed = [id, redshifts]
            params = [[row[i] for row in params_transposed] for i in range(len(id))]
            
            with mp.Pool(processes=num_cores) as pool:
                results = pool.imap_unordered(real, params)

                for result in results: #useless, but otherwise multiprocessing doesn't think it is neccessary to actually run the previous line
                    None
        
        
        else: # nst template
            params_transposed = [id, redshifts, keys, [key_num, ] * len(id), [amp, ] * len(id)]
            params = [[row[i] for row in params_transposed] for i in range(len(id))]
            
            with mp.Pool(processes=num_cores) as pool:
                results = pool.imap_unordered(real_nst, params)

                for result in results: #useless, but otherwise multiprocessing doesn't think it is neccessary to actually run the previous line
                    None        
        
    print((tt() - t1)/60.)