import jax
import jax.numpy as jnp
from scipy.stats import norm
import numpy as np
import pandas as pd

import pocomc

from quasars import prior, prep
from LombScargle import periodogram, psd
from hypothesis_testing.bayes_factor import optimize, quadrature, scheme_d2
from simulations.util import gauss_noise
    


def null_prior_poco():
    mu = np.log(np.array([0.1, 120.]))
    sigma = np.array([0.2, 0.9])
    return pocomc.Prior([norm(loc=mu[i], scale=sigma[i]) for i in range(2)])


def run_pocomc(nlog_lik):
    prior = null_prior_poco()
    # Initialise sampler
    sampler = pocomc.Sampler(
        prior=prior,
        likelihood= lambda x: -nlog_lik(jnp.array(x)),
        vectorize=False,
        random_state=0
    )
    sampler.run()

    # Get the evidence and its uncertainty
    log_ev0, log_ev0_err = sampler.evidence()
    print(sampler.results)
    results = pd.DataFrame.from_dict(sampler.results, index= [0])
    
    return log_ev0, log_ev0_err, results
    
    

def logB(time, data, err_data, freq, nlogpr_logfreq, nlogpr_null, floating_mean= True, temp_func= periodogram.basic):
    
    nlogpost0, nlogpost1, nloglik0, nloglik1, get_amp = psd.nlog_density(time, data, err_data, nlogpr_logfreq, nlogpr_null, floating_mean, temp_func)
  
    ### analyze the null model ###
   
    # pocoMC
    log_ev, log_ev_err, results = run_pocomc(nloglik0)
    
    results['log_ev_pocoMC'] = log_ev
    results['log_ev_err_pocoMC'] = log_ev_err
    
    # quadrature
    y_init = jnp.log(jnp.array([0.1, 120])) # mode of the prior distribution
    map0, success_opt = optimize(nlogpost0, y_init)
    log_ev0, success_quad = quadrature(nlogpost0, map0, scheme_d2)
    
    results['log_ev_quad'] = log_ev0
    
    print(results)
    
    
def gauss(key, time, mag_err):
    sigma, tau = 0.1, 120.
    kernel=  psd.drw_kernel(sigma= sigma, tau= tau)
    cov = psd.covariance(time, kernel, mag_err)
    return gauss_noise(key, cov)


def sim():
    
    # load the time stamps and errors of the real data
    df= pd.read_csv('data/100051.csv')
    time = jnp.array(df['time'])
    mag_err = jnp.array(df['err'])
    redshift= 0.

    # simulate the data
    mag= gauss(key= jax.random.PRNGKey(42), time= time, mag_err= mag_err)
    
    # determine the frequency grid
    T = jnp.max(time) - jnp.min(time)
    freq_bounds = jnp.array([2./T, 1./60.])
    freq = jnp.logspace(*jnp.log10(freq_bounds), 1000)
  
    
    return time, mag, mag_err, freq, redshift
    
    
if __name__ == '__main__':
    
    time, mag, mag_err, freq, redshift = sim()
    
    PriorAlterantive, PriorNull, log_prior_odds = prior.prepare(freq, redshift)
    
    logB(time, mag, mag_err, freq, PriorAlterantive.nlogp, PriorNull.nlogp)
    
