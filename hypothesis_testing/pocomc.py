import jax
import jax.numpy as jnp
from scipy.stats import norm
import numpy as np
import pandas as pd

import pocomc

from quasars import prior, prep
from LombScargle import periodogram, psd
from hypothesis_testing.bayes_factor import optimize, quadrature, scheme_d2



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
    
    ### analyze the null model ###
    y_init = jnp.log(jnp.array([0.1, 120])) # mode of the prior distribution
    map0 = optimize(nlogpost0, y_init)
    log_ev0 = quadrature(nlogpost0, map0, scheme_d2)
    
    results['log_ev_quad'] = log_ev0
    
    results.to_csv('pocoMC_results.csv', sep= '\t', index= False)
    
    
    
if __name__ == '__main__':
    
    id = 100051
    time, data, mag_err, freq = prep.prepare_data(id)
    
    PriorAlterantive = prior.SmoothenEdge(prior.PowerLaw(-11./3., freq[0], freq[-1]), 1.2)
    PriorNull = prior.Normal()
    
    logb, s1, s2 = logB(time, data, mag_err, freq, PriorAlterantive.nlogp, PriorNull.nlogp)
    
