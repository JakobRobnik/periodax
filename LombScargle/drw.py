import jax.numpy as jnp
from . import periodogram



kernel = lambda sigma, tau: lambda t1, t2: jnp.square(sigma) * jnp.exp(-jnp.abs(t2-t1) / tau) # damped random walk kernel


def covariance(t, cov_func, errors):
    """Compute the covariance matrix 
        S_ij = cov_func(t_i, t_j) + delta_ij err(t_i)
    """
    t1, t2 = jnp.meshgrid(t, t)
    return cov_func(t1, t2) + jnp.diag(jnp.square(errors))



def nlog_density(time, data, err, nlogpr_logfreq, nlogpr0, floating_mean= True, temp_func= periodogram.basic):
    """y = log z"""
    
    def nloglik1(y):
        """ -log p(x | z)
            z = (frequency, sigma, tau), phase and amplitude ar maximized analytically
        """
        
        freq, sigma, tau = jnp.exp(y)
        
        # covariance matrix
        cov = covariance(time, kernel(sigma, tau), err)
        sqrt_cov = jnp.linalg.cholesky(cov)
        
        # likelihood ratio (at maximal amplitudes) = log p(x|freq, null_params) / p(x|null_params)
        # note that this is not the maximum log-likelihood ratio, because params are not optimized for the null
        logp_ratio = 0.5* periodogram.lomb_scargle(time, data, floating_mean= floating_mean, sqrt_cov= sqrt_cov, temp_func= temp_func)(freq)[0]
        
        # eliminate p(x | null_params)
        loglik0 = periodogram.loglik_null(data, sqrt_cov)
        return -logp_ratio - loglik0
    
    def nloglik0(y):
        """z = (sigma, tau)"""
        sigma, tau = jnp.exp(y)
        cov = covariance(time, kernel(sigma, tau), err)
        sqrt_cov = jnp.linalg.cholesky(cov)
        return -periodogram.loglik_null(data, sqrt_cov)
    
    
    def nlogpost0(y):
        return nlogpr0(y) + nloglik0(y)

    def nlogpost1(y):
        return nlogpr0(y[1:]) + nlogpr_logfreq(y[0]) + nloglik1(y)
    
    def get_amp(y):
        
        freq, sigma, tau = jnp.exp(y)
        
        # covariance matrix
        cov = covariance(time, kernel(sigma, tau), err)
        sqrt_cov = jnp.linalg.cholesky(cov)
        
        return periodogram.lomb_scargle(time, data, floating_mean= floating_mean, sqrt_cov= sqrt_cov, temp_func= temp_func)(freq)[1]
        
        
        
    return nlogpost0, nlogpost1, nloglik0, nloglik1, get_amp
