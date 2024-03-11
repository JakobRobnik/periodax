import jax.numpy as jnp
from LombScargle import periodogram



def log_likelihood(time, data, cov_func, err):
    
    
    def loglik_alternative(z):
        """ log p(x | z)
        z = (frequency, *parameters of psd), phase and amplitude ar maximized analytically
        """
        # covariance matrix
        cov = periodogram.covariance(time, cov_func(*z[1:])) + jnp.diag(jnp.square(err))
        sqrt_cov = jnp.linalg.cholesky(cov)
        
        # likelihood ratio (at maximal amplitudes) = log p(x|freq, params) / p(x|params)
        # note that this is not the maximum log-likelihood ratio, because params are not separately optimize under the null
        periodogram_score = periodogram.func(time, data, floating_mean= False, sqrt_cov= sqrt_cov)(z[0])
        log_det = jnp.sum(jnp.log(jnp.square(jnp.diag(sqrt_cov))))
        logp_ratio = -0.5 * periodogram_score - 0.5 * log_det 
        
        # eliminate p(x | params)
        logp_null = periodogram.log_prob_null(data, sqrt_cov)
        return logp_ratio + logp_null
    
    def loglik_null(z):
        """z = parameters of psd"""
        cov = periodogram.covariance(time, cov_func(*z)) + jnp.diag(jnp.square(err))
        sqrt_cov = jnp.linalg.cholesky(cov)
        return periodogram.log_prob_null(data, sqrt_cov)
    
    
    return loglik_alternative, loglik_null


