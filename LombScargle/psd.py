import jax.numpy as jnp
from LombScargle import periodogram



drw_kernel = lambda sigma, tau: lambda t1, t2: jnp.square(sigma) * jnp.exp(-jnp.abs(t2-t1) / tau)


def covariance(t, cov_func, errors):
    """Compute the covariance matrix S_ij = cov_func(t_i, t_j) + delta_ij err(t_i)"""
    t1, t2 = jnp.meshgrid(t, t)
    return cov_func(t1, t2) + jnp.diag(jnp.square(errors))



def nlog_density(time, data, err, fmin, fmax, floating_mean= True):
    
    
    def likelihood_alternative(y):
        """ -log p(x | z)
            z = (frequency, sigma, tau), phase and amplitude ar maximized analytically
        """
        
        freq, sigma, tau = jnp.exp(y)
        
        # covariance matrix
        cov = covariance(time, drw_kernel(sigma, tau), err)
        sqrt_cov = jnp.linalg.cholesky(cov)
        
        # likelihood ratio (at maximal amplitudes) = log p(x|freq, params) / p(x|params)
        # note that this is not the maximum log-likelihood ratio, because params are not separately optimize under the null
        nlogp_ratio = 0.5* periodogram.func(time, data, floating_mean= floating_mean, sqrt_cov= sqrt_cov)(freq)[0]
        
        # eliminate p(x | params)
        logp_null = periodogram.log_prob_null(data, sqrt_cov)
        return nlogp_ratio - logp_null
    
    def likelihood_null(y):
        """z = parameters of psd"""        
        sigma, tau = jnp.exp(y)
        cov = covariance(time, drw_kernel(sigma, tau), err)
        sqrt_cov = jnp.linalg.cholesky(cov)
        return -periodogram.log_prob_null(data, sqrt_cov)
    
    
    def prior_null(y):
        log_mu = jnp.log(jnp.array([0.1, 120.]))
        sigma = jnp.array([0.2, 0.9])
        return jnp.sum(0.5 * jnp.square((y - log_mu)/sigma) + 0.5 * jnp.log(2 * jnp.pi * jnp.square(sigma)))    
        
    def prior_alternative(y):
        return prior_null(y[1:]) + jnp.log(jnp.log(fmax/fmin))

    def posterior_null(y):
        return prior_null(y) + likelihood_null(y)

    def posterior_alternative(y):
        return prior_alternative(y) + likelihood_alternative(y)
    
    return posterior_null, posterior_alternative


# def nlogpr(prior_z):
#     return lambda y: - jnp.log(prior_z(jnp.exp(y))) - jnp.sum(y)


# def logNormal(z):
#     y = jnp.log(z)
#     log_mu = jnp.log(jnp.array([0.1, 120.]))
#     sigma = jnp.array([0.2, 0.9])
    
#     return jnp.sum(0.5 * jnp.square((y - log_mu)/sigma) + 0.5 * jnp.log(2 * jnp.pi * jnp.square(sigma)) + y)
