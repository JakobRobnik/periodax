import jax.numpy as jnp
from . import periodogram



drw_kernel = lambda sigma, tau: lambda t1, t2: jnp.square(sigma) * jnp.exp(-jnp.abs(t2-t1) / tau) # damped random walk kernel


def covariance(t, cov_func, errors):
    """Compute the covariance matrix 
        S_ij = cov_func(t_i, t_j) + delta_ij err(t_i)
    """
    t1, t2 = jnp.meshgrid(t, t)
    return cov_func(t1, t2) + jnp.diag(jnp.square(errors))


def multiband_covariance(t, cov_func, errors, drw_amp, sizes):
    _cov = covariance(t, cov_func, errors)
    A = jnp.block([[jnp.ones(shape= (sizes[i], sizes[j])) * drw_amp[i] * drw_amp[j] for j in range(len(sizes))] for i in range(len(sizes))])
    return _cov * A


def sqrt_multiband_drw_covariance(time, errors, params, sizes):
    *sigmas, tau = params
    cov = multiband_covariance(time, drw_kernel(1., tau), errors, drw_amp= sigmas, sizes= sizes)
    return jnp.linalg.cholesky(cov)




def nlog_density(time, data, err, band_mask, temp_func= periodogram.basic):
    """y = log z"""

    sizes = jnp.sum(band_mask, axis= 1).astype(int)

    def nloglik1(y):
        """ -log p(x | z)
            z = (frequency, sigmas, tau), phase and amplitude are maximized analytically
        """
        
        freq, *null_params = jnp.exp(y)

        # covariance matrix        
        sqrt_cov = sqrt_multiband_drw_covariance(time, err, null_params, sizes)
        
        # likelihood ratio (at maximal amplitudes) = log p(x|freq, null_params) / p(x|null_params)
        # note that this is not the maximum log-likelihood ratio, because params are not optimized for the null
        logp = 0.5* periodogram.lomb_scargle(time, data, sqrt_cov= sqrt_cov, temp_func= temp_func, const_only= False, band_mask= band_mask)(freq)[0]

        logdet = periodogram.log_determinant_term(sqrt_cov)
        return -logp - logdet
    

    def nloglik0(y):
        """z = (sigmas, tau)"""
        
        # covariance matrix        
        sqrt_cov = sqrt_multiband_drw_covariance(time, err, jnp.exp(y), sizes)
        
        # likelihood ratio (at maximal amplitudes) = log p(x|freq, null_params) / p(x|null_params)
        # note that this is not the maximum log-likelihood ratio, because params are not optimized for the null
        logp = 0.5* periodogram.lomb_scargle(time, data, sqrt_cov= sqrt_cov, temp_func= temp_func, const_only= True, band_mask= band_mask)(0.)[0]

        logdet = periodogram.log_determinant_term(sqrt_cov)
        return -logp - logdet
    
        
    return nloglik1, nloglik0

